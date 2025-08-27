"""SmartLatent

Create an empty latent or encode an image into a 64-aligned latent with VRAM guardrails.
Adds snap-to-64 policies (pad/downscale/resize/crop) and returns bbox/dims metadata.

This version is robust to any input size (tiny/huge/odd ARs/alpha/batches). It:
- snaps to 64-multiples with selectable strategies,
- pads safely with reflect when legal, else replicate/constant automatically,
- handles alpha cleanly (RGB vs A),
- supports batches where all frames share H×W,
- emits consistent bbox_json + dims_json + width + height.
"""

import json
import math

import torch
import torch.nn.functional as F


# ------------------------------
# Helpers (general, reusable)
# ------------------------------
def _round_up_64(x: int) -> int:
    return max(64, ((int(x) + 63) // 64) * 64)


def _round_down_64(x: int) -> int:
    return max(64, (int(x) // 64) * 64)


def _to_bhwc(img: torch.Tensor) -> torch.Tensor:
    """Accept HWC or BHWC; return BHWC. If BCHW, convert to BHWC.

    Comfy images are usually BHWC in [0,1]. Some nodes may pass BCHW; we coerce.
    """
    if img is None:
        raise ValueError("SmartLatent: image is None")
    if img.ndim == 3:
        # Heuristic: 3D likely HWC
        return img.unsqueeze(0).contiguous()
    if img.ndim != 4:
        raise ValueError(f"SmartLatent: unexpected image tensor ndim = {img.ndim}")
    B, D1, D2, D3 = img.shape
    # If last dim looks like channels (1/3/4), assume BHWC
    if D3 in (1, 3, 4):
        return img.contiguous()
    # Else if second dim looks like channels, assume BCHW -> convert
    if D1 in (1, 3, 4):
        return img.permute(0, 2, 3, 1).contiguous()
    # Fallback: assume BHWC
    return img.contiguous()


def _bhwc_to_bchw(img_bhwc: torch.Tensor) -> torch.Tensor:
    return img_bhwc.permute(0, 3, 1, 2).contiguous()


def _bchw_to_bhwc(img_bchw: torch.Tensor) -> torch.Tensor:
    return img_bchw.permute(0, 2, 3, 1).contiguous()


def _resize_bhwc(img_bhwc: torch.Tensor, W: int, H: int) -> torch.Tensor:
    bchw = _bhwc_to_bchw(img_bhwc)
    bchw = F.interpolate(bchw, size=(H, W), mode="bilinear", align_corners=False)
    return _bchw_to_bhwc(bchw)


def _safe_pad_bchw(
    bchw: torch.Tensor, pad_lrtb: tuple, pad_mode: str, pad_value_float: float
) -> torch.Tensor:
    """Pad BCHW on spatial dims with reflect/replicate/constant, auto-fallback if reflect is illegal.

    If alpha present (C>3): pad RGB with requested mode (fallback replicate),
    and pad alpha with replicate/constant to avoid halos.
    """
    left_pad, right_pad, top_pad, bottom_pad = [int(max(0, v)) for v in pad_lrtb]
    if left_pad == right_pad == top_pad == bottom_pad == 0:
        return bchw
    C = int(bchw.shape[1])
    if C > 3:
        rgb, a = bchw[:, :3], bchw[:, 3:]
        try:
            if pad_mode == "reflect":
                rgb = F.pad(rgb, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect")
            elif pad_mode in ("replicate", "edge"):
                rgb = F.pad(rgb, (left_pad, right_pad, top_pad, bottom_pad), mode="replicate")
            else:
                rgb = F.pad(
                    rgb,
                    (left_pad, right_pad, top_pad, bottom_pad),
                    mode="constant",
                    value=pad_value_float,
                )
        except RuntimeError:
            # reflect may fail if pad >= size; fallback to replicate
            rgb = F.pad(rgb, (left_pad, right_pad, top_pad, bottom_pad), mode="replicate")
        # alpha: prefer replicate/constant (never reflect)
        a = F.pad(
            a,
            (left_pad, right_pad, top_pad, bottom_pad),
            mode=("replicate" if pad_mode not in ("constant",) else "constant"),
            value=pad_value_float,
        )
        return torch.cat([rgb, a], dim=1)
    else:
        try:
            if pad_mode == "reflect":
                return F.pad(bchw, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect")
            elif pad_mode in ("replicate", "edge"):
                return F.pad(bchw, (left_pad, right_pad, top_pad, bottom_pad), mode="replicate")
            else:
                return F.pad(
                    bchw,
                    (left_pad, right_pad, top_pad, bottom_pad),
                    mode="constant",
                    value=pad_value_float,
                )
        except RuntimeError:
            # reflect illegal → replicate
            return F.pad(bchw, (left_pad, right_pad, top_pad, bottom_pad), mode="replicate")


def _compute_snap(target_mode: str, W0: int, H0: int) -> tuple[int, int, str, tuple]:
    """
    Returns (W, H, op, pad_lrtb) for working dims and how to get there.
    target_mode:
      - "pad_up"        : keep content, letterbox up to next 64
      - "downscale_only": fit-inside downscale to <= nearest 64, then pad residuals if any
      - "resize_round"  : direct resize to nearest 64 (may distort AR)
      - "crop_center"   : center-crop down to lower 64 (no resize)
    op in {"pad","resize","crop","pad+resize"} for debugging.
    """
    if target_mode == "pad_up":
        W, H = _round_up_64(W0), _round_up_64(H0)
        left_pad = max(0, (W - W0) // 2)
        right_pad = max(0, W - W0 - left_pad)
        top_pad = max(0, (H - H0) // 2)
        bottom_pad = max(0, H - H0 - top_pad)
        return W, H, "pad", (left_pad, right_pad, top_pad, bottom_pad)

    if target_mode == "downscale_only":
        W_fit, H_fit = _round_down_64(W0), _round_down_64(H0)
        W_fit = max(64, W_fit)
        H_fit = max(64, H_fit)
        # scale to fit inside (W_fit, H_fit) while preserving AR
        sx = W_fit / max(1, W0)
        sy = H_fit / max(1, H0)
        s = min(sx, sy)
        Wr = max(1, int(round(W0 * s)))
        Hr = max(1, int(round(H0 * s)))
        # small residuals → pad
        left_pad = max(0, (W_fit - Wr) // 2)
        right_pad = max(0, W_fit - Wr - left_pad)
        top_pad = max(0, (H_fit - Hr) // 2)
        bottom_pad = max(0, H_fit - Hr - top_pad)
        return W_fit, H_fit, "pad+resize", (left_pad, right_pad, top_pad, bottom_pad)

    if target_mode == "resize_round":
        W, H = _round_down_64(W0 + 32), _round_down_64(H0 + 32)  # nearest-ish
        W = max(64, W)
        H = max(64, H)
        return W, H, "resize", (0, 0, 0, 0)

    if target_mode == "crop_center":
        W, H = _round_down_64(W0), _round_down_64(H0)
        return W, H, "crop", (0, 0, 0, 0)

    # default pad_up
    W, H = _round_up_64(W0), _round_up_64(H0)
    left_pad = max(0, (W - W0) // 2)
    right_pad = max(0, W - W0 - left_pad)
    top_pad = max(0, (H - H0) // 2)
    bottom_pad = max(0, H - H0 - top_pad)
    return W, H, "pad", (left_pad, right_pad, top_pad, bottom_pad)


def _round64(x: int) -> int:
    return max(64, int(math.floor(x / 64) * 64))


def _safe_dims(width: int, height: int, max_pixels: int) -> tuple[int, int, int]:
    """Return (W,H,downscale_steps) downscaled by 64 blocks until under max_pixels."""
    W, H = _round64(width), _round64(height)
    steps = 0
    while (W * H) > max_pixels:
        W = max(64, W - 64)
        H = max(64, H - 64)
        steps += 1
    return W, H, steps


# --- robust VAE tiled helpers -----------------------------------------------
def vae_encode_smart(
    vae, img_bhwc, use_tiled: bool = True, tile_size: int = 320, tile_overlap: int = 32
):
    """Tries multiple encode_tiled signatures, else falls back to vae.encode(img).

    img_bhwc: float BHWC in [0,1]
    """
    if not use_tiled or not hasattr(vae, "encode_tiled"):
        return vae.encode(img_bhwc)

    try:
        # positional (img, tile_size)
        return vae.encode_tiled(img_bhwc, tile_size)
    except TypeError:
        pass
    try:
        # positional with overlap (img, tile_size, tile_overlap)
        return vae.encode_tiled(img_bhwc, tile_size, tile_overlap)
    except TypeError:
        pass
    try:
        # keyword: tile_size
        return vae.encode_tiled(img_bhwc, tile_size=tile_size)
    except TypeError:
        pass
    try:
        # keyword: tile_x/tile_y + overlap
        return vae.encode_tiled(img_bhwc, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
    except TypeError:
        pass
    try:
        # keyword: tile_size + tile_overlap
        return vae.encode_tiled(img_bhwc, tile_size=tile_size, tile_overlap=tile_overlap)
    except TypeError:
        pass

    return vae.encode(img_bhwc)


def vae_decode_smart(
    vae, latent, use_tiled: bool = True, tile_size: int = 320, tile_overlap: int = 32
):
    """Tries multiple decode_tiled signatures, else falls back to vae.decode(latent).

    latent: {"samples": BCHW}
    """
    if not use_tiled or not hasattr(vae, "decode_tiled"):
        return vae.decode(latent)

    try:
        return vae.decode_tiled(latent, tile_size)
    except TypeError:
        pass
    try:
        return vae.decode_tiled(latent, tile_size, tile_overlap)
    except TypeError:
        pass
    try:
        return vae.decode_tiled(latent, tile_size=tile_size)
    except TypeError:
        pass
    try:
        return vae.decode_tiled(latent, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
    except TypeError:
        pass
    try:
        return vae.decode_tiled(latent, tile_size=tile_size, tile_overlap=tile_overlap)
    except TypeError:
        pass

    return vae.decode(latent)


class SmartLatent:
    """Build or encode latents with safe 64-multiple snapping and dims metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "VAE model used to create or encode latents."}),
                "mode": (
                    ["empty", "encode_image"],
                    {"tooltip": "Create an empty latent or encode an input image."},
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Requested width; may downscale by 64s to respect max_pixels.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Requested height; may downscale by 64s to respect max_pixels.",
                    },
                ),
            },
            "optional": {
                "snap_mode": (
                    ["pad_up", "downscale_only", "resize_round", "crop_center"],
                    {
                        "default": "pad_up",
                        "tooltip": "Snap policy for non-64 dims: pad, downscale, resize, or crop.",
                    },
                ),
                "pad_kind": (
                    ["reflect", "edge", "replicate", "constant"],
                    {
                        "default": "reflect",
                        "tooltip": "Padding type for pad_up/downscale (reflect avoids seams). 'edge' behaves like 'replicate'.",
                    },
                ),
                "pad_value": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 255,
                        "tooltip": "Pad value (constant mode only).",
                    },
                ),
                "batch": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "tooltip": "Batch size for empty latent or image batch.",
                    },
                ),
                "image": ("IMAGE", {"tooltip": "Image tensor when mode = encode_image."}),
                "tile_size": (
                    "INT",
                    {
                        "default": 320,
                        "min": 192,
                        "max": 512,
                        "step": 32,
                        "tooltip": "Tile size for VAE.encode_tiled if available.",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Tile overlap for tiled VAE encode/decode.",
                    },
                ),
                "force_bchw": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Ensure image is [B,C,H,W] before encoding."},
                ),
                "use_tiled": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use tiled VAE encode/decode when available."},
                ),
                "seed": ("INT", {"default": 0, "tooltip": "Reserved for future use."}),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1024 * 1024,
                        "min": 256 * 256,
                        "max": 4096 * 4096,
                        "step": 64,
                        "tooltip": "Upper bound on W*H; uniformly downscale to fit if exceeded.",
                    },
                ),
                "max_long_side": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Optional VRAM guard: if >0, downscale input image so max(H,W) <= this before snapping.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "dims_json", "width", "height", "bbox_json")
    FUNCTION = "build"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def build(
        self,
        vae,
        mode,
        width,
        height,
        batch=1,
        image=None,
        tile_size=320,
        tile_overlap=32,
        force_bchw=True,
        use_tiled=True,
        seed=0,
        max_pixels=1024 * 1024,
        snap_mode="pad_up",
        pad_kind="reflect",
        pad_value=128,
        max_long_side=0,
    ):
        # Handle accidental string inputs from UI (e.g., empty or words)
        def _to_int(v, default):
            try:
                if v is None:
                    return default
                if isinstance(v, str):
                    v = v.strip()
                    if not v or not v.lstrip("-+").isdigit():
                        return default
                return int(v)
            except Exception:
                return default

        def _to_bool(v, default=False):
            try:
                if hasattr(v, "numel") and callable(v.numel):
                    try:
                        if v.numel() == 1 and hasattr(v, "item"):
                            return bool(v.item())
                        return bool(v.numel())
                    except Exception:
                        return default
                if isinstance(v, (list, tuple)):
                    return any(_to_bool(x, False) for x in v)
                return bool(v)
            except Exception:
                return default

        # Coerce inputs
        width = _to_int(width, 1024)
        height = _to_int(height, 1024)
        batch = _to_int(batch, 1)
        tile_size = _to_int(tile_size, 320)
        tile_overlap = _to_int(tile_overlap, 32)
        seed = _to_int(seed, 0)
        max_pixels = _to_int(max_pixels, 1024 * 1024)
        max_long_side = _to_int(max_long_side, 0)
        force_bchw = _to_bool(force_bchw, True)
        use_tiled = _to_bool(use_tiled, True)
        snap_mode = str(snap_mode or "pad_up")
        pad_kind = str(pad_kind or "reflect")
        pad_value = _to_int(pad_value, 128)

        # Pre-seed dims_json baseline
        W0_req, H0_req, down = _safe_dims(width, height, max_pixels)
        info = {
            "mode": mode,
            "W": W0_req,
            "H": H0_req,
            "batch": batch,
            "tile": tile_size,
            "down_by_64": down,
        }

        if mode == "empty":
            # allocate latent zeros (Comfy expects a dict with 'samples')
            latent = torch.zeros([batch, 4, H0_req // 8, W0_req // 8], device="cpu")
            bbox = {"x": 0, "y": 0, "w": W0_req, "h": H0_req, "W": W0_req, "H": H0_req}
            return ({"samples": latent}, json.dumps(info), W0_req, H0_req, json.dumps(bbox))

        # encode_image branch (generic; handles any size)
        assert image is not None, "SmartLatent: image input required in encode_image mode"

        # Accept HWC/BHWC/BCHW; convert to BHWC for VAE
        img = _to_bhwc(image)
        B, H0, W0 = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])

        # Optional VRAM guard: pre-limit max long side before snapping
        if max_long_side and max(H0, W0) > max_long_side:
            s = float(max_long_side) / float(max(H0, W0))
            img = _resize_bhwc(img, int(round(W0 * s)), int(round(H0 * s)))
            B, H0, W0 = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])

        # Compute working dims + plan
        W, H, op, pad_lrtb = _compute_snap(snap_mode, W0, H0)

        # Execute plan
        work = img
        if op in ("resize", "pad+resize"):
            # For pad+resize, Wr/Hr = target after resize; compute them:
            Wr = W - (pad_lrtb[0] + pad_lrtb[1])
            Hr = H - (pad_lrtb[2] + pad_lrtb[3])
            work = _resize_bhwc(work, Wr, Hr)

        if op in ("pad", "pad+resize"):
            # pad on BCHW
            bchw = _bhwc_to_bchw(work)
            pad_mode = (
                pad_kind if pad_kind in ("reflect", "replicate", "edge", "constant") else "reflect"
            )
            work = _bchw_to_bhwc(_safe_pad_bchw(bchw, pad_lrtb, pad_mode, float(pad_value) / 255.0))

        if op == "crop":
            # center crop down to W×H
            x0 = max(0, (W0 - W) // 2)
            y0 = max(0, (H0 - H) // 2)
            work = work[:, y0 : y0 + H, x0 : x0 + W, :]

        # Encode (tiled if supported)
        latent = vae_encode_smart(
            vae, work, use_tiled=use_tiled, tile_size=tile_size, tile_overlap=tile_overlap
        )

        # Validate latent shape vs working dims
        samples = latent["samples"] if isinstance(latent, dict) and "samples" in latent else latent
        out_latent = (
            latent if isinstance(latent, dict) and "samples" in latent else {"samples": latent}
        )
        _, _, h8, w8 = (
            int(samples.shape[0]),
            int(samples.shape[1]),
            int(samples.shape[2]),
            int(samples.shape[3]),
        )
        assert w8 * 8 == W and h8 * 8 == H, f"latent mismatch: {(w8*8, h8*8)} vs {(W,H)}"

        # Build bbox for crop-back (record original content region within working)
        left_pad, right_pad, top_pad, bottom_pad = pad_lrtb
        if op == "pad":
            x_off, y_off, w_cont, h_cont = left_pad, top_pad, W0, H0
        elif op == "pad+resize":
            Wr = W - (left_pad + right_pad)
            Hr = H - (top_pad + bottom_pad)
            x_off, y_off, w_cont, h_cont = left_pad, top_pad, Wr, Hr
        elif op in ("resize", "crop"):
            x_off, y_off, w_cont, h_cont = 0, 0, W, H
        else:
            x_off, y_off, w_cont, h_cont = 0, 0, W, H

        bbox = {
            "x": int(x_off),
            "y": int(y_off),
            "w": int(w_cont),
            "h": int(h_cont),
            "W": int(W),
            "H": int(H),
        }
        info = {
            "mode": "encode_image",
            "op": op,
            "pad": [int(x) for x in pad_lrtb],
            "W0": int(W0),
            "H0": int(H0),
            "W": int(W),
            "H": int(H),
            "batch": int(B),
            "tile": int(tile_size),
            "tile_overlap": int(tile_overlap),
            "use_tiled": bool(use_tiled),
        }
        return (out_latent, json.dumps(info), W, H, json.dumps(bbox))
