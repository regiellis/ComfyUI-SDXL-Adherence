"""AlignHintsToLatent

Snap hint images (canny/depth/lineart/etc.) to the latent's exact W×H with
padding, downscale, resize, or crop, keeping hints 1:1 with the UNet grid.
Emits bbox metadata to optionally crop back later.
"""
import json

import torch.nn.functional as F


def _latent_dims(latent):
    t = latent["samples"] if isinstance(latent, dict) else latent
    h8, w8 = t.shape[-2], t.shape[-1]
    return w8 * 8, h8 * 8


def _to_bhwc(img):
    # Comfy IMAGE is usually float [0..1], shape [B,H,W,C]
    if hasattr(img, "ndim") and img.ndim == 3:  # HWC
        img = img.unsqueeze(0)
    return img  # [B,H,W,C]


def _pad_reflect(img, pad):
    left, right, top, bottom = pad
    x = img.permute(0, 3, 1, 2)  # B C H W
    x = F.pad(x, (left, right, top, bottom), mode="reflect")
    return x.permute(0, 2, 3, 1)


def _pad_edge(img, pad):
    left, right, top, bottom = pad
    x = img.permute(0, 3, 1, 2)
    x = F.pad(x, (left, right, top, bottom), mode="replicate")
    return x.permute(0, 2, 3, 1)


def _pad_constant(img, pad, value=0.5):
    left, right, top, bottom = pad
    x = img.permute(0, 3, 1, 2)
    x = F.pad(x, (left, right, top, bottom), mode="constant", value=value)
    return x.permute(0, 2, 3, 1)


def _resize(img, W, H):
    x = img.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(int(H), int(W)), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


class AlignHintsToLatent:
    """Align a hint IMAGE to the latent's working dimensions (64-multiple safe)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latent whose W×H defines the target size."}),
                "image": ("IMAGE", {"tooltip": "Hint image to align (B×H×W×C or H×W×C)."}),
                "snap_mode": (["pad_up", "downscale_only", "resize_round", "crop_center"], {"default": "pad_up", "tooltip": "How to match the latent size: pad, downscale, resize, or crop."}),
                "pad_kind": (["reflect", "edge", "constant"], {"default": "reflect", "tooltip": "Padding mode for pad-up/downscale residual."}),
            },
            "optional": {
                "pad_value": ("INT", {"default": 128, "min": 0, "max": 255, "tooltip": "Constant pad value (0..255)."}),
                "keep_alpha": ("BOOLEAN", {"default": False, "tooltip": "Preserve alpha channel if present."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("image_aligned", "bbox_json", "width", "height")
    FUNCTION = "align"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def align(self, latent, image, snap_mode, pad_kind, pad_value=128, keep_alpha=False):
        Wt, Ht = _latent_dims(latent)
        img = _to_bhwc(image).clone()
        B, H0, W0, C = img.shape

        # Drop alpha if requested
        if C > 3 and not keep_alpha:
            img = img[..., :3]
            C = 3

        W, H = int(Wt), int(Ht)
        left = top = 0
        work = img

        if (W0 == W) and (H0 == H):
            bbox = {"x": 0, "y": 0, "w": int(W0), "h": int(H0), "W": W, "H": H}
            return (work, json.dumps(bbox), W, H)

        if snap_mode == "pad_up":
            left = max(0, (W - W0) // 2)
            right = max(0, W - W0 - left)
            top = max(0, (H - H0) // 2)
            bottom = max(0, H - H0 - top)
            pad = (left, right, top, bottom)
            if pad_kind == "reflect":
                work = _pad_reflect(work, pad)
            elif pad_kind == "edge":
                work = _pad_edge(work, pad)
            else:
                work = _pad_constant(work, pad, value=float(pad_value) / 255.0)

        elif snap_mode == "downscale_only":
            scale = min(W / max(1, W0), H / max(1, H0))
            newW = max(1, int(round(W0 * scale)))
            newH = max(1, int(round(H0 * scale)))
            work = _resize(work, newW, newH)
            left = (W - newW) // 2
            top = (H - newH) // 2
            pad = (left, W - newW - left, top, H - newH - top)
            if pad_kind == "reflect":
                work = _pad_reflect(work, pad)
            elif pad_kind == "edge":
                work = _pad_edge(work, pad)
            else:
                work = _pad_constant(work, pad, value=float(pad_value) / 255.0)

        elif snap_mode == "resize_round":
            work = _resize(work, W, H)

        elif snap_mode == "crop_center":
            x0 = max(0, (W0 - W) // 2)
            y0 = max(0, (H0 - H) // 2)
            x1 = min(W0, x0 + W)
            y1 = min(H0, y0 + H)
            work = work[:, y0:y1, x0:x1, :]
            # If smaller than target, pad out
            if work.shape[1] != H or work.shape[2] != W:
                pad = (0, W - work.shape[2], 0, H - work.shape[1])
                work = _pad_edge(work, pad)

        bbox = {"x": int(left), "y": int(top), "w": int(W0), "h": int(H0), "W": int(W), "H": int(H)}
        return (work.contiguous(), json.dumps(bbox), W, H)
