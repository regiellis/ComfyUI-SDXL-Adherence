# ComfyUI-SDXL-Adherence/nodes/smart_latent.py
import json
import math

import torch


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


class SmartLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "mode": (["empty", "encode_image"],),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 64},
                ),
            },
            "optional": {
                "batch": ("INT", {"default": 1, "min": 1, "max": 16}),
                "image": ("IMAGE",),
                "tile_size": (
                    "INT",
                    {"default": 320, "min": 192, "max": 512, "step": 32},
                ),
                "force_bchw": ("BOOL", {"default": True}),
                "seed": ("INT", {"default": 0}),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1024 * 1024,
                        "min": 256 * 256,
                        "max": 4096 * 4096,
                        "step": 64,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "INT", "INT")
    RETURN_NAMES = ("latent", "dims_json", "width", "height")
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
        force_bchw=True,
        seed=0,
        max_pixels=1024 * 1024,
    ):
        W, H, down = _safe_dims(width, height, max_pixels)
        info = {
            "mode": mode,
            "W": W,
            "H": H,
            "batch": batch,
            "tile": tile_size,
            "down_by_64": down,
        }

        if mode == "empty":
            # allocate latent zeros (Comfy expects a dict with 'samples')
            latent = torch.zeros([batch, 4, H // 8, W // 8], device="cpu")
            return ({"samples": latent}, json.dumps(info), W, H)

        # encode_image
        assert image is not None, "SmartLatent: image input required in encode_image mode"
        img = image
        if force_bchw:
            if hasattr(img, "ndim"):
                if img.ndim == 3:  # CHW
                    img = img.unsqueeze(0)
                elif img.ndim == 4:
                    pass
                else:
                    raise ValueError("SmartLatent: unexpected image tensor shape")

        # Optional safety: if the provided image size exceeds (W,H), allow VAE/resize to handle
        # We do not import external PIL/torchvision here to stay minimal; if your Comfy build
        # provides an image resize node before this, prefer that for resampling quality.

        # Tiled encode if available
        if hasattr(vae, "encode_tiled"):
            latent = vae.encode_tiled(img, tile=tile_size)
        else:
            latent = vae.encode(img)

        # Normalize latent to dict format and derive true dims from samples
        if isinstance(latent, dict) and "samples" in latent:
            samples = latent["samples"]
            out_latent = latent
        else:
            samples = latent
            out_latent = {"samples": latent}
        try:
            h8 = int(samples.shape[-2])
            w8 = int(samples.shape[-1])
            has_shape = hasattr(samples, "shape") and len(samples.shape) >= 1
            B = int(samples.shape[0]) if has_shape else batch
            H_actual = h8 * 8
            W_actual = w8 * 8
            info = {
                "mode": mode,
                "W": W_actual,
                "H": H_actual,
                "batch": B,
                "tile": tile_size,
                "down_by_64": down,
            }
            return (out_latent, json.dumps(info), W_actual, H_actual)
        except Exception:
            # Fallback to requested dims
            return (out_latent, json.dumps(info), W, H)
