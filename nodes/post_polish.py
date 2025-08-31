"""PostPolish (Film Touch)

Tiny tone S-curve + subtle grain to restore micro-texture without changing the look.
"""

import torch
import torch.nn.functional as F


def _s_curve(img: torch.Tensor, strength: float = 0.1):
    # img in [0,1], BHWC
    mid = 0.5
    return torch.clamp((img - mid) * (1 + strength) + mid, 0.0, 1.0)


def _grain(img: torch.Tensor, strength: float = 0.02):
    if strength <= 0:
        return img
    B, H, W, C = img.shape
    noise = torch.randn((B, H, W, 1), device=img.device, dtype=img.dtype)
    # slight blur to avoid harsh speckle
    k = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=img.device, dtype=img.dtype
    )
    k = (k / k.sum()).view(1, 1, 3, 3)
    n_bchw = noise.permute(0, 3, 1, 2)
    n_bchw = F.conv2d(n_bchw, k, padding=1)
    noise = n_bchw.permute(0, 2, 3, 1)
    return torch.clamp(img + strength * noise, 0.0, 1.0)


class PostPolish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "film_touch": ("BOOLEAN", {"default": True}),
                "tone_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "grain_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def apply(self, image, film_touch=True, tone_strength=0.1, grain_strength=0.02):
        if not film_touch:
            return (image,)
        x = image
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = _s_curve(x, float(tone_strength))
        x = _grain(x, float(grain_strength))
        if image.ndim == 3:
            x = x.squeeze(0)
        return (x,)
