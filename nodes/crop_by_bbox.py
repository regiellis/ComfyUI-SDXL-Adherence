"""CropByBBox

Crop an IMAGE using bbox metadata (e.g., from SmartLatent or AlignHintsToLatent).
Optionally feather edges and resize back to original content size.
"""

import json

import torch
import torch.nn.functional as F


def _to_bhwc(img):
    if img.ndim == 3:  # HWC
        img = img.unsqueeze(0)
    return img  # B,H,W,C in Comfy


def _resize_bhwc(img, W, H):
    img = img.permute(0, 3, 1, 2)  # B,C,H,W
    img = F.interpolate(img, size=(int(H), int(W)), mode="bilinear", align_corners=False)
    return img.permute(0, 2, 3, 1)  # B,H,W,C


class CropByBBox:
    """Crop by bbox_json and optionally resize back and feather edges."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to crop (H×W×C or B×H×W×C)."}),
                "bbox_json": (
                    "STRING",
                    {"multiline": False, "tooltip": "BBox JSON: {x,y,w,h,W,H}."},
                ),
            },
            "optional": {
                "resize_back": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Resize crop back to original content size w×h."},
                ),
                "clamp_to_bounds": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Clamp crop rect to image bounds."},
                ),
                "feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Soft edge radius in pixels.",
                    },
                ),
                "expand": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Grow crop rect by N px in all directions.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_cropped", "info_json")
    FUNCTION = "crop"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def crop(self, image, bbox_json, resize_back=True, clamp_to_bounds=True, feather=0, expand=0):
        img = _to_bhwc(image).contiguous()
        B, Hc, Wc, C = img.shape

        # Parse bbox
        try:
            bb = json.loads(bbox_json) if bbox_json else {}
            x, y = int(bb.get("x", 0)), int(bb.get("y", 0))
            w0, h0 = int(bb.get("w", Wc)), int(bb.get("h", Hc))  # original content dims
            Ww, Hw = int(bb.get("W", Wc)), int(bb.get("H", Hc))  # working (padded) dims
        except Exception:
            # passthrough if invalid
            info = {"warning": "invalid bbox_json; passthrough", "W": Wc, "H": Hc}
            return (img, json.dumps(info))

        # Map bbox from working-space (Ww,Hw) to current image-space (Wc,Hc)
        sx = Wc / max(1, Ww)
        sy = Hc / max(1, Hw)
        x_ = int(round(x * sx))
        y_ = int(round(y * sy))
        w_ = int(round(w0 * sx))
        h_ = int(round(h0 * sy))

        # Optional expand (in current image space)
        if expand and int(expand) != 0:
            expand = int(expand)
            x_ -= expand
            y_ -= expand
            w_ += 2 * expand
            h_ += 2 * expand

        # Clamp
        if clamp_to_bounds:
            x_ = max(0, min(x_, Wc - 1))
            y_ = max(0, min(y_, Hc - 1))
            w_ = max(1, min(w_, Wc - x_))
            h_ = max(1, min(h_, Hc - y_))

        # Crop
        crop = img[:, y_ : y_ + h_, x_ : x_ + w_, :]

        # Optional feathered edge (soft mask) to reduce seams if you plan to composite
        if feather and int(feather) > 0:
            feather = int(feather)
            yy = torch.linspace(0, 1, steps=h_, device=crop.device).view(1, h_, 1, 1)
            xx = torch.linspace(0, 1, steps=w_, device=crop.device).view(1, 1, w_, 1)
            # distance to nearest edge normalized, then scaled by feather
            mask_y = torch.clamp(torch.minimum(yy, 1 - yy) * (h_ / max(1, feather)), 0, 1)
            mask_x = torch.clamp(torch.minimum(xx, 1 - xx) * (w_ / max(1, feather)), 0, 1)
            mask = (mask_x * mask_y).clamp(0, 1)
            crop = crop * mask

        # Resize back to original content size if requested and different
        if resize_back and (w_ != w0 or h_ != h0) and w0 > 0 and h0 > 0:
            crop = _resize_bhwc(crop, w0, h0)

        info = {
            "in": {"W": Wc, "H": Hc},
            "bbox": {"x": x, "y": y, "w": w0, "h": h0, "W": Ww, "H": Hw},
            "mapped": {"x": x_, "y": y_, "w": w_, "h": h_},
            "resized_to_original": bool(resize_back),
        }
        return (crop.contiguous(), json.dumps(info))
