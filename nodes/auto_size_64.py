"""AutoSize64

One-control helper to choose a target megapixel size and snap to 64-multiples.

UI: Size = Auto / 1.0MP / 1.5MP / 2.0MP
Inputs: optional IMAGE for aspect inference (else assumes square)
Outputs: width, height, dims_json
"""

import json
import math

try:
    import torch
except Exception:  # noqa: F401
    torch = None


def _round64(x: int) -> int:
    return max(64, int(math.floor(x / 64) * 64))


def _pick_auto_mp() -> float:
    """Heuristic: pick MP based on available VRAM if CUDA present; else 1.2MP."""
    try:
        import torch  # local import in case not available

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            total_gb = getattr(props, "total_memory", 8 * 1024**3) / (1024**3)
            if total_gb < 10:
                return 1.0
            if total_gb < 18:
                return 1.2
            return 1.5
    except Exception:
        pass
    return 1.2


class AutoSize64:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size": (
                    ["Auto", "1.0MP", "1.5MP", "2.0MP"],
                    {"default": "Auto", "tooltip": "Target megapixels (snapped to 64)."},
                )
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image to infer aspect ratio."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "dims_json")
    FUNCTION = "compute"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def compute(self, size: str, image=None):
        # Aspect ratio
        ar = 1.0
        if image is not None:
            try:
                if hasattr(image, "shape") and len(image.shape) == 4:
                    _, H, W, _ = image.shape
                elif hasattr(image, "shape") and len(image.shape) == 3:
                    H, W, _ = image.shape
                else:
                    H, W = 1024, 1024
                ar = max(1e-6, float(W) / float(H))
            except Exception:
                ar = 1.0

        # Pick MP
        if size == "Auto":
            mp = _pick_auto_mp()
        else:
            try:
                mp = float(size.replace("MP", ""))
            except Exception:
                mp = 1.2

        target_pixels = int(mp * 1_000_000)
        # Given ar = W/H, and W*H=target_pixels -> W = sqrt(target_pixels*ar)
        Wf = math.sqrt(max(1, target_pixels) * ar)
        Hf = max(1, target_pixels / max(1e-6, Wf))
        W = _round64(int(Wf))
        H = _round64(int(Hf))

        # Sanity: ensure product near target within 10%
        if abs((W * H) - target_pixels) / max(1, target_pixels) > 0.2:
            # Try adjusting by one 64-step on the minor side
            if W > H:
                H = _round64(int(target_pixels / max(64, W)))
            else:
                W = _round64(int(target_pixels / max(64, H)))

        info = {"W": int(W), "H": int(H), "mp": float(mp), "size": str(size)}
        return (W, H, json.dumps(info))
