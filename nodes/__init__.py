"""ComfyUI-SDXL-Adherence node registrations."""

from .align_hints_to_latent import AlignHintsToLatent
from .crop_by_bbox import CropByBBox
from .dual_clip_encode import SDXLDualClipEncode
from .prompt_styler import SDXLPromptStyler
from .smart_latent import SmartLatent
from .auto_size_64 import AutoSize64
from .negative_prompt_helper import NegativePromptHelper
from .post_polish import PostPolish

NODE_CLASS_MAPPINGS = {
    "AlignHintsToLatent": AlignHintsToLatent,
    "CropByBBox": CropByBBox,
    "SDXLDualClipEncode": SDXLDualClipEncode,
    "SDXLPromptStyler": SDXLPromptStyler,
    "SmartLatent": SmartLatent,
    "AutoSize64": AutoSize64,
    "NegativePromptHelper": NegativePromptHelper,
    "PostPolish": PostPolish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlignHintsToLatent": "Align Hints To Latent",
    "CropByBBox": "Crop By BBox",
    "SDXLDualClipEncode": "SDXL Dual CLIP Encode",
    "SDXLPromptStyler": "SDXL Prompt Styler",
    "SmartLatent": "Smart Latent",
    "AutoSize64": "Auto-Size 64 (MP)",
    "NegativePromptHelper": "Negative Prompt Helper",
    "PostPolish": "Post Polish (Film Touch)",
}

WEB_DIRECTORY = "js"
