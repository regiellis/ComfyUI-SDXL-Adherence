"""ComfyUI-SDXL-Adherence node registrations."""

from .align_hints_to_latent import AlignHintsToLatent
from .auto_size_64 import AutoSize64
from .crop_by_bbox import CropByBBox
from .dual_clip_encode import SDXLDualClipEncode
from .negative_prompt_helper import NegativePromptHelper
from .ksampler_adherence import KSamplerAdherence
from .post_polish import PostPolish
from .prompt_styler import SDXLPromptStyler
from .smart_latent import SmartLatent

NODE_CLASS_MAPPINGS = {
    "AlignHintsToLatent": AlignHintsToLatent,
    "CropByBBox": CropByBBox,
    "SDXLDualClipEncode": SDXLDualClipEncode,
    "SDXLPromptStyler": SDXLPromptStyler,
    "SmartLatent": SmartLatent,
    "AutoSize64": AutoSize64,
    "NegativePromptHelper": NegativePromptHelper,
    "KSamplerAdherence": KSamplerAdherence,
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
    "KSamplerAdherence": "KSampler (Adherence)",
    "PostPolish": "Post Polish (Film Touch)",
}

WEB_DIRECTORY = "js"
