from .nodes.align_hints_to_latent import AlignHintsToLatent
from .nodes.auto_size_64 import AutoSize64
from .nodes.crop_by_bbox import CropByBBox
from .nodes.dual_clip_encode import SDXLDualClipEncode
from .nodes.ksampler_adherence import KSamplerAdherence
from .nodes.negative_prompt_helper import NegativePromptHelper
from .nodes.post_polish import PostPolish
from .nodes.prompt_styler import SDXLPromptStyler
from .nodes.smart_latent import SmartLatent

# Front-end assets directory (relative to this package)
WEB_DIRECTORY = "js"

# Mappings required by ComfyUI
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
