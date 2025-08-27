import os

from .nodes.align_hints_to_latent import AlignHintsToLatent
from .nodes.crop_by_bbox import CropByBBox
from .nodes.dual_clip_encode import SDXLDualClipEncode

# Register SDXL Adherence nodes
from .nodes.prompt_styler import SDXLPromptStyler
from .nodes.smart_latent import SmartLatent

# Optional front-end assets
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

# Mappings required by ComfyUI
NODE_CLASS_MAPPINGS = {
    "SDXLAdherencePromptStyler": SDXLPromptStyler,
    "SDXLAdherenceDualClipEncode": SDXLDualClipEncode,
    "SmartLatent": SmartLatent,
    "AlignHintsToLatent": AlignHintsToLatent,
    "CropByBBox": CropByBBox,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLAdherencePromptStyler": "SDXL Adherence Prompt Styler",
    "SDXLAdherenceDualClipEncode": "SDXL Dual CLIP Encode (pos/neg)",
    "SmartLatent": "Smart Latent (empty or encode)",
    "AlignHintsToLatent": "Align Hints To Latent",
    "CropByBBox": "Crop By BBox",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
