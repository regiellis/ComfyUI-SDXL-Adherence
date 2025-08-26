import os

from .nodes.dual_clip_encode import SDXLDualClipEncode

# Register SDXL Adherence nodes
from .nodes.prompt_styler import SDXLPromptStyler
from .nodes.smart_latent import SmartLatent

# Optional front-end assets
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

# Mappings required by ComfyUI
NODE_CLASS_MAPPINGS = {
    "SDXLPromptStyler": SDXLPromptStyler,
    "SDXLDualClipEncode": SDXLDualClipEncode,
    "SmartLatent": SmartLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLPromptStyler": "SDXL Prompt Styler",
    "SDXLDualClipEncode": "SDXL Dual CLIP Encode (pos/neg)",
    "SmartLatent": "Smart Latent (empty or encode)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
