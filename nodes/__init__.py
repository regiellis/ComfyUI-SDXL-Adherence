"""ComfyUI-SDXL-Adherence node registrations."""

from .align_hints_to_latent import AlignHintsToLatent
from .crop_by_bbox import CropByBBox
from .dual_clip_encode import SDXLDualClipEncode
from .prompt_styler import SDXLPromptStyler
from .smart_latent import SmartLatent

NODE_CLASS_MAPPINGS = {
	"AlignHintsToLatent": AlignHintsToLatent,
	"CropByBBox": CropByBBox,
	"SDXLDualClipEncode": SDXLDualClipEncode,
	"SDXLPromptStyler": SDXLPromptStyler,
	"SmartLatent": SmartLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"AlignHintsToLatent": "Align Hints To Latent",
	"CropByBBox": "Crop By BBox",
	"SDXLDualClipEncode": "SDXL Dual CLIP Encode",
	"SDXLPromptStyler": "SDXL Prompt Styler",
	"SmartLatent": "Smart Latent",
}

WEB_DIRECTORY = "js"
