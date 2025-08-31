"""Negative Prompt Helper

Merges user negatives with a vetted realism pack and de-duplicates.
UI: toggle Realism Negs (on/off), optional custom pack string to extend.
"""

import re


REALISM_NEGS = [
    "cartoon",
    "illustration",
    "anime",
    "cgi",
    "plastic skin",
    "overprocessed",
    "oversharpened",
    "waxy",
    "low detail",
    "smooth skin",
    "extra fingers",
]


def _split(s: str):
    return [w.strip() for w in re.split(r"[,;\n]+", s or "") if w.strip()]


class NegativePromptHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "realism_negs": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_pack": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": ", ".join(REALISM_NEGS),
                        "tooltip": "Editable realism pack (applied when toggle is on).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("neg_text",)
    FUNCTION = "build"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def build(self, negative: str, realism_negs: bool = True, custom_pack: str = ""):
        base = _split(negative)
        if realism_negs:
            pack = _split(custom_pack) or REALISM_NEGS
            base.extend(pack)
        # de-duplicate case-insensitively, preserve order
        out = []
        seen = set()
        for w in base:
            k = w.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(w)
        return (", ".join(out),)
