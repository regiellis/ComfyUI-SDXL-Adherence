# ComfyUI-SDXL-Adherence/nodes/prompt_styler.py
import re

STYLE_PRESETS = {
    "none": {"add": [], "neg_drop": []},
    "cinematic": {
        "add": [
            "cinematic composition",
            "soft rim light",
            "filmic contrast",
            "depth of field",
        ],
        "neg_drop": ["hdr", "award-winning", "bokeh"],
    },
    "product": {
        "add": [
            "studio lighting",
            "clean background",
            "polarizing filter",
            "high microcontrast",
        ],
        "neg_drop": ["hdr", "bokeh", "grain"],
    },
    "portrait": {
        "add": [
            "natural skin tones",
            "subtle skin texture",
            "catchlight eyes",
            "shallow depth",
        ],
        "neg_drop": ["overexposed", "grain"],
    },
    "toon": {
        "add": ["cel-shaded", "bold outlines", "flat colors", "graphic style"],
        "neg_drop": ["photographic", "grain", "noise"],
    },
}


def _split_keywords(s: str) -> list[str]:
    return [w.strip() for w in re.split(r"[,\n;]+", s) if w.strip()]


def _pick_essentials(prompt: str, strict: list[str]) -> str:
    # crude essentials extractor: keep strict + obvious colors/numbers
    if strict:
        return ", ".join(strict)
    keep = []
    for tok in re.split(r"[^\w\-]+", prompt):
        if not tok:
            continue
        if re.match(r"^\d+$", tok):
            keep.append(tok)
        elif tok.lower() in {
            "red",
            "blue",
            "green",
            "black",
            "white",
            "yellow",
            "pink",
            "purple",
            "orange",
            "brown",
            "silver",
            "gold",
        }:
            keep.append(tok)
    return ", ".join(keep[:16])


def _apply_style(prompt: str, style: str) -> str:
    add = STYLE_PRESETS.get(style, STYLE_PRESETS["none"])["add"]
    if not add:
        return prompt
    # Append style bits only if not already present (naive)
    extras = [p for p in add if p.lower() not in prompt.lower()]
    return (prompt + ", " + ", ".join(extras)).strip(", ")


def _clean_neg(neg: str, style: str, normalize: bool) -> str:
    if not normalize:
        return neg
    drops = STYLE_PRESETS.get(style, STYLE_PRESETS["none"])["neg_drop"]
    s = neg
    for w in drops:
        s = re.sub(rf"\b{re.escape(w)}\b", "", s, flags=re.I)
    # keep common factual defects
    keepers = [
        "extra fingers",
        "extra limbs",
        "text",
        "watermark",
        "logo",
        "low contrast",
        "blurry",
    ]
    base = ", ".join(keepers)
    s = re.sub(r"\s+,", ",", s).strip().strip(", ")
    return (base + (", " + s if s else "")).strip(", ")


class SDXLPromptStyler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "style": (list(STYLE_PRESETS.keys()),),
                "token_budget": ("INT", {"default": 75, "min": 48, "max": 150}),
                "strict_keywords": ("STRING", {"multiline": False, "default": ""}),
                "normalize_negatives": ("BOOL", {"default": True}),
            },
            "optional": {
                # Optional: for aspect-aware prompt tweaks
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "aspect_tweaks": ("BOOL", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("early_text", "late_text", "neg_text", "essentials_text")
    FUNCTION = "style"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def style(
        self,
        prompt,
        negative,
        style,
        token_budget,
        strict_keywords,
        normalize_negatives,
        width: int = 1024,
        height: int = 1024,
        aspect_tweaks: bool = False,
    ):
        # 1) style injection
        styled = _apply_style(prompt, style)

        # 1.1) optional aspect-aware tweaks
        if aspect_tweaks and width and height:
            try:
                aspect = float(width) / float(height)
                if style == "cinematic":
                    if aspect > 1.3 and "wide cinematic frame" not in styled.lower():
                        styled += ", wide cinematic frame"
                    elif aspect < 0.8 and "tall frame composition" not in styled.lower():
                        styled += ", tall frame composition"
            except Exception:
                pass

        # 2) naive budgeting: favor early info, move long-tail aesthetics to late
        parts = [p.strip() for p in re.split(r"[.;\n]+", styled) if p.strip()]
        if not parts:
            parts = [styled]
        pivot = max(1, int(len(parts) * 0.4))
        early_text = ", ".join(parts[:pivot])
        late_text = ", ".join(parts[pivot:]) if len(parts) > pivot else ""

        # 3) essentials from strict or auto
        essentials_text = _pick_essentials(styled, _split_keywords(strict_keywords))

        # 4) negative hygiene
        neg_text = _clean_neg(negative, style, normalize_negatives)

        return (early_text, late_text, neg_text, essentials_text)
