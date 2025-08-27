# ComfyUI-SDXL-Adherence/nodes/prompt_styler.py
import json
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


def _ensure_text(x) -> str:
    """Best-effort convert various UI values to a safe string.
    - If Tensor with one element: use item()
    - If list/tuple: join parts by comma
    - Else: str(x) if not None, otherwise empty string
    """
    try:
        if isinstance(x, str):
            return x
        if x is None:
            return ""
        if hasattr(x, "numel") and callable(x.numel):
            try:
                if x.numel() == 1 and hasattr(x, "item"):
                    return str(x.item())
                return ""
            except Exception:
                return ""
        if isinstance(x, (list, tuple)):
            try:
                return ", ".join(_ensure_text(i) for i in x if i is not None)
            except Exception:
                return ""
        return str(x)
    except Exception:
        return ""


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
    # Normalize separators/spaces
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+", " ", s).strip().strip(", ")
    # Deduplicate user negatives and drop ones already in keepers (case-insensitive)
    seen = set(t.lower() for t in keepers)
    rest_parts = []
    if s:
        for part in re.split(r"[,;\n]+", s):
            t = part.strip()
            if not t:
                continue
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            rest_parts.append(t)
    rest = ", ".join(rest_parts)
    out = base + (", " + rest if rest else "")
    return out.strip(", ")


class SDXLPromptStyler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Main prompt (subject and key attrs).",
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt. With normalization adds core defects; drops noisy terms.",
                    },
                ),
                "style": (
                    list(STYLE_PRESETS.keys()),
                    {"tooltip": "Style preset to append cues and clean negatives."},
                ),
                "token_budget": (
                    "INT",
                    {
                        "default": 75,
                        "min": 48,
                        "max": 150,
                        "tooltip": "How much to keep up front. Encoder enforces 77 tokens.",
                    },
                ),
                "strict_keywords": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Comma-separated essentials to preserve.",
                    },
                ),
                "normalize_negatives": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Add core defects and drop noisy style terms in negatives.",
                    },
                ),
            },
            "optional": {
                # Optional: for aspect-aware prompt tweaks
                "width": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Width for aspect-aware tweaks (overridden by dims_json).",
                    },
                ),
                "height": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Height for aspect-aware tweaks (overridden by dims_json).",
                    },
                ),
                "aspect_tweaks": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Small phrasing tweaks based on aspect ratio.",
                    },
                ),
                "dims_json": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Connect SmartLatent.dims_json to auto-use W/H.",
                    },
                ),
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
        width=1024,
        height=1024,
        aspect_tweaks: bool = False,
        dims_json: str = "",
    ):
        # safe parse width/height if provided as strings
        def _to_int(v, default):
            try:
                if v is None:
                    return default
                if isinstance(v, str):
                    v = v.strip()
                    if v == "":
                        return default
                return int(v)
            except Exception:
                return default
        def _to_bool(v, default=False):
            try:
                # Tensor-like values: prefer scalar .item(); else truthy if numel()>0
                if hasattr(v, "numel") and callable(v.numel):
                    try:
                        if v.numel() == 1 and hasattr(v, "item"):
                            return bool(v.item())
                        return bool(v.numel())
                    except Exception:
                        return default
                if isinstance(v, (list, tuple)):
                    return any(_to_bool(x, False) for x in v)
                return bool(v)
            except Exception:
                return default
        width = _to_int(width, 1024)
        height = _to_int(height, 1024)
        normalize_negatives = _to_bool(normalize_negatives, True)
        aspect_tweaks = _to_bool(aspect_tweaks, False)
        # Coerce incoming text-like values to strings
        prompt = _ensure_text(prompt)
        negative = _ensure_text(negative)

        # 1) style injection
        styled = _apply_style(prompt, style)

        # 1.1) optional aspect-aware tweaks
        # If dims_json provided, prefer it for W/H
        if dims_json:
            try:
                info = json.loads(dims_json)
                wj = int(info.get("W", width))
                hj = int(info.get("H", height))
                if wj > 0 and hj > 0:
                    width, height = wj, hj
            except Exception:
                pass
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
