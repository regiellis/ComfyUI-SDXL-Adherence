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


# Lightweight lexicons for essentials extraction (no heavy NLP deps)
COLORS = {
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
    "cyan",
    "magenta",
    "teal",
    "turquoise",
    "beige",
    "ivory",
    "gray",
    "grey",
    "maroon",
    "navy",
    "olive",
    "lime",
    "indigo",
    "violet",
    "cream",
    "bronze",
    "copper",
    "rose",
    "peach",
}

MATERIALS = {
    "wood",
    "metal",
    "steel",
    "iron",
    "bronze",
    "copper",
    "plastic",
    "glass",
    "ceramic",
    "stone",
    "marble",
    "leather",
    "fabric",
    "cloth",
    "silk",
    "cotton",
    "wool",
    "denim",
    "latex",
    "rubber",
    "paper",
    "concrete",
    "chrome",
    "matte",
    "glossy",
    "satin",
    "velvet",
}

STYLE_NOISE = {
    # Common aesthetic/style words we don't want to lock as essentials
    "cinematic",
    "filmic",
    "bokeh",
    "contrast",
    "lighting",
    "studio",
    "composition",
    "hdr",
    "award-winning",
    "microcontrast",
    "toon",
    "cel-shaded",
    "graphic",
    "style",
    "rim",
    "light",
    "shallow",
    "portrait",
    "product",
    "background",
    # Generic/common words we don't want as essentials
    "realistic",
    "photo",
    "photograph",
    "image",
    "picture",
    "head",
    "face",
    "face-shape",
    "body",
    "expression",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "by",
    "to",
    "from",
    "into",
    "over",
    "under",
    "about",
    "as",
    "is",
    "are",
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


def _pick_essentials(
    prompt: str,
    strict: list[str],
    max_out: int = 24,
    strategy: str = "balanced",
    blocklist=None,
    allowlist=None,
) -> str:
    """Extract a compact, order-preserving essentials string from the prompt.

    Heuristics (no heavy NLP):
    - If user provided strict keywords, use those as-is.
    - Include quoted phrases ("..." or '...') verbatim.
    - Prefer subject words before common prepositions (with/in/on/at/of/etc.).
    - Keep hyphenated compounds.
    - Keep numbers, ratios (16:9), and units (e.g., 35mm, 4k).
    - Keep colors and materials.
    - Keep proper nouns (Simple: consecutive Capitalized tokens), but drop style noise.
    - Deduplicate case-insensitively and cap length.
    """
    if strict:
        return ", ".join([s for s in strict if s])

    s = (prompt or "").strip()
    # Remove weight suffixes like :1.15 so decimals don't leak into essentials
    s_clean = re.sub(r":\s*\d+(?:\.\d+)?", "", s)
    keep: list[str] = []
    seen: set[str] = set()
    blockset = set(blocklist or [])

    def _add(x: str):
        x = (x or "").strip().strip(",")
        if not x:
            return
        k = x.lower()
        if k in seen or k in blockset:
            return
        seen.add(k)
        keep.append(x)

    # 1) Quoted phrases
    for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', s_clean):
        phrase = m.group(1) or m.group(2)
        if phrase:
            _add(phrase)

    # 2) Subject chunk: before first common preposition/connector
    subject_chunk = re.split(
        r"\b(with|featuring|wearing|holding|using|in|on|at|of|and)\b",
        s_clean,
        maxsplit=1,
    )[0]
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]*", subject_chunk):
        tl = tok.lower()
        if tl in STOPWORDS or tl in STYLE_NOISE:
            continue
        if len(tl) <= 1:
            continue
        _add(tok)

    # 3) Hyphenated compounds anywhere
    for tok in re.findall(r"\b[\w]+(?:-[\w]+)+\b", s_clean):
        if tok.lower() not in STYLE_NOISE:
            _add(tok)

    # 4) Numbers/ratios/units based on strategy
    strat = (strategy or "balanced").lower()
    if strat in ("balanced", "aggressive"):
        for tok in re.findall(r"\b\d+\s*[:xX]\s*\d+\b", s_clean):
            _add(tok.replace(" ", ""))
        for tok in re.findall(r"\b\d+(?:\.\d+)?\s?(?:mm|cm|k|m|mp|fps)\b", s_clean, flags=re.I):
            _add(tok)
        if strat == "aggressive":
            for tok in re.findall(r"\b\d+(?:\.\d+)?\b", s_clean):
                _add(tok)

    # 5) Colors paired with common nouns; avoid standalone colors
    _color_pat = r"\b(?:" + "|".join(sorted(COLORS)) + r")\b"
    _noun_pat = r"\b(eyes|hair|skin|lips|mouth|beard|mustache|brows|eyebrows|shirt|dress|coat|jacket|pants|fur|feathers)\b"
    for m in re.finditer(_color_pat + r"\s+" + _noun_pat, s_clean, flags=re.I):
        col = m.group(0).split()[0]
        noun = m.group(0).split()[-1]
        _add(f"{col.lower()} {noun.lower()}")
    # Also keep material words (often meaningful)
    for tok in re.findall(r"[A-Za-z]+", s_clean):
        tl = tok.lower()
        if tl in MATERIALS:
            _add(tl)

    # 6) Proper nouns (allow 1–3 capitalized tokens)
    for tok in re.findall(r"\b(?:[A-Z][a-z0-9]+)(?:\s+[A-Z][a-z0-9]+){0,2}\b", s_clean):
        if tok.isupper():
            continue  # avoid acronyms screaming
        if tok.lower() in STYLE_NOISE:
            continue
        _add(tok)

    # Allowlist (force include)
    for t in allowlist or []:
        _add(t)

    # Cap length to avoid bloating the essentials channel
    if len(keep) > max_out:
        keep = keep[:max_out]
    return ", ".join(keep)


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
                # Automation controls
                "auto_essentials": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Generate essentials automatically when strict keywords are empty.",
                    },
                ),
                "max_essentials": (
                    "INT",
                    {
                        "default": 24,
                        "min": 4,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Maximum items to include in essentials.",
                    },
                ),
                "essentials_strategy": (
                    ["off", "conservative", "balanced", "aggressive"],
                    {
                        "default": "balanced",
                        "tooltip": "Controls how essentials are extracted (off uses strict only).",
                    },
                ),
                "noise_blocklist": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated terms to exclude from essentials.",
                    },
                ),
                "noise_allowlist": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated terms to force-include in essentials.",
                    },
                ),
                "auto_pivot": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Auto-adjust early/late split based on prompt length vs token_budget.",
                    },
                ),
                "early_ratio": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.2,
                        "max": 0.8,
                        "step": 0.05,
                        "tooltip": "Manual early split ratio when auto_pivot is off.",
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
        auto_essentials: bool = True,
        max_essentials: int = 24,
        essentials_strategy: str = "balanced",
        noise_blocklist: str = "",
        noise_allowlist: str = "",
        auto_pivot: bool = True,
        early_ratio: float = 0.4,
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

        # 2) budgeting: favor early info, move long-tail aesthetics to late
        # Split on semicolons/newlines (and sentence-ending periods that are NOT decimals)
        parts = [p.strip() for p in re.split(r"(?<!\d)\.(?!\d)|[;\n]+", styled) if p.strip()]
        if not parts:
            parts = [styled]
        # auto pivot based on approximate token load vs budget
        ratio = max(0.2, min(0.8, float(early_ratio)))
        if auto_pivot:
            try:
                approx_words = len(re.findall(r"\w+", styled))
                budget = int(token_budget) if token_budget else 75
                load = approx_words / max(1, budget)
                # map load in [0.5..2.0] to ratio in [0.35..0.65]
                load = max(0.5, min(2.0, load))
                ratio = 0.35 + (load - 0.5) * (0.30 / 1.5)
                ratio = max(0.3, min(0.7, ratio))
            except Exception:
                pass
        pivot = max(1, int(len(parts) * ratio))
        early_text = ", ".join(parts[:pivot])
        late_text = ", ".join(parts[pivot:]) if len(parts) > pivot else ""

        # Light cleanup: collapse duplicate commas/spaces without touching weight syntax
        def _clean_commas(s: str) -> str:
            if not s:
                return s
            s = re.sub(r"\s*,\s*", ", ", s)  # normalize comma spacing
            s = re.sub(r"(?:,\s*){2,}", ", ", s)  # collapse duplicate commas
            return s.strip(" ,")

        early_text = _clean_commas(early_text)
        late_text = _clean_commas(late_text)

        # 3) essentials from strict or auto
        strict_list = _split_keywords(strict_keywords)
        block_list = set(t.lower() for t in _split_keywords(noise_blocklist))
        allow_list = _split_keywords(noise_allowlist)
        if strict_list:
            essentials_text = ", ".join(strict_list)
        else:
            if (not auto_essentials) or (essentials_strategy == "off"):
                essentials_text = ""
            else:
                essentials_text = _pick_essentials(
                    styled,
                    [],
                    max_out=max_essentials,
                    strategy=essentials_strategy,
                    blocklist=block_list,
                    allowlist=allow_list,
                )

        # 4) negative hygiene
        neg_text = _clean_neg(negative, style, normalize_negatives)

        return (early_text, late_text, neg_text, essentials_text)
