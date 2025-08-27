"""SDXL Dual CLIP Encode (pos/neg)

Builds canonical ComfyUI SDXL CONDITIONING lists using the pooled-output path.
Tokenizes and calls clip.encode_from_tokens(..., return_pooled=True)
- Guarantees pooled_output on every entry
- Blends early/late/essentials with per-entry weights
"""

import torch

# --- helpers ---------------------------------------------------------------


def _encode_text_tokens(clip, tokens, clip_skip_g: int, clip_skip_l: int):
    """
    Preferred path: returns (cond[B,T,D], pooled[B,D]) with pooled non-None.
    We try the most specific signature first, then gracefully degrade.
    """
    # Most recent CLIP objects support return_pooled + clip_skip_g/clip_skip_l
    try:
        cond, pooled = clip.encode_from_tokens(
            tokens, return_pooled=True, clip_skip_g=clip_skip_g, clip_skip_l=clip_skip_l
        )
        return cond, pooled
    except TypeError:
        # Older signature (no explicit clip_skip args)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return cond, pooled
    except AttributeError as err:
        # Fallback: no encode_from_tokens method; use encode on string path
        raise RuntimeError(
            "CLIP object missing encode_from_tokens(); use a recent ComfyUI build or adapt encoder to your clip class."
        ) from err


def _encode_text_string(clip, text: str, clip_skip_g: int, clip_skip_l: int):
    """
    Safe encode from string: tokenize first so we control pooled creation.
    """
    if text is None:
        text = ""
    tokens = clip.tokenize(text)
    return _encode_text_tokens(clip, tokens, clip_skip_g, clip_skip_l)


def _cond_entry(cond: torch.Tensor, pooled: torch.Tensor, weight: float = 1.0):
    """
    Canonical CONDITIONING entry for SDXL in Comfy:
      [cond_tensor(B,T,D), {"pooled_output": pooled(B,D), "weight": float}]
    """
    if pooled is None:
        # True fix: pooled MUST exist for SDXL ADM. Synthesize only as last resort.
        if cond.ndim == 3:
            pooled = cond.mean(dim=1)
        elif cond.ndim == 2:
            pooled = torch.zeros(
                (cond.shape[0], cond.shape[1]), device=cond.device, dtype=cond.dtype
            )
        else:
            raise ValueError(
                f"Unexpected 'cond' shape {tuple(cond.shape)}; cannot synthesize pooled."
            )

    # ensure plain float for weight (not a Tensor)
    w = float(weight)
    return [cond, {"pooled_output": pooled, "weight": w}]


def _blend_append(dst: list, src: list, scale: float):
    """
    Append entries from src into dst with a weight scale, preserving pooled_output.
    """
    if not src or scale is None:
        return
    s = float(scale)
    for entry in src:
        # entry must be [cond, meta]
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("Conditioning entry must be [cond, {meta}]")
        cond, meta = entry
        pooled = meta.get("pooled_output", None)
        w = float(meta.get("weight", 1.0)) * s
        dst.append(_cond_entry(cond, pooled, w))


def _encode_as_list(
    clip, text: str, clip_skip_g: int, clip_skip_l: int, weight: float = 1.0
) -> list:
    cond, pooled = _encode_text_string(clip, text or "", clip_skip_g, clip_skip_l)
    return [_cond_entry(cond, pooled, weight)]


def _res_aware(mix: float, lock: float, width: int, height: int) -> tuple[float, float]:
    try:
        long_side = max(int(width), int(height))
    except Exception:
        return mix, lock
    if long_side > 1280:
        return (max(mix, 0.6), max(lock, 0.4))
    if long_side < 896:
        return (min(mix, 0.3), lock)
    return (mix, lock)


# --- node ------------------------------------------------------------------


class SDXLDualClipEncode:
    """Encode positive and negative SDXL conditioning with pooled_output present."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "SDXL CLIP from CheckpointLoader (dual encoders)."}),
                "early_text": (
                    "STRING",
                    {"multiline": True, "tooltip": "Primary prompt (subject, key attributes)."},
                ),
                "late_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Long-tail aesthetics blended with early by early_late_mix.",
                    },
                ),
                "neg_text": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Negative prompt."},
                ),
                "essentials_text": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Keywords to reinforce with extra weight.",
                    },
                ),
                "early_late_mix": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Blend weight for late_text (0=ignore late, 1=equal weight).",
                    },
                ),
                "essentials_lock": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Additional weight applied to essentials_text.",
                    },
                ),
                "clip_skip_openclip": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 2,
                        "tooltip": "OpenCLIP skip (global encoder).",
                    },
                ),
                "clip_skip_clipL": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2, "tooltip": "CLIP-L skip (local encoder)."},
                ),
            },
            "optional": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "tooltip": "Working width for heuristics (feed from SmartLatent).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "tooltip": "Working height for heuristics (feed from SmartLatent).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("cond_positive", "cond_negative")
    FUNCTION = "encode"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def encode(
        self,
        clip,
        early_text,
        late_text,
        neg_text,
        essentials_text,
        early_late_mix,
        essentials_lock,
        clip_skip_openclip,
        clip_skip_clipL,
        width=1024,
        height=1024,
    ):

        # Heuristic tweaks by resolution (does not alter structure)
        mix, lock = _res_aware(early_late_mix, essentials_lock, width, height)

        # --- POSITIVE ---
        pos = []
        # Early (instruction heavy)
        pos_early = _encode_as_list(
            clip, early_text, clip_skip_openclip, clip_skip_clipL, weight=1.0
        )
        pos.extend(pos_early)

        # Late (aesthetic) with mix weight
        if (late_text or "").strip():
            pos_late = _encode_as_list(
                clip, late_text, clip_skip_openclip, clip_skip_clipL, weight=1.0
            )
            _blend_append(pos, pos_late, scale=mix)

        # Essentials lock (hard bias) with lock weight
        if (essentials_text or "").strip() and lock > 0.0:
            pos_lock = _encode_as_list(
                clip, essentials_text, clip_skip_openclip, clip_skip_clipL, weight=1.0
            )
            _blend_append(pos, pos_lock, scale=lock)

        # --- NEGATIVE ---
        neg = _encode_as_list(clip, neg_text, clip_skip_openclip, clip_skip_clipL, weight=1.0)

        # Defensive validation: every entry must have pooled_output
        for name, lst in (("positive", pos), ("negative", neg)):
            if not isinstance(lst, list) or len(lst) == 0:
                raise ValueError(f"{name} conditioning is empty; encode returned 0 entries.")
            for i, entry in enumerate(lst):
                if (not isinstance(entry, (list, tuple))) or len(entry) != 2:
                    raise ValueError(f"{name}[{i}] invalid entry type (must be [cond, meta]).")
                cond, meta = entry
                if meta.get("pooled_output", None) is None:
                    # true fix: never hand KSampler a None pooled_output
                    if cond.ndim == 3:
                        meta["pooled_output"] = cond.mean(dim=1)
                    else:
                        raise ValueError(
                            f"{name}[{i}] has no pooled_output and cond shape {tuple(cond.shape)}"
                        )
                # normalize weight to float
                if "weight" in meta:
                    meta["weight"] = float(meta["weight"])
                else:
                    meta["weight"] = 1.0

        return (pos, neg)
