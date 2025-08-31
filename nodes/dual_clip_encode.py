"""SDXL Dual CLIP Encode (pos/neg)

Builds canonical ComfyUI SDXL CONDITIONING lists using the pooled-output path.
Tokenizes and calls clip.encode_from_tokens(..., return_pooled=True)
- Guarantees pooled_output on every entry
- Blends early/late/essentials with per-entry weights
"""

# SDXLDualClipEncode (full script, integrated)
# - Modes: auto / core_parity / custom_only
# - Long prompts (auto): fused core-like encode + SOFT head/tail ramps (no hard flips)
# - Essentials: early-only, overlap-aware, softly faded; skipped on very long prompts
# - Adaptive ramps by total steps; adaptive caps by CFG; adaptive head/tail slice size by token length
# - Conditioning entries always include pooled_output
# - Resolution-aware tweaks preserved for short/medium (multi-entry) path
# - Optional debug line via env: ADHERENCE_DEBUG=1

import os

import torch

# ----------------- policy base constants (tune lightly) -----------------

LONG_PROMPT_TOKENS = 112  # ~1.5 chunks; use fused/core path at/above this
VERY_LONG_TOKENS = 168  # skip essentials when extremely long

BASE_HEAD_TOKENS = 24  # base head slice
BASE_TAIL_TOKENS = 28  # base tail slice (slightly larger than head)

BASE_HEAD_W = 0.11  # gentle early nudge for subject/context
BASE_TAIL_W = 0.14  # gentle late nudge for tail semantics

ESSENTIALS_BASE_W = 0.15  # micro lock baseline (soft)
BASE_EXTRA_CAP = 0.24  # global cap for sum of all non-core entries

WINDOW_CAP = 0.18  # max overlapping extra weight within any time window

# ----------------- tiny utils -------------------------------------------


def _round4(x: float) -> float:
    return float(round(float(x), 4))


def _debug_enabled() -> bool:
    return os.getenv("ADHERENCE_DEBUG", "0") not in ("0", "", "false", "False")


# ----------------- adaptive policies ------------------------------------


def _adaptive_slice_counts(total_tokens: int):
    """
    Longer prompts → larger tail slice; small bump for head.
    """
    extra = min(16, max(0, (total_tokens - 96) // 16))
    head = BASE_HEAD_TOKENS + extra // 2
    tail = BASE_TAIL_TOKENS + extra
    return head, tail


def _cap_for_cfg(cfg: float | None, base=BASE_EXTRA_CAP) -> float:
    """
    Higher CFG → slightly lower cap (avoid overpowering).
    """
    if cfg is None:
        return base
    return float(min(0.28, max(0.18, base - 0.015 * max(0.0, cfg - 6.0))))


def _ramp_points(total_steps: int | None):
    """
    Return a tuple of (HEAD_RAMP, TAIL_RAMP, ESSENTIALS_RAMP), each a list of
    (t_start, t_end, weight_scale). Ramps adapt to total steps.
    """
    if not total_steps or total_steps <= 0:
        early_end = 0.30
        late_start = 0.55
    else:
        t = max(10, int(total_steps))
        early_end = 0.25 + min(0.35, 6.0 / t)  # shorter ramps at high steps
        late_start = 0.55 - min(0.10, 6.0 / t)  # start a bit earlier for low step counts

    HEAD_RAMP = [
        (0.00, max(0.20, early_end - 0.05), 1.00),
        (max(0.15, early_end - 0.10), early_end, 0.55),
        (max(0.30, early_end - 0.05), min(0.50, early_end + 0.20), 0.25),
    ]
    TAIL_RAMP = [
        (max(0.40, late_start - 0.05), max(0.65, late_start + 0.10), 0.40),
        (max(0.55, late_start + 0.05), min(0.85, late_start + 0.30), 0.75),
        (max(0.70, late_start + 0.20), 1.00, 1.00),
    ]
    ESSENTIALS_RAMP = [
        (0.00, max(0.40, early_end - 0.05), 1.00),
        (max(0.35, early_end - 0.10), min(0.60, early_end + 0.10), 0.60),
        (max(0.55, early_end), min(0.70, early_end + 0.20), 0.30),
    ]

    # clamp & round
    def _norm(r):
        out = []
        for a, b, s in r:
            out.append((_round4(max(0.0, min(1.0, a))), _round4(max(0.0, min(1.0, b))), float(s)))
        return out

    return _norm(HEAD_RAMP), _norm(TAIL_RAMP), _norm(ESSENTIALS_RAMP)


# ----------------- helpers (encode & tokenize) ---------------------------

_token_cache: dict[tuple[int, int], torch.Tensor] = {}


def _text_hash(s: str) -> int:
    return hash(s) & 0x7FFFFFFF


def _encode_text_tokens(clip, tokens, clip_skip_g: int, clip_skip_l: int):
    try:
        return clip.encode_from_tokens(
            tokens, return_pooled=True, clip_skip_g=clip_skip_g, clip_skip_l=clip_skip_l
        )
    except TypeError:
        return clip.encode_from_tokens(tokens, return_pooled=True)
    except AttributeError as err:
        raise RuntimeError(
            "CLIP object missing encode_from_tokens(); update ComfyUI or adapt to your clip class."
        ) from err


def _tokenize_cached(clip, text: str):
    cid = id(clip)
    th = _text_hash(text or "")
    key = (cid, th)
    if key in _token_cache:
        return _token_cache[key]
    toks = clip.tokenize(text or "")
    _token_cache[key] = toks
    # simple size limit
    if len(_token_cache) > 256:
        _token_cache.clear()
    return toks


def _encode_text_string(clip, text: str, clip_skip_g: int, clip_skip_l: int):
    tokens = _tokenize_cached(clip, text)
    return _encode_text_tokens(clip, tokens, clip_skip_g, clip_skip_l)


def _cond_entry(cond: torch.Tensor, pooled: torch.Tensor, weight: float = 1.0):
    if pooled is None:
        if cond.ndim == 3:
            pooled = cond.mean(dim=1)
        else:
            raise ValueError(
                f"Unexpected 'cond' shape {tuple(cond.shape)}; cannot synthesize pooled."
            )
    return [cond, {"pooled_output": pooled, "weight": _round4(weight)}]


def _blend_append(dst: list, src: list, scale: float):
    if not src or scale is None:
        return
    s = float(scale)
    for entry in src:
        cond, meta = entry
        pooled = meta.get("pooled_output", None)
        new_meta = dict(meta)
        new_meta["pooled_output"] = pooled
        new_meta["weight"] = _round4(float(meta.get("weight", 1.0)) * s)
        dst.append([cond, new_meta])


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


def _count_tokens(clip, text: str) -> int:
    if not text:
        return 0
    toks = _tokenize_cached(clip, text)
    if hasattr(toks, "shape"):
        return int(toks.shape[-1])
    return 0


def _slice_tokens(clip, text: str, head_n: int = 0, tail_n: int = 0):
    toks = _tokenize_cached(clip, text or "")
    if not hasattr(toks, "shape"):
        return None
    if toks.ndim == 1:
        toks = toks.unsqueeze(0)
    T = toks.shape[-1]
    if T <= 4:
        return None
    bos = toks[:, :1]
    eos = toks[:, -1:]
    inner = toks[:, 1 : T - 1]
    if head_n > 0:
        if inner.shape[-1] <= 0:
            return None
        slice_part = inner[:, : min(head_n, inner.shape[-1])]
    elif tail_n > 0:
        if inner.shape[-1] <= 0:
            return None
        slice_part = inner[:, max(0, inner.shape[-1] - tail_n) :]
    else:
        return None
    return torch.cat([bos, slice_part, eos], dim=1)


def _overlap_scale(essentials: str, combined: str, base_w: float) -> float:
    if not essentials or not combined:
        return base_w
    E = set(essentials.lower().replace(",", " ").split())
    C = set(combined.lower().replace(",", " ").split())
    if not E:
        return base_w
    j = len(E & C) / max(1, len(E))
    return max(0.06, base_w * (1.0 - 0.6 * j))  # stronger downscale & floor


def _downweight_if_redundant(essentials: str, combined: str, w: float):
    e = (essentials or "").lower()
    c = (combined or "").lower()
    es = e.split()
    big_overlap = 0
    for i in range(max(0, len(es) - 1)):
        bg = f"{es[i]} {es[i+1]}"
        if bg in c:
            big_overlap += 1
    if big_overlap >= 2:
        w *= 0.6
    if (essentials or "").strip().endswith((".", "!", "?")):
        w *= 0.9
    return max(0.06, w)


def _apply_window_cap(entries: list, new_entry: list, window_cap: float):
    """
    Ensure sum of overlapping extras within [t0,t1] does not exceed cap.
    Adjust new_entry weight downward if needed.
    """
    _, meta_new = new_entry
    t0 = float(meta_new.get("timestep_start", 0.0))
    t1 = float(meta_new.get("timestep_end", 1.0))
    w_new = float(meta_new.get("weight", 0.0))
    if w_new <= 0.0:
        return new_entry
    # accumulate overlap
    overlap_sum = 0.0
    for _, m in entries:
        s0 = float(m.get("timestep_start", 0.0))
        s1 = float(m.get("timestep_end", 1.0))
        if max(t0, s0) < min(t1, s1):  # overlaps
            overlap_sum += float(m.get("weight", 0.0))
    if overlap_sum + w_new > window_cap:
        w_new = max(0.0, window_cap - overlap_sum)
        meta_new["weight"] = _round4(w_new)
    return new_entry


def _warn_cfg_rescale(cfg_rescale: float | None):
    if cfg_rescale is None:
        return
    if cfg_rescale > 0.85 or cfg_rescale < 0.5:
        print("[SDXL Adherence] Hint: cfg_rescale ~0.7–0.8 often reduces late-step flips.")


# ----------------- node ----------------------------------------------------


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
                        "tooltip": "Aesthetics / long-tail details.",
                    },
                ),
                "neg_text": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Negative prompt."},
                ),
                "essentials_text": (
                    "STRING",
                    {"multiline": False, "default": "", "tooltip": "Keywords to softly reinforce."},
                ),
                "early_late_mix": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Blend for late_text in custom/short mode.",
                    },
                ),
                "essentials_lock": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Extra weight for essentials in custom/short mode.",
                    },
                ),
                "clip_skip_openclip": (
                    "INT",
                    {"default": 1, "min": 0, "max": 2, "tooltip": "OpenCLIP skip (global)."},
                ),
                "clip_skip_clipL": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2, "tooltip": "CLIP-L skip (local)."},
                ),
                "mode": (
                    ["auto", "core_parity", "custom_only"],
                    {
                        "default": "auto",
                        "tooltip": "auto=fused + soft head/tail ramps on long prompts; core_parity=single fused entry; custom_only=early/late/essentials blending.",
                    },
                ),
            },
            "optional": {
                # purely optional hints (no new visible knobs needed in normal use)
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "tooltip": "For res-aware heuristics (optional).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "tooltip": "For res-aware heuristics (optional).",
                    },
                ),
                "total_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "tooltip": "Optional: sampler total steps for adaptive ramps.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 30.0,
                        "tooltip": "Optional: CFG for adaptive cap.",
                    },
                ),
                "cfg_rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "tooltip": "Optional: warn if extreme (can cause late flips).",
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
        mode="auto",
        width=1024,
        height=1024,
        total_steps=0,
        cfg=0.0,
        cfg_rescale=0.0,
    ):
        # Optional hint
        _warn_cfg_rescale(cfg_rescale if cfg_rescale > 0 else None)

        # Res-aware tweaks for short/medium path
        mix, lock = _res_aware(early_late_mix, essentials_lock, width, height)

        # Count tokens & decide path
        tokens_early = _count_tokens(clip, early_text or "")
        tokens_late = _count_tokens(clip, late_text or "")
        total_tokens = tokens_early + tokens_late
        long_prompt = total_tokens >= LONG_PROMPT_TOKENS
        very_long = total_tokens >= VERY_LONG_TOKENS

        # Adaptive head/tail token slices
        head_n, tail_n = _adaptive_slice_counts(total_tokens)

        # Adaptive ramps (by steps)
        HEAD_RAMP, TAIL_RAMP, ESSENTIALS_RAMP = _ramp_points(
            total_steps if total_steps > 0 else None
        )

        # Adaptive cap (by CFG)
        extra_cap = _cap_for_cfg(cfg if cfg > 0 else None, base=BASE_EXTRA_CAP)

        pos = []

        # ----------------- encoding policy -----------------

        if mode == "core_parity" or (mode == "auto" and long_prompt):
            # ---- CORE-LIKE fused entry ----
            combined = f"{(early_text or '').strip()} {(late_text or '').strip()}".strip()
            cond, pooled = _encode_text_string(clip, combined, clip_skip_openclip, clip_skip_clipL)
            pos.append(_cond_entry(cond, pooled, weight=1.0))

            if mode == "auto":
                extra_weight_budget = extra_cap

                # Early head echo (subject/camera), softly fading out (ramped)
                head_tok = _slice_tokens(clip, combined, head_n=head_n)
                if head_tok is not None and extra_weight_budget > 0.005:
                    h_cond, h_pooled = _encode_text_tokens(
                        clip, head_tok, clip_skip_openclip, clip_skip_clipL
                    )
                    base_w = min(BASE_HEAD_W, extra_weight_budget)
                    for t0, t1, s in HEAD_RAMP:
                        w = min(base_w * s, extra_weight_budget)
                        if w <= 0.0:
                            continue
                        entry = _cond_entry(h_cond, h_pooled, weight=w)
                        entry[1]["timestep_start"] = _round4(t0)
                        entry[1]["timestep_end"] = _round4(t1)
                        # window cap enforcement
                        entry = _apply_window_cap(pos, entry, WINDOW_CAP)
                        w_eff = float(entry[1]["weight"])
                        if w_eff > 0.0:
                            pos.append(entry)
                            extra_weight_budget -= w_eff
                            if extra_weight_budget <= 0.0:
                                break

                # Late tail echo (tail semantics), softly fading in (ramped)
                tail_tok = _slice_tokens(clip, combined, tail_n=tail_n)
                if tail_tok is not None and extra_weight_budget > 0.005:
                    t_cond, t_pooled = _encode_text_tokens(
                        clip, tail_tok, clip_skip_openclip, clip_skip_clipL
                    )
                    base_w = min(BASE_TAIL_W, extra_weight_budget)
                    for t0, t1, s in TAIL_RAMP:
                        w = min(base_w * s, extra_weight_budget)
                        if w <= 0.0:
                            continue
                        entry = _cond_entry(t_cond, t_pooled, weight=w)
                        entry[1]["timestep_start"] = _round4(t0)
                        entry[1]["timestep_end"] = _round4(t1)
                        entry = _apply_window_cap(pos, entry, WINDOW_CAP)
                        w_eff = float(entry[1]["weight"])
                        if w_eff > 0.0:
                            pos.append(entry)
                            extra_weight_budget -= w_eff
                            if extra_weight_budget <= 0.0:
                                break

                # Essentials: early-only, overlap-aware, softly faded; skip on very long
                if (
                    (essentials_text or "").strip()
                    and not very_long
                    and extra_weight_budget > 0.005
                ):
                    base_w = min(lock, ESSENTIALS_BASE_W)
                    eff_w = _overlap_scale(essentials_text, combined, base_w)
                    eff_w = _downweight_if_redundant(essentials_text, combined, eff_w)
                    eff_w = min(eff_w, max(0.0, extra_weight_budget * 0.8))  # keep headroom
                    if eff_w > 0.01:
                        e_list = _encode_as_list(
                            clip, essentials_text, clip_skip_openclip, clip_skip_clipL, weight=1.0
                        )
                        for t0, t1, s in ESSENTIALS_RAMP:
                            w = eff_w * s
                            if w <= 0.0:
                                continue
                            entry = [e_list[0][0], dict(e_list[0][1])]
                            entry[1]["timestep_start"] = _round4(t0)
                            entry[1]["timestep_end"] = _round4(t1)
                            entry[1]["weight"] = _round4(w)
                            entry = _apply_window_cap(pos, entry, WINDOW_CAP)
                            w_eff = float(entry[1]["weight"])
                            if w_eff > 0.0:
                                pos.append(entry)
                        # subtract the *planned* eff_w from budget (ramps consumed it)
                        extra_weight_budget -= eff_w

            # Optional debug
            if _debug_enabled():
                print(
                    f"[Adherence] mode=auto long={long_prompt} tokens={total_tokens} cap={_round4(extra_cap)} head={head_n} tail={tail_n} entries={len(pos)}"
                )

        else:
            # ---- SHORT/MEDIUM prompts: original multi-entry control ----
            pos_early = _encode_as_list(
                clip, early_text, clip_skip_openclip, clip_skip_clipL, weight=1.0
            )
            pos.extend(pos_early)

            if (late_text or "").strip():
                pos_late = _encode_as_list(
                    clip, late_text, clip_skip_openclip, clip_skip_clipL, weight=1.0
                )
                _blend_append(pos, pos_late, scale=mix)

            if (essentials_text or "").strip() and lock > 0.0:
                pos_lock = _encode_as_list(
                    clip, essentials_text, clip_skip_openclip, clip_skip_clipL, weight=1.0
                )
                _blend_append(pos, pos_lock, scale=lock)

            if _debug_enabled():
                print(
                    f"[Adherence] mode={mode} long={long_prompt} tokens={total_tokens} entries={len(pos)}"
                )

        # ----------------- negative (unchanged) -----------------

        neg = _encode_as_list(clip, neg_text, clip_skip_openclip, clip_skip_clipL, weight=1.0)

        # ----------------- validation -----------------

        for name, lst in (("positive", pos), ("negative", neg)):
            if not isinstance(lst, list) or len(lst) == 0:
                raise ValueError(f"{name} conditioning is empty.")
            for i, entry in enumerate(lst):
                if (not isinstance(entry, (list, tuple))) or len(entry) != 2:
                    raise ValueError(f"{name}[{i}] invalid entry type (must be [cond, meta]).")
                cond, meta = entry
                if meta.get("pooled_output", None) is None:
                    if cond.ndim == 3:
                        meta["pooled_output"] = cond.mean(dim=1)
                    else:
                        raise ValueError(
                            f"{name}[{i}] has no pooled_output and cond shape {tuple(cond.shape)}"
                        )
                meta["weight"] = _round4(meta.get("weight", 1.0))
                # Clamp meta timesteps into [0,1]
                if "timestep_start" in meta:
                    meta["timestep_start"] = _round4(
                        max(0.0, min(1.0, float(meta["timestep_start"])))
                    )
                if "timestep_end" in meta:
                    meta["timestep_end"] = _round4(max(0.0, min(1.0, float(meta["timestep_end"]))))

        return (pos, neg)
