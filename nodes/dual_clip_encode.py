# ComfyUI-SDXL-Adherence/nodes/dual_clip_encode.py
import re

# Comfy may expose different clip encode utilities; we keep it generic


def _encode_text(clip, text: str, clip_skip_openclip: int, clip_skip_clipL: int):
    # Attempt to call a unified encode if provided by the CLIP object
    # Fall back to basic encode(text)
    if hasattr(clip, "encode"):
        try:
            return clip.encode(text, clip_skip_g=clip_skip_openclip, clip_skip_l=clip_skip_clipL)
        except TypeError:
            return clip.encode(text)
    raise AttributeError("Provided CLIP object does not support encode(text)")


def _blend_conditionings(cond_a, cond_b, alpha: float):
    if not cond_b or alpha <= 0.0:
        return cond_a
    if alpha >= 1.0:
        return cond_b
    # naive blend by adding entries with weight alpha
    merged = list(cond_a)
    for c in cond_b:
        merged.append(
            {
                "conditioning": c.get("conditioning", c),
                "pooled_output": c.get("pooled_output", None),
                "weight": alpha,
            }
        )
    return merged


def _approx_token_len(text: str) -> int:
    # crude fallback: words + punctuation chunks
    toks = re.findall(r"\w+|[^\s\w]", text)
    return max(1, len(toks))


def _token_len(clip, text: str) -> int:
    # Try model tokenizer, else fallback
    try:
        if hasattr(clip, "tokenize"):
            t = clip.tokenize(text)
            if hasattr(t, "shape"):
                return int(t.shape[-1])
            try:
                return len(t)
            except Exception:
                pass
    except Exception:
        pass
    return _approx_token_len(text)


def _truncate_to_tokens(clip, text: str, budget: int) -> tuple[str, str]:
    if not text or not text.strip():
        return "", ""
    # Greedy by chunks, then words
    chunks = [p.strip() for p in re.split(r"[;\n]+|,(?=\s*[^\d])", text) if p.strip()]
    kept = []
    for i, ch in enumerate(chunks):
        test = ", ".join(kept + [ch]) if kept else ch
        if _token_len(clip, test) <= budget:
            kept.append(ch)
        else:
            # try finer word split for the last chunk
            words = [w for w in re.split(r"\s+", ch) if w]
            for j in range(len(words)):
                test2 = ", ".join(kept + [" ".join(words[: j + 1])])
                if _token_len(clip, test2) <= budget:
                    pass
                else:
                    kept_text = ", ".join(kept + [" ".join(words[:j])]).strip(", ")
                    rest = []
                    if j < len(words):
                        rest.append(" ".join(words[j:]).strip())
                    rest.extend(chunks[i + 1 :])
                    return kept_text, ", ".join([r for r in rest if r]).strip(", ")
            # whole chunk fits if loop didn't return
            kept.append(ch)
    kept_text = ", ".join(kept).strip(", ")
    return kept_text, ""


def _apply_te_loras(clip, paths_str: str, scale_g: float, scale_l: float):
    if not paths_str:
        return
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    for p in paths:
        applied = False
        for meth in ("apply_te_lora", "load_te_lora", "add_te_lora", "load_text_lora"):
            if hasattr(clip, meth):
                try:
                    getattr(clip, meth)(p, scale_g, scale_l)
                    applied = True
                    break
                except Exception:
                    continue
        if not applied and hasattr(clip, "load_lora"):
            try:
                # best-effort generic API
                clip.load_lora(p, text_encoder=True, unet=False, scale_g=scale_g, scale_l=scale_l)
            except Exception:
                pass


class SDXLDualClipEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "role": (["positive", "negative"],),
                "early_text": ("STRING", {"multiline": True}),
                "late_text": ("STRING", {"multiline": True, "default": ""}),
                "neg_text": ("STRING", {"multiline": True, "default": ""}),
                "essentials_text": ("STRING", {"multiline": False, "default": ""}),
                "early_late_mix": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
                "essentials_lock": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0}),
                "token_budget": ("INT", {"default": 77, "min": 48, "max": 150}),
                "clip_skip_openclip": ("INT", {"default": 1, "min": 0, "max": 2}),
                "clip_skip_clipL": ("INT", {"default": 0, "min": 0, "max": 2}),
            },
            "optional": {
                "te_lora_paths": ("STRING", {"default": ""}),
                "te_lora_scales_openclip": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.5},
                ),
                "te_lora_scales_clipL": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.5},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "encode"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def encode(
        self,
        clip,
        role,
        early_text,
        late_text,
        neg_text,
        essentials_text,
        early_late_mix,
        essentials_lock,
        token_budget,
        clip_skip_openclip,
        clip_skip_clipL,
        te_lora_paths="",
        te_lora_scales_openclip=0.7,
        te_lora_scales_clipL=0.6,
    ):
        # negative path
        if role == "negative":
            text = neg_text or ""
            cond = _encode_text(clip, text, clip_skip_openclip, clip_skip_clipL)
            return (cond,)

        # positive path
        # Budget early text to token limit; overflow goes to late
        early_kept, early_over = _truncate_to_tokens(clip, early_text or "", token_budget)
        late_combined = ", ".join([t for t in [early_over, late_text] if t]).strip(", ")

        cond_early = (
            _encode_text(clip, early_kept, clip_skip_openclip, clip_skip_clipL)
            if early_kept
            else []
        )
        cond_late = (
            _encode_text(clip, late_combined, clip_skip_openclip, clip_skip_clipL)
            if late_combined
            else []
        )
        cond = _blend_conditionings(cond_early, cond_late, early_late_mix)

        if essentials_text and essentials_lock > 0.0:
            cond_lock = _encode_text(clip, essentials_text, clip_skip_openclip, clip_skip_clipL)
            for c in cond_lock:
                cond.append(
                    {
                        "conditioning": c.get("conditioning", c),
                        "pooled_output": c.get("pooled_output", None),
                        "weight": essentials_lock,
                    }
                )

        # Best-effort TE-LoRA injection (text encoders only)
        _apply_te_loras(clip, te_lora_paths, te_lora_scales_openclip, te_lora_scales_clipL)

        return (cond,)
