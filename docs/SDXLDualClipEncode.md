# SDXL Dual CLIP Encode (pos/neg)

Purpose: Build canonical SDXL conditioning lists (positive and negative) using the pooled-output path so SDXL ADM always receives pooled_output.

## Inputs

- clip (CLIP)
  - The SDXL dual-encoder CLIP from CheckpointLoader.
- early_text (STRING)
  - Instruction-heavy core subject and key attributes. Always included.
- late_text (STRING)
  - Aesthetic long-tail descriptors. Blended with early by early_late_mix.
- neg_text (STRING)
  - Negative prompt.
- essentials_text (STRING)
  - Extra keywords to reinforce with additional weight.
- early_late_mix (FLOAT)
  - Blend weight for late_text (0 = ignore late, 1 = equal weight).
- essentials_lock (FLOAT)
  - Additional weight applied to essentials_text.
- clip_skip_openclip (INT)
  - Skip for global encoder (OpenCLIP). Usually 0 or 1.
- clip_skip_clipL (INT)
  - Skip for local encoder (CLIP-L). Usually 0.
- width, height (INT, optional)
  - Used by light heuristics: bumps lock for long side >1280, reduces mix on small sizes.

## Outputs

- cond_positive (CONDITIONING): list of [cond, {pooled_output, weight}]
- cond_negative (CONDITIONING): list of [cond, {pooled_output, weight}]

## Why pooled_output matters

SDXL’s ADM pathway requires pooled_output present on each conditioning entry. This node uses tokenize -> encode_from_tokens(..., return_pooled=True) to guarantee pooled presence, with fallbacks that synthesize a safe pooled when needed.

## Tips

- If KSampler feels slower, try setting early_late_mix=0 and essentials_lock=0 to keep a single conditioning entry.
- Keep late_text short and impactful; verbose late_text increases token work without much gain.
