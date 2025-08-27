# SDXL Dual CLIP Encode (pos/neg)

Turn your text into SDXL-friendly conditioning. Positive goes in one side, negative in the other, and we make sure SDXL gets everything it needs.

What it does

- Encodes your 4 texts: early, late, essentials (positive) and negative
- Ensures a special “pooled_output” is always present (prevents SDXL ADM errors)
- Lets you blend late aesthetics softly with early subject

When to use

- Always, right before KSampler (connect to clip from CheckpointLoader)

What to set (simple)

- early_text: your main subject and must-haves
- late_text: vibe/extra aesthetics (optional)
- essentials_text: extra keywords to “lock in” (optional)
- neg_text: things to avoid
- early_late_mix: how much late_text to blend (start at 0.4)
- essentials_lock: how strongly to reinforce essentials (start at 0.35)

Quick recipes

- Fast: only early_text + neg_text; set early_late_mix=0 and essentials_lock=0 (one conditioning entry, slightly faster sampling)
- Balanced: early_text + short late_text, early_late_mix≈0.3–0.5, essentials_lock≈0.25–0.45
- Strong control: add essentials_text (e.g., “red dress, backlighting”) and set essentials_lock≈0.4–0.6

Performance note

- More positive entries (early + late + essentials) can make each sampler step a bit slower. If speed dips, try lowering early_late_mix and/or essentials_lock, or leave late empty.

Troubleshooting

- “ADM pooled_output error” → This node prevents it automatically. If you still see issues, make sure the clip input comes from an SDXL loader.
- “Not following my style” → Move some terms from late into essentials_text and raise essentials_lock.
