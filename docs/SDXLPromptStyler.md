# SDXL Prompt Styler

Helps your prompt “stick” by keeping the subject up front and adding a clean set of style cues. It can also tidy your negatives so they don’t fight the look you want.

What it does

- Splits your prompt into: Early (subject) and Late (aesthetic extras)
- Adds a style preset (cinematic, product, portrait, toon…)
- Normalizes negative prompt (keeps true defects like “extra fingers”, removes noisy style words)
- Optionally adds a tiny hint for wide/tall frames

When to use

- Before Dual CLIP Encode. The outputs plug straight in.

What to set (simple)

- prompt: subject and must-haves
- negative: things to avoid (it will clean this up for you)
- style: pick a preset (or “none”)
- normalize_negatives: On (recommended)
- token_budget: leave at default unless your prompt is very long

Extra knobs (optional)

- strict_keywords: words you absolutely want preserved
- aspect_tweaks: On to add tiny phrasing based on aspect
- dims_json: connect from SmartLatent so aspect tweaks are accurate

Quick recipes

- Cinematic portraits: style=cinematic, normalize_negatives=On
- Clean product shots: style=product, normalize_negatives=On, add essentials later in Dual CLIP
- Toon look: style=toon, keep the prompt simple

Troubleshooting

- “It’s adding words I don’t like” → switch style to “none”, or trim your prompt after this node
- “Late text too long” → lower token_budget or tone down phrasing in the original prompt
