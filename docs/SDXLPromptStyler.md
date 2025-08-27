# SDXL Prompt Styler

Purpose: Protect subject tokens up front, inject style cues, normalize negatives, and optionally tweak phrasing by aspect.

## Inputs

- prompt (STRING)
  - Main prompt: subject and key attributes.
- negative (STRING)
  - Negative prompt; normalization keeps factual defects and removes noisy style terms.
- style (PRESET)
  - One of: none, cinematic, product, portrait, toon.
- token_budget (INT)
  - How much to keep up front; encoder enforces 77 tokens.
- strict_keywords (STRING)
  - Comma-separated essentials to always preserve.
- normalize_negatives (BOOLEAN)
  - When true, adds core defects and drops noisy terms.
- width, height (STRING, optional)
  - For aspect-tweaks if dims_json not provided.
- aspect_tweaks (BOOLEAN)
  - Adds small phrases for wide/tall compositions.
- dims_json (STRING)
  - If connected from SmartLatent, provides W/H for aspect logic.

## Outputs

- early_text, late_text, neg_text, essentials_text (STRINGs)

## How it splits

- Early: subject and key constraints (first 40% by sentence-ish split)
- Late: aesthetic extras (the rest)

## Tips

- Keep prompt focused; let the style preset add most look/feel terms.
- If late_text gets long, reduce token_budget or trim it manually.
