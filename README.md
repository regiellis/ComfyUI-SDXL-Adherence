# SDXL Adherence ⚡

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-00bcd4?logo=pytorch)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/VRAM-24GB-blue)](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)

Custom nodes that make SDXL follow your prompt better and run smoother on single-GPU rigs (3090/4090).

You don’t need a new model to ship good work. This pack patches SDXL’s weak spots: adherence, negatives, and high-res stability.

---

## Why this exists

- Flux and other DiT-style models are great, but most are aimed at 48–80 GB cards or cloud inference.
- SDXL still has the strongest ecosystem (LoRAs, ControlNets, adapters, VAEs, fine-tunes).
- Out of the box, SDXL can drift from your subject, waste tokens, and choke on odd aspect ratios.

These nodes fix that so you can keep moving fast on a single 24 GB GPU.

---

## Quick start (2 minutes)

1. Install in ComfyUI/custom_nodes (see Install below) and restart ComfyUI.
2. Load an SDXL checkpoint → get model, clip, vae.
3. Drop in these nodes:
	- Smart Latent (mode: Empty for new images, or Encode Image for img2img/inpaint)
	- SDXL Prompt Styler (pick a style; keep “Normalize Negatives” on)
	- SDXL Dual CLIP Encode (wire the 4 texts from the styler)
4. KSampler: connect model + latent + cond+/cond-.
5. VAE Decode (tiled if needed). If you padded at encode, use Crop By BBox at the end.

That’s it. Bigger images? Smart Latent snaps sizes to 64 safely and uses tiled VAE automatically when available.

---

## Node cheat sheet

- SDXL Prompt Styler 🎨
	- Protects subject (early/late split), adds style preset, cleans negs.
	- Essentials automation: strategy controls (off/conservative/balanced/aggressive), allow/block lists, and max_essentials cap.
	- Auto pivot for early/late based on prompt length; manual early_ratio available.
	- Good defaults: style=cinematic/portrait; normalize_negatives=on; essentials_strategy=balanced.

- SDXL Dual CLIP Encode 🔗
	- Encodes pos+neg together. Always provides pooled_output (stable SDXL ADM).
	- Good defaults: early_late_mix=0.4; essentials_lock=0.35 (auto-bumps if long side >1280).

- Smart Latent 📐
	- Makes empty latents or encodes images at any size. Snaps to 64 safely.
	- Resolution presets dropdown with aspect ratios; selecting a preset overrides width/height (empty mode).
	- Good defaults: snap_mode=pad_up; tile_size=320; keep alpha-safe padding.

- Align Hints To Latent (helper)
	- Resizes/pads canny/depth/lineart/etc. to exactly match latent W×H.

- Crop By BBox (helper)
	- After decode, crops padded generations back to the original content.

- Auto-Size 64 (MP) (helper)
	- Pick target megapixels and snap W/H to 64-multiples. Size: Auto/1.0MP/1.5MP/2.0MP.

- Negative Prompt Helper (helper)
	- Toggle Realism Negs to merge a vetted pack with your negatives and de-dup.

- Post Polish (Film Touch) (helper)
	- Tiny tone S-curve + subtle grain for micro-texture. Place last.

---

## Install

Clone into your ComfyUI custom_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI-SDXL-Adherence.git
```

Restart ComfyUI.

---

## Any-size + tiled VAE (what this solves)

Smart Latent accepts any H×W and safely snaps to 64-multiples:

- pad_up: letterbox with reflect/edge/constant (alpha-safe), preserves content
- downscale_only: fit inside lower 64-multiple, then pad the small residuals
- resize_round: resize near the nearest 64 (may change aspect)
- crop_center: centered crop down to the lower 64 (no resize)

If your VAE has tiled encode/decode, the wrappers pick the right signature automatically and fall back to non-tiled if needed. This lets you push larger sizes on 24 GB GPUs with fewer OOMs.

Outputs include dims_json and bbox_json so you can align hints 1:1 and crop back after decoding.

---

## Usage diagram

```text
[CheckpointLoader] -> (model, clip, vae)
	|                 |      |
	|                 |      v
	|                 |  [SmartLatent]
	|                 |     └─> (latent, dims_json, bbox_json, W, H)
	|                 v
	|           [SDXL Prompt Styler]
	|                 └─> (early, late, neg, essentials)
	v
[SDXL Dual CLIP Encode]
	└─> (cond_positive, cond_negative)

(cond+, cond-) + (model, latent) -> [KSampler] -> [VAE Decode]
											└─> (image)

Optional helpers:

Image → Auto-Size 64 → Smart Latent/Encode

Neg → Negative Prompt Helper → SDXL Dual CLIP Encode (negative)

… → VAE Decode → Post Polish → (final)
Optional: [Crop By BBox] with bbox_json -> (image cropped)
```

---

## Recommended defaults

- Sampler: DPM++ 2M SDE Karras, 28 steps, CFG 6.0, rescale 0.8, ETA=0
- Smart Latent: snap_mode=pad_up, tile_size=320
- Prompt Styler: normalize_negatives=True, style=cinematic/portrait/etc.
- Dual CLIP Encode: early_late_mix=0.4, essentials_lock=0.35 (auto-bumps on >1280px)

---

## Speed and VRAM tips

- Prefer pad_up over full-resize to keep detail without extra compute.
- If you see OOM, lower tile_size (e.g., 320 → 256) or reduce the long side.
- Keep ControlNet hints aligned via Align Hints To Latent to avoid wasted steps.
- Avoid stacking many heavy LoRAs at once; mix fewer, stronger ones.

---

## Troubleshooting

- NoneType pooled_output or ADM crash: always run conditioning through SDXL Dual CLIP Encode from this pack.
- Sampler shape mismatch: ensure hints go through Align Hints To Latent and that latent W×H match the sampler.
- Washed-out style or broken subject: turn down early_late_mix, bump essentials_lock, or simplify the style.
- Output size wrong after decode: use Crop By BBox to get back to your original content size.

---

## Why not just switch to Flux?

Flux and other DiT-style models are promising:

- Better prompt adherence out of the box
- Longer context window

But today they’re aimed at bigger GPUs + cloud inference. On consumer cards, you’ll hit VRAM ceilings, and the support ecosystem (LoRAs, ControlNets, adapters) is still catching up.

SDXL already has the tools and community. With these nodes, it closes much of the adherence gap so you can keep working efficiently on a single 24 GB GPU until newer archs are both hardware-friendly and ecosystem-ready.

---

## Documentation

- Smart Latent: docs/SmartLatent.md
- SDXL Dual CLIP Encode: docs/SDXLDualClipEncode.md
- SDXL Prompt Styler: docs/SDXLPromptStyler.md
- Align Hints To Latent: docs/AlignHintsToLatent.md
- Crop By BBox: docs/CropByBBox.md

---

## License

MIT — use it, hack it, ship it. Don’t sell it as “secret sauce.”

