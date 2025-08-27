# ComfyUI-SDXL-Adherence ⚡

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-00bcd4?logo=pytorch)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/VRAM-24GB-blue)](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)

Custom nodes to make SDXL prompt-faithful and workflow-friendly on consumer GPUs (RTX 3090, 4090, etc.)

Instead of waiting on every new architecture to trickle down, this pack patches SDXL’s weak spots.

---

## Why this exists

- Flux and other DiT-style models are exciting — they handle longer prompts and follow instructions more tightly. But today they’re built for enterprise-class cards (48–80 GB). On a single 3090/4090, you’ll hit VRAM walls fast.
- Ecosystem reality: SDXL still has the LoRAs, ControlNets, adapters, VAEs, and fine-tunes that make it practical.
- Out of the box, SDXL sometimes struggles with prompt adherence, negatives, and high-res aspect ratios.

This repo adds a minimal set of nodes that solve those gaps, so you can keep shipping with SDXL on “small-guy” rigs.

---

## Nodes included

### 1. SDXL Adherence Prompt Styler 🎨

- Splits your prompt into early vs late chunks
- Injects style presets (cinematic, product, portrait, toon…)
- Normalizes negatives: keeps real defects (extra fingers, watermark), drops style terms that would fight your chosen look
- Can tweak phrasing based on aspect ratio

Why: SDXL burns through tokens fast. This node keeps your subject protected while letting the style do its thing.

---

### 2. SDXL Dual CLIP Encode (pos/neg) 🔗

- Encodes positive + negative in one pass
- Blends early/late/essentials with proper weights
- Always returns valid pooled_output (no more NoneType.shape crashes in SDXL ADM)
- Resolution-aware: bumps essentials lock when long side >1280px

---

### 3. Smart Latent (empty or encode) 📐

- Makes empty latents (fresh gens) or encodes images (img2img/inpaint)
- Snap-to-64 policies: pad_up (default), downscale_only, resize_round, crop_center
- Emits width, height, dims_json, and bbox_json for other nodes

---

### 4. Align Hints To Latent (helper)

- Ensures canny/depth/lineart/etc. are resized/padded to exactly match the latent’s W×H
- Prevents ControlNet drift and sampler shape errors

### 5. Crop By BBox (helper)

- After decode, crops padded generations back to original size
- Supports feathering + resize-back

---

## Install

Clone into your ComfyUI custom_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI-SDXL-Adherence.git
```

Restart ComfyUI.

---

## How to use

1. Load SDXL model with CheckpointLoader → model, clip, vae.
2. SmartLatent:
	- mode="empty" → start fresh
	- mode="encode_image" → img2img/inpaint (handles non-64 dims automatically)
3. SDXL Adherence Prompt Styler: outputs early_text, late_text, neg_text, essentials_text.
4. SDXL Dual CLIP Encode: input clip + 4 texts → cond_positive, cond_negative.
5. KSampler: model + latent + conds.
6. VAE Decode (tiled if needed).
7. (Optional) Crop By BBox with bbox_json if you padded at encode.

---

## Recommended defaults

- Sampler: DPM++ 2M SDE Karras, 28 steps, CFG 6.0, rescale 0.8, ETA=0
- SmartLatent: snap_mode=pad_up, tile_size=320
- PromptStyler: normalize_negatives=True, style=cinematic/portrait/etc.
- DualClipEncode: early_late_mix=0.4, essentials_lock=0.35 (auto-bumps on >1280px)

---

## Why not just switch to Flux?

Flux and other DiT-style models are promising:

- Better prompt adherence out of the box
- Longer context window

But today they’re aimed at bigger GPUs + cloud inference. On consumer cards, you’ll hit VRAM ceilings, and the support ecosystem (LoRAs, ControlNets, adapters) is still catching up.

SDXL already has the tools and community, and with these nodes, it closes the gap on adherence — so you can keep working efficiently on a single 24 GB GPU until newer archs are both hardware-friendly and ecosystem-ready.

---

## License

MIT — use it, hack it, ship it.
Just don’t try to sell it as “secret sauce.”

