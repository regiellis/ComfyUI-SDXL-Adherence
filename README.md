
---

# SDXL Adherence ⚡

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-00bcd4?logo=pytorch)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Custom nodes that make SDXL follow your prompt better and run smoother on single-GPU rigs.

You don’t need a new model to ship good work. This pack patches SDXL’s weak spots: adherence, negatives, and high-res stability.

> [!Caution]
> This node is a work in progress and may not cover all use cases. Please test thoroughly and provide feedback.

> [!IMPORTANT]
> This node was designed and created with my personal needs and goals in mind. This may not fit your specific use case or project and you may need to adapt it accordingly. I encourage you to experiment and modify it to better suit your needs.
>
> If you find it useful, please consider supporting my work by starring the repository or sharing it with others who might benefit from it. But please understand, this is a personal project and released only in an effort to help others and as with most open-source projects, done in my free time with no support or pay.
>
> Please keep that in mind when leaving issues, comments, etc.

---

## Why this exists

* Flux and other DiT-style models are great, but most are aimed at 48–80 GB cards or cloud inference.
* SDXL still has the strongest ecosystem (LoRAs, ControlNets, adapters, VAEs, fine-tunes).
* Out of the box, SDXL can drift from your subject, waste tokens, and choke on odd aspect ratios.
* Recent ComfyUI builds handle long prompt chunking silently (no more 77-token warnings), but they don’t **prioritize which tokens land in which chunk, clean negatives, or snap resolutions safely**. These nodes add that missing layer of *control, consistency, and guardrails*.

These nodes fix those gaps so you can keep moving fast on a single 24 GB GPU or less.

---

## Quick start (2 minutes)

1. Install in `ComfyUI/custom_nodes` (see Install below) and restart ComfyUI.
2. Load an SDXL checkpoint → get model, clip, vae.
3. Drop in these nodes:

   * Smart Latent (mode: *Empty* for new images, or *Encode Image* for img2img/inpaint)
   * SDXL Prompt Styler (pick a style; keep “Normalize Negatives” on)
   * SDXL Dual CLIP Encode (wire the 4 texts from the styler)
4. KSampler: connect model + latent + cond+/cond-.
5. VAE Decode (tiled if needed). If you padded at encode, use Crop By BBox at the end.

That’s it. Bigger images? Smart Latent snaps sizes to 64 safely and uses tiled VAE automatically when available.

---

## Node cheat sheet

* **SDXL Prompt Styler 🎨**

  * Protects subject (early/late split), adds style preset, cleans negs.
  * Essentials automation: strategy controls (off/conservative/balanced/aggressive), allow/block lists, and max\_essentials cap.
  * Auto pivot for early/late based on prompt length; manual early\_ratio available.
  * Good defaults: style=cinematic/portrait; normalize\_negatives=on; essentials\_strategy=balanced.

* **SDXL Dual CLIP Encode 🔗**

  * Builds pos+neg conditioning with pooled_output (stable for SDXL ADM).
  * Modes: `auto` (default), `core_parity`, `custom_only` — see Encoder modes below.
  * Auto adds soft-ramped head/tail assists and early-only essentials on long prompts.
  * Resolution-aware nudges; optionally wire sampler context (steps→`total_steps`, cfg→`cfg`) for adaptive ramps.
  * Good defaults: early_late_mix=0.4; essentials_lock=0.35.

* **Smart Latent 📐**

  * Makes empty latents or encodes images at any size. Snaps to 64 safely.
  * Resolution presets dropdown with aspect ratios; selecting a preset overrides width/height (empty mode).
  * Good defaults: snap\_mode=pad\_up; tile\_size=320; keep alpha-safe padding.

* **Align Hints To Latent (helper)**

  * Resizes/pads canny/depth/lineart/etc. to exactly match latent W×H.

* **Crop By BBox (helper)**

  * After decode, crops padded generations back to the original content.

* **Auto-Size 64 (MP) (helper)**

  * Pick target megapixels and snap W/H to 64-multiples. Size: Auto/1.0MP/1.5MP/2.0MP.

* **Negative Prompt Helper (helper)**

  * Toggle Realism Negs to merge a vetted pack with your negatives and de-dup.
  * Helps counteract SDXL’s tendency toward plastic skin / CGI by merging a curated set automatically.

* **Post Polish (Film Touch) (helper)**

  * Tiny tone S-curve + subtle grain for micro-texture. Place last.

---

### Encoder modes

* **Auto (default)** — Uses a single fused encode (like core) on long prompts, then adds tiny, **soft-ramped** assists (head/tail echoes; early-only essentials) to improve adherence without new knobs.
* **Core parity** — Bypasses assists; behaves like the stock ComfyUI SDXL encoder for maximum predictability.
* **Custom only** — Keeps short/medium-prompt controls (early/late/essentials multi-entry blending) for aesthetic steering.

> [!IMPORTANT]
> **Seeing a sudden, drastic change in the KSampler preview mid-run?**  
> That’s usually a *conditioning schedule* issue (guidance turning on/off too abruptly), or an overly strong guidance stack. Try these:
>
> 1) **Switch `mode`**  
>    * For long prompts: set `mode = core_parity` to match core exactly.  
>    * For short/medium prompts: `mode = custom_only` if you want early/late control without long-prompt assists.
>
> 2) **Pass sampler context (so ramps adapt)**  
>    If available, wire your KSampler **`steps`** and **`cfg`** to the encoder’s optional inputs (`total_steps`, `cfg`).  
>    The encoder will auto-shrink ramps and extra weights at high CFG / low step counts to prevent late flips.
>
> 3) **Keep guidance moderate**  
>    * **CFG** ~ 5.5–7.0  
>    * **CFG rescale** ~ 0.7–0.8 (extremes can cause late instability)
>
> 4) **Make the canvas stable**  
>    * Use **64-safe sizes** (e.g., 1536×896)  
>    * Avoid stacking many heavy **LoRAs**; keep combined LoRA influence reasonable  
>    * If using other conditioning nodes (regional prompts, control adapters), try disabling them to isolate the cause
>
> 5) **Essentials got too pushy?**  
>    Leave `essentials_text` empty for very long prompts, or simplify it. The encoder already *time-limits* essentials to early steps, but redundant terms can still over-constrain.
>
> **Debugging:** set environment variable `ADHERENCE_DEBUG=1` to print a one-line summary (tokens, caps, ramp slices) for each run.


## Install

Clone into your ComfyUI custom\_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI-SDXL-Adherence.git
```

Restart ComfyUI.

---

## Any-size + tiled VAE (what this solves)

Smart Latent accepts any H×W and safely snaps to 64-multiples:

* pad\_up: letterbox with reflect/edge/constant (alpha-safe), preserves content
* downscale\_only: fit inside lower 64-multiple, then pad the small residuals
* resize\_round: resize near the nearest 64 (may change aspect)
* crop\_center: centered crop down to the lower 64 (no resize)

If your VAE has tiled encode/decode, the wrappers pick the right signature automatically and fall back to non-tiled if needed. This lets you push larger sizes on 24 GB GPUs with fewer OOMs.

Outputs include dims\_json and bbox\_json so you can align hints 1:1 and crop back after decoding.

---

## Prompt adherence beyond 77 tokens

* ComfyUI SDXL CLIP silently splits long prompts into 75-token chunks. That means you won’t see the old “77 token” warning, but the limit is still there.
* Extra tokens **do still matter** up to \~225–250, but unmanaged they can dilute subjects.
* These nodes:

  * Bias important tokens (subject, camera, lighting) into the first chunk.
  * Drop filler/duplicates to fit the token budget.
  * Give you optional reporting so you know what stayed, what was dropped, and where it landed.

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

* Sampler: DPM++ 2M SDE Karras, 28 steps, CFG 6.0, rescale 0.8, ETA=0
* Smart Latent: snap\_mode=pad\_up, tile\_size=320
* Prompt Styler: normalize\_negatives=True, style=cinematic/portrait/etc.
* Dual CLIP Encode: early\_late\_mix=0.4; essentials\_lock=0.35 (auto-bumps on >1280px)

---

## Speed and VRAM tips

* Prefer pad\_up over full-resize to keep detail without extra compute.
* If you see OOM, lower tile\_size (e.g., 320 → 256) or reduce the long side.
* Keep ControlNet hints aligned via Align Hints To Latent to avoid wasted steps.
* Avoid stacking many heavy LoRAs at once; mix fewer, stronger ones.

---

## Troubleshooting

* NoneType pooled\_output or ADM crash: always run conditioning through SDXL Dual CLIP Encode from this pack.
* Sampler shape mismatch: ensure hints go through Align Hints To Latent and that latent W×H match the sampler.
* Washed-out style or broken subject: turn down early\_late\_mix, bump essentials\_lock, or simplify the style.
* Output size wrong after decode: use Crop By BBox to get back to your original content size.

---

## Why not just switch to Flux?

Flux and other DiT-style models are promising:

* Better prompt adherence out of the box
* Longer context window

But today they’re aimed at bigger GPUs + cloud inference. On consumer cards, you’ll hit VRAM ceilings, and the support ecosystem (LoRAs, ControlNets, adapters) is still catching up.

Flux Dev, Flux Krea, and Flux ControlNets are improving fast — but licensing is restrictive and VRAM demands are steep. SDXL is still open for commercial work and runs well locally. With these adherence nodes, SDXL remains competitive in image quality while avoiding VRAM and licensing headaches.

---

## Documentation

* Smart Latent: docs/SmartLatent.md
* SDXL Dual CLIP Encode: docs/SDXLDualClipEncode.md
* SDXL Prompt Styler: docs/SDXLPromptStyler.md
* Align Hints To Latent: docs/AlignHintsToLatent.md
* Crop By BBox: docs/CropByBBox.md

---

## License

MIT — use it, hack it, ship it. Don’t sell it as “secret sauce.”
