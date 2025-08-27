# Smart Latent (empty or encode)

Purpose: Create an empty latent or encode an input image into a 64-aligned latent with optional VRAM guardrails. Emits dims_json and bbox_json to align hints and crop back after decode.

## Inputs

- vae (VAE)
  - The VAE model used to create or encode latents.
- mode (empty | encode_image)
  - empty: allocate a latent tensor of the requested size.
  - encode_image: convert IMAGE to latent with safe snapping.
- width, height (INT)
  - Requested size. Will be rounded and/or downscaled by policies below.
- snap_mode (pad_up | downscale_only | resize_round | crop_center)
  - pad_up: Letterbox to the next 64-multiple (preserves content).
  - downscale_only: Fit inside lower 64-multiple, then pad residuals.
  - resize_round: Directly resize near the nearest 64 (may change AR).
  - crop_center: Center-crop down to lower 64 (no resize).
- pad_kind (reflect | edge/replicate | constant)
  - Padding mode used when snap_mode adds padding. Reflect avoids seams; edge/replicate is safest; constant uses pad_value.
- pad_value (INT)
  - Constant padding value (0..255) when pad_kind=constant.
- batch (INT)
  - Batch size for empty latents or image batches.
- image (IMAGE)
  - Source image for encode_image mode. Accepts HWC/BHWC/BCHW.
- tile_size, tile_overlap (INT)
  - Tiled VAE encode/decode hints. Auto-detected when supported; falls back to non-tiled.
- force_bchw (BOOLEAN)
  - Internal safety. Leave default.
- use_tiled (BOOLEAN)
  - Use tiled VAE paths when available to reduce VRAM spikes.
- seed (INT)
  - Reserved for future use.
- max_pixels (INT)
  - Upper bound for W*H. If exceeded, uniformly downscales by 64s.
- max_long_side (INT)
  - If >0, pre-resize the input so max(H,W) <= value before snapping.

## Outputs

- latent (LATENT): {"samples": BCHW}
- dims_json (STRING): {"W": int, "H": int, ...}
- width (INT), height (INT)
- bbox_json (STRING): {"x": int, "y": int, "w": int, "h": int, "W": int, "H": int}

## When to use which snap_mode

- pad_up: Default. Best when you want exact composition preserved.
- downscale_only: When you want to avoid up-rounding resolution.
- resize_round: When matching a specific compute budget matters most.
- crop_center: When you can safely trim edges to hit a 64 multiple.

## Tips

- For 24 GB GPUs, tile_size=320 and use_tiled=True are safe defaults.
- Feed bbox_json to Crop By BBox after decode to remove padding.
- dims_json can be passed to Align Hints and the Styler for aspect-aware tweaks.
