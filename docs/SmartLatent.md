# Smart Latent (empty or encode)

Make a new canvas to paint on (empty) or bring your image in (encode), and the node will snap the size to what SDXL likes (multiples of 64) without messing up your composition.

What it does

- Empty: creates a blank latent at your chosen size.
- Encode: takes any image size and safely snaps it to 64-multiples (pad, resize, or crop) so sampling won’t error.
- Gives you two handy strings: dims_json (final size) and bbox_json (how to crop back later if we padded).

When to use

- Starting a new generation (empty)
- Img2Img, inpaint, or upscaling flows (encode)
- Any time your image is not a clean 64-multiple and you want zero fuss

Recommended settings

- snap_mode: pad_up (keeps composition, adds letterboxing if needed)
- use_tiled: On
- tile_size: 320
- max_long_side: 0 (off) unless you hit VRAM issues with very large images

What to set (only the useful ones)

- mode: empty for fresh images; encode_image for img2img/inpaint
- resolution: quick presets with aspect ratios; selecting one overrides width/height (in empty mode)
- width/height: your target size (the node will adjust as needed); used when resolution is "Use width/height"
- snap_mode:
  - pad_up: preserve composition (default)
  - downscale_only: fit inside the lower 64-multiple, then tiny pads
  - resize_round: change size directly (may shift aspect)
  - crop_center: trim edges to hit a 64 multiple
- pad_kind: reflect (nicest), edge (safest), or constant (use pad_value)
- use_tiled + tile_size: keeps VRAM stable for big images

Outputs (what you use)

- latent: plug into KSampler
- width / height: final working size (64-multiples)
- dims_json: useful to tell other nodes the exact size (optional)
- bbox_json: use with Crop By BBox after decoding to remove padding (optional)

Quick recipe

1) For a new generation: mode=empty, width=1024, height=1024, snap_mode=pad_up, use_tiled=On.
2) For img2img: mode=encode_image, plug your IMAGE into image, keep snap_mode=pad_up. Later, use Crop By BBox with bbox_json to remove letterboxing.

Common issues

- “My output is slightly bigger than I set” → pad_up rounded up to the nearest 64. If you want to avoid that, use downscale_only.
- “Gray bars on the sides” → that’s the padding. Use Crop By BBox after decoding to remove it.
- “Out of memory at big sizes” → turn on use_tiled and try tile_size=320; also set max_long_side (e.g., 1536) to gently pre-shrink huge inputs.

Tip

- Pass dims_json to Align Hints To Latent and the Prompt Styler (aspect tweaks) so everything stays in sync.
