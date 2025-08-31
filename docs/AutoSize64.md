# Auto-Size 64 (MP)

One-control helper that picks a target megapixel size and snaps W/H to 64-multiples.

- Size: Auto / 1.0MP / 1.5MP / 2.0MP
- Inputs: optional image (to infer aspect)
- Outputs: width, height, dims_json
- Default: Auto (uses GPU VRAM to pick ~1.2–1.5MP)

Usage: Place before VAE Encode (or Smart Latent empty mode) and wire width/height.
