# Align Hints To Latent

Purpose: Snap ControlNet hint images (canny, depth, lineart, etc.) to exactly match the latent’s W×H so the UNet grid and hints stay in lockstep.

## Inputs

- latent (LATENT)
  - The latent whose W×H defines target size.
- image (IMAGE)
  - Hint image (H×W×C or B×H×W×C).
- snap_mode (pad_up | downscale_only | resize_round | crop_center)
  - Same semantics as SmartLatent.
- pad_kind (reflect | edge | constant)
  - Padding mode for pad_up/downscale-only residuals.
- pad_value (INT)
  - Constant pad value for pad_kind=constant.
- keep_alpha (BOOLEAN)
  - Preserve alpha if present.

## Outputs

- image_aligned (IMAGE): B×H×W×C
- bbox_json (STRING): {x,y,w,h,W,H}
- width, height (INT)

## Tips

- If ControlNet throws size mismatches, feed dims_json from SmartLatent to ensure your hint node keys off the same W×H.
- Use pad_up for composition-sensitive hints (like edge maps) to avoid AR changes.
