# Crop By BBox

Purpose: After decoding an image that was padded or resized, crop back to original content region using bbox_json produced by SmartLatent or AlignHintsToLatent.

## Inputs

- image (IMAGE)
  - Input image to crop (H×W×C or B×H×W×C).
- bbox_json (STRING)
  - {x,y,w,h,W,H} where (W,H) are working dims and (w,h) are content size.
- resize_back (BOOLEAN)
  - If true, resizes the cropped region back to content size (w×h).
- clamp_to_bounds (BOOLEAN)
  - Keep crop within image bounds.
- feather (INT)
  - Soft edge radius in pixels. Useful for compositing.
- expand (INT)
  - Grow crop rect by N px in all directions before cropping.

## Outputs

- image_cropped (IMAGE)
- info_json (STRING): includes original/mapped rects and whether resized back.

## Tips

- Use after VAE decode when SmartLatent used pad_up or downscale_only. It removes letterboxing so your final output matches original content size.
