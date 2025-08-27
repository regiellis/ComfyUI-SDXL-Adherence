# Align Hints To Latent

Make your hint images (canny, depth, lineart…) exactly match the latent size, so ControlNet and the UNet grid stay in sync.

What it does

- Resizes/pads/crops your hint image to the latent’s width/height
- Returns bbox_json so you can crop back after decoding if you need

What to set

- latent: connect from SmartLatent
- image: your hint image
- snap_mode: usually pad_up to keep the hint’s composition
- pad_kind: reflect (nicest) or edge (safest)
- keep_alpha: On if your hint has an alpha you want to preserve

Outputs

- image_aligned: exact B×H×W×C to feed ControlNet
- width/height: the size it produced
- bbox_json: for crop-back later

Tips

- If ControlNet complains about size, make sure its input is this image_aligned and that dims_json/latent come from the same SmartLatent.
- For delicate edges/lines, prefer pad_up so you don’t squeeze or stretch.
