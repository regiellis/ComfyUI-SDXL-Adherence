# Crop By BBox

Remove any padding that was added earlier so your final image is exactly the original content size.

When to use

- After VAE Decode, if you used SmartLatent snap_mode=pad_up or downscale_only

What to set

- image: the decoded image
- bbox_json: connect from SmartLatent (or AlignHints if that was your source)
- resize_back: On to get back to the original content width/height
- feather: add a soft edge for smoother compositing (optional)
- expand: grow the crop a little before cutting (optional)

Outputs

- image_cropped: your final image without letterbox padding
- info_json: details about the crop it applied

Troubleshooting

- “Nothing changes when I crop” → Your snap_mode likely didn’t add padding. That’s fine.
- “Edges look too sharp” → Increase feather a bit.
