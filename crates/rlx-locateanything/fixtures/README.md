# Sample image

`sample.jpg` — 640×360 RGB JPEG (from [Ultralytics yolov5 `zidane.jpg`](https://github.com/ultralytics/yolov5/blob/master/data/images/zidane.jpg)).

Used by:

- `rlx_locateanything::fixtures::sample_image_path()` (Rust)
- Default CLI / example when `--image` is omitted
- HF parity `*_real` probes (unless `RLX_LOCATEANYTHING_IMAGE` is set)

Override with any local path:

```bash
export RLX_LOCATEANYTHING_IMAGE=/path/to/your.jpg
```
