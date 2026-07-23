# rlx-ppocrv6

PP-OCRv6 **tiny** and **small** text detection + recognition for RLX.

Runtime inference loads **safetensors** and builds offline-decomposed HIR
([`src/native/`](src/native/)). Host DB post-process and CTC decoding finish the
pipeline. There is **no ONNX Runtime** and **no runtime `rlx-onnx-import`**.

| Tier | Det | Rec | Notes |
|------|-----|-----|--------|
| **tiny** | LCNetV4 + FPN + DB | LCNetV4 + CTC | Fastest; often drops spaces (`HelloOCR`) |
| **small** | wider LCNetV4 + FPN + DB | LCNetV4 + LightSVTR + CTC | Better spacing (`Hello OCR`) |

Medium tier, doc-unwarp, and textline orientation are out of scope.

## Quick start

```bash
just fetch-ppocrv6-tiny   # or fetch-ppocrv6-small
just ppocrv6 -- --tier tiny --model-dir .cache/ppocrv6/tiny --image page.png --device cpu
```

Or with Cargo:

```bash
cargo run -p rlx-ppocrv6 --release --features apple-silicon -- \
  --tier small --model-dir .cache/ppocrv6/small --image page.png --device metal
```

### Model directory

```text
det/model.safetensors              # or ppocrv6_{tier}_det.safetensors
rec/model.safetensors              # or ppocrv6_{tier}_rec.safetensors
rec/keys.txt                       # optional (bundled dicts are the fallback)
```

`just fetch-ppocrv6-*` downloads official Paddle ONNX, rewrites ops for RLX, and
exports safetensors (including `model.safetensors`). Any leftover `.onnx` files
are **export/emit inputs only** — the runner does not load them.

## API

```rust
use rlx_ppocrv6::{PpOcrV6Runner, Tier};
use rlx_runtime::Device;

let runner = PpOcrV6Runner::builder()
    .tier(Tier::Tiny)
    .model_dir(".cache/ppocrv6/tiny")
    .device(Device::Cpu)
    .build()?;
let out = runner.predict_path("page.png")?;
println!("{}", out.text);
# anyhow::Ok(())
```

Facade (optional): `rlx_models::ppocrv6` with feature `ppocrv6`.

## Architecture

| Piece | Role |
|-------|------|
| [`native`](src/native/) | Offline `rlx-onnx-decompose` HIR builders (tiny/small × det/rec) |
| [`rlx`](src/rlx/) | Compile cache + session run (safetensors → HIR) |
| [`detection`](src/detection/) | Host DB box post-process |
| [`recognition`](src/recognition/) | CTC greedy decode + char dicts |
| [`backbone`](src/backbone/) | LCNetV4 tier channel configs |

## Backends

Verified on `hello.png` → readable `Hello OCR` / `HelloOCR`:

| Backend | Feature | Notes |
|---------|---------|--------|
| CPU | (default) | Reference |
| Metal | `metal` / `apple-silicon` | |
| MLX | `mlx` / `apple-silicon` | Lazy fallback for FPN `ConvTranspose2d` |
| wgpu | `gpu` / `apple-silicon` | `--device gpu` |
| CoreML / ANE | `coreml` / `apple-silicon` | Defaults to Neural Engine; avoid `RLX_COREML_UNITS=gpu` on small (SVTR Softmax) |
| Vulkan | `vulkan` | |
| CUDA | `cuda` / `nvidia-gpu` | Verified on NVIDIA (e.g. RTX 3080 Ti) |

```bash
cargo build -p rlx-ppocrv6 --release --features apple-silicon
cargo build -p rlx-ppocrv6 --release --features cuda
```

## Dev: re-emit native graphs

Only needed when regenerating HIR after emit/decompose changes:

```bash
# after: just fetch-ppocrv6-tiny (or small)
python3 scripts/ppocrv6_emit_native.py --tier tiny --task det \
  --onnx .cache/ppocrv6/tiny/det/inference_rlx.onnx --h 96 --w 320
python3 scripts/ppocrv6_emit_native.py --tier tiny --task rec \
  --onnx .cache/ppocrv6/tiny/rec/inference_rlx.onnx --h 48 --w 160
# small: --tier small (rec includes LightSVTR)
```

Fetch/export rewrites (emit input): `HardSigmoid` → `Clip`; `Conv`/`MaxPool`
`auto_pad=SAME_*` → explicit `Pad` + valid op.

## License

Same as the workspace (GPL-3.0).
