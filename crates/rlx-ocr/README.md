# rlx-ocr

OCR engine for RLX — native compiled **text detection + recognition** graphs. Loads HuggingFace [`robertknight/ocrs`](https://huggingface.co/robertknight/ocrs) weights as `.safetensors` and runs detection + recognition on every standard RLX backend. No Python and (on the default `rlx` feature) no RTen engine — host-side pad/resize plus RLX graphs.

## Quick start

```bash
# Detection + recognition on a single image (models auto-resolved from --model-dir).
just ocr --model-dir .cache/ocrs --image page.png --device cpu

# or directly:
cargo run -p rlx-ocr --bin rlx-ocr --release -- \
  --model-dir .cache/ocrs --image page.png --device metal
```

CLI flags: `--model-dir DIR` (expects `ocr-*-full.safetensors`) or both `--detection-model PATH` and `--recognition-model PATH`; `--image PATH`; `--device cpu|metal|mlx|cuda|rocm|gpu|vulkan`; `--dry` (skip inference). The `rlx-ocr-convert` bin (feature `convert-rten`) exports legacy `.rten` checkpoints to `.safetensors`.

## Public API

```rust
use rlx_ocr::{OcrRunner, OcrOutput};
use rlx_runtime::Device;

let runner = OcrRunner::builder()
    .model_dir(".cache/ocrs")
    .device(Device::Cpu)
    .build()?;

let out: OcrOutput = runner.predict_path("page.png".as_ref())?;
println!("{}", out.text);

// Or straight from decoded RGB bytes:
let text: String = runner.predict_text("page.png".as_ref())?;
# anyhow::Ok(())
```

Lower-level entry points: [`OcrEngine`](src/engine.rs) / `OcrEngineParams` (detection + recognition pipeline), `ocr_rgb_bytes`, and [`prepare_image`](src/preprocess.rs). Text geometry is returned as [`TextLine` / `TextWord` / `TextChar`](src/text.rs) with `RotatedRect` boxes; CTC decoding lives in [`ctc`](src/ctc.rs).

## Backends & features

| Feature | Enables |
|---------|---------|
| `rlx` (default) | Native RLX detection + recognition graphs + host pad/resize |
| `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan` | Forwarded to `rlx-runtime`; `all-backends` builds one binary accepting all `--device` values |
| `blas-accelerate` | Apple Accelerate (AMX) CPU matmul/conv |
| `convert-rten` | `rlx-ocr-convert` — export `.rten` → `.safetensors` |
| `rten-inference` / `parity-ocrs` | Legacy RTen graph inference / upstream `ocrs` parity oracle (baseline + tests only) |

## Tests & benches

```bash
just test-ocr-backends       # detection + recognition + pipeline on each backend (needs OCR_MODEL_DIR)
just bench-ocr-rlx           # native rlx-ocr latency vs upstream ocrs
just test-ocr-parity         # rlx pipeline vs ocrs reference (parity-ocrs feature)
```
