# rlx-locateanything

[NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) on RLX — MoonViT-SO-400M + MLP projector + Qwen2.5-3B-Instruct with MTP box decoding.

## Setup

**Hugging Face cache (recommended)** — weights live under `~/.cache/huggingface/hub/` (or `$HF_HOME`):

### Speed

This is a **3B VLM** (299 vision tokens at 640px side + hybrid MTP decode). Expect **minutes on CPU**.

| Cause | Mitigation |
|--------|------------|
| Default build is **CPU-only** (`just locateanything`) | `just locateanything-demo` uses smaller image + fewer tokens |
| First run **compiles** MoonViT + LM graphs | Skip `--warmup` for one-shot; reuse session for many images |
| Large images → many patches | `--max-image-side 384` (or `320`) |
| Long decode | `--max-tokens 32`, `--generation-mode fast` |
| **OOM** on WGPU / multi-backend bench | Mmap LM + prompt-only embed rows; WGPU uses **one KV bucket** (1024 cap) and GPU-resident KV (no per-step host mirror). Bench: `just bench-locateanything-backends` (one subprocess per backend). Single backend: `--device wgpu --no-isolate`. |

```bash
just locateanything-demo   # ~2–4 min CPU typical (384px, 32 tokens)
```

**GPU backends:** build with `--features apple-silicon` (Metal + MLX + wgpu), `cuda`, or `all-backends`. `just locateanything-demo` picks `apple-silicon` on macOS and `nvidia-gpu` on Linux when available. MoonViT uses decomposed 2D RoPE on MLX/wgpu/Vulkan/Metal. Until per-backend GPU parity lands, **vision + LM graphs run on the CPU host** for every non-CPU `--device` (Metal/MLX/CUDA/WGPU/Vulkan) so grounding output matches CPU/HF; the CLI still reports your chosen device. GPU-resident KV is disabled while on the CPU host path.

## Setup

```bash
# RLX (writes into HF cache via hf-hub)
just fetch-locateanything

# Or Hugging Face CLI
huggingface-cli download nvidia/LocateAnything-3B

# Or rlx-locateanything binary
cargo run -p rlx-locateanything --features hf-download --release -- --download
```

No env var required for Rust/CLI: [`default_model_dir()`] / [`LocateAnythingSession::open_default()`] resolve the cached snapshot.

Optional override:

```bash
export RLX_LOCATEANYTHING_DIR=/path/to/snapshot   # explicit dir
export HF_HOME=~/.cache/huggingface               # cache root
```

Legacy layout (still supported): `.cache/locateanything/LocateAnything-3B` from an older fetch. Run `just fetch-locateanything-tokenizer` if `tokenizer.json` is missing (processor prompts).

## Sample image

One JPEG ships with this crate:

| Path | API |
|------|-----|
| `fixtures/sample.jpg` | `rlx_locateanything::fixtures::sample_image_path()` |

640×360 RGB (from [yolov5 zidane.jpg](https://github.com/ultralytics/yolov5/blob/master/data/images/zidane.jpg)). Used by the CLI (when `--image` is omitted), examples, and HF `*_real` parity probes.

Override:

```bash
export RLX_LOCATEANYTHING_IMAGE=/path/to/your.jpg
```

## Fastest path (CLI)

No image path needed — uses the bundled sample:

```bash
just locateanything-demo
```

Equivalent:

```bash
just locateanything -- --model-dir "$RLX_LOCATEANYTHING_DIR" \
  --task ground-single --phrase "person" \
  --device auto --max-image-side 640 --warmup
```

GPU build:

```bash
just features=all-backends locateanything-demo
```

### CLI flags

| Flag | Default | Notes |
|------|---------|--------|
| `--model-dir` | *(required)* | Directory with `config.json` + safetensors |
| `--image` | `fixtures/sample.jpg` | Any JPEG/PNG |
| `--device` | `auto` | `cpu`, `metal`, `cuda`, … or `RLX_DEVICE` |
| `--task` | `ground-single` | See `prompts` module |
| `--phrase` | `object` | Target text for ground-* tasks |
| `--processor-prompt` | on | HF processor layout (boxes) |
| `--rlx-prompt` | off | RLX Qwen chat layout |
| `--generation-mode` | `hybrid` | `fast` / `slow` / `hybrid` |
| `--max-tokens` | `64` | New tokens cap |
| `--temperature` | `0` | Greedy grounding |
| `--max-image-side` | none | Resize longest edge before patchify |
| `--warmup` | off | Compile vision + LM prefill first |
| `--preload-lm` | off | Load LM at session open |
| `--dry` | | Validate checkpoint only |

Environment:

| Variable | Purpose |
|----------|---------|
| `RLX_LOCATEANYTHING_DIR` | Checkpoint directory (overrides HF cache) |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | Hugging Face cache root |
| `RLX_LOCATEANYTHING_IMAGE` | Replace bundled sample |
| `RLX_DEVICE` | Default device when `--device` omitted |

## Rust API

High-level session (recommended):

```rust
use rlx_locateanything::{LocateAnythingSession, fixtures};

// HF Hub cache (after `huggingface-cli download nvidia/LocateAnything-3B`)
let mut session = LocateAnythingSession::open_default()?;

let out = session.ground_path(fixtures::sample_image_path(), "person")?;
for b in &out.boxes {
    println!("({:.0},{:.0})-({:.0},{:.0})", b.x1, b.y1, b.x2, b.y2);
}
```

Hub id or filesystem path also work:

```rust
let mut session = LocateAnythingSession::open("nvidia/LocateAnything-3B")?;
// let mut session = LocateAnythingSession::open("/path/to/snapshot")?;
```

With options:

```rust
use rlx_locateanything::{InferenceOptions, PromptStyle, resolve_device};

let opts = InferenceOptions::for_grounding()
    .device(resolve_device(Some("auto"))?)
    .max_image_side(640)
    .prompt_style(PromptStyle::Processor);
let mut session = LocateAnythingSession::open_with_options(
    rlx_locateanything::default_model_dir()?,
    opts,
)?;
```

Lower-level runner: `LocateAnythingRunner::builder()` → `generate()`, `encode_vision_cached()`, `warmup_compile()`.

### Modules

| Module | Role |
|--------|------|
| `infer` | `LocateAnythingSession`, `InferenceOptions`, `PromptStyle` |
| `hub` | `default_model_dir()`, `hf_snapshot_dir()`, `default_hf_cache_dir()` |
| `fixtures` | `sample_image_path()`, `probe_image_path()` |
| `device` | `resolve_device`, `pick_auto_device` |
| `runner` | Vision + LM generation, compile caches |
| `preprocess` | Native-resolution patchify (`in_token_limit` from JSON) |
| `processor_prompt` | HF `LocateAnythingProcessor` token layout |
| `parse` | `<box>`, `<ref>` output parsing |
| `prompts` | Task strings (`ground_single`, `detect`, …) |
| `generation` / `mtp` | `slow` / `hybrid` / `fast` decode |

## Example binary

```bash
cargo run -p rlx-models --example locateanything_ground --release -- \
  --model-dir "$RLX_LOCATEANYTHING_DIR" --phrase person --device auto
```

Uses `fixtures/sample.jpg` unless `--image` is passed.

## Validation

```bash
just test-locateanything-checkpoint    # layout + CPU vision encode
just test-locateanything-parity        # 28 HF tensor/generate probes
just test-locateanything-parity-real   # real-photo subset only
just features=all-backends test-locateanything-backends
```

Crate tests on the sample (need `RLX_LOCATEANYTHING_DIR`):

```bash
RLX_LOCATEANYTHING_DIR=$RLX_LOCATEANYTHING_DIR \
  cargo test -p rlx-locateanything --test preprocess_real --release
```

## Backends

| Feature set | Devices |
|-------------|---------|
| `metal` | Apple GPU (Metal) |
| `mlx` | Apple MLX |
| `cuda` / `rocm` | NVIDIA / AMD |
| `gpu` / `vulkan` | wgpu (Vulkan) |
| `apple-silicon` | Metal + MLX + wgpu |
| `all-backends` | All of the above |

```bash
cargo build -p rlx-locateanything --features apple-silicon
cargo build -p rlx-locateanything --features all-backends
just features=all-backends test-locateanything-backends
just features=all-backends test-locateanything-moonvit-backends
```

`--device auto` (or `RLX_DEVICE`) picks the first compiled backend: CUDA → Metal → MLX → ROCm → wgpu → Vulkan → CPU.

MoonViT, projector, and Qwen2.5 LM graphs compile per device; caches live on `MoonVitCache` and `LmSessionCaches`. See `compile_support` for per-backend compile options.

## HF reference

The HF checkpoint uses custom `transformers` code (`trust_remote_code=True`). Task strings and generation modes match the model card (`rlx_locateanything::prompts`, `GenerationMode`).

Parity harness: `crates/rlx-models/tests/locateanything_hf_parity.rs` + `locateanything_parity_helpers/hf_reference.py`. Set `RLX_LOCATEANYTHING_PYTHON` to a venv with `transformers`, `torch`, `safetensors`.

## Architecture notes

- **Preprocess** — rescale when patch count exceeds `preprocessor_config.json` `in_token_limit` (25600), then pad to merge kernel grid.
- **MTP** — incremental prefill on trailing `block_size` window; hybrid falls back to AR on decode errors.
- **Coordinates** — model emits integers in `[0, 1000]`; `parse_grounding` maps to pixel space using `PreprocessedImage.pixel_w` / `pixel_h`.

## See also

- Main repo [README](../../README.md#locateanything)
- [AGENTS.md](../../AGENTS.md) — `just fetch-locateanything`, parity and backend tests
