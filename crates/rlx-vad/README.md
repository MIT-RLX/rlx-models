# rlx-vad

Voice activity detection on RLX — pure Rust inference, no ONNX Runtime or PyTorch.

Two backends ship with **embedded weights** (no downloads at runtime):

| Backend | Reference | Embedded file | Size | Frame @ 16 kHz |
|---------|-----------|---------------|------|----------------|
| **Earshot** | [pykeio/earshot](https://github.com/pykeio/earshot) | `weights/earshot_weights.bin` | ~75 KiB | 256 samples (16 ms) |
| **Silero** | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | `weights/silero_vad_16k.safetensors` | ~920 KiB | 512 samples + 64 context |

Facade re-export: `rlx_models::vad` (see [crates/rlx-models/src/lib.rs](../rlx-models/src/lib.rs)).

## Quick start

```bash
# Earshot — embedded bin (~5–6 µs/frame on Apple Silicon)
cargo run -p rlx-vad --release -- \
  --backend earshot --wav assets/jfk/jfk_rust_speech.wav

# Silero — embedded safetensors
cargo run -p rlx-vad --release -- \
  --backend silero --wav assets/jfk/jfk_rust_speech.wav

# Optional: print segment boundaries in seconds
cargo run -p rlx-vad --release -- \
  --backend silero --wav assets/jfk/jfk_rust_speech.wav --seconds

# Noise / latency / quality bench (assets/jfk + white-noise sweep)
cargo run -p rlx-vad --example jfk_bench --release

# Sweep RLX device slots (cpu, metal, mlx, wgpu, …)
just bench-vad-jfk-all-devices
# or: cargo run -p rlx-vad --example jfk_bench --release --features all-backends -- --devices all
```

CLI flags: `--backend earshot|silero`, `--wav PATH`, `--threshold` (override preset), `--device cpu|metal|…`, `--weights PATH` (Silero override only), `--seconds`.

Bench flags: `--devices all|apple-silicon|cpu,metal,…` (see [Benchmarks](#benchmarks-assetsjfk)).

## Benchmarks (assets/jfk)

Measured with `cargo run -p rlx-vad --example jfk_bench --release` on Apple Silicon (release, CPU BLAS / Accelerate). Clips: `assets/jfk/jfk_rust_speech.wav` (12.1 s) and `jfk_voice_clone.wav` (5.2 s), each wrapped in 1.5 s silence pads for labeled-region scoring.

### Latency (clean audio)

| VAD | Mean / frame | p99 / frame | RTF | Notes |
|-----|--------------|-------------|-----|-------|
| **Earshot** | ~5–6 µs | ~6 µs | ~0.00034 | 256-sample hop; BLAS MinGRU |
| **Silero** | ~365 µs | ~460 µs | ~0.011 | 512-sample hop + 64-sample context |

RTF = wall time ÷ audio duration (lower is faster). Both are well under real-time for streaming.

### Quality (clean audio, algorithm-specific segment presets)

Frame metrics use each algorithm’s `SegmentParams` preset (see below). **Segment IoU** measures overlap with the labeled speech region — the primary quality metric for Silero, where LSTM state keeps frame probabilities elevated on trailing silence (frame-level silence specificity is misleading).

| VAD | Frame acc | Speech recall | Silence spec | **Seg IoU** | Mean speech prob |
|-----|-----------|---------------|--------------|-------------|------------------|
| **Earshot** | ~70% | ~82% | ~48% | **0.96–0.97** | ~0.67 |
| **Silero** | ~57% | ~90% | ~0%* | **0.99–1.00** | ~0.82 |

\*Silero frame silence specificity is low on padded silence because probabilities decay slowly after speech; segment IoU remains near 1.0 with official min-speech / min-silence settings.

### Noise sweep (jfk_rust_speech.wav, SNR vs white noise)

| SNR | Earshot rec / spec / IoU | Silero rec / IoU |
|-----|--------------------------|------------------|
| clean | 82% / 48% / 0.97 | 90% / 0.99 |
| 20 dB | 82% / 47% / 0.97 | 93% / 0.99 |
| 10 dB | 71% / 60% / 0.91 | 93% / 0.99 |
| 5 dB | 75% / 38% / 0.95 | 93% / 0.99 |
| 0 dB | 91% / 26% / 0.96 | 93% / 0.99 |
| −5 dB | 94% / 26% / 0.96 | 89% / 0.99 |

Earshot is faster and lighter; Silero holds higher speech recall and segment IoU under noise at the cost of ~60× higher per-frame latency.

### RLX device compatibility

`jfk_bench --devices all` validates each RLX backend slot (`cpu`, `metal`, `mlx`, `cuda`, `wgpu`, …) and reports identical probabilities across slots (parity checked in `tests/backend_quick_check.rs`, tolerance `< 1e-6`).

Streaming inference runs on **CPU BLAS for every device slot** — 256–512 sample frames make GPU transfer dominate latency (same policy as Whisper decode on Metal/MLX). `--device` still validates that the requested RLX backend is available in the build.

```bash
just test-vad-backends                               # per-device segment + prob parity
cargo run -p rlx-vad --example jfk_bench --release --features apple-silicon -- --devices apple-silicon
```

### Segment presets (quality tuning)

CLI and bench pick defaults via `SegmentParams::for_algorithm()`:

| Preset | `threshold` | `neg_threshold` | `min_speech` | `min_silence` |
|--------|-------------|-----------------|--------------|---------------|
| `SegmentParams::earshot()` | 0.35 | 0.20 | 100 ms | 50 ms |
| `SegmentParams::silero()` | 0.5 | threshold − 0.15 | 250 ms | 100 ms |

Override on the CLI with `--threshold`. Library callers can clone a preset and adjust fields.

## Cargo features

| Feature | Default | Backend |
|---------|---------|---------|
| `earshot` | yes | pykeio/earshot CNN + MinGRU (~77 KiB embedded bin) |
| `silero` | yes | Silero ONNX 16 kHz branch (~944 KiB embedded safetensors; pulls `rlx-core`) |
| `all-backends` | no | Forward GPU features to `rlx-runtime` for `--device metal\|cuda\|…` validation |

Build one backend only:

```bash
cargo build -p rlx-vad --no-default-features --features earshot
cargo build -p rlx-vad --no-default-features --features silero
cargo test -p rlx-vad --release                    # both (default)
just test-vad                                        # default + each backend alone
just test-vad-backends                               # CPU/Metal/CUDA/… slot checks
```

`enabled_backends()` / `default_backend()` reflect the compile-time VAD algorithm set.

### RLX execution devices

`--device` validates the RLX backend is available (`resolve_device`). Streaming frame inference runs on **CPU BLAS** for all device slots; probabilities are identical across slots. See [Benchmarks](#rlx-device-compatibility) for multi-device bench commands.

## Silero embedded weights

### What is embedded

The crate embeds **`weights/silero_vad_16k.safetensors`** at compile time:

```rust
// crates/rlx-vad/src/silero/embedded.rs
const SAFETENSORS: &[u8] = include_bytes!("../../weights/silero_vad_16k.safetensors");
```

On first use, bytes are parsed with [`rlx_core::embedded_safetensors::EmbeddedSafetensors`](../../rlx-models-core/src/embedded_safetensors.rs) and cached in a `OnceLock`. No filesystem access unless you call `SileroWeights::load(path)`.

### Not the same as the HF download

Hugging Face hosts a file also named `silero_vad_16k.safetensors`, but it matches the **8 kHz** ONNX branch (STFT `(258, 1, 256)`, conv1 in `(128, 129, 3)`). That graph is **not** interchangeable with 16 kHz streaming inference.

The embedded file is exported from the official **`silero_vad.onnx` 16 kHz branch** (`then_branch` when `sr == 16000`):

| Tensor | Shape | Notes |
|--------|-------|-------|
| `stft_conv.weight` | `(130, 1, 128)` | STFT as conv, stride **64**, +32 reflect pad |
| `conv1.weight` / `bias` | `(128, 65, 3)` / `(128,)` | magnitude → 128 ch |
| `conv2` … `conv4` | … | stride-2 middle layers |
| `lstm_cell.weight_ih` / `weight_hh` | `(512, 128)` | PyTorch LSTM layout |
| `lstm_cell.bias_ih` / `bias_hh` | `(512,)` | |
| `final_conv.weight` / `bias` | `(1, 128, 1)` / `(1,)` | sigmoid speech prob |

Use the export script below — do not copy the HF artifact into `weights/`.

### Regenerate embedded safetensors

Requires Python 3 + `pip install onnx numpy safetensors`.

```bash
curl -sL -o /tmp/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

python3 scripts/export_silero_onnx_weights.py /tmp/silero_vad.onnx \
  crates/rlx-vad/weights/silero_vad_16k.safetensors
```

Then rebuild `rlx-vad` (the new blob is picked up via `include_bytes!`).

Legacy RLXV blob export (same tensors, custom header) still exists in `scripts/export_silero_embedded.py` for experiments; **safetensors is the supported embed format**.

## Earshot embedded weights

`weights/earshot_weights.bin` — custom layout from [pykeio/earshot](https://github.com/pykeio/earshot) (FFT tables + CNN + MinGRU). Parsed once at startup; no external files.

## Library API

```rust
use rlx_vad::{
    earshot,
    silero::{SileroConfig, SileroSession, SileroWeights},
    SegmentParams,
    resolve_device,
};

// Earshot — frame-at-a-time
let mut det = earshot::Detector::default();
let prob = det.predict_f32(&frame_256);

// Silero — streaming session (512-sample frames, 64-sample context)
let mut session = SileroSession::new(SileroWeights::embedded(), SileroConfig::default());
let prob = session.predict_frame(&frame_512)?;

// Segments with tuned presets (requires matching Cargo features)
let _dev = resolve_device("cpu")?;
let segs = rlx_vad::speech_segments_earshot(&pcm, &SegmentParams::earshot());
let segs = rlx_vad::speech_segments_silero(&mut session, &pcm, &SegmentParams::silero())?;
```

Segment helpers merge frame scores into `[start, end)` sample ranges. Use `SegmentParams::earshot()` or `::silero()` rather than bare defaults when quality matters.

## Tests

```bash
cargo test -p rlx-vad --release
just test-vad
just test-vad-backends                               # needs --features all-backends
cargo test -p rlx-vad --test e2e_jfk --release   # assets/jfk end-to-end + CLI
cargo run -p rlx-vad --example jfk_bench --release -- --devices all
```

Integration tests use `assets/jfk/jfk_rust_speech.wav` when present.

## Implementation notes

- **Shared ops** — `src/ops.rs`: Conv1d, LSTM cell, BLAS `gemv` (via `rlx-cpu`).
- **Silero STFT** — reflect-pad 32 samples right; conv stride 64; magnitude from 65 bins of 130 STFT channels.
- **Streaming** — Silero expects `context || chunk` (576 samples @ 16 kHz per step); LSTM state carried in `SileroSession`.
- **Backends** — `--device` validates RLX backend availability; streaming inference uses CPU BLAS on all slots (see `streaming_execution_device`).

## See also

- [`rlx_core::embedded_safetensors`](../rlx-models-core/src/embedded_safetensors.rs) — reusable in-memory safetensors loader for other small embedded models
- [README.md](../../README.md) — workspace overview
- [AGENTS.md](../../AGENTS.md) — agent command cheat sheet
