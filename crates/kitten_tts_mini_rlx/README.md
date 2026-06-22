# kitten_tts_mini_rlx

RLX native decomposition of [Kitten TTS mini](https://huggingface.co/KittenML/kitten-tts-mini-0.8) ONNX.

Published on crates.io as **`kitten_tts_mini_rlx`** (`weights/**` excluded from the crate tarball — export or download weights locally).

## Layout

| Path | Purpose |
|------|---------|
| `weights/model.safetensors` | Native weights (decomposed from ONNX) |
| `weights/model.gguf` | Optional GGUF container (`just export-kitten-gguf`) |
| `weights/rlx_bundle/` | Legacy ONNX bundle (`graph.json` + safetensors) |
| `src/native/` | Native feature: config, compile cache, module map |
| `src/graph.rs` | Full Rust HIR builder (`--features native`) |
| `src/bundle_compile.rs` | Legacy bundle → HIR path |
| `src/kernels.rs` | Model-specific CPU kernels (`onnx.QMatMul`, …) |
| `decompose_report.json` | Op coverage from export |

## Native weights (recommended)

No `graph.json` at runtime — architecture lives in Rust (`graph.rs` + `native/flow`).

```bash
just export-kitten-native-weights   # bundle export + rlx-onnx-decompose
just export-kitten-gguf             # optional GGUF copy of safetensors
just test-kitten-native-compile     # compile check (needs weights/)
```

Deploy layout:

```text
checkpoint/
  model.safetensors   # or model.gguf
  voices.npz
  config.json
```

Build with `--features native`. `rlx-kittentts --features native` enables this path automatically when `model.safetensors` is present.

Set `KITTEN_RLX_FORCE_BUNDLE=1` to keep using `rlx_bundle/graph.json` instead.

## Legacy bundle export

```bash
just export-kitten-rlx-bundle
```

The export script bakes `__onnx_import__/duration_carry` into `/Expand_1` and `/Where_1`
so RLX import matches ORT single-pass duration semantics.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `KITTEN_RLX_FORCE_BUNDLE` | Prefer `rlx_bundle/graph.json` over native weights |
| `RLX_ONNX_BUNDLE` | Override bundle directory |
| `RLX_ONNX_SEQUENCE_LENGTH` | Active token count for compile-time shape restoration |
| `KITTEN_RLX_BUNDLE` | Legacy alias for `RLX_ONNX_BUNDLE` |
| `KITTEN_RLX_AOT_CACHE` | AOT compile cache directory |
| `KITTEN_RLX_SKIP_FUSION` | Set `1` to disable fusion (debug / probes) |

## Usage

```rust
use kitten_tts_mini_rlx::{compile, GraphOptions};
use rlx_runtime::Device;

let graph = compile(
    Device::Cpu,
    weights_dir.as_ref(),
    &GraphOptions { sequence_length: 128, max_waveform_samples: 24_000 },
)?;
```

## Module map

See [`src/native/config.rs`](src/native/config.rs) (`ModuleKind`) and [`src/native/flow/modules.rs`](src/native/flow/modules.rs) for semantic boundaries (bert, text encoder, mel decoder, predictor, duration, vocoder).

## Tests

```bash
cargo test -p kitten_tts_mini_rlx --features native
cargo test -p rlx-kittentts --features native native_infer_smoke
```
