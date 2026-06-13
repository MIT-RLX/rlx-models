# Native RLX backend (`native` feature)

Runs the decomposed [`kitten_tts_mini_rlx`](../kitten_tts_mini_rlx) graph via `rlx-runtime` (no ONNX Runtime).

## Build

```bash
cargo build -p rlx-kittentts --features native --release
```

## Bundle + weights

Primary path: `crates/kitten_tts_mini_rlx/weights/rlx_bundle/` (ONNX export + safetensors).

For Hugging Face–only deployments, place `rlx_bundle/{graph.json,manifest.json,weights.safetensors}` under the KittenML checkpoint directory; ONNX on disk is optional when the native bundle is present.

```bash
# From rlx-models root
just export-kitten-rlx-bundle
```

Or decompose from an existing RLX bundle:

```bash
rlx-onnx-decompose --bundle /path/to/rlx_bundle \
  -o crates/kitten_tts_mini_rlx --crate-name kitten_tts_mini_rlx
```

## Environment

| Variable | Purpose |
|----------|---------|
| `RLX_ONNX_BUNDLE` | Bundle directory (`manifest.json`, `graph.json`, `weights.safetensors`) |
| `RLX_ONNX_SEQUENCE_LENGTH` | Active token count for compile-time LSTM/seq shapes |
| `KITTEN_RLX_WEIGHTS` | Weights dir: `model.safetensors` or `rlx_bundle/graph.json` (HF bundle layout) |
| `KITTEN_RLX_BUNDLE` | Legacy alias for `RLX_ONNX_BUNDLE` |
| `KITTEN_SEQUENCE_LENGTH` | Legacy alias for `RLX_ONNX_SEQUENCE_LENGTH` |
| `KITTEN_VOICES_NPZ` | Voice style table for parity tests |

## Usage

```rust
use rlx_kittentts::{KittenTTS, Device};

let tts = KittenTTS::load_native(
    "crates/kitten_tts_mini_rlx/weights".as_ref(),
    "path/to/voices.npz".as_ref(),
    Default::default(),
    Default::default(),
    Device::Cpu,
    128,      // max IPA token length (compile-time)
    24_000,   // max waveform samples binding
)?;
let audio = tts.generate_from_ipa("həˈloʊ", "default", 1.0, 6)?;
```

## Test

```bash
export KITTEN_RLX_WEIGHTS=crates/kitten_tts_mini_rlx/weights
export KITTEN_VOICES_NPZ=/path/to/voices.npz
cargo test -p rlx-kittentts --features native --release native_infer_smoke -- --nocapture
cargo test -p kitten_tts_mini_rlx --test bundle_hir
```
