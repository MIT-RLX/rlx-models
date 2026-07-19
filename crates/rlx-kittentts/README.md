# rlx-kittentts

KittenTTS ONNX / native RLX text-to-speech. Optional **`espeak`** feature adds plain-text input via [espeak-ng-rs](https://github.com/eugenehp/espeak-ng-rs). Higher-level preprocessing and mobile FFI live in [kittentts-rs](https://github.com/eugenehp/kittentts-rs).

## Quick start

```bash
just fetch-kittentts
just kittentts-demo              # IPA
just kittentts-text-demo         # plain English (--features espeak)
just test-kittentts-e2e          # fetch + unit tests + ONNX/native/text synthesis
```

## CLI

```bash
# Positional IPA works
just kittentts -- "həˈloʊ" --voice Jasper --out-wav out.wav

# Plain English (rebuild with espeak feature)
cargo run -p rlx-kittentts --features espeak --release -- \
  --text "Hello world" --voice Jasper --out-wav out.wav

# Explicit flags
just kittentts -- --ipa "həˈloʊ" --device metal

# Native RLX graph (default) — see NATIVE.md
just kittentts -- --ipa "həˈloʊ" --out-wav out.wav

# Optional ONNX Runtime path
cargo run -p rlx-kittentts --features onnx --release -- \
  --ipa "həˈloʊ" --out-wav out.wav
```

**Path resolution** (first match wins):

1. `--model-dir` / `RLX_KITTENTTS_DIR` / `KITTENTTS_MODEL_DIR`
2. `.cache/kittentts-mini-0.8` (from `just fetch-kittentts`)
3. Hugging Face hub cache (`models--KittenML--kitten-tts-mini-0.8`)

## Library

```rust
use rlx_kittentts::{KittenTTS, Device};

let tts = KittenTTS::load_from_dir(".cache/kittentts-mini-0.8".as_ref(), Device::Cpu)?;
let audio = tts.generate_from_ipa("həˈloʊ", "Jasper", 1.0, 6)?;

// With `espeak` feature:
// let audio = tts.generate_from_text("Hello world", "Jasper", 1.0, "en")?;
```

## Features

| Feature | Purpose |
|---------|---------|
| `native` (default) | Decomposed `kitten_tts_mini_rlx` graph |
| `onnx` | Optional ONNX Runtime inference |
| `espeak` | Plain text → IPA via `espeak-ng` 0.1.2 (`bundled-data-en`; GPL-3.0) |
| `hf-download` | `--download` via Hugging Face Hub |
| `metal` / `cuda` / … | Backend forwarding to ORT + RLX runtime |

Native backend details: [NATIVE.md](NATIVE.md).

## Native RLX env vars

| Variable | Purpose |
|----------|---------|
| `RLX_ONNX_BUNDLE` | RLX bundle dir (`weights/rlx_bundle` under decomposed weights) |
| `RLX_ONNX_SEQUENCE_LENGTH` | Active IPA token count during native compile/run |
| `KITTEN_RLX_WEIGHTS` | Decomposed safetensors directory |
| `KITTEN_RLX_BUNDLE` | Legacy alias for `RLX_ONNX_BUNDLE` |
| `KITTEN_SEQUENCE_LENGTH` | Legacy alias for `RLX_ONNX_SEQUENCE_LENGTH` |

Re-export bundle after ONNX changes: `just export-kitten-rlx-bundle` (see `kitten_tts_mini_rlx/README.md`).

## See also

- Main repo [README](../../README.md#per-crate-readmes)
- [kitten_tts_mini_rlx/README.md](../kitten_tts_mini_rlx/README.md) — RLX bundle layout
- [AGENTS.md](../../AGENTS.md) — `just fetch-kittentts`, `just kittentts`
