# rlx-kittentts

KittenTTS native RLX text-to-speech. Optional **`espeak`** feature adds plain-text input via [espeak-ng-rs](https://github.com/eugenehp/espeak-ng-rs). Higher-level preprocessing and mobile FFI live in [kittentts-rs](https://github.com/eugenehp/kittentts-rs).

## Quick start

```bash
just fetch-kittentts
just kittentts-demo              # IPA
just kittentts-text-demo         # plain English (--features espeak)
just test-kittentts-native       # native smoke (needs decomposed weights)
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

# NVIDIA (MSI): cuda / vulkan / gpu (wgpu)
cargo run -p rlx-kittentts --release --features native,cuda,gpu,vulkan -- \
  --ipa "həˈloʊ" --device vulkan --out-wav out.wav
```

**Path resolution** (first match wins):

1. `--model-dir` / `RLX_KITTENTTS_DIR` / `KITTENTTS_MODEL_DIR`
2. `.cache/kittentts-mini-0.8` (from `just fetch-kittentts`)
3. Hugging Face hub cache (`models--KittenML--kitten-tts-mini-0.8`)

Native weights resolve via `KITTEN_RLX_WEIGHTS` or `crates/kitten_tts_mini_rlx/weights` (`model.safetensors` + `rlx_bundle/`).

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
| `espeak` | Plain text → IPA via `espeak-ng` 0.1.2 (`bundled-data-en`; GPL-3.0) |
| `hf-download` | `--download` via Hugging Face Hub |
| `metal` / `cuda` / `gpu` / `vulkan` / … | Backend forwarding to RLX runtime |

Native backend details (placement, discrete NVIDIA caps, QMatMul, full env list): **[NATIVE.md](NATIVE.md)**.

## Backends (native)

| Device | Status | Prefer for |
|--------|--------|------------|
| Cpu | ✅ | Reference |
| Cuda / Rocm | ✅ | Long IPA (single-pass) |
| Metal / Mlx | ✅ | Apple Silicon |
| Vulkan | ✅ | Native GPU when CUDA unavailable (80 k wave cap) |
| Gpu (wgpu) | ✅ | **→ Cuda** when linked; else CPU vocoder. `KITTEN_RLX_GPU_WAVE=1` → Vulkan |
| Ane | ✅ | Duration+wave on CPU |

Discrete NVIDIA policy (`device_policy::prepare`): Vulkan 80 k + frames=8; Gpu wave on CPU by default. Details in [NATIVE.md § Discrete NVIDIA](NATIVE.md#discrete-nvidia-gpu--vulkan).

## Native RLX env vars (short list)

| Variable | Purpose |
|----------|---------|
| `RLX_ONNX_BUNDLE` | RLX bundle dir (`weights/rlx_bundle` under decomposed weights) |
| `RLX_ONNX_SEQUENCE_LENGTH` | Active IPA token count during native compile/run |
| `KITTEN_RLX_WEIGHTS` | Decomposed safetensors directory |
| `KITTEN_RLX_BUNDLE` | Legacy alias for `RLX_ONNX_BUNDLE` |
| `KITTEN_SEQUENCE_LENGTH` | Legacy alias for `RLX_ONNX_SEQUENCE_LENGTH` |
| `KITTEN_RLX_INFER` | `production` (default with `native-fast`) or `parity` |
| `KITTEN_RLX_WGPU_WAVEFORM_CAP` | Discrete wgpu wave clamp (default 32000; `0` disables) |
| `KITTEN_RLX_VULKAN_WAVEFORM_CAP` | Vulkan wave clamp (default 80000) |
| `KITTEN_RLX_NATIVE_QMATMUL` | `0`/`1` force quantized vs native GEMM |

Full table: [NATIVE.md](NATIVE.md).

Re-export bundle after ONNX changes: `just export-kitten-rlx-bundle` (see `kitten_tts_mini_rlx/README.md`).

## See also

- Main repo [README](../../README.md#per-crate-readmes)
- [NATIVE.md](NATIVE.md) — backends, discrete NVIDIA, QMatMul, env
- [kitten_tts_mini_rlx/README.md](../kitten_tts_mini_rlx/README.md) — RLX bundle layout + `device_policy`
- [AGENTS.md](../../AGENTS.md) — `just fetch-kittentts`, `just kittentts`
