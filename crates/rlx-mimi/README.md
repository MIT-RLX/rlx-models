# rlx-mimi

Native Rust inference for [Kyutai Mimi](https://huggingface.co/kyutai/mimi) — a streaming neural audio codec at **12.5 Hz** / **24 kHz** mono.

## Quick start

```bash
# Download weights (~385 MB)
just fetch-mimi

# Encode + decode roundtrip (CPU eager — HF safetensors layout)
just mimi -- --model-dir .cache/mimi --in-wav speech.wav --out-wav /tmp/mimi-roundtrip.wav

# GPU codec (Metal/CUDA) — needs Moshi Candle sidecar weights
export RLX_MOSHI_DIR=.cache/moshiko   # contains tokenizer-e351c8d8-checkpoint125.safetensors
cargo run -p rlx-mimi --features "gpu-codec,metal" --release -- \
  --bench --in-wav speech.wav --device metal
```

## Backends

| Device | Engine |
|--------|--------|
| `cpu` | Native eager (HF `kyutai/mimi` weights) |
| `metal` / `cuda` | Kyutai `moshi::mimi` on Candle |
| `mlx` | Same as Metal (Apple GPU via Candle) |
| `wgpu` / `vulkan` | CPU fallback until RLX compiled codec lands |

Build: `cargo build -p rlx-mimi --features "gpu-codec,metal"`

GPU path loads `tokenizer-e351c8d8-checkpoint125.safetensors` from `RLX_MOSHI_DIR` or the mimi dir (not `model.safetensors` HF layout).

## Library

```rust
use rlx_mimi::{MimiCodec, MimiCodes, SAMPLE_RATE};
use rlx_runtime::Device;

let mut codec = MimiCodec::open_on_with_moshi(".cache/mimi", Some(".cache/moshiko".as_ref()), Device::Metal, Some(8))?;
let codes: MimiCodes = codec.encode_wav("speech.wav", Some(8))?;
let recon = codec.decode_codes(&codes)?;
```

Set `RLX_MIMI_DIR` to skip `--model-dir`. Input WAVs are resampled to 24 kHz.

## Weights

| Artifact | HuggingFace |
|----------|-------------|
| HF Mimi (CPU) | [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) |
| Candle Mimi (GPU) | sidecar in [`kyutai/moshiko-candle-bf16`](https://huggingface.co/kyutai/moshiko-candle-bf16) |

## Tests

```bash
export RLX_MIMI_DIR=.cache/mimi
export RLX_MOSHI_DIR=.cache/moshiko
cargo test -p rlx-mimi --release

RLX_MIMI_GPU_SMOKE=1 cargo test -p rlx-mimi --test gpu_codec_smoke --features "gpu-codec,metal"
```
