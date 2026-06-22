# rlx-dac

Native Rust inference for [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec) — high-fidelity neural audio compression at 8–16 kbps.

## Quick start

```bash
# Download 24 kHz safetensors (~285 MB, HuggingFace mirror)
cargo run -p rlx-dac --features hf-download --release -- --fetch --model-type 24khz

# Encode + decode roundtrip (CPU eager)
export RLX_DAC_DIR=.cache/dac/24khz
cargo run -p rlx-dac --release -- \
  --in-wav speech.wav --out-wav /tmp/dac-roundtrip.wav
```

Supported variants: `24khz` (default), `16khz`, `44khz`.

## Library

```rust
use rlx_dac::{DacCodec, DacCodes};

let codec = DacCodec::open(".cache/dac/24khz")?;
let codes: DacCodes = codec.encode_pcm(&pcm, Some(32))?;
let recon = codec.decode_codes(&codes)?;
```

Weights load from `model.safetensors` + `config.json` under `RLX_DAC_DIR` (default `.cache/dac/24khz`).

## Weights

| Variant | HuggingFace (safetensors) |
|---------|---------------------------|
| 24 kHz | [`hance-ai/descript-audio-codec-24khz`](https://huggingface.co/hance-ai/descript-audio-codec-24khz) |
| 16 kHz | [`hance-ai/descript-audio-codec-16khz`](https://huggingface.co/hance-ai/descript-audio-codec-16khz) |
| 44.1 kHz | [`hance-ai/descript-audio-codec-44khz`](https://huggingface.co/hance-ai/descript-audio-codec-44khz) |

Official `.pth` releases: [descript-audio-codec releases](https://github.com/descriptinc/descript-audio-codec/releases). Convert to safetensors externally if needed; this crate loads safetensors only.

## Tests

```bash
export RLX_DAC_DIR=.cache/dac/24khz
cargo test -p rlx-dac --release
```

Reference parity uses baked fixture `tests/fixtures/dac_24khz_synthetic.json` (no Python at test time). Tests skip gracefully when weights are missing.

## Reference

- Paper: [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546)
- Upstream: [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)
