# rlx-encodec

**Meta EnCodec** neural audio codec (`facebook/encodec_24khz`) for RLX — encode 24 kHz audio to discrete codes and decode back, running on any RLX backend (CPU / Metal / MLX / wgpu).

## Public API

```rust
use rlx_encodec::{EncodecCodec, EncodecConfig};
use rlx_runtime::Device;

let cfg = EncodecConfig::default();          // facebook/encodec_24khz
let mut codec = EncodecCodec::load(weights, cfg, Device::Cpu)?;
let codes = codec.encode(&pcm_24khz)?;       // discrete tokens
let pcm = codec.decode(&codes)?;             // reconstructed waveform
# anyhow::Ok(())
```

Modules: `codec`, `config`, `model`, `graph` (compiled IR path), `eager` (reference), `lstm`.

## Quick start

```bash
cargo run -p rlx-encodec --example bench
```

## How it fits

One of the RLX neural-codec crates alongside [rlx-snac](../rlx-snac), [rlx-dac](../rlx-dac), [rlx-mimi](../rlx-mimi), [rlx-speechtokenizer](../rlx-speechtokenizer), used by TTS/audio pipelines for waveform ↔ token conversion.
