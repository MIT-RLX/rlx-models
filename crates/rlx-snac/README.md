# rlx-snac

**SNAC** multi-scale neural audio codec (`hubertsiuzdak/snac`) for RLX — the codec behind Orpheus-style TTS. Decodes (and encodes) SNAC's hierarchical RVQ codes to/from waveform on any RLX backend (CPU / Metal / MLX / wgpu).

## Public API

```rust
use rlx_snac::{SnacDecoder, SnacConfig, build_decode_graph};
use rlx_runtime::Device;

let cfg = SnacConfig::default();
let mut dec = SnacDecoder::load(weights, cfg, Device::Cpu)?;
let pcm = dec.decode(&codes)?;   // multi-scale RVQ codes -> 24 kHz waveform
# anyhow::Ok(())
```

Modules: `codec` (`SnacDecoder`), `config`, `model`, `graph` (`build_decode_graph` / `build_encode_graph`, `SnacDecoderGraph` / `SnacEncoderGraph`), `eager`.

## Quick start

```bash
cargo run -p rlx-snac --example bench
```

## How it fits

Used by the Orpheus TTS decode path; a sibling of the other RLX codecs — [rlx-encodec](../rlx-encodec), [rlx-dac](../rlx-dac), [rlx-mimi](../rlx-mimi).
