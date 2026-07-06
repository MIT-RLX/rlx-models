# rlx-xcodec

**XCodec2** (`HKUSTAudio/xcodec2`) decoder for RLX — a multi-backend **RoFormer + Vocos** decoder that reconstructs waveform from XCodec2 codes via an ISTFT head. Runs on any RLX backend (CPU / Metal / MLX / wgpu).

## Public API

```rust
use rlx_xcodec::{XcodecDecoderGraph, XcodecWeights, head_to_wav};

let weights = XcodecWeights::load(path)?;
// build/compile the RoFormer-Vocos decoder graph, run to a spectral head, then:
let pcm = head_to_wav(&head_out, t, &window);   // ISTFT overlap-add -> waveform
# anyhow::Ok(())
```

Modules: `graph` (`XcodecDecoderGraph`), `model` (`XcodecWeights`), `istft`.

## Quick start

```bash
cargo run -p rlx-xcodec --example bench
```

## How it fits

A sibling of the other RLX neural codecs — [rlx-encodec](../rlx-encodec), [rlx-snac](../rlx-snac), [rlx-wavtokenizer](../rlx-wavtokenizer) — supplying waveform reconstruction for TTS/audio pipelines.
