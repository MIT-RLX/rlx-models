# rlx-wavtokenizer

**WavTokenizer** (`novateur/WavTokenizer`) single-token audio codec for RLX, with a multi-backend **Vocos** decoder. Reconstructs waveform from WavTokenizer codes via an ISTFT head, running on any RLX backend (CPU / Metal / MLX / wgpu).

## Public API

```rust
use rlx_wavtokenizer::{WavtokEncoder, WavtokDecoderGraph, WavtokWeights, head_to_wav};

let weights = WavtokWeights::load(path)?;
// build/compile the Vocos decoder graph, run it to a spectral head, then:
let pcm = head_to_wav(&head_out, t, &window);   // ISTFT overlap-add -> waveform
# anyhow::Ok(())
```

Modules: `encoder` (`WavtokEncoder`), `graph` (`WavtokDecoderGraph`), `model` (`WavtokWeights`), `istft`.

## How it fits

A sibling of the other RLX neural codecs — [rlx-encodec](../rlx-encodec), [rlx-snac](../rlx-snac), [rlx-xcodec](../rlx-xcodec) — supplying waveform reconstruction for TTS/audio pipelines.
