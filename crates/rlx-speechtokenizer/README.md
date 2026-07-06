# rlx-speechtokenizer

**SpeechTokenizer** (`fnlp/SpeechTokenizer`) RVQ speech codec for RLX — a residual-vector-quantized codec that disentangles content/acoustic information across quantizer layers. Runs on any RLX backend (CPU / Metal / MLX / wgpu).

## Public API

```rust
use rlx_speechtokenizer::{SpeechTokenizerCodec, SpeechTokenizerConfig};
use rlx_runtime::Device;

let cfg = SpeechTokenizerConfig::default();
let mut codec = SpeechTokenizerCodec::load(weights, cfg, Device::Cpu)?;
let codes = codec.encode(&pcm)?;   // RVQ codes (layered)
let pcm = codec.decode(&codes)?;   // reconstructed waveform
# anyhow::Ok(())
```

Modules: `codec`, `config`, `model`, `graph` (compiled IR path), `eager`, `lstm`.

## Quick start

```bash
cargo run -p rlx-speechtokenizer --example bench
```

## How it fits

A sibling of the other RLX neural codecs — [rlx-encodec](../rlx-encodec), [rlx-snac](../rlx-snac), [rlx-dac](../rlx-dac), [rlx-mimi](../rlx-mimi).
