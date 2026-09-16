# rlx-diarize

Native RLX **speaker diarization**: slide a window over mono PCM, embed each
window, then agglomeratively cluster into speaker turns.

## Backends

| Feature | Embeddings |
|---------|------------|
| *(default)* | Mel energy statistics (no neural weights) |
| `wespeaker` | WeSpeaker ResNet34-LM **on RLX** (`rlx-wespeaker`) — Metal / CoreML / CPU / … |

ONNX under `onnx/wespeaker.onnx` is only for RLX import / packing; runtime prefers
`graphs/wespeaker.rlxp`.

## Public API

```rust
use rlx_diarize::{DiarizeSession, DiarizeConfig, SpeakerTurn};
use rlx_runtime::Device;

let cfg = DiarizeConfig::default()
    .with_auto_wespeaker(&[std::path::Path::new("weights")])
    .with_device(Device::Metal);
let mut session = DiarizeSession::new(cfg)?;
let turns: Vec<SpeakerTurn> = session.diarize(&pcm_16k)?;
# anyhow::Ok(())
```

- `DiarizeConfig { window_sec, hop_sec, cluster_threshold, wespeaker_dir, device }`
  — defaults `1.5 s` / `0.75 s` / `0.35` / `None` / `Cpu`.
- `DiarizeSession::mel_stat(cfg)` — force mel-stat (always succeeds).

## Tests

```bash
cargo test -p rlx-diarize --release
cargo run -p rlx-diarize --release --example diarize_wav --features "wespeaker,metal" -- \
  clip.wav weights/wespeaker-voxceleb-resnet34-LM metal
```
