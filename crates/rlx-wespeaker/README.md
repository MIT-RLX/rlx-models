# rlx-wespeaker

Speaker embeddings from **WeSpeaker ResNet34-LM** (256-d) on **native RLX**
(Metal / CoreML / CPU / … via `TinyModel` + `rlx-onnx-import`).

## Weights

```text
weights/wespeaker-voxceleb-resnet34-LM/
  onnx/wespeaker.onnx          # fixed-T=148 graph for RLX import
  onnx/wespeaker_ref.onnx      # upstream packed ONNX (ORT reference only)
  graphs/wespeaker.rlxp        # packed native RLX weights (preferred)
```

Prepare / pack:

```bash
# bake + pack (after placing onnx/wespeaker.onnx)
cargo run -p rlx-wespeaker --release --example pack_rlxp --features pack -- \
  weights/wespeaker-voxceleb-resnet34-LM
```

## API

```rust
use rlx_runtime::Device;
use rlx_wespeaker::WeSpeaker;

let mut spk = WeSpeaker::open_on("weights/wespeaker-voxceleb-resnet34-LM", Device::Metal)?;
let emb = spk.embed_pcm(&pcm_16k)?; // L2-normalized 256-d
```

Windows are fixed at **148** fbank frames (~1.5 s). Shorter audio is zero-padded.

## Features

| Feature | Role |
|---------|------|
| `native` (default) | RLX TinyModel path |
| `onnx` | Optional ORT parity vs upstream packed ONNX |
| `metal` / `coreml` / … | RLX backends |
| `pack` | `pack_rlxp` example |

Used by `rlx-diarize` (`wespeaker` feature).
