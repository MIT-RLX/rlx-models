# rlx-hviske

**Hviske** Danish ASR on RLX — Whisper-large-v3 finetunes by Syv.ai
(`syvai/hviske`, `-v2`, `-v3`). Architecturally identical to Whisper-large-v3, so
this crate is a thin **preset over [`rlx-whisper`](../rlx-whisper)**:

- pins the correct large-v3 config (128 mel bins, `d_model` 1280, 32/32 layers, vocab 51866),
- defaults decode language to Danish (`da`),
- forwards backend feature flags, so Hviske inherits **all rlx-whisper backends**
  (CPU / Metal / MLX / CoreML / CUDA / ROCm / Vulkan / wgpu).

```rust
use rlx_hviske::{danish_builder, HviskeVariant};
use std::path::Path;

let runner = danish_builder()
    .weights(Path::new("hviske-v3/model.safetensors"))
    .config_path(Path::new("hviske-v3/config.json"))
    .tokenizer_path(Path::new("hviske-v3/tokenizer.json"))
    .build()?;
```

Backends: `cargo build -p rlx-hviske --features all-backends` (or a single backend,
e.g. `--features metal`).

## Status

Preset + config, CPU smoke (4 tests: variant metadata, large-v3 config dims,
shared config, Danish defaults). Real-weight transcription + per-backend parity are
validated once a Hviske checkpoint is available — the compute path is entirely
`rlx-whisper`, which is already multi-backend.
