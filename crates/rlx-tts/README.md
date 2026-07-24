# rlx-tts — RLX FastSpeech2 + WaveRNN

Native RLX text-to-speech: Hydra frontend → FastSpeech2 mel → WaveRNN vocoder.

Weights ship as a single packed file:

```text
weights/tts/rlx-tts/rlx-tts.rlxp
```

Hub: [`eugenehp/rlx-tts`](https://huggingface.co/eugenehp/rlx-tts) (`just fetch-rlx-tts`).
Legacy `rlx-tts.gguf` still opens locally. Override with `RLX_TTS_BUNDLE`. A loose
directory bundle (safetensors + `frontend/`) still loads if present; `.rlxp` is
preferred when both exist.

```bash
just fetch-rlx-tts                       # download Hub .rlxp
just tts-prepare SRC=/path/to/unpacked   # optional: import loose bundle
just export-rlx-tts-rlxp                 # pack → rlx-tts.rlxp
just export-rlx-tts-gguf                 # optional legacy GGUF
just tts-pack-only                       # delete loose files; keep pack only
```

```bash
cargo run -p rlx-tts --release -- --pack-rlxp --bundle weights/tts/rlx-tts
# or re-pack an existing GGUF:
cargo run -p rlx-tts --release -- --pack-rlxp --bundle weights/tts/rlx-tts/rlx-tts.gguf
```

## Quick start

```bash
just fetch-rlx-tts
cargo run -p rlx-tts --release -- --probe-bundle
cargo run -p rlx-tts --release -- --text "Hello from our system." --out /tmp/rlx-tts.wav
just tts-demo
just tts-asr-whisper-check               # longer sentences → Whisper (+ ASR)
just tts-asr-validate-suite              # multi-model short/long WAV → Whisper + ASR
```

```rust,ignore
use rlx_tts::{RlxTts, VarianceControls, WaveRnnOpts, write_wav};

let tts = RlxTts::open_default()?;
let audio = tts.synthesize_text(
    "Hello from our system.",
    &VarianceControls::default(),
    &WaveRnnOpts::product_default(),
)?;
write_wav(&audio, std::path::Path::new("out.wav"))?;
```

## What’s inside the pack

- Neural: `encoder.*`, `decoder.*`, `wavernn.*` (`.rlxp` tensors; GGUF legacy)
- Frontend tables + TorchN G2P (materialized to `$TMPDIR` on open)
- Metadata: sample rate, voice id, format `rlx-tts-rlxp-v1`

## Product path

| Stage | Implementation |
|-------|----------------|
| Frontend | Hydra tables + TorchN G2P |
| Acoustic | FastSpeech2 |
| Vocoder | WaveRNN h448 — fused GRU via `rlx_cpu::vmath` |
| Sampler | NativeBnns Gumbel seed 16807, β=0.01 |
| Post | μ-law+IIR + output volume + leading silence |

Backends: CPU everywhere; Metal/MLX on Apple; CUDA / wgpu / Vulkan on Linux/Windows GPU hosts (`--features cuda` / `gpu` / `all-backends`).

## Stress / backends

```bash
just tts-stress -- --n 1000 --resume
just tts-backends
just tts-backends-whisper
```
