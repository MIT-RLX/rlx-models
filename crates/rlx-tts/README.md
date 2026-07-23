# rlx-tts — RLX FastSpeech2 + WaveRNN

Native RLX text-to-speech: Hydra frontend → FastSpeech2 mel → WaveRNN vocoder.

Weights ship as a single packed file:

```text
weights/tts/rlx-tts/rlx-tts.gguf
```

Override with `RLX_TTS_BUNDLE`. A loose directory bundle (safetensors + `frontend/`)
still loads if present; GGUF is preferred when both exist.

```bash
just tts-prepare SRC=/path/to/unpacked   # optional: import loose bundle
just export-rlx-tts-gguf                 # pack → rlx-tts.gguf (Rust, no Python)
just tts-pack-only                       # delete loose files; keep GGUF only
```

## Quick start

```bash
cargo run -p rlx-tts --release -- --probe-bundle
cargo run -p rlx-tts --release -- --text "Hello from RLX." --out /tmp/rlx-tts.wav
just tts-demo
```

```rust,ignore
use rlx_tts::{RlxTts, VarianceControls, WaveRnnOpts, write_wav};

let tts = RlxTts::open_default()?;
let audio = tts.synthesize_text(
    "Hello from RLX.",
    &VarianceControls::default(),
    &WaveRnnOpts::product_default(),
)?;
write_wav(&audio, std::path::Path::new("out.wav"))?;
```

## What’s inside the GGUF

- Neural: `encoder.*`, `decoder.*`, `wavernn.*`
- Frontend tables + TorchN G2P (materialized to `$TMPDIR` on open)
- Metadata: sample rate, voice id, format `rlx-tts-gguf-v1`

## Product path (macOS)

| Stage | Implementation |
|-------|----------------|
| Frontend | Hydra tables + TorchN G2P |
| Acoustic | FastSpeech2 |
| Vocoder | WaveRNN h448 — fused GRU via `rlx_cpu::vmath` |
| Sampler | NativeBnns Gumbel seed 16807, β=0.01 |
| Post | μ-law+IIR + output volume + leading silence |

## Stress / backends

```bash
just tts-stress -- --n 1000 --resume
just tts-backends
just tts-backends-whisper
```
