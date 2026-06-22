# rlx-aec

Pure Rust acoustic echo cancellation at **16 kHz mono** — FDAF overlap-save NLMS (`rustfft` via `rlx-fft`) plus optional per-bin RLX residual suppression.

Pre-ASR front-end: feed cleaned PCM to Whisper / Voxtral unchanged.

## CLI

```bash
cargo run -p rlx-aec --release -- \
  --mic-wav echoed_mic.wav \
  --ref-wav speaker_ref.wav \
  --out-wav cleaned.wav
```

Flags: `--step-size`, `--max-delay-ms`, `--no-residual`.

## Library

```rust
use rlx_aec::{AecConfig, AecSession};

let mut aec = AecSession::new(AecConfig::default())?;
let clean = aec.process(&mic_chunk, &far_end_chunk)?;
```

Duplex: `aec.push_reference(tts_pcm)` on playback; `aec.process_mic(mic_chunk)` on capture.

## Bench

```bash
just test-aec
just bench-aec
just bench-aec-parity   # Rust echo_bench + Python NLMS baseline
```

## Weights

Embedded `weights/residual_aec.safetensors` (identity mask v1). Regenerate:

```bash
python3 scripts/aec_synthetic_dataset.py --out-dir /tmp/aec_dataset
python3 scripts/aec_train_residual_export.py --dataset-dir /tmp/aec_dataset \
  --out crates/rlx-aec/weights/residual_aec.safetensors
```

## Tests

| Test | What |
|------|------|
| `synthetic_echo` | Delay, FDAF MSE improvement, session offline |
| `fdaf_parity` | vs `fdaf-aec` crate (dev-dep) |
| `whisper_e2e` | `RLX_WHISPER_DIR` + JFK echo fixture |
