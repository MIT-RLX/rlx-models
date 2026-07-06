# rlx-voxtral-tts-train

Native RLX **autodiff training for Voxtral voice cloning** — fine-tunes [rlx-voxtral-tts](../rlx-voxtral-tts) to a target voice. Two phases:

1. **Codec encoder** training (reconstruction + VQ auxiliary losses).
2. **LoRA adapters** on the 4B LM (embedding distillation).

Trained weights export/inject into `consolidated.safetensors` for inference.

## Quick start

```bash
# Training driver
cargo run -p rlx-voxtral-tts-train --bin rlx-voxtral-tts-train --release -- --help

# Codec-encoder micro-benchmark
cargo run -p rlx-voxtral-tts-train --example bench_encoder
```

## Modules

- `codec_graph` / `audio_metrics` — codec-encoder training + metrics.
- `asr_loss` — ASR-based distillation loss (via [rlx-whisper](../rlx-whisper)).
- `adam`, `backward_prep`, `compile`, `checkpoint`, `config` — the training loop.

## How it fits

- [rlx-voxtral-tts](../rlx-voxtral-tts) — the inference model these adapters/weights target.
- [rlx-tune](../rlx-tune) — shared LoRA machinery.
- Built on `rlx-autodiff` / `rlx-opt` / `rlx-compile`.
