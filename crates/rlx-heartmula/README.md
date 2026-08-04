# rlx-heartmula

**HeartMula** music generation on RLX, native Rust. A MusicGen-style codec-token
LM: a transformer emits the RVQ codebooks of a neural music codec (**HeartCodec**),
interleaved with the delay pattern, then the codec decodes to a waveform.

| Component | Reuse |
|-----------|-------|
| LM backbone | `rlx-llama32` |
| RVQ delay pattern | `rlx-audio-blocks::codec` |
| HeartCodec decode | RVQ + GAN codec (`rlx-dac` / `rlx-encodec` patterns) |

## What's here (checkpoint-free, tested)

- `HeartMulaConfig` — LM backbone + condition dim + HeartCodec RVQ (num_codebooks,
  codebook_size, frame_rate) + `max_seconds`.
- `frames_for_duration` — duration→codec-frame control (clamped to `max_seconds`).
- `delay_encode` / `delay_decode` — RVQ codebook delay interleave via the shared
  `rlx-audio-blocks::codec` delay pattern.

3 CPU smoke tests: config/validate, duration clamping, delay round-trip.

> HeartMula's exact dims come from its checkpoint; values here follow MusicGen-family
> conventions.

## Next step

Wire the LM backbone + HeartCodec decode (with text/melody conditioning) for
end-to-end generation, then per-backend parity. Needs a HeartMula checkpoint.
