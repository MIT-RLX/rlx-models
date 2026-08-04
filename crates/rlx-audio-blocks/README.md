# rlx-audio-blocks

Shared, reusable building blocks for RLX audio models — the RLX analogue of
[`audio.cpp`](https://github.com/0xShug0/audio.cpp)'s `framework/` layer. The goal
is **"improve once, benefit many families"**: a new audio-model port becomes mostly
wiring of these components plus a thin model-specific glue crate.

Many blocks already live in dedicated crates, which stay the canonical home for
their weights/graphs:

| Block | Canonical crate(s) |
|-------|--------------------|
| BigVGAN vocoder | `rlx-neutts`, `rlx-facodec` |
| HiFiGAN / HiFT | `rlx-nanocodec`, `rlx-tsac`, `rlx-metavoice` |
| Vocos + ISTFT | `rlx-wavtokenizer`, `rlx-luxtts` |
| CAM++ speaker encoder | `rlx-funasr` |
| WavLM SSL | `rlx-miratts` |
| Native T5 encoder | `rlx-parlertts` |
| Conformer | `rlx-wav2vec2-bert`, `rlx-nemotron-asr` |
| RNN-T greedy decode | `rlx-nemotron-asr` |

This crate is deliberately **checkpoint-free**: it collects the pure,
model-agnostic *algorithms* (decode loops, samplers, DSP math) that those crates
and new ports both need, and — as the port campaign proceeds — re-exports the
canonical graph modules behind a single import surface.

## Modules

### `decoders`

- **`tdt`** — Token-and-Duration Transducer greedy decoder. Augments RNN-T with a
  duration head so decoding skips a learned number of encoder frames per step.
  Unlocks **Parakeet-TDT** and TDT-variant Nemotron models. The caller supplies a
  [`TdtDecoderCore`] (prediction + joint argmax); this crate owns the decode loop.
  Port of audio.cpp `framework/decoders/tdt_*`.

  ```rust
  use rlx_audio_blocks::decoders::{run_tdt_greedy_duration_loop, TdtDecoderCore};
  // impl TdtDecoderCore for your model's predictor+joint, then:
  let out = run_tdt_greedy_duration_loop(
      &mut core, &encoder_projected, frames, hidden, blank_id, &durations, max_symbols,
  )?;
  // out.token_ids / out.token_timestamps / out.token_durations
  ```

## Roadmap

Added as the port campaign reaches each family:

- `sampling` — torch-compatible RNG + diffusion schedulers (parity for flow/diffusion ports)
- `vocoders`, `speech_encoders`, `text_encoders` — re-exports of the canonical crates above
- streaming KV / conv primitives for real-time (TTFT) paths
