# rlx-confucius

**Confucius4-TTS** — multilingual voice-cloning TTS on RLX, native Rust. An LM
backbone emits neural-codec tokens conditioned on target text + a reference
utterance (transcript + codec frames); a neural codec renders audio in the
reference speaker's voice.

| Component | Reuse |
|-----------|-------|
| LM backbone | `rlx-llama32` |
| Neural codec | `rlx-dac` / `rlx-snac` |

## What's here (checkpoint-free, tested)

- `ConfuciusConfig` — LM backbone + codec + multilingual + cloning flag.
- `plan_clone` — the **voice-cloning prompt planner**: orders reference-text →
  reference-audio → target-text → target-audio-start so the backbone is primed with
  the paired (text, audio) reference before generating.

3 CPU smoke tests: config/validate, clone-plan ordering + counts, rejection of
missing reference audio / empty target.

## Next step

Wire the LM backbone + codec decode with reference conditioning for end-to-end
cloning, then per-backend parity. Needs a Confucius4-TTS checkpoint.
