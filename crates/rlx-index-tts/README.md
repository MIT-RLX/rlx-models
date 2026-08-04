# rlx-index-tts

**IndexTTS-2** on RLX — native Rust. A controllable TTS with explicit **duration
control** and **emotion control**: a GPT-style AR backbone predicts semantic tokens
from text (+ speaker + emotion), a **flow-matching S2A** stage turns those into a
mel, and a BigVGAN vocoder renders the waveform.

| Component | Reuse |
|-----------|-------|
| GPT AR backbone | `rlx-llama32` |
| S2A flow head + CFG | `rlx-audio-blocks::sampling` (`FlowMatchEuler::ascending`, `classifier_free_guidance`) |
| Vocoder (BigVGAN) | `rlx-neutts` |

## What's here (checkpoint-free, tested)

- `IndexTtsConfig` — GPT backbone + semantic vocab/rate + S2A flow + mel + speaker
  & emotion dims.
- `tokens_for_duration` — the model's signature **duration control** (seconds →
  semantic-token count).
- `s2a_scheduler` (noise→data flow) + `guided` (CFG) + `blend_emotion` (emotion
  interpolation via the shared guidance blend).

4 CPU smoke tests: config/validate, duration→token mapping, S2A schedule, CFG +
emotion blend endpoints.

## Next step

Wire the GPT AR backbone, the S2A flow DiT, and BigVGAN for end-to-end synthesis,
then per-backend parity. Needs an IndexTTS-2 checkpoint.
