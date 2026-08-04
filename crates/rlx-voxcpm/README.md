# rlx-voxcpm

**VoxCPM** on RLX — native Rust. A **tokenizer-free** multilingual TTS: a
MiniCPM-style LM backbone conditions a **local flow-matching** head that generates
continuous acoustic latents per frame (no discrete codebook), then a vocoder
renders them.

| Component | Reuse |
|-----------|-------|
| MiniCPM backbone | `rlx-minicpm5` / `rlx-llama32` |
| Local flow head (noise→data) + CFG | `rlx-audio-blocks::sampling` (`FlowMatchEuler::ascending`, `classifier_free_guidance`) |
| Vocoder | `rlx-neutts` BigVGAN / `rlx-tsac` HiFT |

## What's here (checkpoint-free, tested)

- `VoxCpmConfig` — MiniCPM backbone dims + acoustic latent + `frames_per_step` +
  `flow_steps` + `cfg_scale`, with GQA-divisibility validation.
- `local_flow_scheduler` — the noise→data flow-matching schedule.
- `guided` — classifier-free guidance at the model's scale.
- `acoustic_frames` — LM-step → acoustic-frame expansion.

5 CPU smoke tests: config/validate, GQA check, frame expansion, noise→data schedule,
CFG scaling.

Also added to the foundation this round: `FlowMatchEuler::ascending` (the
conditional-flow-matching noise→data schedule, shared with Seed-VC-style heads).

## Next step

Wire the MiniCPM backbone + local flow DiT + vocoder for end-to-end synthesis, then
per-backend parity. Needs a VoxCPM checkpoint.
