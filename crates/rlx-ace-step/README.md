# rlx-ace-step

**ACE-Step** music generation on RLX — native Rust. A **flow-matching DiT** over a
music autoencoder (DCAE) latent, conditioned on a UMT5 text/tag encoder + a lyric
encoder, sampled with an SD3-shifted flow schedule and classifier-free guidance.

| Component | Reuse |
|-----------|-------|
| Flow Euler sampler + SD3 shift + CFG | `rlx-audio-blocks::sampling` |
| Text/tag + lyric conditioner (UMT5) | native T5 in `rlx-parlertts` |
| Flow-matching DiT | `rlx-flux2` / `rlx-vlash` patterns |
| Music autoencoder (DCAE) | conv codec stack (`rlx-dac` / `rlx-encodec`) |

## What's here (checkpoint-free, tested)

- `AceStepConfig` — audio/DCAE latent + DiT dims + UMT5/lyric conditioning +
  `flow_shift` (3.0) + `guidance_scale` (7.0).
- `flow_scheduler` — `FlowMatchEuler` over `sd3_shifted_sigmas` (this model's shift).
- `guided` — classifier-free guidance at the model's scale.

3 CPU smoke tests: config/validate, SD3-shifted schedule vs linear, CFG scaling.

Also promoted into the foundation this round: `sampling::classifier_free_guidance`
and `sampling::{sd3_time_shift, sd3_shifted_sigmas}` — shared by every SD3/Flux-style
flow generator.

## Next step

Wire the flow-matching DiT, the UMT5 + lyric conditioner, and the DCAE decode for
end-to-end music generation, then per-backend parity. Needs an ACE-Step checkpoint.
