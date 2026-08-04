# rlx-stable-audio

**Stable Audio Open** on RLX — text-to-audio via a **rectified-flow DiT** over an
autoencoder latent, conditioned on a T5 text embedding + timing (seconds) features.
All-Rust, reusing rlx:

| Component | Reuse |
|-----------|-------|
| RF Euler sampler | `rlx-audio-blocks::sampling::FlowMatchEuler` + this crate's schedule |
| Text conditioner (T5) | native T5 in `rlx-parlertts` |
| Diffusion transformer (DiT) | flow-matching DiT patterns in `rlx-flux2` / `rlx-vlash` |
| Audio autoencoder (SAME VAE) | conv codec stack (`rlx-dac` / `rlx-encodec`) |

## What's here (checkpoint-free, tested)

- `StableAudioConfig` — audio/AE + conditioner + DiT dims + sampler shift.
- `sampler` — the rectified-flow schedule: length-dependent timestep shift
  (`LogSnr` default, `Full`, `None`), `make_schedule` (sigmas 1→0),
  `effective_latent_length`.
- `StableAudioConfig::flow_match_scheduler` — builds a shared `FlowMatchEuler`
  over the RF schedule for a given clip length + step count.

7 CPU smoke tests: config defaults, logsnr/full/none shift curves (endpoints +
monotonicity), schedule descent, latent length, and the `FlowMatchEuler` reuse.

## Next step

Wire the DiT (reuse `rlx-flux2` flow-matching transformer), the T5 + seconds
conditioner, and the SAME autoencoder decode, then per-backend parity. Needs a
Stable Audio Open checkpoint.
