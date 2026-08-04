# rlx-seed-vc

**Seed-VC** zero-shot voice conversion on RLX — native Rust. Seed-VC converts a
source utterance to a reference timbre with a **conditional flow-matching (CFM)**
DiT that predicts a mel/latent from source content + reference speaker embedding
(+ F0 for singing), then a vocoder. Composition of rlx pieces:

| Component | Reuse |
|-----------|-------|
| CFM Euler sampler + CFG | `rlx-audio-blocks::sampling::FlowMatchEuler` + this crate |
| Speaker embedding (CAM++) | `rlx-funasr` |
| Content encoder (Whisper/HuBERT) | `rlx-whisper` / `rlx-neutts` (Wav2Vec2-BERT) |
| Vocoder (BigVGAN) | `rlx-neutts` |

## What's here (checkpoint-free, tested)

- `SeedVcConfig` — mel geometry, content/speaker dims, DiT dims, F0 flag,
  `diffusion_steps`, `cfg_rate` (0.7).
- `cfm_scheduler` — the `0 → 1` flow-matching schedule over the shared
  `FlowMatchEuler` (ascends toward data, unlike the descending diffusion denoise).
- `cfg_blend` / `cfm_guided_step` — classifier-free guidance
  `v = v_uncond + rate·(v_cond − v_uncond)` and the guided Euler step.

4 CPU smoke tests: config/validate, ascending CFM schedule, CFG endpoints/midpoint,
guided-step integration.

## Next step

Wire the CFM DiT (reuse `rlx-flux2` / `rlx-vlash` flow-matching transformer), the
CAM++ speaker encoder + content encoder, and the BigVGAN vocoder for end-to-end
conversion, then per-backend parity. Needs a Seed-VC checkpoint.
