# rlx-inflect-v2

**Inflect v2** on RLX — a VITS-style end-to-end flow TTS (espeak phonemes → text
encoder + stochastic duration predictor + normalizing flow → HiFiGAN-style decoder,
24 kHz).

> ⚠️ Inflect v2 is a **different architecture** from Inflect-Nano v1
> (`rlx-inflect-nano`, a mel-acoustic model + separate vocoder). v2 is a **VITS2**
> family model, so its synthesis graph is shared with
> [`rlx-tiny-tts`](../rlx-tiny-tts) (MeloTTS / VITS2), which already runs on all
> RLX backends. This crate should drive that graph rather than reimplement it.

## What's here (checkpoint-free, tested)

- `InflectV2Config` — architecture config ported from audio.cpp
  `community_models/inflect_v2` (vocab 178, sr 24000, hop 256, flow_count 4, …).
- `GenerationOptions` — VITS controls (`speaking_rate`, `variation` = noise scale
  0.667, `seed`).
- `sample_flow_prior` — the VITS latent prior `z ~ variation · N(0, I)`, drawn from
  `rlx-audio-blocks::sampling::Rng` (seeded ⇒ identical noise across backends).

6 CPU smoke tests: config defaults vs audio.cpp, frame-rate, generation defaults,
prior determinism/shape, variation scaling, zero-variation.

## Next step

Map `InflectV2Config` → `rlx_tiny_tts::BundleConfig`, wire the espeak phoneme
frontend (`rlx_tiny_tts::frontend`), and drive `rlx_tiny_tts::TinyModel` for
end-to-end synthesis, then per-backend parity. Needs a v2 checkpoint.
