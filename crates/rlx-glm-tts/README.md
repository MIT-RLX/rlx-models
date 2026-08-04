# rlx-glm-tts

**GLM-TTS** (Zhipu GLM-4-Voice family) on RLX — native Rust. Zero-shot TTS / voice
cloning: a GLM backbone (Llama-shaped) emits low-rate single-codebook speech tokens
in a **streaming** text/audio interleave; a CosyVoice-style **flow-matching
token→mel** decoder produces a mel; a HiFiGAN vocoder renders audio.

> Note: `rlx-glm` is the GLM *chat* LLM; this crate is the GLM *TTS*, which reuses
> that backbone.

| Component | Reuse |
|-----------|-------|
| GLM backbone | `rlx-glm` (Llama-shaped) / `rlx-llama32` |
| token→mel flow + CFG | `rlx-audio-blocks::sampling` |
| Vocoder (HiFiGAN/BigVGAN) | `rlx-nanocodec` / `rlx-neutts` |

## What's here (checkpoint-free, tested)

- `GlmTtsConfig` — GLM backbone + single-codebook speech vocab/rate + streaming
  interleave (`text_chunk` 13 / `audio_chunk` 26) + flow decoder + CFG.
- `streaming_schedule` — the GLM-4-Voice text/audio interleave blocks.
- `tokens_for_duration`, `token2mel_scheduler` (noise→data), `guided` (CFG).

4 CPU smoke tests: config/validate, streaming interleave (13/13/4 text : 26 audio),
empty text, duration + flow + guidance.

## Next step

Wire the GLM backbone, the flow token→mel decoder, and HiFiGAN for end-to-end
synthesis, then per-backend parity. Needs a GLM-TTS checkpoint.
