# rlx-outetts

**OuteTTS** (OuteAI) multilingual TTS / voice cloning on RLX. The 1.0 line is a
**Llama-3 backbone** that autoregressively emits interleaved text + audio tokens,
where audio tokens are two **DAC** codebooks (`<|c1_N|>` / `<|c2_N|>`) decoded to a
24 kHz waveform. Pure composition of existing RLX crates:

| Component | Reuse |
|-----------|-------|
| LM backbone (Llama-3 + Llama-3 RoPE scaling) | `rlx-llama32` |
| Codec (2-codebook DAC decoder) | `rlx-dac` (bit-exact CPU/Metal/MLX/wgpu) |

Both already run on every RLX backend, so OuteTTS inherits full backend coverage
once wired.

## What's here (checkpoint-free, tested)

- `OuteTtsConfig` / `GenerationConfig` — faithful port of audio.cpp
  `community_models/outetts` (codebooks 2, codebook_size 1024, hop 320, sr 24000,
  rope_theta 5e5, rope factor 32; sampler temp 0.4 / top_k 40 / top_p 0.9 / min_p 0.05).
- `build_prompt_string` — the OuteTTS prompt
  `<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|>\n`.
- `AudioCodeMap` + `collect_codebooks` — the `<|c1_N|>` / `<|c2_N|>` token ↔ DAC
  codebook mapping (`append_audio_code` mirrors upstream, including the top-code
  guard) and stream de-interleaving.

6 CPU smoke tests: config/gen defaults, prompt wrapping, code-map round-trip,
codebook routing + top-code guard, stream de-interleave.

## Next step

Drive `rlx_llama32` with these prompts + a repetition-penalised sampler, route
generated tokens through `AudioCodeMap`, and decode the two codebooks with
`rlx_dac`. Then per-backend parity. Needs an OuteTTS checkpoint.
