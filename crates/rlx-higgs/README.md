# rlx-higgs

**Higgs-Audio v2** (Boson AI) on RLX — native Rust. A unified audio-language model
over a **Llama-3.2 backbone** with a **DualFFN** audio adapter and an **RVQ audio
tokenizer**; the one model does TTS (text→audio) and STT (audio→text).

| Component | Reuse |
|-----------|-------|
| Backbone (Llama-3.2 + DualFFN) | `rlx-llama32` |
| RVQ codebook delay pattern | `rlx-audio-blocks::codec` (added here) |
| Audio tokenizer decode | RVQ + GAN codec stack (`rlx-dac` / `rlx-neutts`) |

## What's here (checkpoint-free, tested)

- `HiggsConfig` — Llama-3.2-3B backbone dims + DualFFN flag + RVQ tokenizer
  (num_codebooks, codebook_size, frame_rate) + validate (incl. GQA divisibility).
- `HiggsMode` — `TextToAudio` (TTS) / `AudioToText` (STT).
- `delay_encode` / `delay_decode` — RVQ codebook delay interleave via the shared
  `rlx-audio-blocks::codec` delay pattern.

4 CPU smoke tests here (+ 4 for the shared delay pattern in `rlx-audio-blocks`).

Also promoted into the foundation this round: `rlx-audio-blocks::codec`
(`build_delay_pattern` / `revert_delay_pattern`) — the RVQ interleaving shared by
MusicGen / Parler / Higgs-style AR audio LMs.

## Next step

Wire the Llama-3.2 backbone + DualFFN branch and the RVQ tokenizer decode for
end-to-end TTS/STT, then per-backend parity. Needs a Higgs-Audio v2 checkpoint.
