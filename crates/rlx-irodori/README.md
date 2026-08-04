# rlx-irodori

**Irodori-TTS** — Japanese voice-design TTS on RLX, native Rust. An LM backbone
emits neural-codec tokens conditioned on text + a voice-design (style/timbre)
embedding; a neural codec renders audio.

| Component | Reuse |
|-----------|-------|
| LM backbone | `rlx-llama32` |
| Neural codec | `rlx-dac` / `rlx-snac` |

## What's here (checkpoint-free, tested)

- `IrodoriConfig` — LM backbone + codec + voice-design dim + `tokens_per_mora`.
- `count_morae` — a correct Japanese **mora** counter (Japanese TTS is timed in
  morae, not characters): base kana = 1, yōon/small-vowel kana attach (0), sokuon
  っ/ッ = 1, moraic nasal ん/ン = 1, chōonpu ー = 1, non-kana ignored.
- `tokens_for_kana` — a mora-based acoustic-token budget.

4 CPU smoke tests: config/validate, mora counting (とうきょう=4, がっこう=4, きゃ=1,
ラーメン=4, …), non-kana ignored, token budget.

## Next step

Wire the LM backbone + codec decode (with voice-design conditioning) and a full
Japanese g2p/kana frontend for end-to-end synthesis, then per-backend parity. Needs
an Irodori checkpoint.
