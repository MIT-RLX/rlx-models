# rlx-omnivoice

**OmniVoice** — massively-multilingual (**646+ languages**) voice-design TTS on RLX,
native Rust. An LM backbone emits neural-codec tokens conditioned on text, a
language id, and a voice-design (description → style) embedding; a neural codec
renders audio.

| Component | Reuse |
|-----------|-------|
| LM backbone | `rlx-llama32` |
| Neural codec | `rlx-dac` / `rlx-snac` |

## What's here (checkpoint-free, tested)

- `OmniVoiceConfig` — LM backbone + codec + `num_languages` (646) + language &
  voice-design embedding dims.
- `normalize_language` — canonical **ISO 639-3** handling (three-letter codes, since
  646 languages need 639-3): trims, lowercases, validates.

3 CPU smoke tests: config/validate, 639-3 acceptance (eng/ENG/jpn/cmn), malformed
rejection (639-1 two-letter, too-long, non-alpha).

## Next step

Wire the LM backbone + codec decode with language-id + voice-design conditioning
for end-to-end synthesis, then per-backend parity. Needs an OmniVoice checkpoint.
