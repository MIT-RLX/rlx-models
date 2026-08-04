# rlx-dramabox

**DramaBox** — expressive TTS + voice cloning on RLX, native Rust. An LM backbone
emits neural-codec tokens conditioned on text, an inline **expressive** style track,
and an optional reference voice; a neural codec renders audio.

| Component | Reuse |
|-----------|-------|
| LM backbone | `rlx-llama32` |
| Neural codec | `rlx-dac` / `rlx-snac` |

## What's here (checkpoint-free, tested)

- `DramaBoxConfig` — LM backbone + codec + emotion-conditioning dim + cloning flag.
- `parse_expressive` — inline expressive-tag parser: `"Hello [happy]world [sad]bye"`
  → styled spans (neutral prefix, then `happy`, then `sad`). Unclosed `[` is literal.
- `strip_tags` — the plain spoken text.

4 CPU smoke tests: config/validate, tag parsing, plain text + strip, unclosed bracket.

## Next step

Wire the LM backbone + codec decode with expressive/emotion + reference conditioning
for end-to-end synthesis, then per-backend parity. Needs a DramaBox checkpoint.
