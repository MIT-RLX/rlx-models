# rlx-vevo

**Vevo2** (Amphion) controllable zero-shot voice/style imitation on RLX, native
Rust. Vevo **disentangles** speech into content, style, and timbre and recombines
them from different references: an AR content→content-style transformer (style ref)
feeds a flow-matching content-style→mel stage (timbre ref), then a vocoder.

| Component | Reuse |
|-----------|-------|
| AR content→style transformer | `rlx-llama32` |
| Flow token→mel + CFG | `rlx-audio-blocks::sampling` |
| Vocoder | `rlx-neutts` BigVGAN |

## What's here (checkpoint-free, tested)

- `VevoConfig` — content/style vocabs + AR + flow + mel + CFG; `flow_scheduler`
  (noise→data) + `guided` (CFG).
- `collapse_repeats` / `expand_units` — the reduced-unit RLE the content tokenizer
  emits (consecutive duplicate units → `units + durations`, and inverse).
- `VevoControl` — disentangled-control presets (`voice_conversion`, `style_transfer`,
  `full_imitation`) with `converts_style` / `converts_timbre` queries.

4 CPU smoke tests: config + flow/guidance, control presets, unit RLE round-trip +
edge cases.

## Next step

Wire the AR transformer, the flow content-style→mel stage, and the vocoder for
end-to-end conversion, then per-backend parity. Needs a Vevo2 checkpoint.
