# rlx-roformer-sep

**BS-RoFormer** and **Mel-Band-RoFormer** music source separation on RLX, native
Rust. STFT the mixture → split frequency bins into bands → RoFormer (RoPE
transformer, time/band attention) → complex mask per band → apply → ISTFT to
per-stem waveforms. The two variants differ only in the **band-split scheme**
(fixed bin widths vs mel-spaced).

| Component | Reuse |
|-----------|-------|
| STFT / ISTFT | `rlx-fft` |
| RoPE transformer | `rlx-llama-base` RoPE / `rlx-flow` attention |

## What's here (checkpoint-free, tested)

- `RoformerSepConfig` + `BandSplit` (`Fixed(widths)` for BS-RoFormer, `Mel(n)` for
  Mel-Band-RoFormer) + `num_freqs`.
- `fixed_band_ranges` / `mel_band_ranges` — the two band partitions (contiguous,
  full-coverage; mel uses the standard 2595·log₁₀(1+hz/700) scale).
- `apply_complex_mask` — complex `spec × mask` masking.

5 CPU smoke tests: config, fixed-band coverage (+ over-wide rejection), mel-band
contiguity/coverage, mel-scale monotonicity + round-trip, complex mask multiply.

## Next step

Wire the RoFormer graph (time/band attention over the band features) + mask
estimation head, and the `rlx-fft` STFT/ISTFT pipeline, then per-backend parity.
Needs a RoFormer checkpoint.
