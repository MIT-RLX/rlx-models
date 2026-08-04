# rlx-rvc

**RVC** (Retrieval-based Voice Conversion) on RLX, native Rust. Extract HuBERT /
ContentVec **content features** from the source, **retrieve** the nearest
target-speaker features from a trained index and blend them (the "retrieval" that
fixes timbre), condition on **F0** (optionally transposed), and synthesize with an
**NSF-HiFiGAN** generator.

| Component | Reuse |
|-----------|-------|
| Content encoder (HuBERT/ContentVec) | `rlx-neutts` / `rlx-wav2vec2-bert` |
| NSF-HiFiGAN generator | `rlx-nanocodec` / `rlx-neutts` |

## What's here (checkpoint-free, tested)

- `RvcConfig` — content/output sample rates, content_dim, `index_rate`, transpose.
- `retrieval_blend` — the k-NN feature-index blend: `(1−rate)·query + rate·Σ wᵢ·featᵢ`
  with inverse-square-distance weights (closer neighbours dominate).
- `transpose_f0` — pitch shift `f0·2^(semitones/12)` (unvoiced frames stay 0).

5 CPU smoke tests: config validate, retrieval rate-0/full/half/weighting, F0 octave
shift.

## Next step

Wire the HuBERT content encoder + NSF-HiFiGAN generator (with F0 conditioning) for
end-to-end conversion, then per-backend parity. Needs an RVC model + index.
