# rlx-demucs

**Hybrid-Transformer Demucs** (`htdemucs`) music source separation on RLX, native
Rust. A dual-branch model — a **time branch** (1D conv U-Net on the waveform) and a
**spectral branch** (2D conv U-Net on the STFT) — joined by a **cross-domain
Transformer**, outputting 4 stems (drums/bass/other/vocals). Long audio is split
into overlapping segments with triangular overlap-add.

| Component | Reuse |
|-----------|-------|
| STFT / ISTFT (spectral branch) | `rlx-fft` |
| Conv U-Nets + transformer | `rlx-flow` |

## What's here (checkpoint-free, tested)

- `DemucsConfig` — stems + STFT + conv U-Net (`base_channels`, `growth`, `depth`) +
  cross-domain transformer + segmentation (`segment_length`, `overlap`).
- `encoder_channels` — the U-Net channel progression (`base · growth^i`).
- `segment_starts` / `transition_weight` — the overlap-add inference segmentation
  (stride = `segment·(1−overlap)`) and its triangular cross-fade weight.

4 CPU smoke tests: config, channel doubling, segment coverage, symmetric triangle.

## Next step

Wire the two conv U-Net branches, the cross-domain transformer, and the `rlx-fft`
STFT/ISTFT + overlap-add for end-to-end separation, then per-backend parity. Needs
an htdemucs checkpoint.
