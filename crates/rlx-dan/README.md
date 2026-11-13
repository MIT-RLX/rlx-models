# rlx-dan

[DAN](https://github.com/FactoDeepLearning/DAN) — the Document Attention
Network (Coquenet, Chatelain, Paquet, TPAMI 2023) — natively in RLX:
whole-page handwritten text recognition, no line segmentation, with layout
tokens marking the page structure.

A fully convolutional encoder (six conv blocks and four depthwise-separable
blocks, InstanceNorm, to 256 channels at 1/32 × 1/8 resolution) and an
8-layer transformer decoder that reads the page character by character with a
100-token self-attention window. Two graphs per page size: the encoder, which
also emits every layer's cross-attention K/V, and a one-token decoder step
with a ring of the last 99 tokens' K/V.

## Models

| Checkpoint ([Zenodo 7244382](https://zenodo.org/records/7244382)) | Pages | Size |
|---|---|---|
| `dan_rimes_page.pt` | RIMES 2009 — French letters, 150 dpi | 85 MB |
| `dan_read_page.pt`, `dan_read_double_page.pt` | READ 2016 — German | 85 MB |

## Quick start

```bash
just fetch-dan rimes
just dan -- --weights weights/ocr/dan-rimes/dan_rimes_page.pt --image page.png [--layout]
```

Input is the page as RGB 0–255, as the authors' `predict_example.py` feeds it
(the training-set mean/std are not in the checkpoint; the encoder's
InstanceNorm absorbs the scale). `DanRunner::with_normalization` applies them
for checkpoints that have them. The RIMES model was trained on 150 dpi pages.

## Parity

`scripts/dan_reference.py` is a literal plain-torch transcription of the
authors' encoder, decoder and greedy loop — including its cache of per-layer
outputs, which this crate replaces with the equivalent per-layer K/V cache.

```bash
just test-dan-parity page.png [DEVICE]
```

On a RIMES letter page (1240×792, 371 tokens): features within 5.7e-7,
decision logits within 7.1e-7 at every step, all 371 tokens identical on CPU;
within 3.3e-6 on Metal, MLX and wgpu. ~2.7 s per page on CPU. The READ 2016
checkpoint on the same page (300 tokens, capped): logits within 1.2e-6, all
tokens identical on CPU.

### NVIDIA / AMD

Also at parity on CUDA (RTX 3080 Ti), ROCm (Radeon 780M) and Vulkan (both),
with identical transcriptions. CUDA multiplies in TF32 by default — about
1e-4 relative, which can exceed the parity tolerance while the text stays
identical — so run CUDA parity with `RLX_CUDA_PARITY=1` (strict FP32).

Vulkan needed two upstream fixes this model exposed: GroupNorm (DAN's
InstanceNorm) ran one thread per group — wrong by ~1e-3 on NVIDIA and a
`DEVICE_LOST` watchdog reset on RADV for a full page — and attention ran one
thread per query row, so a decode step's 4-head cross-attention over 3.9K
feature positions took 0.7 s; 371 steps now take 13 s instead of 264 s.
