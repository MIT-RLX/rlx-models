# rlx-trocr

[TrOCR](https://arxiv.org/abs/2109.10282) — Microsoft's transformer text
recognizer — natively in RLX.

A `VisionEncoderDecoderModel`: a DeiT (small) or ViT (base, large) encoder
over a 384×384 line image and a post-norm transformer decoder that writes the
text. Two graphs: the encoder, which also emits every decoder layer's
cross-attention K/V once per image, and a KV-cached one-token decoder step
with the output projection.

| Checkpoint | Encoder | Decoder | Params | Status |
|---|---|---|---|---|
| `microsoft/trocr-small-handwritten` | DeiT-S | 6 × 256 | 62 M | parity CPU, Metal, MLX, wgpu |
| `microsoft/trocr-base-handwritten` | ViT-B | 12 × 1024 | 334 M | parity CPU, Metal |
| `microsoft/trocr-large-handwritten` | ViT-L | 12 × 1024 | 558 M | parity CPU, Metal |

The Microsoft checkpoints are trained on IAM (English); they misspell French
accents. Fine-tunes of the same architecture load the same way.

## Quick start

```bash
just fetch-trocr small
just trocr -- --weights weights/ocr/trocr-small-handwritten --image line.jpg
```

Preprocessing is the HF processor's: Pillow-exact bicubic (small) or bilinear
(base, large) resize to 384×384, then `(v / 255 − 0.5) / 0.5`. Decoding is
greedy, `--max-new-tokens` (default 64).

## Parity

`scripts/trocr_reference.py` runs transformers 5.3 (`.venv-trocr`).

```bash
just test-trocr-parity line.jpg [DEVICE] [small|base|large]
```

Small on five lines (IAM + RIMES), base on three, large on two: identical
greedy token ids and text; decoder logits at every position within 4.8e-5.
Peak RSS on CPU: 2.7 GB (base), 3.4 GB (large). Encoder states agree
to ~1e-5 except on near-blank patches, where the final LayerNorm (eps 1e-12)
amplifies float noise to ~1e-3 on one or two rows.

### NVIDIA / AMD

Also at parity on CUDA (RTX 3080 Ti), ROCm (Radeon 780M) and Vulkan (both),
with identical transcriptions. CUDA multiplies in TF32 by default — about
1e-4 relative, which can exceed the parity tolerance while the text stays
identical — so run CUDA parity with `RLX_CUDA_PARITY=1` (strict FP32).
