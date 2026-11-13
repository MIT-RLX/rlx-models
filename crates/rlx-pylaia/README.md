# rlx-pylaia

[PyLaia](https://gitlab.teklia.com/atr/pylaia) — Teklia's CRNN for handwritten
text recognition — natively in RLX. One text line in, text out.

`LaiaCRNN`: four VGG-style conv blocks (conv, batchnorm, LeakyReLU, max-pool),
a column sequencer, a 3-layer bidirectional LSTM and a linear layer, decoded
with greedy CTC. BatchNorm is folded into the convs at load; the whole forward
is one RLX graph per line width. ~4 M parameters, 42 MB checkpoint.

## Models

Every `Teklia/pylaia-*` release on the Hugging Face Hub, for example:

| Model | Language | |
|---|---|---|
| `Teklia/pylaia-rimes` | French (modern letters) | CER 4.5 % (RIMES test, no LM) |
| `Teklia/pylaia-iam` | English | |
| `Teklia/pylaia-norhand-v3`, `-belfort`, `-popp`, `-himanis`, … | Norwegian, French archives, Latin, … | |

The architecture comes from the `model` file beside `weights.ckpt`
(read without executing the pickle), so other `LaiaCRNN` hyperparameters,
GRU layers and the `avgpool-N` sequencer work too.

## Quick start

```bash
just fetch-pylaia rimes
just pylaia -- --weights weights/ocr/pylaia-rimes --image line.jpg
```

Lines must be at the model's training height (128 px for the Teklia models),
as PyLaia expects; `--height 128` rescales with Pillow's Lanczos.

## Parity

`scripts/pylaia_reference.py` is a plain-torch transcription of PyLaia
1.0.7's modules (PyLaia itself pins torch 1.13); the checkpoint loads into it
with `strict=True`.

```bash
just test-pylaia-parity line.jpg [DEVICE]
```

On six RIMES test lines: logits within 7e-7 (relative to scale) and identical
transcriptions on CPU, and within 1e-6 on Metal, MLX and wgpu. CPU is the
fastest backend for a line (~70–120 ms for a 2,500 px line, warm).

### NVIDIA / AMD

Also at parity on CUDA (RTX 3080 Ti), ROCm (Radeon 780M) and Vulkan (both),
with identical transcriptions. CUDA multiplies in TF32 by default — about
1e-4 relative, which can exceed the parity tolerance while the text stays
identical — so run CUDA parity with `RLX_CUDA_PARITY=1` (strict FP32).
