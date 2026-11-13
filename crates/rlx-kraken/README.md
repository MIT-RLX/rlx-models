# rlx-kraken

[Kraken](https://kraken.re) recognition models — the HTR engine of
eScriptorium and the HTR-United / CATMuS model zoo — natively in RLX.

A Kraken model is a [VGSL](https://kraken.re/main/vgsl.html) network
serialized as a Core ML `.mlmodel` or, from Kraken 6 on, a `.safetensors`
with the metadata in its header. This crate reads both directly (no
coremltools, no Python), parses the VGSL spec from its metadata, and
builds the network as one RLX graph per line width: convolutions, max-pools,
Kraken's generalized reshape, uni/bidirectional LSTMs along x or y, and the
linear CTC output, decoded greedily through the model's codec.

Covered blocks: `C` (every activation, stride, dilation), `Mp`, `Do`, `S`,
`L{f,r,b}{x,y}[s]`, `O1{c,s,l}[a]`. Rejected with an error naming the block:
parallel/serial sub-networks, GRUs, group norm, transposed convolutions and
legacy (clstm/ocropy) LSTMs. Segmentation (`blla`) is a different model; lines
come in pre-cropped.

## Models

| Model | Languages | Size |
|---|---|---|
| CATMuS Medieval 1.5 ([10.5281/zenodo.12743230](https://doi.org/10.5281/zenodo.12743230)) | Old & Middle French, Latin, Old Spanish, Italian, … | 16 MB |

and the rest of the Kraken zoo on Zenodo (`kraken list`).

## Quick start

```bash
just fetch-kraken-catmus
just kraken -- --model weights/ocr/kraken-catmus-medieval/catmus-medieval.mlmodel --image line.png
```

Input follows Kraken's line transform for baseline segmentations: grayscale,
Pillow-exact Lanczos resize to the model height, 16 px of white padding each
side (or the model's `pad`), and `max − x` inversion.

## Parity

`scripts/kraken_reference.py` runs Kraken 7.1.1 itself (`.venv-kraken`).

```bash
uv venv -p python3.12 .venv-kraken && VIRTUAL_ENV=.venv-kraken uv pip install kraken==7.1.1
just test-kraken-parity line.png [DEVICE]
```

On four 13th-century lines from CATMuS: the preprocessed input matches
Kraken's to float rounding, logits within 8e-7, transcriptions identical on
CPU; within 1.1e-6 on Metal, MLX and wgpu.

### NVIDIA / AMD

Also at parity on CUDA (RTX 3080 Ti), ROCm (Radeon 780M) and Vulkan (both),
with identical transcriptions. CUDA multiplies in TF32 by default — about
1e-4 relative, which can exceed the parity tolerance while the text stays
identical — so run CUDA parity with `RLX_CUDA_PARITY=1` (strict FP32).
