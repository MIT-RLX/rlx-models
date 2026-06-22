# rlx-nanocodec

[NVIDIA NanoCodec](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)
(NeMo `nemo-nano-codec-22khz-*`) decoder ported to [rlx-runtime](../rlx-runtime)
graphs, running natively and **bit-exactly on every backend** (CPU / Metal / MLX /
CUDA / ROCm / wgpu / Vulkan).

NanoCodec is a low-bitrate (0.6 kbps, 12.5 fps) 22 kHz speech codec built from a
Group Finite-Scalar-Quantizer (FSQ) and a causal HiFi-GAN generator. This crate
ports the decode path:

- **Group-FSQ dequant** — 4 groups × 4 dims, levels `[9,8,8,7]`; each group index
  splits into per-dim levels and maps to a centered continuous code. Pure host
  arithmetic (no learned codebook).
- **CausalHiFiGAN decoder** — `WNConv1d(16→864,k7)` → 5 up-sampling stages
  (rates 7/7/6/3/2; channels 864→432→216→108→54→27) each = HalfSnake →
  **grouped causal transposed conv** (groups = out-channels) → HiFiGAN residual
  layer (mean of 3 kernel blocks {3,7,11}, each chaining dilations {1,3,5}) →
  HalfSnake → `WNConv1d(27→1,k3)` → clamp `[-1,1]`.

All convolutions are causal (left-padded). **HalfSnake** applies Snake
(`x + sin(αx)²/(α+1e-9)`) to the first half of the channels and LeakyReLU(0.01) to
the rest. Weight-norm is folded into plain `.weight` when the fixture is generated.

The reference fixture transcribes NeMo's verbatim forward math from
`audio_codec_modules.py` + `common/parts/utils.py` (no NeMo runtime needed — the
`.nemo` archive is just `model_config.yaml` + `model_weights.ckpt`).

## Backend parity

```
nanocodec decoder Cpu   vs official: max|Δ| = 8.85e-6
nanocodec decoder Metal vs official: max|Δ| = 5.75e-6
nanocodec decoder Mlx   vs official: max|Δ| = 5.75e-6
```

## Regenerating the fixture

```bash
python3 crates/rlx-nanocodec/scripts/gen_fixture.py   # downloads the .nemo, no NeMo install
cargo test -p rlx-nanocodec --features metal,mlx -- --nocapture
```

The `*.safetensors` / `*.bin` fixtures are gitignored.
