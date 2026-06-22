# rlx-facodec

[FACodec](https://huggingface.co/amphion/naturalspeech3_facodec) (NaturalSpeech 3
factorized codec) decoder ported to [rlx-runtime](../rlx-runtime) graphs, running
natively and **bit-exactly on every backend** (CPU / Metal / MLX / CUDA / ROCm /
wgpu / Vulkan).

FACodec factorizes speech into prosody / content / residual / timbre subspaces.
This crate ports `FACodecDecoder` — the HiFi-GAN/BigVGAN-style generator that turns
the summed VQ latent back into a waveform:

- **timbre AdaIN** — LayerNorm over channels (no affine) followed by a per-channel
  scale + shift derived from the speaker embedding (`timbre_linear`);
- **anti-aliased SnakeBeta** — BigVGAN's `Activation1d`: 2× upsample (replicate-pad
  + depthwise transposed conv with a shared kaiser-sinc FIR) → `x + sin(eᵃx)²/eᵇ`
  → 2× downsample (replicate-pad + strided depthwise conv);
- **4 DecoderBlocks** — SnakeBeta → transposed conv (strides 5/5/4/2) → 3 residual
  MRF units (dilations 1/3/9);
- a final SnakeBeta → `WNConv1d` → `tanh`.

Weight-norm is folded into plain `.weight` when the fixture is generated, and the
per-speaker timbre affine is precomputed on the host (a 512×256 mat-vec); the full
generator runs on-device. The VQ encoder / SSL front-end are out of scope.

## Backend parity

`tests/decoder_matches_official_real_weights` checks the decoder against the
official Amphion `FACodecDecoder.inference` on real weights:

```
facodec decoder Cpu vs official: max|Δ| = 1.5e-5
```

## Regenerating the fixture

```bash
# fetch Amphion's ns3_codec module (models/codec/ns3_codec) into $NS3_CODEC_DIR
python3 crates/rlx-facodec/scripts/gen_fixture.py
cargo test -p rlx-facodec --features metal,mlx -- --nocapture
```

The `*.safetensors` / `*.bin` fixtures are gitignored.
