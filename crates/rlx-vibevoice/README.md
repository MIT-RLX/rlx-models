# rlx-vibevoice

[VibeVoice](https://huggingface.co/microsoft/VibeVoice-1.5B)'s **acoustic σ-VAE
tokenizer decoder** ported to [rlx-runtime](../rlx-runtime) graphs, running
natively and **bit-exactly on every backend** (CPU / Metal / MLX / CUDA / ROCm /
wgpu / Vulkan).

VibeVoice tokenizes speech into a continuous 7.5 Hz acoustic latent (σ-VAE,
`vae_dim=64`) plus a semantic stream. This crate ports the acoustic
`TokenizerDecoder` — a ConvNeXt-style causal upsampler that turns the latent back
into a waveform:

- **causal stem** — `Conv1d(64→2048, k7)`, left-padded;
- **7 ConvNeXt stages** (depths 8/3/3/3/3/3/3; dims 2048→1024→512→256→128→64→32)
  interleaved with **6 causal transposed-conv upsamplers** (strides 8/5/5/4/2/2);
- **causal head** — `Conv1d(32→1, k7)`.

Each ConvNeXt block is `x + γ·mixer(RMSNorm(x))` then `x + γ_ffn·FFN(RMSNorm(x))`,
where the mixer is a depthwise causal conv (k7) and the FFN is `Linear→GELU→Linear`
(4× expansion). RMSNorm is weight-only (eps 1e-5); the layer-scales γ are
per-channel. No weight-norm (plain convs). The encoder, semantic tokenizer, and
diffusion LM head are out of scope.

The reference fixture instantiates the **original** VibeVoice
`modular_vibevoice_tokenizer.TokenizerDecoder` (its weight keys match the
checkpoint — the transformers-bundled port renames them) and decodes a random
latent.

## Backend parity

```
vibevoice decoder Cpu   vs official: max|Δ| = 8.20e-7
vibevoice decoder Metal vs official: max|Δ| = 6.52e-7
vibevoice decoder Mlx   vs official: max|Δ| = 1.09e-6
```

## Regenerating the fixture

```bash
python3 crates/rlx-vibevoice/scripts/gen_fixture.py   # fetches the modular file + weights
cargo test -p rlx-vibevoice --features metal,mlx -- --nocapture
```

The `*.safetensors` / `*.bin` fixtures are gitignored.
