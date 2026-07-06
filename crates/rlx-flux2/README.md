# rlx-flux2

**FLUX.2** (Black Forest Labs) rectified-flow text-to-image model for RLX. Ships the denoiser transformer trunk plus VAE, text encoder, and flow-matching scheduler: config parsing, BFL / NVFP4 packed-weight adaptation, a native CPU forward, and compiled HIR on GPU backends. Runs on `cpu`, `metal`/`mps`, `mlx`, `cuda`, `rocm`/`hip`, `gpu`/`wgpu`, and `vulkan`.

## Quick start

```bash
# Text-to-image (denoiser + VAE decode → PNG). Needs FLUX.2 weights on disk.
just flux2 --weights .cache/flux2/transformer \
  --text-encoder .cache/flux2/text_encoder --vae .cache/flux2/vae \
  --prompt "a red fox in the snow" --steps 28 --output out.png --device metal

# or directly:
cargo run -p rlx-flux2 --bin rlx-flux2 --release -- \
  --weights .cache/flux2/transformer --prompt "..." --dry
```

Key flags (see [`cli.rs`](src/cli.rs)): `--weights`, `--hf-repo`, `--config`, `--text-encoder`, `--vae`, `--tokenizer`, `--prompt` / `--negative-prompt`, `--cfg-scale`, `--width` / `--height` (latent grid) or `--pixel-width` / `--pixel-height`, `--steps`, `--seed`, `--output`, `--lora` / `--lora-scale`, `--packed` (NVFP4), `--device`, `--dry`. `rlx-flux2-serve` reads one JSON request per stdin line for a warm reusable session.

## Public API

Build a runner, encode the prompt, then drive the rectified-flow sampler to RGB:

```rust
use rlx_flux2::{Flux2Runner, Flux2SampleParams, generate_to_rgb};
use rlx_runtime::Device;

let runner = Flux2Runner::builder()
    .weights(".cache/flux2/transformer")
    .text_encoder_dir(".cache/flux2/text_encoder")
    .vae_dir(".cache/flux2/vae")
    .device(Device::Metal)
    .build()?;

let (hidden, txt_ids) = runner.encode_prompt("a red fox in the snow")?;

let params = Flux2SampleParams {
    encoder_hidden_states: &hidden,
    encoder_negative: None,
    txt_ids: &txt_ids,
    neg_txt_ids: None,
    num_inference_steps: 28,
    cfg_scale: 1.0,
    guidance: None,
    latent_h: 64,
    latent_w: 64,
    seed: 0,
    init_timestep: 0,
    initial_latents: None,
    reference: None,
};

let (rgb, h, w) = generate_to_rgb(&runner, &params)?; // HWC u8
# anyhow::Ok(())
```

`sample_rectified_flow` returns raw latents ([`Flux2SampleOutput`](src/pipeline.rs)) if you want to VAE-decode yourself.

Lower-level pieces are all re-exported from the crate root:

| Area | Items |
|------|-------|
| Config | [`Flux2Config`](src/config.rs) (`flux2_dev`, `flux2_klein_9b`, `tiny`) |
| Transformer forward | [`flux2_transformer_forward`](src/forward.rs), `Flux2ForwardInput`; graph builders in [`hir_builder`](src/hir_builder.rs) / [`builder`](src/builder.rs) |
| Weights | [`extract_flux2_weights`](src/weights.rs), `adapt_bfl_weights`, NVFP4 packed loaders in [`packed`](src/packed.rs) / [`packed_gguf`](src/packed_gguf.rs) |
| Text encoder | [`encode_flux2_prompt`](src/text_encoder.rs), `Flux2TextEncoderGraph` |
| VAE | [`flux2_vae_decode`](src/vae.rs) / `flux2_vae_encode`, `Flux2VaeConfig` |
| Scheduler / flow | [`flow_match_euler_step`](src/scheduler.rs), `sample_rectified_flow`, `generate_to_rgb` |
| Guidance | Diamond posterior-sampling guidance in [`diamond`](src/diamond.rs) (`sample_rectified_flow_diamond`, reward hooks from [rlx-diamond](../rlx-diamond)) |

## Compile policy

Denoiser + VAE use compiled HIR on non-CPU backends. The text encoder uses compiled HIR on **Metal / MLX only** — CUDA/ROCm/wgpu encode once on native CPU, then drop TE weights before compiling the denoiser. Session/AOT caching lives in [`session`](src/session.rs) and [`compile_util`](src/compile_util.rs).

## Features

`flux2-tokenizer` + `flux2-image` (default) · `hf-download` (Hub fetch) · backends `metal` / `mlx` / `cuda` / `rocm` / `gpu` / `vulkan` (+ `all-backends`, `apple-silicon`, `nvidia-gpu`, …).

## How it fits

- The text encoder is a Qwen3-shaped trunk from [rlx-qwen3](../rlx-qwen3); guidance rewards come from [rlx-diamond](../rlx-diamond).
- Packed weights load through [rlx-gguf](../rlx-gguf); graphs compile and run via [rlx-ir](../rlx-ir) / [rlx-flow](../rlx-flow) / [rlx-runtime](../rlx-runtime).
