# rlx-qwen3-vl

Alibaba **[Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)** vision tower for RLX. The catalog target is `Qwen3-VL-30B-A3B-Instruct` — a 30B MoE (A3B active-experts routing on the LM side) with a SigLIP-variant ViT. This crate implements the **vision half**: a pre-LN SigLIP ViT (separate Q/K/V, GELU FFN, no LayerScale, no CLS) plus the multimodal projector (LayerNorm → 2× linear with GELU), producing `[num_patches, lm_hidden]` embeddings the caller interleaves into the Qwen3 text stream at `<|image_pad|>` positions.

The LM text path uses the existing [`rlx-qwen3`](../rlx-qwen3) MoE runner; the interleave glue lives in `rlx_cli::mtmd`. This crate's responsibility ends at the projected vision embeds.

## Status

STUB (PLAN.md M7). The vision tower, image preprocessing, and projector are implemented; the end-to-end multimodal LM wiring is in progress. The binary only validates the GGUF `general.architecture` (one of `qwen3vl` / `qwen3vlmoe` / `qwen3_vl` / `qwen3-vl`) and points you at the library API.

## Public API

```rust
use rlx_qwen3_vl::{Qwen3VlVisionRunner, Qwen3VlVisionConfig};
use rlx_runtime::Device;
use std::path::Path;

// Vision tower hyperparameters come from the HF config.json (or build one directly).
let mut runner = Qwen3VlVisionRunner::builder()
    .mmproj("mmproj-qwen3-vl.gguf")     // vision weights GGUF
    .hf_config("config.json")           // or .config(Qwen3VlVisionConfig { .. })
    .device(Device::Cpu)
    .build()?;

// image -> [num_patches, projector_output_dim] LM-aligned embeddings
let embeds: Vec<f32> = runner.embed_image_path(Path::new("photo.jpg"))?;
# anyhow::Ok(())
```

`Qwen3VlVisionRunner` implements the [`rlx-vlm-base`](../rlx-vlm-base) `VisionTower` + `Projector` traits and also exposes `embed_image_bytes` / `embed_patches`; `Qwen3VlImagePreprocessor` does bicubic resize + SigLIP normalization (mean/std = 0.5) + host-side patch embedding. The graph builder `build_qwen3_vl_vision` (and `_with_packed`) returns a `Qwen3VlVisionBuilt` for custom compilation.

## How it fits

- Shared multimodal traits: [`rlx-vlm-base`](../rlx-vlm-base).
- LM text path: [`rlx-qwen3`](../rlx-qwen3) (Qwen3 MoE, A3B routing).
- Sibling vision/omni runners: [`rlx-lfm-vl`](../rlx-lfm-vl), [`rlx-nemotron-omni`](../rlx-nemotron-omni), [`rlx-qwen25-vl`](../rlx-qwen25-vl).
