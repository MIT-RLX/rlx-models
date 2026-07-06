# rlx-lfm-vl

LiquidAI **[LFM2.5-VL](https://huggingface.co/LiquidAI)** vision tower for RLX (catalog row `LFM2.5-VL-1.6B`). LFM2-VL pairs a Google **SigLIP2** vision tower (separate-Q/K/V, pre-LN ViT) with a LLaVA-style 2-layer GELU projector into the LFM2.5 LM hidden dim. This crate implements the vision half, producing `[num_patches, lm_hidden]` embeddings the caller interleaves with text-token embeds at the `<image>` placeholder positions.

The text path uses [`rlx-lfm`](../rlx-lfm) (the LFM2.5 LM). Weight names follow the HF LFM2-VL layout (`vision_tower.vision_model.encoder.layers.{i}.…`, `multi_modal_projector.linear_{1,2}.…`).

## Status

PLAN.md M7. The SigLIP2 vision tower, image preprocessing, and projector are implemented. The binary only validates the GGUF `general.architecture` (one of `lfm2-vl` / `lfm25-vl` / `lfm2_5_vl` / `lfm-vl`) and directs you to the library API for image inference.

## Public API

```rust
use rlx_lfm_vl::{LfmVlVisionRunner, LfmVlVisionConfig};
use rlx_runtime::Device;
use std::path::Path;

let mut runner = LfmVlVisionRunner::builder()
    .mmproj("mmproj-lfm2-vl.gguf")     // vision weights GGUF
    .hf_config("config.json")          // or .config(LfmVlVisionConfig { .. })
    .device(Device::Cpu)
    .build()?;

// image -> [num_patches, projector_output_dim] LM-aligned embeddings
let embeds: Vec<f32> = runner.embed_image_path(Path::new("photo.jpg"))?;
# anyhow::Ok(())
```

`LfmVlVisionRunner` implements the [`rlx-vlm-base`](../rlx-vlm-base) `VisionTower` + `Projector` traits and also exposes `embed_image_bytes` / `embed_patches` and a `preprocessor()`. `LfmVlImagePreprocessor` does bicubic resize + SigLIP normalization (mean/std = 0.5) + host-side patch embedding. The graph builder `build_lfm_vl_vision` (and `_with_packed`) returns an `LfmVlVisionBuilt` for custom compilation.

## How it fits

- Shared multimodal traits: [`rlx-vlm-base`](../rlx-vlm-base).
- LM text path: [`rlx-lfm`](../rlx-lfm) (LFM2.5).
- Sibling vision/omni runners: [`rlx-qwen3-vl`](../rlx-qwen3-vl), [`rlx-nemotron-omni`](../rlx-nemotron-omni), [`rlx-qwen25-vl`](../rlx-qwen25-vl).
