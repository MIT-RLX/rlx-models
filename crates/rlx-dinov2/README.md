# rlx-dinov2

Meta's [**DINOv2**](https://github.com/facebookresearch/dinov2) self-supervised ViT for RLX (with optional register tokens). Runs the ViT-S/B/L encoder to produce patch + CLS tokens, or the linear classifier head when `num_classes > 0`. Weight keys match Meta / candle safetensors, so HF Hub checkpoints (e.g. `lmz/candle-dino-v2`) load with no remapping; a bundled converter handles HuggingFace `transformers` layouts.

## Quick start

```bash
just dinov2 --weights dinov2_vitb14.safetensors --variant base --img-size 518
# or:
cargo run -p rlx-dinov2 --release --bin rlx-dinov2 -- \
  --weights dinov2_vitb14.safetensors --variant base --img-size 518 --device cpu

# Convert a HuggingFace transformers checkpoint → Meta key layout:
cargo run -p rlx-dinov2 --release --bin rlx-dinov2-convert-hf -- src.safetensors [dst.safetensors]

# Batch-encode every *.png in a dir → per-view "features" safetensors:
cargo run -p rlx-dinov2 --release --features metal --bin rlx-dinov2-batch -- \
  --weights dinov2_vitl14.meta.safetensors --variant large --img-size 518 \
  --views ./views --out ./features
```

`rlx-dinov2` flags: `--weights`, `--device`, `--variant small|base|large`, `--img-size`, `--batch`, `--dry`.

## Public API

```rust
use rlx_dinov2::{DinoV2Runner, DinoV2Variant, DinoV2Output};
use rlx_runtime::Device;

let mut runner = DinoV2Runner::builder()
    .weights("dinov2_vitb14.safetensors")
    .device(Device::Cpu)
    .variant(DinoV2Variant::Base)   // Small | Base | Large
    .img_size(518)
    .batch(1)
    .build()?;

// rgb: row-major [H*W*3] u8.
match runner.predict_image(&rgb, h, w)? {
    DinoV2Output::Tokens { per_batch, seq, hidden } => {
        // per_batch[b]: [seq*hidden], token 0 = CLS
    }
    DinoV2Output::Logits { per_batch, num_classes } => { /* classifier head */ }
}
# anyhow::Ok(())
```

Lower-level exports (`src/lib.rs`): [`DinoV2Config`] (`vit_small` / `vit_base` / `vit_large`), the graph builder [`build_dinov2_graph_sized`] and [`DinoV2Flow`] / `build_dinov2_built`, host preprocessing [`assemble_hidden`] / [`rgb_u8_to_imagenet_nchw`], and GGUF loading via [`load_dinov2_from_gguf`] / `gguf_has_packed_linears`.

## How it fits

A vision backbone used elsewhere in the workspace (e.g. per-view feature extraction consumed by splat/clustering tooling — the `rlx-dinov2-batch` output matches `rlx-splat-anim cluster-by-features`). Sits alongside other vision encoders: [rlx-vjepa2](../rlx-vjepa2) (video ViT), [rlx-sam](../rlx-sam) / [rlx-sam2](../rlx-sam2) / [rlx-sam3](../rlx-sam3) (segmentation).

## Tests

```bash
cargo test -p rlx-dinov2   # encoder-only + classifier graphs build; register tokens; assemble_hidden math
```
