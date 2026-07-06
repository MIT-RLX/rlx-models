# rlx-sam

Meta's [**Segment Anything (SAM v1)**](https://github.com/facebookresearch/segment-anything) for RLX. Loads the official `sam_vit_{b,l,h}` safetensors (candle / Meta key layout, no remapping), builds an RLX IR graph for the ViT image encoder + neck, then runs the prompt encoder, two-way transformer, and mask decoder to turn point/box prompts into masks.

## Quick start

```bash
# CPU (default features)
just sam1 --weights sam_vit_b_01ec64.safetensors --device cpu --point 512,512
# or directly:
cargo run -p rlx-sam --release --bin rlx-sam1 -- \
  --weights sam_vit_b_01ec64.safetensors --device cpu --point 512,512

# Metal / MLX / CUDA … (pick the matching feature)
cargo run -p rlx-sam --release --features metal --bin rlx-sam1 -- \
  --weights sam_vit_b_01ec64.safetensors --device metal
```

CLI flags: `--weights`, `--device (cpu|metal|mlx|cuda|rocm|gpu|vulkan|tpu)`, `--point X,Y`, `--dry`. The encoder variant is selected with `RLX_SAM_VARIANT=vit_b|vit_l|vit_h` (default `vit_b`). The `rlx-sam-batch` bin runs the same model over a batch.

## Public API

```rust
use rlx_sam::{Sam, SamConfig, Device};

let mut sam = Sam::from_safetensors_on(
    "sam_vit_b_01ec64.safetensors",
    SamConfig::vit_b(),   // or ::vit_l() / ::vit_h()
    Device::Cpu,
)?;

// rgb: row-major [H*W*3] u8; one positive point prompt at (cx, cy).
let (pred, _low_res) = sam.forward(
    &rgb, h, w,
    Some((&[cx, cy], &[1.0f32])),  // point coords + labels (1 = foreground)
    None,                          // boxes
    None,                          // mask input
    /*multimask_output=*/ true,
)?;

// pred.mask_logits: [num_masks, mask_side, mask_side] (threshold > 0 for binary)
// pred.iou_pred:    [num_masks] self-estimated quality
println!("{} masks, side {}", pred.num_masks, pred.mask_side);
# anyhow::Ok(())
```

Key exports (see `src/lib.rs`): [`Sam`] / [`MaskPrediction`], the graph builder [`build_sam_encoder_graph`], and the staged pieces [`preprocess_image`], [`prompt_encoder_forward`], [`two_way_transformer_forward`], [`mask_decoder_forward`]. Config constants and `SamConfig` / `SamEncoderConfig` / `SamDecoderConfig` live in `src/config.rs`.

## How it fits

| Crate | Relationship |
|---|---|
| [rlx-sam-ir](../rlx-sam-ir) | shared mask-decoder IR (two-way transformer, hyper matmul, ReLU MLP) |
| [rlx-sam2](../rlx-sam2) | SAM 2 (Hiera) — reuses this crate's compile profiles |
| [rlx-sam3](../rlx-sam3) | SAM 3 (segment with concepts) — depends on this crate |

## Tests

```bash
cargo test -p rlx-sam                       # synthetic-weight encoder graph build + preprocess shapes
```

Real-weight numerical parity vs candle's `ImageEncoderViT::forward()` lives in `tests/sam_parity.rs` (max |Δ| ≈ 7e-6 on the 1×256×64×64 image embeddings for ViT-B at 1024×1024); the `RLX_SAM_DEBUG_*` env vars there bisect the encoder.
