# rlx-sam2

Meta's [**Segment Anything Model 2 (SAM 2)**](https://github.com/facebookresearch/sam2) for RLX — image **and** video segmentation on a **Hiera** backbone. Mirrors `facebookresearch/sam2` so the published `sam2_hiera_{t,s,b+,l}.{pt,safetensors}` checkpoints load with no weight-key remapping.

## Quick start

```bash
just sam2 --weights sam2_hiera_tiny.safetensors --device cpu --point 512,512
# or:
cargo run -p rlx-sam2 --release --bin rlx-sam2 -- \
  --weights sam2_hiera_tiny.safetensors --device cpu --point 512,512
```

CLI flags: `--weights`, `--device (cpu|metal|mlx|cuda|rocm|gpu|vulkan)`, `--point X,Y`, `--dry`. Variant is chosen with `RLX_SAM2_VARIANT=tiny|small|base_plus|large` (default `tiny`).

## Public API

```rust
use rlx_sam2::{Sam2, Sam2Config};
use rlx_runtime::Device;

let mut sam = Sam2::from_safetensors_on(
    "sam2_hiera_base_plus.safetensors",
    Sam2Config::hiera_base_plus(),   // ::hiera_tiny() / ::hiera_small() / ::hiera_large()
    Device::Cpu,
)?;

// Single image: rgb is row-major [H*W*3] u8, one foreground point.
let pred = sam.predict_image(
    &rgb, h, w,
    Some((&[cx, cy], &[1.0f32])),  // points + labels
    None,                          // boxes
    None,                          // mask input
    /*multimask_output=*/ true,
)?;
println!("{} masks at {}x{}", pred.num_masks, pred.h_out, pred.w_out);

// Video: sam.predict_video_frame(...) threads a Sam2VideoState across frames.
# anyhow::Ok(())
```

The top-level [`Sam2`] wraps all stages; individual components are also public: the Hiera encoder graph ([`build_sam2_image_encoder_graph`]), [`FpnNeck`](src/fpn_neck.rs) (`apply_fpn_neck`), [`prompt_encoder_forward`], [`two_way_transformer_forward`], [`mask_decoder_forward`], plus the video path [`memory_encoder_forward`] / [`memory_attention_forward`]. Outputs: [`Sam2ImagePrediction`], [`Sam2VideoState`].

## Components

| Stage | Modules |
|---|---|
| Hiera image encoder + FPN neck | `image_encoder`, `fpn_neck`, `preprocess` |
| Prompt encoder + two-way transformer + mask decoder | `prompt_encoder`, `transformer`, `mask_decoder` (object-pointer / object-score / high-res mask path) |
| Video memory | `memory_encoder`, `memory_attention` (axial RoPE) |

## How it fits

Shares the two-way-transformer / hyper-matmul / MLP IR with [rlx-sam](../rlx-sam) via [rlx-sam-ir](../rlx-sam-ir), and reuses [rlx-sam](../rlx-sam)'s compile profiles (`sam2_profile_default`, `sam2_profile_near_weights`). See also [rlx-sam3](../rlx-sam3).

## Tests

```bash
cargo test -p rlx-sam2      # every Hiera variant builds; FPN IR-vs-host and memory-encoder parity
```

Synthetic-weight tests exercise the encoder, prompt encoder, decoder, and memory enc/attn end to end for all four Hiera sizes. PyTorch numerical parity is wired in `tests/sam2_parity.rs` behind the `parity-pytorch` feature.
