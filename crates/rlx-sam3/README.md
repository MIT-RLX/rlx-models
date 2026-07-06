# rlx-sam3

Meta's [**SAM 3** — Segment Anything with Concepts](https://github.com/facebookresearch/sam3) for RLX. Targets the base SAM 3 image + video architecture: a DETR-style **detector** (ViT-L vision trunk + text encoder → decoder queries) that turns an image and a text prompt into instance masks and scores. Loads safetensors or GGUF checkpoints; SAM 3.1 multiplex is intentionally out of scope.

## Quick start

```bash
just sam3 --weights sam3_base.safetensors --device cpu --text-tokens 0,1,2,3
# or:
cargo run -p rlx-sam3 --release --bin rlx-sam3 -- \
  --weights sam3_base.safetensors --device cpu

# Metal:
just sam3-metal --weights sam3_base.safetensors --device metal
```

CLI flags: `--weights`, `--device (cpu|metal|mlx|cuda|rocm|gpu|vulkan)`, `--text-tokens A,B,C` (comma-separated token ids; a placeholder `0..32` is used if omitted), `--dry`.

## Public API

```rust
use rlx_sam3::{Sam3, Sam3Config};
use rlx_runtime::Device;

let mut sam = Sam3::from_checkpoint_on(       // safetensors or GGUF
    "sam3_base.safetensors",
    Sam3Config::base(),
    Device::Cpu,
)?;

// rgb: row-major [H*W*3] u8; text_tokens: pre-tokenized concept prompt.
let pred = sam.predict_image_text(&rgb, h, w, &text_tokens)?;
println!(
    "{} instances, mask_shape={:?}",
    pred.num_instances, pred.mask_shape,   // pred.scores[..] = per-instance confidence
);
# anyhow::Ok(())
```

Other exports (`src/lib.rs`): [`Sam3EncodedImage`], the video path [`Sam3VideoState`] / [`Sam3VideoFramePrediction`]; graph builders `build_sam3_detector_encoder_graph` and the [`flow`](src/flow.rs) `Sam3DetectorEncoderFlow` / `Sam3DetectorDecoderFlow`; GGUF loading via [`load_sam3_from_gguf`] / `gguf_has_packed_linears`; and `preprocess_image` / `assemble_patch_tokens`. Config: `Sam3Config` + `Sam3VitConfig` / `Sam3DetectorConfig` / `Sam3TextConfig` / `Sam3TrackerConfig` (base = 1008px input, patch 14, embed 1024, 200 detector queries).

## Example — vision-trunk timing / parity

```bash
cargo run -p rlx-sam3 --release --example vision_timing -- <weights>
```

Times the ViT-L vision trunk per backend and compares against the host reference (max-abs / mean-abs / cosine). Env: `RLX_SAM3_VISION_ITERS`, `RLX_SAM3_VISION_DEVICES`.

## How it fits

Depends on [rlx-sam](../rlx-sam) (compile profiles `sam3_profile_default` / `sam3_profile_near_weights`) and [rlx-gguf](../rlx-gguf) for packed GGUF weights. Sits alongside [rlx-sam2](../rlx-sam2) in the segmentation family; host tensor kernels are shared via `rlx_core::host_kernels`.

## Tests

```bash
cargo test -p rlx-sam3      # base config dims, preprocess weight extraction, patch-token shapes
```
