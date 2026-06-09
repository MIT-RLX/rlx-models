# SAM v1 (Segment Anything) for rlx-models

Meta's image-segmentation model, ported from
`candle-transformers::models::segment_anything`. Bit-equivalent
outputs vs candle on CPU, MLX, and Metal; runs on weights from
`lmz/candle-sam` directly.

## Layout

```
sam/
├── mod.rs              Public API + integration tests
├── config.rs           SamConfig / SamEncoderConfig / SamDecoderConfig +
│                       constants (1024×1024, patch 16, ImageNet stats)
├── preprocess.rs       Host-side: resize→1024 + pad + patchify + pos_embed
├── image_encoder.rs    IR-graph ViT-B/L/H encoder (windowed + global
│                       attention, decomposed rel_pos, neck on host)
├── prompt_encoder.rs   Host-side prompt encoder (points, boxes, masks)
├── transformer.rs      Host-side two-way transformer (BLAS-backed)
├── mask_decoder.rs     Host-side decoder: two-way xformer →
│                       ConvTranspose2d upscale → hypernet MLPs → masks
├── sam.rs              `Sam` orchestrator (encoder + prompt + decoder)
└── README.md           This file
```

## Quick start

```rust
use rlx_models::sam::{Device, Sam, SamConfig};

// Load + JIT-compile the encoder once. `Device::Cpu | Metal | Mlx` —
// see "Backends" below.
let mut sam = Sam::from_safetensors_on(
    "sam_vit_b_01ec64.safetensors",
    SamConfig::vit_b(),
    Device::Cpu,
)?;

// Foreground point prompt at (512, 512) on a synthetic image.
let rgb: Vec<u8> = /* H × W × 3 bytes */;
let (pred, _resized) = sam.forward(
    &rgb, h, w,
    Some((&[512.0, 512.0], &[1.0])), // (coords [N·2], labels [N])
    None,                            // no box prompt
    None,                            // no mask prompt
    /*multimask=*/ true,             // 3 candidate masks
)?;

// `pred.mask_logits`: [num_masks, 256, 256] f32 (threshold > 0 ⇒ binary mask)
// `pred.iou_pred`:    [num_masks]            self-estimated quality
```

The encoder lives in the IR graph (the 99% compute hotspot — backend
matters). Prompt encoder + mask decoder are pure host-side Rust with
BLAS-backed matmuls (`rlx_cpu::blas::sgemm_auto`).

## Backends

| Device | feature | When | Speed (SAM ViT-B @ 1024², this machine) |
|---|---|---|---:|
| `Cpu` | default | reproducibility, fastest CPU | 31 s encoder |
| `Metal` | `metal` | fastest single forward | **9.85 s** encoder |
| `Mlx` | `mlx` | bit-equivalent to CPU on GPU | ~18 s encoder |

```bash
cargo add rlx-models --features metal,mlx  # add the backends you want
```

## Numerical parity vs candle

| Pair | image_emb cosine | mask logits cosine | binary mask agreement |
|---|---:|---:|---:|
| **CPU ↔ candle CPU** | 1.000000000 | 1.000000000 | **99.993%** |
| **Metal ↔ CPU** | 1.000000000 | 1.000000000 | **100.000%** |
| **MLX ↔ CPU** | 1.000000000 | 1.000000000 | **100.000%** |

f32 per-element mismatch is ~10⁻⁵–10⁻⁶ across all backends (BLAS
reduction-order noise). For tightest reproducibility against candle on
CPU, enable `parity-gemm` — routes sgemm through the same `gemm` crate
candle uses (no AMX, slower).

```bash
cargo test -p rlx-models --features parity-candle,parity-gemm --release sam_
```

## Running the parity test

Download the same `lmz/candle-sam` checkpoint candle uses:

```bash
hf download lmz/candle-sam sam_vit_b_01ec64.safetensors --local-dir /tmp/rlx_sam

RLX_SAM_WEIGHTS=/tmp/rlx_sam/sam_vit_b_01ec64.safetensors \
RLX_SAM_DEVICE={cpu|metal|mlx} \
  cargo test -p rlx-models --features parity-candle,metal,mlx --release \
  sam_ -- --nocapture
```

Debug bisect options (encoder-only parity test):

```bash
RLX_SAM_DEBUG_DEPTH=N         # only run the first N blocks
RLX_SAM_DEBUG_NO_RELPOS=1     # disable decomposed relative position bias
RLX_SAM_DEBUG_FORCE_GLOBAL=1  # force every block to use global attention
RLX_SAM_DEBUG_ZERO_RELH=1     # zero-out the rel_h tensor
RLX_SAM_DEBUG_ZERO_RELW=1     # zero-out the rel_w tensor
```

## Architecture notes

### Image encoder (`image_encoder.rs`)

Pre-norm ViT-B/L/H with **windowed + global** attention and
**decomposed relative positional bias**. Implemented as a single IR
graph passed to `rlx-runtime::Session`. Key shapes:

- Input: `[1, 4096, 768]` (BHWC, 64×64 patches at 16-pixel stride, 768-dim)
- 12 attention blocks, 4 of which use global attention (idx 2/5/8/11),
  the other 8 use 14×14 windowed attention.
- Output: `[1, 256, 64, 64]` NCHW after the neck

The **neck** (Conv 1×1 → LN2d → Conv 3×3 padding=1 → LN2d) runs on
the host via `apply_neck_host` — Conv 3×3 with padding isn't in
`rlx-ir` today and the neck is < 1% of compute.

**Windowed attention** is expressed in pure IR via pad-via-concat-zeros
+ reshape + transpose (no `Pad` op needed). The decomposed rel_pos add
uses unrolled-per-h_q matmuls + concat-tile (workaround for an MSL
broadcast kernel bug — see `mod.rs` comments).

### Prompt encoder + mask decoder

Host-side Rust. Mirrors candle's `prompt_encoder.rs` /
`mask_decoder.rs` / `transformer.rs` exactly. BLAS-backed matmuls via
`rlx_cpu::blas::sgemm_auto` — the decoder is ~1% of total compute, so
keeping it host-side avoids growing the IR surface with
ConvTranspose2d, 4-D layer norm, etc.

## Weights & key naming

Compatible with `lmz/candle-sam`'s safetensors layout exactly
(`image_encoder.blocks.0.attn.qkv.weight`, `prompt_encoder.pe_layer.*`,
`mask_decoder.transformer.layers.0.self_attn.q_proj.weight`, …).
No remapping needed — `WeightMap::from_file` loads directly.

`rlx-models::WeightMap` was hardened during the SAM port to handle
safetensors with **non-4-byte-aligned f32 tensors** (SAM ViT-B has
them); previously it panicked in `bytemuck::cast_slice`.

## Bugs found + fixed during the SAM port

| Crate | Bug | Fix |
|---|---|---|
| `rlx-models::weight_map` | Panic on non-aligned f32 in safetensors | Manual LE decode when unaligned |
| `rlx-cpu` | `Activation::GeluApprox` aliased to erf-GELU in forward | Added real tanh-approx kernel |
| `rlx-cpu` | f32 3-D batched matmul used shared-RHS flatten | Added `Thunk::BatchedSgemm` |
| `rlx-cpu` | `BiasAdd` misroute for mid-shape singleton broadcasts | `is_trailing_bias_broadcast` gate |
| `rlx-metal` | f32 3-D batched matmul (same as CPU) | Metal `Thunk::BatchedSgemm` |
| `rlx-metal` | `Op::Binary` had no shape-aware broadcast | `Thunk::BinaryBroadcast` + MSL kernel |
| `rlx-metal` | `Op::Concat` only supported last-axis (debug_assert was a no-op in release) | `inner` stride + host fallback for mid-axis |
| `rlx-metal` | `MPSMatrix` cache aliased Buffer wrappers across `Sam` instances → NaN on second backend | `invalidate_caches()` on every `MetalBackend::compile` |
| `rlx-runtime` | MLX defaulted to `Lazy` mode → re-trace on every `run()` | Default to `Compiled` + `warm_compile` at compile time (6× faster) |

## What's NOT supported (yet)

- `sam_vit_l` / `sam_vit_h` weights load and the encoder builder
  handles them, but only `vit_b` is regularly tested.
- TinyViT backbone (MobileSAM) — not ported.
- Variable input resolution (we hardcode 1024×1024); supporting
  arbitrary input sizes would need bicubic interpolation of `pos_embed`
  + dynamic shapes in the IR graph.
- Batched encoder (graph hardcodes `batch=1`). DINOv2 already takes
  `batch` as a build-time param; SAM needs ~50 LoC to thread it through
  the encoder, preprocess, and neck.
- True bit-exact CPU↔Metal parity (would need a custom MSL sgemm
  matching NEON's reduction order — BLAS-noise floor at ~10⁻⁵ is the
  practical limit otherwise).

## License

GPL-3.0-only (matches the workspace).
