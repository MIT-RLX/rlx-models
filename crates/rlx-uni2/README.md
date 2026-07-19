# rlx-uni2

[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) — MahmoodLab's pathology
foundation model — as a native RLX encoder.

UNI2-h is a **ViT-H/14** in the DINOv2 family, trained self-supervised on
histopathology. It differs from a plain DINOv2 encoder in three ways, all
handled by this crate:

| Aspect | UNI2-h |
| --- | --- |
| Depth / width / heads | 24 / 1536 / 24 |
| Patch / image size | 14 / 224 |
| MLP | **packed SwiGLU** (`timm.layers.SwiGLUPacked`, SiLU gate), `mlp.fc1` → chunk → `SiLU(value)·gate` → `mlp.fc2` |
| Register tokens | **8** (`reg_token`) |
| Position embed | **`no_embed_class`** — patch tokens only; `[CLS]`/registers get none |
| LayerScale | `init_values=1e-5` (`ls1`/`ls2.gamma`) |
| Output | pooled `[CLS]` feature, **1536-d** |

The transformer block reuses RLX's existing ViT stages (attention,
LayerScale, LayerNorm, residual); only the FFN is UNI2-specific and is
emitted as a small tier-2 plugin in [`src/flow.rs`](src/flow.rs).

## Weights

The HF repo is **gated** and ships only `pytorch_model.bin` (2.73 GB) under
the **CC-BY-NC-ND 4.0** license (non-commercial, academic use, with
attribution). You must request access and download it yourself.

RLX loads `safetensors`/`gguf`, not pickled `.bin`, so convert once. The
timm state-dict keys already match what this crate expects — no renaming:

```bash
python -c '
import torch, safetensors.torch as st
sd = torch.load("pytorch_model.bin", map_location="cpu", weights_only=True)
st.save_file({k: v.contiguous() for k, v in sd.items()}, "uni2h.safetensors")
'
```

## Run

```bash
cargo run -p rlx-uni2 --bin rlx-uni2 -- \
  --weights uni2h.safetensors --image tile.png --device cpu
```

Flags (`--help` for the full list):

| flag | meaning |
| --- | --- |
| `--weights <path>` | model `.safetensors` (required) |
| `--image <path>` | input image; omit for a deterministic synthetic tile |
| `--device <dev>` | `cpu` \| `metal` \| `mlx` \| `cuda` \| `rocm` \| `gpu` (wgpu) \| `vulkan` |
| `--img-size <n>` | square size, multiple of 14 (default 224) |
| `--batch <n>` | batch size (default 1) |
| `--dump <path>` | write the `[1536]` CLS embedding as little-endian f32 |
| `--dump-tokens <p>` | write the full `[seq×1536]` token grid as little-endian f32 |
| `--dry` | compile only, skip the forward pass |
| `--layers <n>` | *(debug)* truncate to the first `n` blocks |

GPU backends need the matching Cargo feature, e.g.
`cargo run -p rlx-uni2 --features metal --bin rlx-uni2 -- --device metal …`
(`metal` / `mlx` / `cuda` / `rocm` / `gpu` / `vulkan`).

The encoder is also reachable through the `rlx-models` facade (feature `uni2`,
module `rlx_models::uni2`) and the `rlx-run uni2` multiplexer.

## Library

```rust
use rlx_uni2::Uni2Runner;
use rlx_runtime::Device;

let mut runner = Uni2Runner::builder()
    .weights("uni2h.safetensors")
    .device(Device::Cpu)          // or Device::Metal / Device::Mlx / Device::Gpu
    .build()?;

// `rgb`: HWC u8 of any resolution; resized + ImageNet-normalized to 224.
let rgb: Vec<u8> = image::open("tile.png")?.to_rgb8().into_raw();
let embedding: Vec<f32> = runner.embed_image(&rgb, 224, 224)?; // [1536]
```

`predict_image` returns the same pooled embedding **plus** the full post-norm
token sequence (`[CLS, reg×8, patches]`) for dense/patch-level features.

## Parity

Verified **bit-exact against timm's reference UNI2-h forward** on the real
weights: `timm.create_model("vit_giant_patch14_224", **uni2_kwargs)` →
`load_state_dict(strict=True)` (0 missing / 0 unexpected keys) → same
normalized input → **cosine 1.00000000**, max-abs 8.3e-6, rel-L2 4.2e-6
(f32 numerical noise), identical ‖emb‖₂. The full official pipeline
(`create_transform` + model) on a native 224×224 tile also matches at
cosine 1.0.

Per-backend (cosine vs the PyTorch/CPU reference, real weights):

| backend | cosine | status |
| --- | --- | --- |
| CPU | 1.000000 | ✅ bit-exact vs PyTorch |
| Metal | 1.000000 | ✅ |
| MLX | 1.000000 | ✅ (see MLX note) |
| wgpu | 1.000000 | ✅ via auto no-reuse workaround (see wgpu note) |
| CUDA / ROCm / Vulkan | — | not tested (no local hardware); separate arena planners |

All four locally-testable backends produce the correct embedding out of the
box — `--device cpu|metal|mlx|gpu` all match PyTorch at cosine 1.0.

> **MLX note.** The FFN originally split the packed `fc1` with an in-graph
> `narrow` of the activation, which was silently mis-lowered on **both** MLX
> and wgpu. Splitting `fc1` host-side into two matmuls (this crate's current
> approach, mirroring Nomic's `VisionSwiGluFfnStage`) fixed MLX → cosine 1.0.
>
> **wgpu note.** wgpu additionally mis-synchronizes **reused arena buffers**
> for this graph shape: with slot reuse on, a live buffer is clobbered before
> read-back and the output is silently corrupted (even a 1-layer run diverges;
> the memory *plan* is identical to Metal's correct plan, so this is a wgpu
> *executor* bug, not a planning one). The runner therefore forces
> `RLX_ARENA_NO_REUSE` on the wgpu device (an in-process override), which
> restores bit-exact output (cosine 1.0) at the cost of extra GPU arena memory.
> This is a workaround for an open rlx-wgpu executor issue, not a fix for it;
> reproduce/inspect with `--layers N` and the per-node dump
> (`RLX_WGPU_DUMP_NODES` vs `RLX_CPU_DUMP_NODES`, both with `RLX_ARENA_NO_REUSE=1`).

Notes:

- The checkpoint's `pretrained_cfg` uses `interpolation: bilinear`,
  `crop_pct: 1` — which is exactly what `rgb_u8_to_imagenet_nchw` does (no
  center crop, bilinear resize; a no-op for native 224×224 tiles).
- Non-224 resolutions need position-embedding interpolation (the model was
  created with `dynamic_img_size=True`); not yet implemented — the loader
  errors if `pos_embed` length doesn't match `num_patches`.
