# rlx-dinov3

Native [RLX](https://github.com/MIT-RLX) port of **Meta DINOv3** — the
self-supervised vision transformer with **2-D axial RoPE**, register tokens,
LayerScale, and an optional gated (GeGLU) MLP. The same graph runs on every RLX
backend (CPU / Metal / MLX / wgpu / CUDA / Vulkan / ROCm).

The architecture matches HuggingFace `transformers.models.dinov3_vit`
(`DINOv3ViTModel`) exactly, and the weight keys are the HF safetensors keys
verbatim — no remapping or offline repacking.

## What makes DINOv3 different from DINOv2

| | DINOv2 / UNI2 | **DINOv3** |
|---|---|---|
| position | learned `pos_embed` added at embed | **none** — 2-D axial RoPE inside attention |
| attention | fused `qkv` | separate `q/k/v/o` (no key bias) |
| MLP | GELU / SwiGLU-packed | GELU **or** gated GeGLU (biased) |
| patch size | 14 | 16 |
| extras | — | LayerScale, register tokens |

RoPE is applied **only to patch tokens** (the CLS + register prefix is skipped).
This port precomputes the `[seq, head_dim/2]` cos/sin tables on the host with
**identity rows for the prefix**, so the stock NeoX `rope` op runs over the whole
sequence with no in-graph slicing (an MLX/wgpu hazard) — see
[`src/rope.rs`](src/rope.rs) for the exact HF-equivalence derivation. The whole
port lives in this crate: attention and MLP are emitted as `rlx-flow` plugins, so
no changes to the shared framework were needed.

## Usage

```rust
use rlx_dinov3::{DinoV3Config, DinoV3Runner};
use rlx_runtime::Device;

let mut runner = DinoV3Runner::builder()
    .weights("dinov3-vitb16.safetensors")
    .config(DinoV3Config::vit_b16(224))    // or ::from_file("config.json")
    .device(Device::Cpu)                   // Cpu / Metal / Mlx / Gpu (wgpu) / Cuda …
    .build()?;

// rgb: HWC u8, any resolution — resized + ImageNet-normalized internally.
let embedding: Vec<f32> = runner.embed_image(&rgb, height, width)?;   // pooled [CLS], len = hidden_size
```

- [`DinoV3Runner::predict_image`] returns both the pooled `[CLS]` embedding and
  the full post-norm token grid ([`DinoV3Output`]).
- [`DinoV3Runner::forward_nchw`] takes an already-normalized `NCHW` tensor — the
  rigorous entry point for matching a reference `pixel_values` exactly.

### CLI

Also available as `rlx-run dinov3 …` when the `dinov3` feature is enabled.

```bash
cargo run -p rlx-dinov3 --release -- \
    --weights dinov3-vitb16.safetensors --variant vitb16 \
    --image cat.jpg --device metal --dump emb.bin
```

Variant presets: `vit_s16`, `vit_b16` (the default
`facebook/dinov3-vitb16-pretrain-lvd1689m` embedder), `vit_l16`. For the gated /
larger variants (ViT-H+/16, ViT-7B/16) or a non-default register count, pass the
checkpoint's `config.json` via `--config` (CLI) or
[`DinoV3Config::from_file`] (API).

## Cargo features

Backend selection forwards to `rlx-runtime`; the crate code is backend-agnostic.

| feature | backend |
|---|---|
| *(default)* | CPU |
| `metal` | Apple Metal |
| `mlx` | Apple MLX |
| `gpu` | wgpu (portable GPU) |
| `cuda` | NVIDIA CUDA |
| `vulkan` | Vulkan |
| `rocm` | AMD ROCm |
| `all-backends` | all of the above |
| `apple-silicon` | `metal, mlx, gpu` |

## Weights

The pretrained DINOv3 weights are **gated** on the HuggingFace Hub (accept the
license, then `hf auth login`). Use the downloaded safetensors as-is — the keys
already match: `embeddings.{cls_token,register_tokens,mask_token,patch_embeddings.*}`,
`layer.{i}.{norm1,norm2}`, `layer.{i}.attention.{q,k,v,o}_proj`,
`layer.{i}.mlp.{up,down[,gate]}_proj`, `layer.{i}.layer_scale{1,2}.lambda1`,
`norm.*`. Both `.safetensors` and `.gguf` load.

## Parity

Verified against HuggingFace `DINOv3ViTModel` (same weights, same
`pixel_values`) by [`tests/dinov3_parity.rs`]. The reference dump comes from
[`scripts/dump_reference.py`](scripts/dump_reference.py) — a tiny random-weight
fixture, so the architecture can be validated with **no gated download**:

| backend | last_hidden cos | pooled cos | max_abs |
|---|---|---|---|
| CPU  | 1.0000000 | 1.0000000 | 6.7e-8 |
| Metal | 1.0000000 | 1.0000000 | 6.0e-8 |
| MLX  | 1.0000000 | 1.0000000 | 6.7e-8 |
| wgpu | 1.0000000 | 1.0000000 | 6.0e-8 |

All four backends are **bit-exact** vs HF, for both the standard-GELU and
gated-GeGLU MLP. CUDA / Vulkan / ROCm use the same primitive ops (`mm` / `add` /
`rope` / `attention` / `layer_norm` / `gelu` / `mul`) that those backends already
lower, so they are supported by construction (not run on the Apple-Silicon dev
host).

> **Metal note.** Reaching bit-exactness required a backend fix: Apple's MPSGraph
> (and the fused `sgemm` epilogue) mis-execute an **erf-GELU fused onto a
> matmul+bias** — an O(1) divergence, hidden under the trailing LayerNorm.
> tanh-GELU is unaffected, which is why DINOv2 never showed it. `rlx-metal` now
> splits `FusedMatMulBiasAct{Gelu}` into a fused matmul+bias plus a standalone
> (correct) GELU, which also fixes any other erf-GELU FFN (BERT-style) model on
> Metal.

Reproduce:

```bash
python crates/rlx-dinov3/scripts/dump_reference.py --out /tmp/dv3fix
DINOV3_FIXTURES=/tmp/dv3fix DINOV3_DEVICES=cpu,metal,mlx,wgpu \
  cargo test -p rlx-dinov3 --test dinov3_parity --features metal,mlx,gpu -- --nocapture
```

## License

Code is GPL-3.0 (the RLX runtime). The DINOv3 **weights** are Meta's, under the
DINOv3 license (gated) — obtaining and using them is the user's responsibility.
