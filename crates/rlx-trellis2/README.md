# rlx-trellis2

Native RLX port of [Microsoft TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) (image → textured 3D mesh).

## Pipeline

```text
image ─▶ preprocess (alpha crop) ─▶ DINOv3 ViT-L/16
       ├─ sparse-structure DiT + dense conv3d decoder → active voxels
       ├─ shape SLat DiT (+ optional 512→1024 cascade) + FlexiDualGrid VAE → mesh
       └─ texture SLat DiT + SparseUnet VAE → PBR voxels (optional)
```

Host modules (`dit_host`, sparse VAEs, dual-grid mesh) are parity-checked against PyTorch dumps when env fixtures are set. End-to-end orchestration lives in [`Trellis2Runner`](src/pipeline.rs).

**DiT backends:** on `--device metal` / `mlx` / `cuda` the generative DiTs compile through [`dit_flow`](src/dit_flow.rs) (AdaLN + backend SDPA + GatedResidual). Pass `--eager-dit` to force the host reference. Structure DiT uses fixed `n_pos = 16³`; shape/tex SLat DiTs pad to a token bucket. CPU still uses the host path.

Prefer `--pipeline-type 512 --shape-only --device metal` for practical full Euler runs.

## Checkpoints

| Component | Source |
|---|---|
| Flow DiTs + shape/tex VAEs | `microsoft/TRELLIS.2-4B` (`pipeline.json` + `ckpts/*`) |
| Sparse-structure decoder | `microsoft/TRELLIS-image-large` (`ckpts/ss_dec_conv3d_16l8_fp16`) |
| Image conditioner | `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated); mirror `camenduru/dinov3-vitl16-pretrain-lvd1689m` |

BiRefNet / RMBG-2.0 rembg is **not** bundled. Prefer an RGBA cutout, or pass `--no-rembg` to treat RGB as opaque foreground.

Texture needs `ckpts/tex_dec_*` and `ckpts/slat_flow_imgshape2tex_*_512` from TRELLIS.2-4B.
Export formats:
- **PLY** — vertex RGB
- **GLB** — UV atlas + `baseColorTexture` + `metallicRoughnessTexture` (packed per-vertex texels from PBR voxels)
- **OBJ** — geometry only

This is not the official `o_voxel` remesh/UV-unwrap bake; atlas UVs are one texel per mesh vertex.

## CLI

```bash
# Inventory what is on disk (omit --shape-only to require tex ckpts)
cargo run -p rlx-trellis2 --release -- \
  --model-dir ~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/<rev> \
  --pipeline-type 512 --dry

# Shape-only 512³ on Metal (compiled DiT + DINOv3)
cargo run -p rlx-trellis2 --release --features apple-silicon -- \
  --model-dir /path/to/TRELLIS.2-4B \
  --ss-decoder-dir /path/to/TRELLIS-image-large/ckpts \
  --dinov3-weights /path/to/dinov3-vitl16.safetensors \
  --dinov3-config /path/to/config.json \
  --image cutout.png --no-rembg \
  --pipeline-type 512 --shape-only --device metal --steps 12 \
  --output /tmp/trellis.obj

# Force host DiT (parity / debugging)
cargo run -p rlx-trellis2 --release --features apple-silicon -- \
  … --device metal --eager-dit --steps 1

# Textured 512³ → vertex-colored PLY/GLB (host DiT is slow; use --steps N for a short run)
cargo run -p rlx-trellis2 --release -- \
  --model-dir /path/to/TRELLIS.2-4B \
  --ss-decoder-dir /path/to/TRELLIS-image-large/ckpts \
  --dinov3-weights /path/to/dinov3-vitl16.safetensors \
  --dinov3-config /path/to/config.json \
  --image cutout.png --no-rembg \
  --pipeline-type 512 \
  --output /tmp/trellis.glb
```

Or via just:

```bash
just trellis2 -- --model-dir … --dry
```

## Library

```rust
use rlx_trellis2::{PipelineType, PreprocessOptions, Trellis2Input, Trellis2Runner};

let mut runner = Trellis2Runner::builder()
    .model_dir("/path/to/TRELLIS.2-4B")
    .dinov3_weights("/path/to/dinov3.safetensors")
    .pipeline_type(PipelineType::Res512)
    .shape_only(true)
    .build()?;

let img = image::open("cutout.png")?;
let out = runner.generate(Trellis2Input {
    image: &img,
    seed: 42,
    preprocess: PreprocessOptions { allow_rgb_fallback: true, ..Default::default() },
})?;
std::fs::write("out.obj", out.mesh.to_obj())?;
// With texture: out.mesh.to_ply() or out.mesh.to_glb() for vertex colors
```

## Tests

Component parity tests are env-gated:

| Test | Env |
|---|---|
| `ssflow_parity` | `RLX_TRELLIS2_SSFLOW_CKPT`, `RLX_TRELLIS2_SSFLOW_REF` |
| `ssdec_parity` | `RLX_TRELLIS2_SSDEC_CKPT`, `RLX_TRELLIS2_SSDEC_REF` |
| `shape_dec_parity` | `RLX_TRELLIS2_SHAPEDEC_CKPT`, `RLX_TRELLIS2_SHAPEDEC_REF` |
| `sparse_ops_parity` | `RLX_TRELLIS2_SUBM_REF` |
| `structure_e2e_parity` | `RLX_TRELLIS2_SSFLOW_CKPT`, `SSDEC_CKPT`, `SS_INJECT`, `SS_OCC_REF`; optional `SS_SAMPLE` (prefer float32 Python dump), `DIT_DEVICE=metal`, DINO envs |

Structure inject parity locks Python `cond`/`noise` and compares occupancy (and optional sample cosine). Use a Python DiT with `convert_to(float32)` for the sample/occ refs — default bf16 torso dumps sit ~0.991 vs host f32 even when the sampler is correct. Trellis image cond uses non-affine DINOv3 final LN (`final_layer_norm_affine = false`).

```bash
cargo test -p rlx-trellis2
```
