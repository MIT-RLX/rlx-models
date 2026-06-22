# rlx-florence2

Native [RLX](https://github.com/MIT-RLX/rlx) implementation of Microsoft
[Florence-2](https://huggingface.co/microsoft/Florence-2-large) — a
vision-language model that pairs a **DaViT** (Dual-Attention Vision Transformer)
backbone with a **BART** encoder-decoder, driven by task prompts
(`<CAPTION>`, `<OD>`, `<OCR>`, `<REFERRING_EXPRESSION_SEGMENTATION>`, …).

Runs on every RLX backend (CPU / Metal / MLX / CUDA / ROCm) from one IR graph —
no per-backend code.

## Pipeline

```
image ──▶ CLIP preprocess (PIL bicubic) ──▶ DaViT vision tower ──▶ 577 visual tokens
                                                                        │
task prompt ──▶ BART tokenizer ──▶ text embeds ──────────────┐         │
                                                              ▼         ▼
                                          inputs_embeds = [image ‖ text] (BART encoder)
                                                              │
                                                              ▼
                       BART decoder (cross-attention) ──▶ greedy / beam search
                                                              │
                                                              ▼
                          task post-processor ──▶ caption / boxes / quads / polygons
```

## Use

```rust
use rlx_florence2::Florence2Session;
use rlx_runtime::Device;

let mut sess = Florence2Session::from_dir("Florence-2-large".as_ref(), Device::Cpu)?;
let result = sess.run(&rgb, height, width, "<CAPTION>", 64)?;
println!("{result:?}");
```

CLI:

```bash
just fetch-florence2
just florence2-demo                       # caption the bundled sample
cargo run -p rlx-florence2 --release -- \
    --weights Florence-2-large --image photo.jpg --task '<OD>' --device metal
```

## Parity

Verified against HuggingFace `transformers` (script: `scripts/florence2_hf_parity.py`,
reference venv `.venv-florence2`):

| Stage | Metric vs HF |
|-------|--------------|
| CLIP preprocessing (bicubic) | cosine 1.000000, max abs 0.0 |
| DaViT vision features (577×1024) | cosine 1.000000 |
| BART encoder hidden | cosine 1.000000 |
| Decoder step-0 logits | cosine 1.000000, top-1 match |
| Greedy generation | exact token match |
| Beam search (num_beams=3) | exact token match |
| OD post-processor (boxes + labels) | exact |

All of the above are bit-exact on **CPU, Metal, and MLX** (vision features +
decoder logits cosine 1.000000, top-1 token match).

```bash
just florence2-ref-venv      # one-time: python3.11 + torch + transformers==4.44.2
just test-florence2-parity   # caption + OD staged + e2e parity (CPU)
just test-florence2-backends # CPU vs Metal/MLX on the real checkpoint
```

### Backend status

| Backend | Status |
|---------|--------|
| **CPU** | ✅ 100% parity, end-to-end |
| **Metal** | ✅ vision + logits match CPU (cos 1.0) |
| **MLX** | ✅ vision + logits match CPU (cos 1.0) |
| **CUDA / ROCm** | Code-complete; validated on the rig (`../rlx/rig.sh`) |

Bringing up Metal/MLX surfaced two genuine framework backend bugs (now fixed
upstream in `../rlx`), both exercised by DaViT's grouped channel attention at
the deep stage (1024 ch, 32 groups, 2304 tokens):

- **rlx-metal**: the tiled `transpose_swap12_batched_trail_f32` kernel (used for
  a `[B,g,N,Cg]→[B,N,g,Cg]` head-merge when both swapped dims ≥ 32) wrote to a
  transposed destination index — `rr`/`cc` were swapped in the store.
- **rlx-mlx**: `mlx_align_concat_inputs` (a padded-decode seq-alignment helper)
  narrowed inputs along the *concat* axis, truncating a genuine
  `[1,1,C] ‖ [1,N,C]` pooling concat to 2 tokens. It now never aligns the axis
  being concatenated.

Every op is regression-tested in isolation
(`cargo test -p rlx-florence2 --features apple-silicon --test florence2_op_diag`),
and `tests/florence2_backend_parity.rs` checks the real checkpoint across
backends (set `RLX_FLORENCE2_BISECT=metal` for a per-stage/per-depth bisection).

## Architecture notes

- **DaViT**: 4 stages (depths 1/1/9/1, dims 256/512/1024/2048), each a ConvEmbed
  stride-downsample plus alternating `SpatialBlock` (depthwise conv + window
  attention, window 12) and `ChannelBlock` (depthwise conv + grouped channel
  attention). Real in-graph `Conv` (strided + depthwise) on all backends.
- **Vision projection**: learned 2D position + cosine temporal embeddings,
  `spatial_avg_pool ‖ temporal_avg_pool` → `image_projection` → LayerNorm.
- **BART**: post-norm, learned positions (offset 2), `layernorm_embedding`,
  tied LM head + `final_logits_bias`.

Weight keys match the published checkpoint
(`vision_tower.*`, `language_model.model.*`, `image_projection`, …).
