# rlx-mllama — Llama-3.2-Vision (mllama) for RLX

Native RLX port of Meta's **Llama-3.2-Vision** (`mllama`). Unlike the embed-splice
VLMs (Pixtral, Qwen2.5-VL), mllama fuses vision by **cross-attention**: a ViT
vision tower produces per-tile features that a subset of the Llama-3.2 decoder
layers attend to as K/V (with tanh output gates); the `<|image|>` token stays a
normal text token.

## Layout

| module | role |
|---|---|
| `config.rs` | nested vision + text config (HF `config.json`) |
| `preprocess.rs` | host image pipeline: aspect-ratio tiling → resize/normalize → patch-embed → tile/position embeddings (ports `image_processing_mllama.py`) |
| `vision.rs` | ViT tower (32 local + 8 gated-global layers) + multi-modal projector as a native `ModelFlow` → `cross_states [1, tiles·1025, 4096]` |
| `cross_attn.rs` | the cross-attention decoder layer as a `FlowStage` (GQA repeat_kv, q/k RMSNorm, tanh attn/mlp gates, no RoPE; vision K/V from the `cross_states` graph input) |
| `runner.rs` | ties it together; loads the sharded checkpoint, splits vision/text weights, drives generation |
| `cli.rs` | `--weights <dir> --image <path> --prompt <text>` |

The text decoder reuses `rlx-llama32` unchanged — cross-attention layers are
inserted purely via the existing `Llama32Flow::layer()` hook (cross indices →
`cross_attn_stage`, others → `ctx.default_stage()`), and `cross_states` is a
shared graph input declared via `patch_flow`.

**Equivalence simplification:** the tower runs with the image's *exact* tile
count and no /8 patch padding. HF masks both padded tiles and alignment-pad
patches, so an exact run is numerically equivalent while dropping all masks.

**v1 decode:** full-sequence (no KV cache) — the text graph compiles once at
`max_len` and each step re-runs the padded sequence, reading logits at the true
last position (self-attention is causal, so trailing pad never affects it).
Correct but O(L²); a KV-cache version (self layers cached, cross K/V cached once)
is future work.

## Status

- Vision-encoder graph and cross-attention text integration both **compile and
  run on CPU** (`cargo test -p rlx-mllama` — `vision_smoke`, `text_smoke`).
- Numerical parity vs HF is pending a real-checkpoint run (see below).

## Validating against a real checkpoint

Llama-3.2-11B-Vision is gated + large (bf16 ≈ 22 GB; RLX loads f32 ≈ 44 GB).
Lead with **vision-encoder parity** (only ≈ 2 GB), then attempt the full model
where RAM allows.

```bash
# 1. HF reference (needs the gated checkpoint + HF_TOKEN):
python3 scripts/mllama_hf_dump.py --ckpt <ckpt> --image <img> --out out/mllama_ref

# 2. RLX vision output:
rlx-mllama --weights <ckpt> --image <img> --device cpu --dump-vision out/mllama_rlx_vision

# 3. Compare vision (cosine per tile):
python3 scripts/mllama_vision_compare.py --ref out/mllama_ref_cross_states.npy --rlx out/mllama_rlx_vision

# 4. Full generation:
rlx-mllama --weights <ckpt> --image <img> --prompt "Describe this image." --device cpu --max-tokens 32
```
