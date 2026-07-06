# rlx-grounding-dino

[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)
(`IDEA-Research/grounding-dino-base`) on RLX — open-vocabulary object detection:
an image plus a free-text prompt (e.g. `"a cat. a remote control."`) →
bounding boxes, grounded labels, and scores.

## Architecture

A faithful re-implementation of HF `transformers`
`GroundingDinoForObjectDetection`:

- **Swin-B** vision backbone (windowed + shifted-window attention, relative
  position bias, patch merging) → 3 multi-scale feature maps ([`swin`]).
- **BERT** (`bert-base-uncased`) text backbone with the phrase block-diagonal
  self-attention mask, projected to `d_model=256` ([`text_encoder`],
  [`tokenizer`]).
- **Neck** — `input_proj_vision` (1×1 conv + GroupNorm ×3, 3×3 stride-2 conv +
  GroupNorm ×1) + sine position embeddings → 4 levels ([`neck`]).
- **Feature enhancer** (6 layers): bidirectional vision↔text fusion + text
  self-attention + vision multi-scale **deformable** self-attention
  ([`enhancer`], [`deform_attn`]).
- **Language-guided query selection** (two-stage, top-900) ([`query_select`]).
- **Cross-modality decoder** (6 layers): query self-attention + text
  cross-attention + image deformable cross-attention + iterative box refinement
  ([`decoder`]).
- **Heads**: parameter-free contrastive class head + shared box MLP, with full
  postprocessing (cxcywh→xyxy, rescale, token grounding) ([`postprocess`]).

See `ARCHITECTURE.md` for the authoritative weight-name map.

## Backends

The pipeline is **hybrid**: the heavy linear algebra — the Swin backbone, BERT
text encoder, feature enhancer, and decoder — runs **on-device as compiled IR
graphs** (CPU/Metal/MLX/CUDA/WGPU/Vulkan), while the scalar glue (window
geometry, query selection, postprocess) and the fused deformable-attention op
run on the host. A pure CPU-native path is retained as the byte-for-byte parity
anchor (`SwinBackbone::forward_native`, etc.). All backends are validated to
produce identical detections.

> **Performance.** Measured end-to-end on a 320×240 image (release, M-series),
> all backends bit-identical:
>
> | CPU | MLX | Metal | WGPU |
> |-----|-----|-------|------|
> | 4.9s | **2.9s** | 4.0s | 5.5s |
>
> MLX is the fastest (~1.7× CPU). The remaining host cost is the multi-scale
> deformable attention over all image tokens (BLAS-backed; bounded). Set
> `RLX_GDINO_PROFILE=1` for a per-stage breakdown (swin / neck / text / enhancer
> / decoder, plus per-Swin-stage timings). First-run Metal/WGPU includes one-time
> shader-pipeline compilation that amortizes over repeated inference.

## Usage

```bash
# Downloads the checkpoint into the HF cache on first run (hf-cache feature).
cargo run -p rlx-grounding-dino --release -- \
    --image cat.jpg --text "a cat. a remote control." --device cpu

# Explicit checkpoint + config + tokenizer:
cargo run -p rlx-grounding-dino --release -- \
    --weights model.safetensors --config config.json \
    --image cat.jpg --text "a cat." --box-threshold 0.3 --text-threshold 0.25
```

GPU feature builds (host path, device reported): `--features apple-silicon`
(Metal+MLX+wgpu), `--features cuda`, or `--features all-backends`.

## Library

```rust
use rlx_grounding_dino::{GroundingDino, GroundingDinoConfig, text_tokens_from_ids};

let model = GroundingDino::from_checkpoint(path, GroundingDinoConfig::base())?;
let tokens = text_tokens_from_ids(input_ids); // from a BERT wordpiece tokenizer
let dets = model.detect(&rgb, h, w, &tokens, 0.3, 0.25);
for d in &dets {
    println!("{:?} {:.3} {:?}", d.bbox, d.score, d.token_indices);
}
```

## Parity testing

`reference_backend.py` dumps HF `transformers` detections for an image + prompt;
the ignored `tests/e2e_real.rs` probe compares this crate's output against it
(box IoU + count). See the test header for the `RLX_GDINO_*` environment setup.

```bash
cargo test -p rlx-grounding-dino           # unit + synthetic full-pipeline tests
cargo test -p rlx-grounding-dino -- --ignored   # real-weights parity (needs setup)
```

## Status

- ✅ Full CPU-native reference pipeline (the byte-for-byte parity anchor), with
  heavy compute additionally running on-device as compiled IR graphs.
- ✅ CLI + binary, HF download, tokenization, postprocessing, parity harness.
- ✅ IR (on-device) building blocks, parity-tested against the host reference:
  `linear`, masked `mha`, relu-FFN ([`ir`]), and the **BERT text backbone as a
  compiled graph** ([`text_encoder_ir`]). These compose only ops present on every
  backend, so the graph compiles + runs on CPU/Metal/MLX/CUDA/WGPU/Vulkan with no
  host fallback. The enhancer text self-attention + FFN and the decoder
  self/text-cross attention + FFN are direct compositions of these validated
  blocks.
- ✅ **Fused deformable-attention op** (`Op::Custom("gdino.ms_deform_attn")`,
  [`deform_op`] / [`deform_attn_ir`]). Runs the whole DETR-style module in one
  kernel — memory-bounded, no `[nq·heads·levels·points·4, head_dim]` blow-up.
  Dispatched through the device; native-vs-IR parity test on `Cpu`. The CPU
  kernel is self-contained here (no engine GPU symbols on the default path); the
  matching engine host-delegate kernels are shipped in `../rlx` for **CPU, Metal
  and MLX** (`rlx_cpu` / `rlx_metal` / `rlx_mlx` `ms_deform_attn`), enabled via
  the `engine-apple` feature + the `../rlx` patch.
- ✅ **WGPU**: engine `Step::MsDeformAttnHost` (lowering + host-delegate executor in
  `rlx-wgpu`), **tested on `Device::Gpu`** — `ir_deform_matches_native_wgpu` passes
  (native-vs-wgpu `< 1e-4`).
- ✅ **CUDA / ROCm**: `Step::MsDeformAttnHost` + lowering + executor + host module
  (`ms_deform_attn_host`) in `rlx-cuda`/`rlx-rocm`, staging the arena D→H→D and
  running the shared `rlx_cpu::ms_deform_attn::execute_in_arena`. Faithful mirrors
  of the verified WGPU/`llada2_gate` patterns; **compile-unverified on this macOS
  host** (no `nvcc`/ROCm toolchain) — verify on the respective hardware.

The fused deformable op is now wired on **all six backends** (CPU, Metal, MLX, WGPU,
CUDA, ROCm) via the shared `rlx_cpu` compute.
- ✅ **Full decoder layer on-device** ([`decoder_ir`]): one HIR graph composing
  query self-attention + text cross-attention + the fused deformable custom op +
  FFN, dispatched through a `Device`; native-vs-IR parity (`< 1e-4`).
- ✅ **Whole feature enhancer on-device** ([`enhancer_ir`]): all 6 layers —
  bidirectional vision↔text BiMultiHeadAttention (shared score matrix, dual
  softmax), text self-attention, the fused deformable custom op, and the FFNs —
  compose into **one** HIR graph so vision/text stay on-device across layers
  (per-component param sub-prefixes avoid name collisions); native-vs-IR parity.
- ✅ **Swin block fully on-device** ([`swin`]/[`swin_ir`]): each block is **one**
  fused HIR graph — `ln_before` → pad → cyclic shift → window partition → batched
  per-window attention (`[n_win·heads, ws², hd]` matmuls + relative-position bias)
  → reverse → crop → residual → FFN. Window geometry is expressed with
  reshape/transpose/narrow/concat ops (the shift mask + rel-pos bias are
  data-independent, so precomputed on the host); native-vs-IR parity on all
  backends.
- ✅ **Model integration**: `GroundingDino::detect_preprocessed_on(device)` and
  `Decoder::forward_on_device(device)` run the decoder's attention/FFN stack
  on-device per layer (`decoder_ir`) with the scalar glue on the host; the
  `on_device_decoder_matches_native` test confirms the full model's detections
  match the native path on `Cpu`.

> **Build note.** The pinned engine version (`rlx* = =0.2.11`) is not yet on
> crates.io, so the workspace builds against a sibling `../rlx` checkout via
> `.cargo/config.toml` (copied from `.cargo/config.toml.example`). The GPU engine
> kernels added for the deformable op live in that checkout.

### Enabling the on-device deformable op on Apple GPUs

```bash
cp .cargo/config.toml.example .cargo/config.toml   # patch rlx* → ../rlx
cargo test -p rlx-grounding-dino --features engine-apple   # Metal + MLX kernels
cargo test -p rlx-grounding-dino --features gpu \
    deform_attn_ir::tests::ir_deform_matches_native_wgpu   # WGPU on-device
```
