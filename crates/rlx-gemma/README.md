# rlx-gemma

Google **Gemma**, **Gemma 2**, **Gemma 3**, and **Gemma 4** causal LMs on RLX.
Single graph layer, every backend (CPU / Metal / MLX / CUDA / ROCm / wgpu /
Vulkan), no Python.

## Versions and features

| Family | Status | Notes |
|---|---|---|
| Gemma 1 (`2b`, `7b`) | ✅ | Reference dense decoder. |
| Gemma 2 (`9b`, `27b`) | ✅ | Pre/post FFN RMS norms, attention soft-cap, logit soft-cap, alternating sliding-window mask. |
| Gemma 3 (`1b`–`27b`) | ✅ | Stride-6 sliding-window pattern (5 sliding + 1 full causal). |
| Gemma 4 (`E2B`, `E4B`, `12B` unified) | ✅ LM, ⚠️ multimodal | See below. |
| Gemma 4 12B Coder GGUF ([Composer/Fable fine-tune](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF)) | ✅ packed GGUF | Thinking chat template; `just fetch-gemma4-12b-coder-gguf`. |

### Gemma 4 unified (`google/gemma-4-12B`)

The 12B "unified" model is encoder-free: image patches and audio frames are
projected directly into the LM embedding space. This crate implements:

- **Per-layer attention shape.** Sliding layers use `head_dim=256`,
  `num_kv_heads=8`. Full-attention layers use `global_head_dim=512`,
  `num_global_key_value_heads=1`. KV cache shapes diverge per layer in the
  decode flow.
- **Split RoPE.** Sliding layers run plain RoPE at `theta=10_000`. Full-attention
  layers run **proportional RoPE** with `partial_rotary_factor=0.25` at
  `theta=1_000_000`. Both prefill (static) and decode (static + dynamic) paths
  emit two RoPE tables and dispatch per layer.
- **`attention_k_eq_v`.** The V projection is aliased to K (pre-RoPE), so V
  matmul and `v_proj.weight` load are skipped at runtime.
- **`final_logit_softcapping=30.0`** wired through the existing `LogitSoftcap`
  stage.
- **Unified token vocab** (`vocab_size=262_144`) with `image_token_id=258_880`,
  `audio_token_id=258_881`, `video_token_id=258_884`, and the surrounding
  `boi`/`eoi`/`boa`/`eoa` markers.

Multimodal status:

- ✅ Standalone vision + audio projector graphs (matmul + RMS norm + LM
  projection). Numerically verified on CPU, Metal, MLX, and wgpu.
- ✅ Preprocessing: JPEG/PNG → patches (ImageNet mean/std by default), WAV →
  16 kHz mono f32 (linear resampler), prompt template → media-placeholder
  token stream.
- ✅ Safetensors loader for `vision_tower.*` / `audio_tower.*` weights
  (F32 / F16 / BF16).
- ⚠️ Vision pool-to-soft-tokens uses a placeholder matmul; the reference
  cross-attention "Q-Former-style" reduction from P patches to
  `num_soft_tokens` is not yet wired (output shape is `[B, P,
  output_proj_dims]` today). Fix-in-place when reference projector weights are
  pinned — see [`build_vision_projection_hir`].

### Gemma 4 12B Coder (GGUF)

[yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF)
is a Python-focused coding fine-tune with native thinking (Composer 2.5 + Fable 5
chain-of-thought distillation). Same `gemma4` graph as the base 12B IT model.

```bash
just fetch-gemma4-12b-coder-gguf
just gemma4-coder-demo
```

Packed `Q4_K_M` (~7 GB) is the practical default on 16 GB hosts. The CLI applies
the embedded GGUF chat template (thought channel) by default; pass `--raw-prompt`
to tokenize verbatim. Model-card sampling: `temp 1.0`, `top_p 0.95`, `top_k 64`.

```bash
just test-gemma4-coder   # env-gated; needs .cache/gemma4-12b-coder
```

## Backend coverage

`cargo build -p rlx-gemma --features <feature>`. Feature → backend:

| Feature | `Device` it enables |
|---|---|
| (default) | CPU |
| `metal` | Apple Metal |
| `mlx` | Apple MLX |
| `cuda` | NVIDIA CUDA |
| `rocm` | AMD ROCm |
| `gpu` | wgpu (Metal/Vulkan/DX12) |
| `vulkan` | Vulkan compute |
| `apple-silicon` | metal + mlx + gpu |
| `all-backends` | every device |

### Verified parity (M-series macOS, `apple-silicon`)

```
cargo test -p rlx-gemma --features apple-silicon
# 106 / 106 pass:
#   54 lib  (config, flow, multimodal, runner)
#    8 gemma4_backend_parity         (projector vs CPU on Metal/MLX/wgpu)
#   28 gemma4_lm_backend_parity      (prefill + Gemma 1/2/3 regression)
#    4 gemma4_decode_backend_parity  (full per-layer KV divergence)
#    4 gemma4_decode_throughput      (tok/s regression detection)
#    8 gemma4_reference_fixture      (3 auto-skip without RLX_GEMMA4_FIXTURE; 5 offline flow tests)
```

Numerical parity vs CPU on the Gemma-4 LM graph (k_eq_v + split RoPE +
per-layer head_dim variation, tiny 6-layer config):

| Backend | LM `max\|Δ\|` | Projector `max\|Δ\|` |
|---|---|---|
| MLX | **8.6e-8** (essentially bit-exact) | < 1e-3 |
| wgpu | **1.2e-6** | < 1e-3 |
| Metal causal | **1.1e-6** (precise scalar sgemm) | < 1e-3 |
| Metal sliding | **1.1e-6** (SDPA `mask_kind=4` wired) | < 1e-3 |
| CUDA / ROCm / Vulkan | compile-time gated, not run on this host | compile-time gated |

### Metal precision: a knob, not a floor

Apple Silicon's `simdgroup_float8x8` matmul units use reduced-precision
internal accumulators (~fp16 class for the tensor-multiply step) — fine
for production inference where the resulting ~1e-3 relative error is
bounded by stable softmax + layer norms, but visible (~1e-1 absolute)
on tiny parity-test logits. Set `RLX_METAL_SGEMM_VARIANT=naive` (or
`RLX_METAL_PRECISE=1`) to route through the scalar fp32 sgemm kernel
and recover full CPU parity. The parity tests do this automatically.

This work also tightened `rlx-metal`'s SDPA kernels to use
`precise::sqrt` + `precise::exp` (avoiding the relaxed-precision
`metal::` namespace defaults; see `kernels.rs::sdpa{,_long,_h}`) and
wired `MaskKind::SlidingWindow(w)` end-to-end so Gemma 3 / 4 alternating
attention now runs on Metal where it previously panicked.

- CUDA / ROCm / Vulkan have compile-time parity (`is_available()`-gated
  tests auto-skip without a live driver) and will execute when a host
  with the matching backend runs `cargo test --features cuda` (or
  `rocm`, `vulkan`).

## Examples

### Load a config and run the text LM

```rust
use rlx_gemma::{GemmaConfig, GemmaRunner, GemmaConfigSource};
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;

let mut runner = GemmaRunner::builder()
    .weights("/path/to/gemma-4-12b/model.safetensors")
    .device(Device::Metal)
    .max_seq(512)
    .sample(SampleOpts::greedy())
    .build()?;

let prompt_ids = rlx_gemma::encode_prompt_auto(
    runner.config_ref(),  // resolved weights path
    None,                  // tokenizer auto-found
    "Explain attention in one paragraph.",
)?;
runner.generate(&prompt_ids, 64, |tok| print!("{tok} "))?;
```

### Multimodal projector

```rust
use rlx_gemma::{GemmaMultimodalConfig, GemmaMultimodalRunner, MultimodalWeights};
use rlx_runtime::Device;

let cfg = GemmaMultimodalConfig::from_file("/path/to/gemma-4-12b/config.json")?;
let weights = MultimodalWeights::from_safetensors("/path/to/gemma-4-12b/model.safetensors")?;

let mut mm = GemmaMultimodalRunner::new(cfg, /* lm_hidden */ 3840, Device::Metal, None, None)?;

// JPEG/PNG → ImageNet-normalized patches → projector → soft tokens.
let image_soft = mm.project_image_file("photo.jpg", &weights, /* max_side_patches */ 32)?;

// WAV → 16 kHz mono → frames → projector → audio soft tokens.
let audio_soft = mm.project_audio_file("clip.wav", &weights)?;

// Tokenize "describe <image> while listening to <audio>" with the right
// number of placeholder tokens for each slot.
let encode = |s: &str| rlx_gemma::encode_prompt_auto("...".as_ref(), None, s);
let ids = mm.tokenize_prompt(
    "describe <image> while listening to <audio>",
    /* image_count */ 1,
    &[audio_samples_len],  // total samples in the audio clip
    encode,
)?;

// After embedding the LM token stream, splice the media rows in.
mm.fuse_text_and_media(&mut text_embeds, &ids, &image_soft, &audio_soft)?;
// `text_embeds` now feeds the first LM decoder block.
```

### Inspect per-layer dispatch

```rust
let cfg = GemmaConfig::from_file("/path/to/gemma-4-12b/config.json")?;
assert_eq!(cfg.arch, rlx_gemma::GemmaArch::Gemma4);

// Sliding layer (layer index 0).
assert_eq!(cfg.layer_head_dim(0), 256);
assert_eq!(cfg.layer_num_kv_heads(0), 8);
assert_eq!(cfg.layer_n_rot(0), 256);
assert!((cfg.layer_rope_theta(0) - 10_000.0).abs() < 1e-3);

// Full-attention layer (every 6th, 1-indexed).
assert!(cfg.is_full_attention_layer(5));
assert_eq!(cfg.layer_head_dim(5), 512);
assert_eq!(cfg.layer_num_kv_heads(5), 1);
assert_eq!(cfg.layer_n_rot(5), 128);   // 512 * 0.25
assert!((cfg.layer_rope_theta(5) - 1_000_000.0).abs() < 1e-3);
```

## Architecture notes

### Per-layer attention without new IR ops

Every Gemma 4 quirk is expressible with the existing IR op surface:

- Per-layer `head_dim` / `num_kv_heads` ⇒ per-layer `gemma_attn_spec(...)`
  with different shapes. Each backend already lowers `MatMul` at arbitrary
  shapes.
- Partial RoPE (p-RoPE) ⇒ `Op::Rope { head_dim, n_rot }` with `n_rot <
  head_dim`. This was added for Qwen3.5 MRoPE; every backend already
  implements it.
- Split RoPE tables ⇒ a second `RopeTablesStage::param_named("global", …)`
  publishes `(cos, sin)` into `FlowState::named` under the `global` slot.
  Self-attn and decode emitters opt in via `SelfAttnPrefillSpec::rope_table` /
  `GemmaDecodeLayerSpec::rope_table`.
- `attention_k_eq_v` ⇒ the emitter skips the V matmul and the V weight load.
  Backend kernels see only Q and K — V aliases K's HIR node id.

### Multimodal pipeline

```text
                                    +----------------+
   prompt template  ───────────────►│ tokenize_with_ │
   "<image> see <audio>"            │     media      │──► [u32] token stream
                                    +----------------+
                                              │
                                              ▼
 photo.jpg ──► load_image_patches  ──► [B, P, F]
                                              │
                                              ▼  build_vision_projection_graph
                                       compiled vision projector
                                              │
                                              ▼
 clip.wav ──► load_wav + framing   ──► [B, frames, samples_per_token]
                                              │
                                              ▼  build_audio_projection_graph
                                       compiled audio projector
                                              │
                                              ▼
 LM embed(token_ids) ────────────► fuse_multimodal_embeddings
                                              │
                                              ▼
                                       LM decoder blocks
```

Each box is a separately-runnable graph or pure-CPU function, so callers can
swap in their own image decoder, tokenizer, or fusion policy without touching
the rest.

## Weight formats

### Safetensors

`GemmaRunner::builder().weights("model.safetensors")` accepts F32, F16,
and BF16 safetensors directly. The runner uses
`rlx-models-core::WeightMap` which decodes the source dtype to F32 at
load time. Sharded checkpoints (`model-*.safetensors`) are loaded
through `from_safetensors_dir`.

`MultimodalWeights::from_safetensors` carries the same decoder for the
`vision_tower.*` / `audio_tower.*` projector weights.

### GGUF (quantized)

Gemma 4 GGUF support is wired through `rlx-models-core`'s
`GgufModelFamily::Gemma` — accepts `gemma`, `gemma2`, `gemma3`,
`gemma3n`, `gemma4`, `gemma4moe`, and `gemma4_unified` arch tags.
`GemmaConfig::from_gguf` reads metadata keys under the matching arch
prefix.

The standard GGUF path (`GemmaRunner::builder().weights("model.gguf")`)
**dequantizes** weights to F32 at load time, then runs F32 inference.
This works for `Q8_0`, `Q4_K_M`, `Q5_K_S`, and other GGUF tensor
formats — the dequant table is in `rlx-gguf`. Memory cost is full F32
(~48 GB for Gemma 4 12B); this format choice trades memory for
implementation simplicity.

### Packed quantized inference

`GemmaRunner::builder().packed_weights(true)` is the **memory-saving**
path — weights stay in their GGUF-quantized layout and
`Op::DequantMatMul` runs on-the-fly during matmul. The packed builder
covers every Gemma 1/2/3/4 path the F32 builder does: per-layer
head_dim / num_kv_heads / n_rot, split RoPE, `attention_k_eq_v`,
attention soft-cap, final logit soft-cap, alternating sliding-window
mask. Q/K/V/O and the gate/up/down MLP projections all route through
`DequantMatMul` when the source tensor is K-quantized; tensors that
the GGUF loader returns as F32 (RMS norm gammas, embed_tokens) stay
on the regular `MatMul` path.

### Multimodal weights (separate GGUF)

llama.cpp ships multimodal projector weights as a **separate** GGUF
file (the `mmproj.gguf` companion next to the LM `model.gguf` — used
for LLaVA, MiniCPM-V, Gemma 3 vision, and the upcoming Gemma 4
unified). Both loaders are wired:

```rust
// HF safetensors path (one tensor decoder, F32/F16/BF16):
let w = MultimodalWeights::from_safetensors("model.safetensors")?;

// llama.cpp mmproj.gguf path (decodes any K-quant rlx-gguf supports):
let w = MultimodalWeights::from_mmproj_gguf("mmproj.gguf")?;
```

`from_mmproj_gguf` drains every tensor whose name starts with
`vision_tower.` / `audio_tower.` and dequantizes to F32 at load. The
runner doesn't care which loader supplied the bytes — both produce
the same `MultimodalWeights` map.

### Quantization story summary

| Format | Memory | Status |
|---|---|---|
| F32 safetensors | 48 GB (Gemma 4 12B) | ✅ |
| F16 / BF16 safetensors | 24 GB | ✅ (dequant to F32 at load) |
| GGUF `Q8_0` / `Q4_K_M` (default load) | 6–13 GB on disk → 48 GB in RAM | ✅ (dequant at load) |
| GGUF + `packed_weights(true)` | 6–13 GB in RAM | ✅ (`Op::DequantMatMul` per projection) |
| mmproj.gguf (multimodal) | depends on K-quant | ✅ via `MultimodalWeights::from_mmproj_gguf` |

## See also

- Main repo [README](../../README.md#whats-here) — Gemma entry under **What's here**
- [AGENTS.md](../../AGENTS.md) — `just` recipes and CI commands

## License

GPL-3.0. See workspace LICENSE.
