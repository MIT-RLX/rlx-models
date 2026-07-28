# rlx-minimax

**MiniMax M2.5 / M2.7** text runner for RLX — a Lightning Attention language model. Provides GGUF + HF config parsing and the per-layer decode block that emits the Lightning-Attention step plus its state-out side output.

> **Status.** `MiniMaxConfig` and the decode-layer plugin are in place. Still pending: a `MiniMaxRunner` that allocates per-layer state buffers, binds them as model inputs each step, and reads back the `minimax.state_out_{layer}` side outputs; plus a weights loader for embedding + LM head + per-layer projections.

## Public API

```rust
use rlx_minimax::{MiniMaxConfig, minimax_decode_layer_plugin};

let cfg = MiniMaxConfig::from_gguf_path("minimax.gguf")?;
// minimax_decode_layer_plugin(...) emits a LightningAttentionStepStage
// + state side-output per layer when building the decode graph.
# anyhow::Ok(())
```

`rlx_minimax::cli_run(&args)` provides the command-line entry point.

## How it fits

- [rlx-ssm](../rlx-ssm) — shared linear-attention / state-step kernels.
- [rlx-llama-base](../rlx-llama-base) — shared Llama-shaped config helpers.
- [rlx-lfm](../rlx-lfm) — sibling SSM LM sharing the state-wiring pattern.

---

# MiniMax-M3 (MSA) — `rlx_minimax::m3`

**MiniMax-M3** (`minimax_m3_vl` / `MiniMaxM3SparseForCausalLM`) is a different
architecture from M2 above — a natively-multimodal ~428B/23B-active MoE with
**MSA (MiniMax Sparse Attention)**. It lives under the [`m3`](src/m3) module.

The text backbone is a mixed dense/sparse 128-expert MoE on a GQA backbone:
per-head **Gemma `(1+w)` QK-norm**, **partial NeoX RoPE** (`rotary_dim=64` of
`head_dim=128`), **SwiGLU-OAI** (gpt-oss clamped) experts, a sigmoid top-k
router with per-expert routing bias + shared expert, and **MSA**: a lightning
indexer scores keys, max-pools them into blocks, force-includes the local block,
and top-k selects the blocks the main attention may see — realized in-graph as a
block-sparse additive attention bias (`MaskKind::Bias`). Layers `0..3` are dense
+ full-attention; the rest are MoE + MSA (from `moe_layer_freq` /
`sparse_attention_freq`). A CLIP-style vision tower (Conv patch-embed, 3D RoPE,
biased encoder, spatial-merge projector) is included.

```rust
use rlx_minimax::m3::{MiniMaxM3Config, MiniMaxM3Runner, build_m3_text_flow};
use rlx_runtime::Device;

let runner = MiniMaxM3Runner::from_pretrained("model.safetensors", None, Device::Cpu)?;
let logits = { let mut r = runner; r.forward_last_logits(&[1, 2, 3])? };
# anyhow::Ok(())
```

Dispatched as `minimax-m3` (arch `minimax-m3` / HF `model_type` `minimax_m3_vl`).
CLI (token-id driven for now): `rlx-run minimax-m3 --weights <path> --prompt-ids 1,2,3 --max-tokens 16`.

## Status

- **Text decoder** (config, Gemma norm, SwiGLU-OAI, GQA attention, **MSA indexer +
  block-sparse bias**, MoE, dense MLP, full flow) compiles and runs **finite on
  CPU and Metal**. `cargo test -p rlx-minimax`: `m3_text_flow_smoke`,
  `m3_indexer_smoke` (asserts causal + local-block-visible + all-blocks⇒pure-causal),
  `m3_runner_smoke` (predict_logits + re-prefill greedy generate + expert stacking).
- **Vision tower + projector** compile and run finite on CPU (`m3_vision_smoke`).
- **Weights loader** normalizes HF names (strips `language_model.`, stacks
  per-expert `w1/w3/w2` → `gate_up_proj`/`down_proj`).
- **Runner**: correctness-first **prefill** LM (`impl LmRunner`); the default
  re-prefill `generate` works. Registered for auto-dispatch + a `minimax-m3` CLI.
- **VL runner** (`MiniMaxM3VlRunner`): encodes an image through the tower +
  projector, splices the projected features into the prompt's token embeddings
  at the `image_token_index` rows, and prefills the decoder via an
  `inputs_embeds` flow — `prefill_multimodal` / `generate_multimodal` run finite
  end-to-end on CPU (`m3_vl_runner_smoke`).
- **KV-cache incremental decode** (`decode.rs` + `MiniMaxM3Runner::decode_*`):
  single-token decode over a cached (post-RoPE) K/V/idx_k store, with the MSA
  indexer running over the cache. Validated by **prefill-equivalence** — decode
  reproduces batched prefill's last-token logits, and KV-cached greedy matches
  re-prefill greedy (`m3_decode_smoke`). The CLI uses this path.
- **Host image preprocessing** (`M3ImagePreprocessor`): RGB → bilinear resize →
  CLIP-normalize → patchify → `pixel_values` + grid, feeding the vision encoder
  (`m3_vl_runner_smoke::m3_image_preprocess_feeds_vision_encoder`).
- **GGUF config + names** (`MiniMaxM3Config::from_gguf` + `gguf_to_flow_name`):
  parses the `minimax-m3.*` / `minimax.*` metadata and maps ggml `blk.*` tensor
  names to flow names, best-effort per llama.cpp #24908 (`m3_gguf_smoke`).

## Remaining / deferred

- **Real-weight parity**: M3 is 428B — it cannot be loaded (f32) or run on a
  local machine; parity is deferred to adequate hardware. Tests use tiny
  random-weight configs (the established pattern for `rlx-deepseek`/`rlx-llama4`).
- **GGUF weight *loading***: config + name mapping are in place, but stacking the
  per-expert `ffn_{gate,up}_exps` halves into `experts.gate_up_proj` from the real
  ggml byte layout is a follow-up pending an actual M3 GGUF (untestable locally).
- **Decode perf**: decode still recompiles one graph per distinct `past_len`
  (correctness-first); a fixed-max-length bucketed graph + the fast MSA
  `top_k`+gather kernel (per llama.cpp #24908) are the perf follow-ups.
