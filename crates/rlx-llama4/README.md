# rlx-llama4 — Llama-4 (Scout / Maverick) for RLX

Native RLX port of Meta's **Llama-4**. Text tower: a mixture-of-experts decoder
with **iRoPE** — RoPE layers apply complex/interleaved rotary + L2 qk-norm and do
chunked-window attention, while periodic NoPE layers do full attention with
temperature-tuned scaling. Vision is **early-fusion** (embed-splice), added next.

## Layout

| module | role |
|---|---|
| `config.rs` | `Llama4TextConfig` + derived `no_rope`/`moe` layer schedules |
| `moe.rs` | top-1 MoE FFN (`Op::GroupedMatMul` experts, input-scaled by `sigmoid(top_logit)`, + shared expert) |
| `attention.rs` | GptJ rope + L2 qk-norm (RoPE layers), causal GQA attention |
| `flow.rs` | text prefill graph: `token_embed → N×(RMSNorm→attn→+res→RMSNorm→FFN→+res) → norm → lm_head` |
| `rope.rs` | interleaved rotary cos/sin tables `[seq, head_dim/2]` |
| `runner.rs` + `cli.rs` | load checkpoint, generate; `--weights <dir> --prompt <text>` |

**v1 simplifications (valid for `seq < attention_chunk_size`, default 8192):**
temperature tuning is exactly 1.0 and chunked attention equals full causal, so all
layers use plain causal attention. Decode is full-sequence (no KV cache) — the
graph compiles once at `max_len`, each step re-runs the padded sequence and reads
logits at the true last position.

### Vision (`vision.rs`)

2D-axial-RoPE ViT (34 layers, biased q/k/v/o, GptJ rope, full attention) → drop
class token → pixel-shuffle (N→N/4, C→C·4) → adapter MLP2 → multi-modal projector
→ image features `[1, N/4, text_hidden]`. The 2D rope lives entirely in the host
cos/sin (`rope::build_vision_rope_tables`).

### Multimodal (`preprocess.rs`, `vl_runner.rs`, `cli.rs --image`)

Host image preprocessing (unfold patch-embed + class token + position embedding),
an `inputs_embeds` text-flow variant, and `Llama4VlRunner` which runs the vision
tower, prepends the image feature rows to the prompt token embeddings, and decodes
on `inputs_embeds`.

## Status

- Config, MoE FFN, text prefill graph, vision tower, image preprocessing, and the
  full VLM wiring all compile and run on CPU (`cargo test -p rlx-llama4`: `config`,
  `preprocess`, `moe_smoke`, `text_flow_smoke`, `vision_smoke`).
- Scout is all-MoE (109B); real-checkpoint validation is out of local/NVIDIA RAM,
  deferred with mllama.

## Remaining (refinements)

Aspect-ratio image tiling + global thumbnail (v1 uses one global tile); proper
`<|image|>` placeholder matching (v1 prepends the image); KV-cache decode; `run.rs`
dispatch wiring.
