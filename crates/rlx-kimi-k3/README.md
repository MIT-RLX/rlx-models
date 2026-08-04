# rlx-kimi-k3

Kimi-K3 (Moonshot AI) for RLX — a multimodal MoE model.

**Text backbone (`KimiLinear`, 93 layers, hidden 7168):**
- **KDA** (Kimi Delta Attention) — gated delta-net linear attention (short causal
  conv on q/k/v + silu, L2-normed keys, gated-RMSNorm output) via `Op::GatedDeltaNet`.
- **MLA** (NoPE) — DeepSeek-style multi-head latent attention, **no** rotary
  embedding, optional sigmoid output gate. Interleaved with KDA per `linear_attn_config`.
- **LatentMoE** — 896 routed experts (16 active, MXFP4-packed) + shared latent
  up/down projection + 2 shared experts; sigmoid `noaux_tc` grouped-topk router.
- **situ** activation (custom GLU), **Attention Residuals** every `attn_res_block_size`.

**Vision:** ViT tower (patch 14, RMSNorm, gelu-tanh, video-aware pos-emb) + a
patch-merger projector, spliced into the text stream at the media placeholder token.

## Status
Port in progress. Verified via device-parametrized finite-logits smoke tests on
synthetic configs across all RLX backends, plus a single-block real-weight check.
Real full-model validation is deferred (the checkpoint is ~1.6 TB bf16).

## Cluster inference & expert-compute performance
The disaggregated expert-parallel path (recurrent backbone on one node; stateless MoE
experts fanned out to workers over TCP, one rank per compute engine) and its optimized
MXFP4 expert kernels — parallel paging, fused CPU decode-matmul, and **native CUDA/HIP
register-decode GEMM** — are documented in
[`docs/MOE_EXPERT_OPTIMIZATION.md`](docs/MOE_EXPERT_OPTIMIZATION.md). Fleet launch/measure
tooling is in [`scripts/`](scripts/README.md).
