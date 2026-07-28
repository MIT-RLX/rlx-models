# rlx-jamba — Jamba (Mamba-1 + attention hybrid) for RLX

Native RLX port of Jamba's hybrid decoder: most layers are **Mamba-1 SSM
mixers**, every `attn_layer_period` layer is **attention**, and (in the full
model) every `expert_layer_period` layer is MoE.

- **`mamba.rs`** — the Mamba-1 mixer as a flow subgraph: `in_proj → causal
  depthwise conv1d → SiLU → x_proj → dt/B/C RMSNorm → dt_proj →
  softplus + Op::SelectiveScan + D-skip → SiLU(z) gate → out_proj`. Builds on
  `rlx-ssm`'s selective-scan op.
- **`attention.rs`** — plain causal GQA with **no RoPE** (position comes from the
  Mamba layers).
- **`flow.rs`** — interleaves the mixers per `layer_is_attention` schedule with a
  dense SwiGLU FFN, standard pre-norm.

## Status

- Selective scan, the full Mamba-1 mixer, and the interleaved Mamba/attention
  text flow compile and run (`cargo test -p rlx-jamba`: `scan_smoke`,
  `mamba_block_smoke`, `jamba_flow_smoke`).

## Remaining

Jamba's sparse-MoE FFN (softmax top-k, v1 uses dense MLP everywhere); runner +
CLI; real-checkpoint parity. **Zamba / Falcon-H1** are Mamba-**2** hybrids — they
need a Mamba-2 *prefill* scan stage first (rlx-ssm currently wires only the
Mamba-2 decode step; `Op::Mamba2` exists).
