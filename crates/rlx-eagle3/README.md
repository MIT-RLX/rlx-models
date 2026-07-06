# rlx-eagle3

**EAGLE3** speculative-decoding primitives for RLX — draft-model config, draft→target vocab mapping, weight loading, and the `Speculator` scaffold used to propose candidate tokens for a larger target model.

> **Status.** Parsing/mapping/weights are tested; the draft graph + end-to-end proposer are still being wired.

| Module | Status | Notes |
|---|---|---|
| `config` | ✅ tested | Parses RedHatAI / vLLM-speculators `Eagle3SpeculatorConfig`. |
| `d2t` | ✅ tested | Draft-vocab → target-vocab scatter (LUT-based). |
| `weights` | ✅ tested (synthetic) | Reads draft `model.safetensors`, surfaces named tensors. |
| `speculator` | ⚠️ scaffold | `Eagle3Speculator<H>` implements `Speculator`; `propose()` needs the draft graph. |
| `draft` | ⚠️ not yet built | HIR graph for the 1-layer Llama draft + fc fusion + lm_head over draft vocab. |

## Quick start

```bash
# Draft-step / lm-head / end-to-end propose micro-benchmarks
cargo run -p rlx-eagle3 --example bench_draft_step_backends
cargo run -p rlx-eagle3 --example load_real_draft
```

## How it fits

- [rlx-qwen3](../rlx-qwen3) / [rlx-gemma](../rlx-gemma) — target models the draft accelerates.
- Builds on the core [rlx-ir](../../rlx) / [rlx-flow](../../rlx) graph layer for the draft network.
