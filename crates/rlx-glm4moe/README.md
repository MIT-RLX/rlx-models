# rlx-glm4moe

**GLM-4.5 / GLM-4.6** (`glm4_moe`) for RLX: a pre-norm decoder with partial-RoPE GQA attention
(optional qk-norm) and DeepSeek-style fine-grained MoE, reusing [rlx-deepseek](../rlx-deepseek)'s
router and experts.

| Item | Notes |
|---|---|
| `Glm4MoeConfig` | HF `config.json` for `glm4_moe`. |
| `emit_glm_attention` / `GlmAttnDims` | Partial-RoPE GQA attention (`rope_cos` / `rope_sin` tables). |
| `build_glm_text_flow` | Text prefill graph for a fixed `seq`, optional LM head. |

The mlx-community GLM-4.5 / 4.7 checkpoints that route here are listed in
[docs/mlx-community-support.md](../../docs/mlx-community-support.md).

## Tests

```bash
# Full text prefill graph with tiny synthetic weights, compiled and run on CPU
cargo test -p rlx-glm4moe
```

## Features

Backend features forward to `rlx-runtime`: `metal`, `mlx`, and the other backends listed in
`Cargo.toml`.
