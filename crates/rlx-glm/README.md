# rlx-glm

**GLM 5.1** text-generation runner for RLX. GLM ships as `general.architecture = glm4` / `glm5` / `chatglm` / `glm4moe` in its GGUF converters — Llama-shaped with a GLM-specific RoPE layout and RMSNorm placement. This crate is a thin, arch-validating wrapper over [`rlx_llama32::Llama32Runner`](../rlx-llama32).

> **Status.** GLM-specific RoPE (`rope_ratio`) and the post-LayerNorm-on-FFN-input placement are not yet wired into `rlx-llama32`. Runs produce *some* tokens but won't match the upstream reference for the first prompts until those deltas land (PLAN.md M5 follow-up).

## Public API

```rust
use rlx_glm::{GlmRunner, GlmRunnerBuilder};
use rlx_runtime::Device;

let mut runner = GlmRunner::builder()
    .weights("glm.Q4_K_M.gguf")
    .device(Device::Cpu)
    .build()?;
runner.generate(&prompt_ids, 32, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

`rlx_glm::cli_run(&args)` exposes the shared `rlx-llama32` CLI (`--weights`, `--tokenizer`, `--prompt-ids`, `--device`, …).

## How it fits

- [rlx-llama32](../rlx-llama32) — the underlying Llama-family runner.
- [rlx-llama-base](../rlx-llama-base) — shared Llama config / weight-layout helpers.
