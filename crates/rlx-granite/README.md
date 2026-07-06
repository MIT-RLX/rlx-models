# rlx-granite

**Granite** (IBM) text-generation runner for RLX. Granite ships as `general.architecture = granite` / `granitemoe` / `granitehybrid` in its GGUF converters — Llama-shaped with attention-scale and embedding-scale multipliers. This crate is a thin, arch-validating wrapper over [`rlx_llama32::Llama32Runner`](../rlx-llama32).

> **Status (stub).** Granite's `attention.scale` and `embedding.scale` keys aren't yet read or applied in `rlx-llama32`, so runs produce *some* tokens but won't match the upstream reference until those land. `granitemoe` / `granitehybrid` additionally need MoE / hybrid wiring (PLAN.md M4 + M5).

## Public API

```rust
use rlx_granite::{GraniteRunner, GraniteRunnerBuilder};
use rlx_runtime::Device;

let mut runner = GraniteRunner::builder()
    .weights("granite.Q4_K_M.gguf")
    .device(Device::Cpu)
    .build()?;
runner.generate(&prompt_ids, 32, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

`rlx_granite::cli_run(&args)` exposes the shared `rlx-llama32` CLI (`--weights`, `--tokenizer`, `--prompt-ids`, `--device`, …).

## How it fits

- [rlx-llama32](../rlx-llama32) — the underlying Llama-family runner.
- [rlx-llama-base](../rlx-llama-base) — shared Llama config / weight-layout helpers.
