# rlx-gpt-oss

**gpt-oss-20b** text-generation runner for RLX. gpt-oss ships as `general.architecture = gpt-oss` (or `gpt_oss`) in its GGUF converters; the 20B variant is an attention-only, Llama-shaped decoder. This crate is a thin, arch-validating wrapper over [`rlx_llama32::Llama32Runner`](../rlx-llama32).

## Public API

```rust
use rlx_gpt_oss::{GptOssRunner, GptOssRunnerBuilder};
use rlx_runtime::Device;

let mut runner = GptOssRunner::builder()
    .weights("gpt-oss-20b.Q4_K_M.gguf")
    .device(Device::Cpu)
    .build()?;
runner.generate(&prompt_ids, 32, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

`rlx_gpt_oss::cli_run(&args)` exposes the shared `rlx-llama32` CLI (`--weights`, `--tokenizer`, `--prompt-ids`, `--device`, …).

## How it fits

- [rlx-llama32](../rlx-llama32) — the underlying Llama-family runner.
- [rlx-llama-base](../rlx-llama-base) — shared Llama config / weight-layout helpers.
