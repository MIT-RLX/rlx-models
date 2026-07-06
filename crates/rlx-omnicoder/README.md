# rlx-omnicoder

**OmniCoder** (Tesslate/OmniCoder-9B and siblings) text-generation runner for RLX. OmniCoder is Qwen3-coder-shaped and ships as `general.architecture = qwen3` in its GGUF converters, so this crate is a thin, arch-validating wrapper over [`rlx_qwen3::Qwen3Runner`](../rlx-qwen3) — it gives users a discoverable per-family entry point and makes misrouted weights fail loudly at `build()`.

> **Status (stub).** Inference is entirely `rlx-qwen3`; this crate adds the arch check and CLI wiring.

## Public API

```rust
use rlx_omnicoder::{OmniCoderRunner, OmniCoderRunnerBuilder};
use rlx_runtime::Device;

let mut runner = OmniCoderRunner::builder()
    .weights("omnicoder-9b.Q4_K_M.gguf")
    .device(Device::Cpu)
    .build()?;
runner.generate(&prompt_ids, 32, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

`rlx_omnicoder::cli_run(&args)` exposes the shared `rlx-qwen3` CLI.

## How it fits

- [rlx-qwen3](../rlx-qwen3) — the underlying Qwen3 runner (prefill + KV-cache decode).
- [rlx-llama-base](../rlx-llama-base) — shared arch-tag presets.
