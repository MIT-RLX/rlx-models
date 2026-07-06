# rlx-bonsai

Runner for the **Bonsai** small-reasoning model family (1.7B / 2B / 4B / 8B). Bonsai ships as `general.architecture = llama` in its GGUF converters — a standard Llama-shaped decoder with hyperparameters tuned for small-context reasoning — so this crate is a thin wrapper over [`rlx_llama32::Llama32Runner`](../rlx-llama32) that validates the GGUF arch tag (misrouted files fail loudly at `build()` instead of producing garbage) and reserves a spot for Bonsai-specific tuning later. Status: stub (PLAN.md M4).

## Public API

```rust
use rlx_bonsai::BonsaiRunner;

let mut runner = BonsaiRunner::builder()
    .weights("bonsai-2b.gguf") // must be `general.architecture = llama`
    .max_seq(4096)
    .packed_weights(true)
    .build()?; // Err if the GGUF arch tag isn't `llama`

let out = runner.generate_packed(&prompt_ids, 128, |tok| {
    // stream each new token id
    let _ = tok;
})?;
# anyhow::Ok(())
```

`BonsaiRunner::config()` borrows the parsed [`LlamaBaseConfig`](../rlx-llama-base) (dims, RoPE, GQA); `inner()` / `inner_mut()` expose the underlying `Llama32Runner` for callers that need to drive prefill/decode directly. `cli_run(&args)` delegates to `rlx_llama32::cli::run` after checking the arch tag, so every Llama-shaped flag (`--weights`, `--tokenizer`, `--prompt`, `--packed`, …) works.

## Features

`tokenizer` and the backend flags (`metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `all-backends`) all forward to [rlx-llama32](../rlx-llama32).

## How it fits

- Delegates every forward pass to [rlx-llama32](../rlx-llama32); config parsing comes from [rlx-llama-base](../rlx-llama-base).
- Same pattern as other per-family Llama wrappers in the workspace — a validation shell around the shared Llama-3.2 runner rather than a re-implementation.
