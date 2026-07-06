# rlx-lfm

**LiquidAI LFM2.5** text runner for RLX — an LFM state-space (SSM) language model. Provides GGUF + HF config parsing and the per-layer decode block that emits the LFM SSM step plus its state-out side output.

> **Status.** `LfmConfig` and the decode-layer plugin are in place; `LfmRunner` with state-buffer binding across decode calls is the remaining piece (mirrors the [rlx-minimax](../rlx-minimax) follow-up).

## Public API

```rust
use rlx_lfm::{LfmConfig, lfm_decode_layer_plugin};

let cfg = LfmConfig::from_gguf_path("lfm2.5.gguf")?;
// lfm_decode_layer_plugin(...) emits an LfmSsmStepStage + state side-output
// per layer when building the decode graph.
# anyhow::Ok(())
```

`rlx_lfm::cli_run(&args)` provides the command-line entry point.

## How it fits

- [rlx-ssm](../rlx-ssm) — shared state-space step kernels (`LfmSsmStepStage`).
- [rlx-llama-base](../rlx-llama-base) — shared Llama-shaped config helpers.
- [rlx-lfm-vl](../rlx-lfm-vl) — the vision-language LFM variant.
