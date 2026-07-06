# rlx-minimax

**MiniMax M2.5 / M2.7** text runner for RLX — a Lightning Attention language model. Provides GGUF + HF config parsing and the per-layer decode block that emits the Lightning-Attention step plus its state-out side output.

> **Status.** `MiniMaxConfig` and the decode-layer plugin are in place. Still pending: a `MiniMaxRunner` that allocates per-layer state buffers, binds them as model inputs each step, and reads back the `minimax.state_out_{layer}` side outputs; plus a weights loader for embedding + LM head + per-layer projections.

## Public API

```rust
use rlx_minimax::{MiniMaxConfig, minimax_decode_layer_plugin};

let cfg = MiniMaxConfig::from_gguf_path("minimax.gguf")?;
// minimax_decode_layer_plugin(...) emits a LightningAttentionStepStage
// + state side-output per layer when building the decode graph.
# anyhow::Ok(())
```

`rlx_minimax::cli_run(&args)` provides the command-line entry point.

## How it fits

- [rlx-ssm](../rlx-ssm) — shared linear-attention / state-step kernels.
- [rlx-llama-base](../rlx-llama-base) — shared Llama-shaped config helpers.
- [rlx-lfm](../rlx-lfm) — sibling SSM LM sharing the state-wiring pattern.
