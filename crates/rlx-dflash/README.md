# rlx-dflash

**DFlash** speculative-decoding drafter for RLX (`general.architecture = "dflash"`).

DFlash is an Eagle-style draft head: it has no embedding or LM head of its own. It reads the
target model's residual streams at `dflash.target_layers`, fuses them through `fc`, and proposes
`dflash.block_size` tokens per step, which the target verifies in one batched forward. DFlash2
checkpoints (`dflash.selector_top_k > 0`) add a grouped dynamic depthwise convolution and a
low-rank selector lattice over adjacent candidate pairs.

| Module | Status | Notes |
|---|---|---|
| `config` | ✅ | `DflashConfig` / `Dflash2Config` from GGUF; `from_hf_json` + `hf_to_dflash_name` for the safetensors drafters. |
| `conv`, `selector` | ✅ | DFlash2 modules, each checked against an independent CPU reference. |
| `builder` | ✅ | Encoder (`fc` + `enc.output_norm`), KV injection, and the non-causal `[anchor, MASK, …]` decoder. |
| `runner` | ✅ | `DflashDrafter` (graphs + KV cache), `DflashTarget`, `DflashLoop` (`seed` / `generate`); `CallbackTarget` plugs in an existing decode loop. |

Validated against the released `z-lab/Qwen3.8-27B-DFlash2` GGUF: the config parses, all three
graphs build, the packed K-quant tensors load, and the loop runs rounds with consistent cache
bookkeeping. Greedy speculation reproduces plain decoding token for token against a mock target.

**Limits.**

- Sliding-window attention past the window is unsupported; `DflashDrafter::new` refuses rather than
  diverging. The released checkpoints set a 2048 window, so that is the drafter's context ceiling.
- Reduced draft vocabularies (`d2t`) are rejected at build time.

The crate has no model dependency. Target-side residual taps live in the model crate
(`rlx-qwen3`: `Qwen3Flow::with_tap_layers`).

Not to be confused with `rlx-laguna`'s `dflash` module, which is poolside's block-diffusion draft
checkpoint under the same product name.

## Examples

```bash
# One full draft step (encoder → KV injection → decoder) on synthetic taps
cargo run --release -p rlx-dflash --example dflash_forward -- <drafter.gguf> [device]

# Propose/verify loop against a stub target (cache bookkeeping, not acceptance)
cargo run --release -p rlx-dflash --example dflash_loop -- <drafter.gguf> [device] [n]

# Acceptance rate against a real Qwen3 target
cargo run --release -p rlx-dflash --example dflash_measure -- \
    <drafter-dir> <target.gguf> [device] [n_tokens]
```

`dflash_measure` reports acceptance, not speedup: the target re-decodes the drafted block one token
at a time rather than verifying it in one batched forward.

## Features

Backend features forward to `rlx-runtime`: `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`,
`all-backends`.

## How it fits

- [rlx-qwen3](../rlx-qwen3) — target model with residual taps.
- [rlx-eagle3](../rlx-eagle3) — EAGLE3 drafter primitives.
- `speculative_accept_sparse` (verification of sampled drafts) comes from
  [rlx-runtime](https://docs.rs/rlx-runtime)'s `spec_decode`.
