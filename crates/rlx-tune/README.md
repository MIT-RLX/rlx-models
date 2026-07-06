# rlx-tune

Generic **LoRA / DoRA fine-tuning** for RLX models — model-agnostic, host-side adapters, dataset loaders, graph injection, and a trainer loop. The forward pass uses RLX's first-class `LoraMatMul` op (which lowers on every backend), so training runs on CPU / Metal / MLX like inference.

## Modules

- `adapter` — LoRA/DoRA specs + host-side merge ([`fuse_lora`], [`fuse_dora`]).
- `dataset` — text / chat / completions JSONL loaders with prompt masking (mirrors mlx-lm's `datasets.py`).
- `inject` — graph-rewrite `inject_lora` over a model's forward graph.
- `trainer` — compile `grad_with_loss` once, accumulate, step (via `rlx-optim` / `rlx-autodiff`).
- `dwq` — distilled weight quantization support.

## Public API

```rust
use rlx_tune::adapter::{LoraSpec, LoraInit, fuse_lora};

let spec = LoraSpec { rank: 16, alpha: 32.0, /* … */ ..Default::default() };
// inject into a forward graph, train, then fuse the adapter back into base weights:
// fuse_lora(&mut weights, &adapter)?;
```

## Quick start

```bash
cargo run -p rlx-tune --example bench_inject
```

## How it fits

Backend-agnostic; builds on `rlx-autodiff` + `rlx-optim`. Adapters fuse into any RLX model's weights — e.g. the model-specific trainers [rlx-qwen3-tts-train](../rlx-qwen3-tts-train) and [rlx-voxtral-tts-train](../rlx-voxtral-tts-train) use the same LoRA machinery.
