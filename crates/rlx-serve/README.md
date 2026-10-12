# rlx-serve

**OpenAI-compatible HTTP library** for RLX language models. Exposes
`/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`, with
SSE streaming. Everything above the `engine::Engine` seam is host-side and
`Device`-agnostic.

**Canonical multi-model binary:** [`rlx-openai`](../rlx-openai) (`RegistryBackend`).
The `rlx-serve` binary remains a Qwen3-only compatibility entrypoint.

## Modules

- `engine` — the [`Engine`] trait + `SingleEngine`; turns `GenRequest`s into `StreamItem`s.
- `backend` — `ModelBackend`, `SingleBackend`, **`RegistryBackend`** (multi-model by request `model` id).
- `openai` / `routes` — request/response types and the HTTP routing.
- `batch` — request batching (Qwen3 fused / continuous).
- `sampling_map` — maps OpenAI sampling params to RLX sampling.
- `stop` — stop-sequence handling.
- `serve_http` — shared bind + listen helper.

## Public API

```rust
use std::sync::Arc;
use rlx_serve::{
    Engine, RegistryBackend, SingleEngine, build_router, build_router_backend, host_drivable,
    serve_http,
};

// Single model
let app = build_router(engine, 256);

// Multi-model (preferred for production hosts)
let backend = RegistryBackend::new()
    .register(qwen_engine)
    .register(laguna_engine);
let app = build_router_backend(Arc::new(backend), 256);
serve_http(app, "127.0.0.1", 8080).await?;
```

## Serving a family that has no host-driven decode

`run_generation` drives a model through `LmRunner::prefill_logits` then
`decode_logits` per token, because sampling, logit bias, log-probs and stop
handling all belong above the model. The trait's **default for that pair is
`unsupported`**, and it is called with `?` — so a family that has not
implemented it fails on the first request rather than running slowly. Of the ten
model families in this workspace, two implement the pair (`rlx-qwen3`,
`rlx-qwen4exp`) and one implements session export (`rlx-qwen3`), which
`RunnerBatchRunner::new` requires and bails without.

`host_drivable` closes that for the rest:

```rust
use rlx_serve::{host_drivable, run_generation};

// Anything with `predict_logits` becomes servable.
let mut runner = host_drivable(my_runner);
run_generation(runner.as_mut(), &req, &mut detok, &mut emit)?;
```

It wraps the runner in `rlx_core::lm_host_driven::HostDriven`, which tracks the
context and answers each step by re-running `predict_logits` over all of it —
correct, and **quadratic**. A family that implements the pair natively is
detected on the first prefill and passed through untouched, so wrapping is
always safe. Prefer implementing the pair.

## How it fits

- [`rlx-openai`](../rlx-openai) — multi-model CLI (`--engine qwen3|laguna|…`).
- [rlx-qwen3](../rlx-qwen3) — `LmRunner` behind `SingleEngine` / batching.
- [rlx-laguna](../rlx-laguna) — custom `LagunaEngine` (packed generate).
- [rlx-text](https://docs.rs/rlx-text) — tokenizer + chat templating.
