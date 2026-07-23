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
    Engine, RegistryBackend, SingleEngine, build_router, build_router_backend, serve_http,
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

## How it fits

- [`rlx-openai`](../rlx-openai) — multi-model CLI (`--engine qwen3|laguna|…`).
- [rlx-qwen3](../rlx-qwen3) — `LmRunner` behind `SingleEngine` / batching.
- [rlx-laguna](../rlx-laguna) — custom `LagunaEngine` (packed generate).
- [rlx-text](../rlx-text) — tokenizer + chat templating.
