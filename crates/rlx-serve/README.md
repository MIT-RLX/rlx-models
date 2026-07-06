# rlx-serve

**OpenAI-compatible HTTP server** for RLX language models. Exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`, with SSE streaming. Everything above the `engine::Engine` seam is host-side and `Device`-agnostic — the model runs wherever its `LmRunner` runs (CPU / Metal / MLX).

## Modules

- `engine` — the [`Engine`] trait + `SingleEngine`; turns `GenRequest`s into `StreamItem`s.
- `openai` / `routes` — request/response types and the HTTP routing.
- `batch` — request batching.
- `sampling_map` — maps OpenAI sampling params to RLX sampling.
- `stop` — stop-sequence handling.

## Public API

```rust
use rlx_serve::engine::{Engine, GenRequest, SingleEngine, StreamItem};

// Wrap any LM runner in a SingleEngine, then mount the OpenAI routes
// (see routes.rs) on your HTTP server. Streaming yields StreamItem deltas.
```

## How it fits

- [rlx-qwen3](../rlx-qwen3) — a concrete `LmRunner` backing the engine.
- [rlx-text](../rlx-text) — tokenizer + chat templating.
