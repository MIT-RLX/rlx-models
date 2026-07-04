# rlx-nomic

NomicBERT **encoder graph builder** for RLX — the text tower behind [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). Given a `NomicBertConfig` + weights, it emits an RLX IR graph: embeddings → N NomicBERT layers → `hidden_states`.

Like [`rlx-bert`](../rlx-bert), this crate **only builds the graph** — no tokenizer, loader, or runner. Compiling and running it (tokenize → compile → pool) lives in [`rlx-embed`](../rlx-embed).

## What's different from BERT

NomicBERT is a modernized BERT encoder:

- **Rotary positions (RoPE)** instead of learned position embeddings (so there's a `.rope_tables(...)` stage and no `position_embeddings`).
- **Fused QKV** — one `encoder.layers.{i}.attn.Wqkv.weight` `[3H, H]` instead of separate query/key/value.
- **SwiGLU gated FFN** (`mlp.fc11` / `mlp.fc12` → gate·up → `mlp.fc2`).
- **Biasless** linear layers.
- Token-type embeddings + an embedding LayerNorm (`emb_ln`) are retained.

## Public API

```rust
use rlx_nomic::{NomicFlow, build_nomic_built};
use rlx_core::config::NomicBertConfig;
use rlx_core::weight_map::WeightMap;

let built = build_nomic_built(&cfg, &mut weights, /*batch*/ 1, /*seq*/ 8)?;
// or the configurable builder:
let built = NomicFlow::new(&cfg, 1, 8).with_profile(profile).build(&mut weights)?;
# anyhow::Ok(())
```

- `NomicFlow` / `build_nomic_built(cfg, w, batch, seq) -> BuiltModel` — the production path (native `rlx_flow::ModelFlow`).
- `build_nomic_graph_sized` / `build_nomic_diagnostic_graph` — lowered `Graph` + per-op checkpoints, **diagnostics only** (per-element parity bisection).

Dimensions are baked into the graph; build again for another `batch`×`seq`. The graph takes `input_ids`, `attention_mask`, `token_type_ids` (each `[batch, seq]`) and produces `hidden_states` `[batch, seq, hidden]`.

## Build & test

```sh
cargo test -p rlx-nomic                    # tiny-graph build
cargo build -p rlx-nomic --features metal  # metal | mlx | cuda | rocm | gpu | vulkan
```

To actually embed text, use [`rlx-embed`](../rlx-embed) (mean-pooling, L2-normalize, tokenizer).

## License

GPL-3.0-only. © 2026 Eugene Hauptmann, Nataliya Kosmyna.
