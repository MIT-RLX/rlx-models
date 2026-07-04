# rlx-embed

Text + image **embedding runtime** for RLX. This is the crate that turns the encoder *graph builders* ([`rlx-bert`](../rlx-bert), [`rlx-nomic`](../rlx-nomic), [`rlx-vision`](../rlx-vision)) into a working pipeline: **tokenize → compile → run → pool → normalize → vectors**. (Migrated from `burnembed`.)

## Quickstart

```rust
use rlx_models::embed::{Pooling, RlxBertModel, BertTokenizer, embed_with_rlx};

let tok = BertTokenizer::from_dir(model_dir, 512)?;
let mut model = RlxBertModel::load(&config, &weights)?;
let vecs = embed_with_rlx(&mut model, &tok, &["hello", "world"], Pooling::Mean)?;
// vecs: Vec<Vec<f32>> — one embedding per input
# anyhow::Ok(())
```

## Supported models

Auto-detected via `detect_arch` (from `config.json` / GGUF metadata) into one of three architectures — `Bert`, `NomicBert`, `NomicVision`:

| Model | Arch | Notes |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2`, `-L12-v2` | Bert | mean pooling |
| `BAAI/bge-{small,base,large}-en-v1.5`, `bge-small-zh-v1.5` | Bert | CLS pooling |
| `nomic-ai/nomic-embed-text-v1.5` | NomicBert | 768-dim, RoPE + SwiGLU, mean pooling |
| `nomic-ai/nomic-embed-vision-v1.5` | NomicVision | 768-dim ViT, 224px, CLS pooling |

`EmbeddingModel::list_supported()` / `models_map()` enumerate the built-in registry (HF repo + weight file + pooling).

## How it works

1. **Tokenize** — `BertTokenizer::encode_batch` → `input_ids` / `attention_mask` / `token_type_ids` (`TokenizedBatch`).
2. **Compile** — `RlxBertModel` / `RlxNomicModel` / `RlxVisionModel` build the encoder graph (via the sibling builder crate) and compile it for a `Device`; the compiled graph is cached per `batch`×`seq` (`recompile` on a new shape).
3. **Run** — inputs are marshaled to `f32` and fed to the compiled graph → `hidden_states` `[B, S, H]`.
4. **Pool** — `Pooling::{Mean, Cls, None}` (`pool_embeddings`), then optional `l2_normalize_in_place`.

Lower-level entry points: `RlxEmbed`, `compile_model` / `compile_model_cpu`, `assemble_vision_hidden`.

## Features

`embed` (tokenizers, **default**) · `hf-download` (Hub fetch) · backends `metal` / `mlx` / `cuda` / `rocm` / `gpu` / `vulkan` / `coreml` (+ `all-backends`, `apple-silicon`, …).

```sh
cargo test -p rlx-embed
cargo build -p rlx-embed --features apple-silicon --release
```

## License

GPL-3.0-only. © 2026 Eugene Hauptmann, Nataliya Kosmyna.
