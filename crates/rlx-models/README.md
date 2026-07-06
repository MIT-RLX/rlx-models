# rlx-models

The umbrella **facade** crate for the RLX model zoo. It re-exports the shared config/weight/graph plumbing (`rlx_core`, `rlx_flow`) and, behind cargo features, the per-family model modules (`qwen3`, `gemma`, `whisper`, `bert`, `nomic`, `vision`, …). Depend on this crate to reach several model families through one dependency — or depend on a specific `rlx-<family>` member directly when you only need one.

## Features

Model modules are **feature-gated** and off by default — the default features (`qwen35-tokenizer`, `flux2-tokenizer`, `flux2-image`, `embed`) do **not** pull in the model modules. Enable what you need:

```bash
cargo build -p rlx-models --features "qwen3,qwen35,gemma"
```

## Usage

```rust
// with --features embed
use rlx_models::embed::{Pooling, RlxBertModel, BertTokenizer, embed_with_rlx};

let tok = BertTokenizer::from_dir(model_dir, 512)?;
let mut model = RlxBertModel::load(&config, &weights)?;
let vecs = embed_with_rlx(&mut model, &tok, &["hello", "world"], Pooling::Mean)?;
# anyhow::Ok(())
```

Each gated module (`rlx_models::qwen3`, `::gemma`, `::whisper`, …) simply forwards to the corresponding member crate.

## Binary

```bash
# rlx-run: multiplexer dispatching to per-family runners
cargo run -p rlx-models --bin rlx-run -- <family> [args…]
```

## How it fits

Sits on top of every model member — e.g. [rlx-qwen3](../rlx-qwen3), [rlx-gemma](../rlx-gemma), [rlx-whisper](../rlx-whisper), [rlx-embed](../rlx-embed), [rlx-sam](../rlx-sam) — plus [rlx-cli](../rlx-cli) for the command-line front-ends.
