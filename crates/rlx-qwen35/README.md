# rlx-qwen35

Alibaba **Qwen3.5 / Qwen3.6** for RLX — a hybrid **Gated DeltaNet** ("linear attention") + standard-attention trunk. Most layers are a gated-DeltaNet SSM block; full attention is inserted every `full_attention_interval` layers, with an optional **MTP** (multi-token-prediction) head on top for speculative decoding. Dense (`qwen35`/`qwen36`) and MoE (`qwen35moe`) GGUFs load through the shared GGUF metadata reader.

> **Status:** dense and MoE GGUF forward is wired end-to-end on CPU (prefill, bucketed decode cache, optional MTP head, packed K-quants). Every standard backend accepts `--device`, though some GPU paths still run the GDN recurrence or dequant matmul on the host. See `PLAN.md` § Qwen3.5 for the remaining MoE / VLM-parity gaps.

## Quick start

```bash
just qwen35 -- --weights model.gguf --prompt "Hello" --max-tokens 32
# or directly:
cargo run -p rlx-qwen35 --release -- --weights model.gguf --prompt-ids 1,2,3
```

Key flags (`src/cli.rs`): `--device`, `--prompt` / `--prompt-ids` (`;`-separated rows for batching), `--max-seq`, `--mtp`, `--spec-decode --spec-n N`, `--packed`, `--mmproj`/`--image` (VLM, needs the `qwen35-vlm` feature). The default `tokenizer` feature enables the text `--prompt` path.

## Public API

```rust
use rlx_qwen35::{Qwen35Runner};
use rlx_runtime::Device;

let mut runner = Qwen35Runner::builder()
    .weights("model.gguf")
    .device(Device::Cpu)
    .max_seq(512)
    .enable_mtp(true)          // build the MTP head
    .packed_weights(true)      // keep K-quants packed in the arena
    .build()?;

let out = runner.generate(&[1, 2, 3], 32, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

Also exported: [`Qwen35Config`](src/config.rs) (`from_gguf`, arch-prefix aware for `qwen35`/`qwen36`), the prefill / decode graph builders (`build_qwen35_graph_sized`, `build_qwen35_prefill_cache_graph`, `build_qwen35_decode_graph`), the [`Qwen35DecodeCache`](src/cache.rs), [`Qwen35SpecRunner`](src/spec_runner.rs) for MTP speculative decode, the MoE expert-offload API (`build_moe_offload`, `MoeOffloadState`), and the multimodal prefill helpers (`MultimodalPrompt`, `Qwen35VisionEncoder`).

## How it fits

| Crate | Relationship |
|---|---|
| [rlx-qwen3](../rlx-qwen3) | reuses `Qwen3` sampling (`SampleOpts`, `sample_token`) |
| [rlx-llada2](../rlx-llada2) | re-exports its **TIDE** predictive expert-offload API for MoE checkpoints |

## Features

| Feature | Enables |
|---|---|
| `tokenizer` (default) | text `--prompt` encode/decode via `tokenizers` |
| `qwen35-vlm` | image preprocessing + `--image` multimodal prefill |
| `parity-llama` | llama.cpp reference oracle (`llama-cpp-2`) for numeric parity tests |
| `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `coreml`, `all-backends` | forwarded to `rlx-runtime` |

Parity vs llama.cpp is env-gated (`QWEN35_GGUF_PATH`, optional `parity-llama`).
</content>
