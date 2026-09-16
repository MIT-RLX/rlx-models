# rlx-hy-mt

[Tencent HY-MT1.5](https://huggingface.co/tencent/HY-MT1.5-1.8B) on-device multilingual translation in RLX.

The checkpoint is **Hunyuan dense** (`hunyuan_v1_dense` / GGUF `hunyuan-dense`): GQA + QK-norm + SwiGLU — the same shape as Qwen3. This crate validates HY-MT metadata and runs on [`rlx-qwen3`](../rlx-qwen3) across all standard RLX backends (`cpu`, `metal`, `mlx`, `cuda`, …).

## Weights

| Variant | Hugging Face | Notes |
|--------|----------------|-------|
| 1.8B safetensors | [`tencent/HY-MT1.5-1.8B`](https://huggingface.co/tencent/HY-MT1.5-1.8B) | HF uses `query_layernorm` / `key_layernorm` (aliased to Qwen `q_norm` / `k_norm`) |
| 1.8B GGUF | [`tencent/HY-MT1.5-1.8B-GGUF`](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF) | `Q4_K_M` / `Q6_K` / `Q8_0` — preferred for phones |
| 7B | [`tencent/HY-MT1.5-7B`](https://huggingface.co/tencent/HY-MT1.5-7B) | Same arch, larger |

Extreme 1.25-bit / 2-bit AngelSlim quants need STQ kernels not yet in RLX — use the official K-quants above.

## Quick start

```rust
use rlx_hy_mt::{HyMtRunner, TranslatePrompt, target_language_name};

let mut runner = HyMtRunner::builder()
    .weights("/path/to/HY-MT1.5-1.8B-Q4_K_M.gguf")
    .packed_weights(true)
    .device(rlx_runtime::Device::Cpu)
    .build()?;

let user = TranslatePrompt::xx_to_xx(
    "The secret sauce is in the windings.",
    target_language_name("fr"),
)
.render_user();
// tokenize with the model tokenizer, then runner.generate(...)
# anyhow::Ok(())
```

```sh
cargo build -p rlx-hy-mt --release
cargo run -p rlx-hy-mt --bin rlx-hy-mt --features metal --release -- \
  --weights /path/to/HY-MT1.5-1.8B-Q4_K_M.gguf --packed --device metal \
  --prompt "Translate the following segment into French, without additional explanation.

Hello."
```

## Backend parity

Synthetic Qwen3-shaped graphs (same topology as HY-MT) are covered by `just test-qwen3-backends`. HY-MT-specific registration / config tests:

```sh
cargo test -p rlx-hy-mt
cargo test -p rlx-models-core mlx_coverage -- --nocapture
```

## Prompt templates

From the model card:

- **XX↔XX** (EN→FR, …): `Translate the following segment into {lang}, without additional explanation.`
- **ZH↔XX**: Chinese instruction template (`format_zh_to_xx_prompt`)
