# rlx-qwen35

Alibaba **Qwen3.5 / Qwen3.6** for RLX — hybrid **Gated DeltaNet** ("linear attention") + full attention every `full_attention_interval` layers, optional **MTP** head for speculative decode. Dense (`qwen35` / `qwen36`) and MoE (`qwen35moe`) GGUFs load through the shared GGUF metadata reader.

**Status:** dense and MoE GGUF prefill + bucketed decode run on CPU and on GPU backends (`metal`, `mlx`, `cuda`, …) with `--packed` for K-quants / Q1_0. Some ops may still fall back to host on a given device; see root [README.md](../../README.md) backend matrix. Remaining gaps (MoE offload polish, VLM parity): `PLAN.md` § Qwen3.5.

## Quick start

```bash
just qwen35 -- --weights model.gguf --prompt "Hello" --max-tokens 32
# or:
cargo run -p rlx-qwen35 --release --features apple-silicon -- \
  --weights model.gguf --packed --device metal --fast \
  --prompt "What is the capital of France?"
```

`--fast` turns on ChatML with thinking disabled, a tight `max_seq` (prompt + tokens), and a `prefill_seq` equal to the prompt length so prefill GEMMs are not padded to decode capacity.

### Useful CLI flags

| Flag | Role |
|------|------|
| `--device` | `cpu`, `metal`, `mlx`, `cuda`, … |
| `--packed` | Keep GGUF K-quants / Q1_0 packed (required for large Bonsai-class files) |
| `--prompt` / `--prompt-ids` | Text (needs `tokenizer`) or raw ids (`;` separates batch rows) |
| `--chat` / `--system` / `--messages-json` | ChatML formatting |
| `--no-think` / `--think` / `--thinking-budget N` / `--show-thinking` | Reasoning on/off and budget |
| `--fast` | Low-latency QA: `--no-think` + tight seqs + `prefill_seq` |
| `--max-seq` / `--max-tokens` | Decode capacity / generation length |
| `--mtp` / `--spec-decode` / `--spec-n` | MTP / speculative decode |
| `--mmproj` / `--image` | VLM (`qwen35-vlm` feature) |

`rlx-qwen35 --help` lists the same set.

### Env (optional)

| Variable | Effect |
|----------|--------|
| `RLX_QWEN35_BENCH=1` | Prefill/decode ms and tok/s on stderr |
| `RLX_QWEN35_DECODE_TRACE=1` | Per-token run/cache/lm timings |
| `RLX_QWEN35_WARM_DECODE=1` | Force decode-bucket warm (drops prefill first on long / non-short GPU contexts) |
| `RLX_QWEN35_KEEP_PREFILL=1\|0` | Override keep/drop of the prefill arena after seed |
| `RLX_LOW_MEM_COMPILE=1` | Stream packed uploads; skip broad warm (short Metal/MLX/CUDA still warm one decode bucket) |
| `RLX_QWEN35_HOST_EMBED=1\|0` | Force host-gathered token embeddings on/off |

Short `max_seq` (≤ 128) on Metal, MLX, and CUDA keeps the prefill graph and warms one decode bucket by default so the first generate does not pay a cold compile or a prefill rebuild.

Do not use `--dynamic-prefill` on CUDA for packed Q1/K-quant GGUFs yet — specialize-per-length paths can produce garbage; prefer static `--fast` / `prefill_seq`.

## Public API

```rust
use rlx_qwen35::Qwen35Runner;
use rlx_runtime::Device;

let mut runner = Qwen35Runner::builder()
    .weights("model.gguf")
    .device(Device::Metal)
    .max_seq(512)
    .prefill_seq(128)          // optional: compile prefill tighter than decode
    .enable_mtp(true)
    .packed_weights(true)
    .build()?;

let out = runner.generate(&[1, 2, 3], 32, |tok| {
    print!("{tok} ");
    true
})?;
# anyhow::Ok(())
```

Also exported: [`Qwen35Config`](src/config.rs), graph builders (`build_qwen35_graph_sized`, `build_qwen35_prefill_cache_graph`, `build_qwen35_decode_graph`), [`Qwen35DecodeCache`](src/cache.rs), [`Qwen35SpecRunner`](src/spec_runner.rs), MoE offload (`build_moe_offload`, `MoeOffloadState`), ChatML helpers (`format_chatml_with`, `split_thinking`, …), and multimodal prefill (`MultimodalPrompt`, `Qwen35VisionEncoder`).

## How it fits

| Crate | Relationship |
|---|---|
| [rlx-qwen3](../rlx-qwen3) | Shared sampling (`SampleOpts`, `sample_token`) |
| [rlx-llada2](../rlx-llada2) | TIDE predictive expert-offload API for MoE checkpoints |

## Features

| Feature | Enables |
|---|---|
| `tokenizer` (default) | Text `--prompt` encode/decode via `tokenizers` |
| `qwen35-vlm` | Image preprocess + `--image` multimodal prefill |
| `parity-llama` | llama.cpp reference (`llama-cpp-2`) for numeric parity tests |
| `metal`, `mlx`, `cuda`, `rocm`, `gpu`, `vulkan`, `coreml`, `all-backends` | Forwarded to `rlx-runtime` |

Parity vs llama.cpp is env-gated (`QWEN35_GGUF_PATH`, optional `parity-llama`). Performance notes: [src/BENCHMARKS.md](src/BENCHMARKS.md).
