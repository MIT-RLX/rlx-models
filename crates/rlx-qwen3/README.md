# rlx-qwen3

Alibaba [**Qwen3**](https://github.com/QwenLM/Qwen3) dense causal-LM family (0.6B / 1.7B / 4B / 8B / 14B / 32B) for RLX — GQA + QK-norm + SwiGLU MLP + RoPE. Ships a compiled prefill graph, a bucketed KV-cache decode loop, and an optional MTP speculative-decode head. Weight keys mirror HF's `Qwen3ForCausalLM`, so safetensors checkpoints load with no remapping; GGUF works too via the `rlx_core::weight_loader` adapter.

## Quick start

The CLI takes **token ids** (there is no built-in tokenizer yet):

```bash
just qwen3 -- --weights model.gguf --prompt-ids 1,2,3 --max-tokens 32
# or directly:
cargo run -p rlx-qwen3 --release -- --weights model.gguf --prompt-ids 1,2,3
```

Key flags (see `--help` / `src/cli.rs`): `--device cpu|metal|mlx|cuda|rocm|gpu|vulkan`, `--config config.json`, `--format safetensors|gguf`, `--max-seq`, `--precision f32|f16-lm`, `--packed`, `--use-mtp`, `--temperature`, `--top-p`, `--prefer-quant`.

Large GGUFs (≥ 256 MB) auto-enable **packed weights** — K-quant tensors stay packed in the arena and each matmul emits `Op::DequantMatMul`, cutting host memory ~6× so 14B+ models fit on commodity hardware.

## Public API

```rust
use rlx_qwen3::{Qwen3Runner, Precision};
use rlx_runtime::Device;

let mut runner = Qwen3Runner::builder()
    .weights("model.gguf")          // safetensors or gguf (autodetected)
    .device(Device::Metal)
    .max_seq(128)
    .precision(Precision::F32)
    .build()?;

let prompt_ids: &[u32] = &[1, 2, 3];
let out = runner.generate(prompt_ids, 32, |tok| print!("{tok} "))?;
# anyhow::Ok(())
```

Lower-level entry points are also public:

- [`Qwen3Config`](src/config.rs) — HF `config.json` / GGUF-metadata deserialiser.
- [`build_qwen3_graph_sized`](src/builder.rs) — emit the prefill IR graph for a concrete `(batch, seq)`.
- [`build_qwen3_decode_graph_sized`](src/builder.rs) — the incremental KV-cache decode graph.
- [`Qwen3Generator`](src/generator.rs) — streaming prefill + cached decode driver.
- [`Qwen3Speculator`](src/spec.rs) — MTP-head draft/verify speculative decoding.
- [`SampleOpts`] / [`sample_token`](src/sampling.rs) — temperature / top-p sampling.

Dimensions are baked into each compiled graph, so consumers keep a shape→graph cache and recompile on a new `(batch, seq)` bucket.

## Backends

Prefill and KV-cache decode run on every standard RLX backend when the matching `rlx-runtime` feature is enabled: `metal`, `mlx`, `cuda`, `rocm`, `gpu` (wgpu), `vulkan`, `coreml`. Build with `all-backends` (or `macos-gpu` / `nvidia-gpu` / `amd-gpu` / `portable-gpu`) for a single binary that accepts every `--device` value. [`validate_device`](src/capabilities.rs) reports what the current build supports.

## Examples

```bash
cargo run -p rlx-qwen3 --release --example decode_bench         -- --weights model.gguf
cargo run -p rlx-qwen3 --release --example roundtrip_bench      -- --weights model.gguf
cargo run -p rlx-qwen3 --release --example qwen3_pipeline       -- --weights model.gguf
# batched / serving throughput curve (PACKED=1, RESIDENT=1, RAGGED=1 env toggles):
cargo run -p rlx-qwen3 --release --features metal --example batched_decode_bench -- <weights-dir>
```

`pipeline_bench` / `pipeline_multiproc` exercise the block-pipelined ([`BlockSpec`](src/pipeline.rs)) prefill/decode stages.

## Performance & throughput

Packed (Q4) decode, the SIMD quant-GEMV kernels, resident KV, flash-decode, and
the **batched / serving throughput** stack (uniform + ragged, packed +
resident-KV — fuse concurrent requests at mixed lengths) are documented in
[`docs/qwen3-packed-decode-throughput.md`](../../docs/qwen3-packed-decode-throughput.md).
The earlier F16-weight single-stream path is in
[`rlx/docs/metal-qwen3-decode-perf.md`](../../../rlx/docs/metal-qwen3-decode-perf.md).

## Export as an `.rlxp` package

With feature `export`, `export::export_rlxp` writes a Hugging Face checkpoint
as a package that rlx runs with no rlx-models code.
`Model.open("qwen3-0.6b.rlxp").chat(messages)` then works from any rlx
language binding:

```rust,ignore
rlx_qwen3::export::export_rlxp("Qwen3-0.6B".as_ref(), "qwen3-0.6b.rlxp".as_ref(), &Default::default())?;
```

The package holds the decode graph at cache capacities 64 to 4096, a
`generate` entry, the RoPE tables, the tokenizer and chat template, and the
`README.md` as its model card. Its weights are referenced from
`model.safetensors`, so Qwen3-0.6B becomes a 4.9 MB package. To check that
it greedy-decodes the same tokens as `Qwen3Runner`:
`RLX_TEST_QWEN3_DIR=<checkpoint dir> cargo test -p rlx-qwen3 --features export --test export_rlxp`.

## How it fits

- [rlx-qwen35](../rlx-qwen35) — Qwen3.5 / 3.6 hybrid trunk, reuses `rlx_qwen3::sampling`.
- [rlx-omnicoder](../rlx-omnicoder) — OmniCoder wrapper over [`Qwen3Runner`] with arch validation.
</content>
</invoke>
