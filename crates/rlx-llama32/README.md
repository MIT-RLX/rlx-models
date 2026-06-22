# rlx-llama32

LLaMA 3.2–shaped causal LMs in RLX (runner, CLI, GGUF packed prefill).

**Workspace 0.2.6** — Metal + GGUF prefill is configurable via [`MetalGgufPrefillMode`] (`auto`, `cpu`, `packed`, `metal`) on [`Llama32Generator::with_metal_gguf_prefill_mode`] or env (`RLX_METAL_PACKED_PREFILL`, `RLX_METAL_F32_PREFILL_CPU`). Default: CPU F32 (parity).

## CLI

```sh
cargo run -p rlx-llama32 --features tokenizer --release -- \
  --weights /path/to/model.gguf \
  --packed --device metal \
  --prompt-ids 1,42 --max-tokens 16
```

## Packed GGUF

When building a packed prefill graph (`Op::DequantMatMul`), use the shared helpers from `rlx_core`:

- `compile_options_for_packed_gguf_prefill(device)` — Llama 3.2 prefill profile
- `packed_gguf_compile_guard(device, || compile…)` — Metal / MLX env overrides
- `packed_gguf_execution_device(device)` — native CPU/Metal/MLX; wgpu/CUDA → CPU prefill

See [README.md](../../README.md) gotchas and [crates/rlx-minicpm5/README.md](../rlx-minicpm5/README.md).

## See also

- [README.md](../../README.md)
- [AGENTS.md](../../AGENTS.md)
