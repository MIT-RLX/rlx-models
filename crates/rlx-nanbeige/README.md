# rlx-nanbeige

[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) — compact **Looped Transformer** causal LM — on RLX backends.

The checkpoint is Llama-shaped (`NanbeigeForCausalLM`: GQA + RoPE + SwiGLU + RMSNorm) with `num_loops = 2`: the same 22 physical layers run twice with **shared weights** and **separate KV slots** (44 effective depth). Mid-loop `model.norm` matches the upstream HF modeling code (`skip_loop_final_norm = false`).

This crate validates HF/GGUF metadata and delegates inference to [`rlx-llama32`](../rlx-llama32) (loop unrolling lives in the shared decoder).

## Download

```sh
just fetch-nanbeige
# → /tmp/rlx-weights/Nanbeige4.2-3B/  (override with NANBEIGE_MODEL_DIR=…)
```

Weights are ~8 GB BF16 safetensors (two shards).

## CLI

```sh
DIR=/tmp/rlx-weights/Nanbeige4.2-3B

just nanbeige -- \
  --weights "$DIR" \
  --device cpu \
  --prompt-ids 1,42 \
  --max-tokens 16 \
  --max-seq 512
```

Pass the **model directory** (needs `model.safetensors.index.json` + shards) or any shard path whose parent holds the index.
Tokenizer / chat template: use the HF `tokenizer.json` next to the weights (`--tokenizer …`). Recommended sampling for reasoning/chat: temperature `0.6`, top-p `0.95`, top-k `20` (see the model card).

GPU:

```sh
just features=apple-silicon nanbeige -- --weights "$WEIGHTS" --device metal --prompt-ids 1,42 --max-tokens 8
```

## Library

```rust,ignore
use rlx_nanbeige::NanbeigeRunner;
use rlx_runtime::Device;

let mut runner = NanbeigeRunner::builder()
    .weights("/path/to/model-00001-of-00002.safetensors")
    .device(Device::Cpu)
    .max_seq(512)
    .build()?;

let logits = runner.predict_logits(&[1, 42])?;
let tokens = runner.generate(&[1, 42], 16, |_| {})?;
```

Preset dims: [`nanbeige42_3b_preset`](src/config.rs) (`hidden=3072`, `layers=22`, `loops=2`, `heads=48/8`, `head_dim=128`, `rope_theta=7e7`).

## Backends

Prefill + decode with `num_loops = 2` is exercised on every standard RLX device
(CPU, Metal, MLX, CUDA, ROCm, WGPU, Vulkan; CoreML validates when enabled):

```sh
just features=all-backends test-nanbeige-backends
just features=apple-silicon test-nanbeige-backends
cargo run -p rlx-nanbeige --example backend_matrix --features all-backends --release
```

Unavailable devices are skipped; present ones must match CPU logits (cosine > 0.99).

## Bench (OOM-aware)

Per-backend plans live in [`device_policy`](src/device_policy.rs): max_seq, bucketed
decode, and whether full BF16/F32 3B is allowed (wgpu/Vulkan → synth/GGUF only).

```sh
# Synthetic looped graphs (safe on every backend)
just features=all-backends bench-nanbeige-backends
just features=apple-silicon bench-nanbeige-backends

# Real 3B F32 (mmap-on-take; ~1× device-resident params + KV). MLX sets
# RLX_MLX_COMPILE_MAX_NODES=4096 so looped graphs stay Compiled (not Lazy).
just fetch-nanbeige
just features=apple-silicon bench-nanbeige-backends -- \
  --weights /tmp/rlx-weights/Nanbeige4.2-3B --device mlx
```

Overrides: `RLX_NANBEIGE_MAX_SEQ`, `RLX_NANBEIGE_PROMPT_LEN`,
`RLX_NANBEIGE_DECODE_TOKENS`, `RLX_NANBEIGE_MEM_BUDGET_BYTES`,
`RLX_MLX_COMPILE_MAX_NODES` (default 4096 via `prepare` when unset),
`RLX_SOFT_MEMORY_FRACTION` (default 0.95 on Metal/MLX via `prepare`).

Observed on Apple M4 Pro (synth looped graphs): **MLX** wins accelerator prefill/decode;
Metal prefill is much slower on short prompts; wgpu/Vulkan use a smaller portable graph
and are not used for full BF16/F32 3B (storage-buffer / RAM limits). Full F32 3B uses
safetensors mmap-on-take (no eager host drain) so peak RSS is roughly one device copy
(~16 GiB) plus KV — suitable on ≥32 GiB unified-memory Macs.

## Notes

- Advanced Nanbeige4.5 features in `modeling_nanbeige.py` (LoopSplit, mHC, depth attention, n-gram embeddings) are **not** enabled on the 4.2-3B config and are not implemented here.
- Community GGUF quants need the Nanbeige `llama.cpp` fork for loop-aware runtime; RLX accepts `general.architecture ∈ {llama, nanbeige}` and applies loop unrolling from HF `config.json` / GGUF `llama.num_loops` when present.
