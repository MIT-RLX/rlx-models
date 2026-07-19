# Bonsai-27B backend bench (post Q1_0 GEMM + packed fallback fix)

Complex prompt (~597 chars → **155** chat-template tokens), `max_tokens=16`, greedy,
`RLX_KV_CACHE_MAX_RESIDENT=1`, `RLX_QWEN35_BENCH=1`, `RLX_LOW_MEM_COMPILE=1`.

Hardware:
- **Mac** `mm4.local`: Apple M4 Pro (Metal / MLX / wgpu / CoreML / Vulkan)
- **MSI** `msi.local`: NVIDIA GeForce RTX 3080 Ti Laptop GPU 16GB (CUDA)

## Packed GGUF device routing (updated)

`packed_gguf_execution_device` keeps the **requested** backend by default for
CPU / Metal / MLX / CUDA / ROCm / wgpu / **Vulkan** / **CoreML**. Opt into CPU
with `RLX_PACKED_GGUF_{WGPU,VULKAN,COREML}_HOST=1`.

CoreML Q1_0 defaults to `RLX_COREML_Q1_MODE=lut` (≈3.8 GiB weight.bin for
Bonsai-27B; F32 unfold OOMs disk).

| backend | host | prompt_tok | new_tok | prefill_ms | decode_ms | ms/tok | tok/s | notes |
|---------|------|------------|---------|------------|-----------|--------|-------|-------|
| **cuda** | MSI RTX 3080 Ti | 155 | 16 | **15860.6** | **8719.6** | **545.0** | **1.835** | matches MLX token-for-token |
| mlx | Mac | 155 | 16 | **8498.7** | 21886.7 | 1367.9 | 0.731 | fastest prefill; matches CUDA tokens |
| metal | Mac | 155 | 16 | 30522.3 | 19239.9 | 1202.5 | 0.832 | `"the answer is 1000…"` then zeros |
| gpu (wgpu) | Mac | 155 | 16 | 94476.9 | 30879.6 | 1930.0 | 0.518 | sharded 20 GiB → 5×4 GiB; coherent think text |
| **vulkan** | Mac | 1 (Hi) | 2 | **15634** | **1606** | 803 | **1.25** | **native** (no CPU redirect); tokens `[271, 248068]` match Metal |
| coreml | Mac | 1 (Hi) | — | — | — | — | — | **native** (no CPU redirect); Lut bake 3.8 GiB; Espresso AOT still compiling after 17 min on first load |

## Ranking (decode tok/s, complex prompt — prior matrix)

1. **CUDA** 1.835 tok/s — MSI
2. **Metal** 0.832 tok/s — Mac
3. **MLX** 0.731 tok/s — Mac
4. **wgpu** 0.518 tok/s — Mac (sharded)

## Token agreement (first 16 ids, complex prompt)

| backend | text prefix |
|---------|-------------|
| CUDA / MLX | `<think>\nHere's a structured brief that: (1) contrasts latency vs` |
| wgpu | `Wait, the user wants a structured brief that: (1) contrasts latency` |
| Metal | `the answer is 100000000000` (repeats) |

## Reproduce

```bash
# Mac: metal + mlx + gpu + vulkan + coreml (native packed)
scripts/matrix/bonsai27b_bench.sh

# Force old CPU redirect if needed:
# RLX_PACKED_GGUF_VULKAN_HOST=1 / RLX_PACKED_GGUF_COREML_HOST=1

# MSI CUDA
bash scripts/matrix/sync_to_msi.sh
scripts/matrix/bonsai27b_bench.sh --remote-cuda
```
