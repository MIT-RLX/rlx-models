# CUDA / ROCm packed GGUF decode (Orpheus / Llama32)

Greedy parity path for large-vocab tied-embedding GGUF on NVIDIA/AMD GPUs.
Validated with `examples/cuda_token_probe` on Orpheus 3B Q4_K_M.

## Architecture

```
prefill (PackedGguf mode)
  └─ host-greedy: CPU F32 reference hidden + KV  (mmap weights, no 12 GiB drain)
decode (packed Q4 bucketed graph)
  └─ resident K/V on device between steps (lazy KV)
  └─ host tied-lm_head argmax (skip in-graph vocab matmul)
bucket rollover
  └─ flush missing rows from outgoing bucket → host cache
  └─ H2D bind padded K/V into wider bucket
```

Device packed prefill **hidden** and **KV** still diverge from CPU on Orpheus-scale
models today. Native prefill (`MetalGgufPrefillMode::PackedGguf` /
`ORPHEUS_CUDA_NATIVE_PREFILL=1`) therefore uses CPU F32 reference tensors for
seed while keeping weights mmap'd via `gguf_defers_f32_drain`.

## Recommended environment

```bash
ORPHEUS_BUCKET_DECODE=1
ORPHEUS_RESIDENT_KV=1
ORPHEUS_COMPILE_SEQ_CAP=128
RLX_CUDA_ARENA_POOL=0
ORPHEUS_CUDA_NATIVE_PREFILL=1   # TTS / parity-safe native prefill
```

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `ORPHEUS_CUDA_NATIVE_PREFILL` | off | Use `MetalGgufPrefillMode::PackedGguf` (mmap + reference CPU prefill for host-greedy). |
| `ORPHEUS_CUDA_F32_PREFILL=1` | off | Force full CPU F32 prefill-with-cache (slow, baseline). |
| `ORPHEUS_CUDA_LAZY_KV` | on | Keep K/V on GPU between decode steps; flush only at bucket boundaries. Set `0` to sync host every step. |
| `ORPHEUS_CUDA_KV_DEVICE_REBIND=1` | off | Defer evicting the outgoing bucket until after the wider bucket is compiled/rebound. Parity-safe; same H2D bind as default. |
| `ORPHEUS_CUDA_KV_HOST_REBIND=1` | off | Disables device-rebind mode (force default flush+bind). |
| `ORPHEUS_CUDA_NATIVE_DECODE=0` | on | Fall back decode to CPU (parity escape hatch). |
| `ORPHEUS_CUDA_PACKED_PREFILL=0` | on | Disable packed GPU prefill attempts in non-host-greedy paths. |
| `ORPHEUS_CUDA_GPU_KV=1` | off | Experimental: GPU KV + CPU F32 logits only. |
| `ORPHEUS_PREFILL_PERSIST=1` | off | Keep prefill graph resident alongside decode (high VRAM). |

## Key code paths

| Concern | Location |
|---------|----------|
| Deferred GGUF drain (enables host-greedy on CUDA) | `gguf_defers_f32_drain` in `generator.rs` |
| CUDA reference prefill seed | `seed_cuda_host_greedy_reference_prefill` |
| Bucket flush + resident bind | `maybe_flush_resident_kv_before_bucket`, `bind_resident_kv_from_host_cache` |
| Per-step KV fold-back | `feed_kv_row` in `rlx-cuda` `backend.rs` |
| Future D2D bucket seed | `copy_resident_kv_rows_from` in `rlx-cuda` (not wired in generator yet) |

## Parity probe

```bash
cargo build -p rlx-llama32 --example cuda_token_probe --features cuda --release
ORPHEUS_BUCKET_DECODE=1 ORPHEUS_RESIDENT_KV=1 ORPHEUS_COMPILE_SEQ_CAP=128 \
ORPHEUS_CUDA_NATIVE_PREFILL=1 \
ORPHEUS_GGUF_PATH=/path/to/orpheus.gguf ORPHEUS_STEPS=32 \
target/release/examples/cuda_token_probe
```

## Open performance work

- Wire `copy_resident_kv_rows_from` into bucket rollover (true D2D, no host H2D).
- Fix device packed prefill KV/hidden so reference CPU prefill is not required.
