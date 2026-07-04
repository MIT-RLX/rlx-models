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
- `packed_gguf_execution_device(device)` — native CPU/Metal/MLX; wgpu/CUDA → CPU prefill unless `PackedGguf` mode

### CUDA / ROCm (Orpheus TTS)

Greedy packed decode + native prefill are documented in [docs/cuda-gguf-decode.md](docs/cuda-gguf-decode.md).
Quick probe: `examples/cuda_token_probe` with `ORPHEUS_CUDA_NATIVE_PREFILL=1`.

See [README.md](../../README.md) gotchas and [crates/rlx-minicpm5/README.md](../rlx-minicpm5/README.md).

## Decode

Incremental (KV-cache) decode compiles one graph per **shape bucket** (a
power-of-two `past_seq` ladder via `BucketedCompileCache`) so one compiled
graph serves every position in its range. `Llama32Generator::step`/`step_cached`
dispatch (priority order) picks:

1. **host-greedy** (`decode_step_greedy_host`) — greedy sampling + tied
   embeddings on a GGUF quant checkpoint: runs a hidden-only graph and does the
   `lm_head` argmax on the host (skips the in-graph vocab matmul). Gated by
   `llama32_host_greedy_lm_enabled`; opt out with `RLX_LLAMA32_GRAPH_LM_HEAD=1`.
   The caller's per-step logit adjustment (via `step_cached_adjust`, e.g.
   Orpheus's SNAC-slot mask + repetition penalty) **is applied** before the host
   argmax — skipping it was a bug that derailed structured (audio-codebook)
   greedy decode into non-speech. Fastest whisp-correct Orpheus/Metal path
   (RTF ~19.5 vs ~22 for the in-graph lm_head).
2. **resident** (`decode_step_bucketed_packed_resident`) — Metal/Vulkan/CUDA:
   K/V lives in the device arena across steps (fed in place, logits-only
   readback). Opt out with `ORPHEUS_RESIDENT_KV=0`.
3. **packed / dynamic / bucketed / oneshot** fallbacks.

### Cross-utterance decode-graph reuse

By default the generator **keeps compiled decode-bucket graphs (and their
uploaded weights) resident across utterances** — `prefill` drops only the
per-utterance K/V bindings (via `soft_release_decode_kv_bindings`), not the
graphs. A warm/2nd+ utterance then skips the multi-second per-bucket recompile +
weight upload. Measured on Orpheus 3B Q4_K_M (Metal, 64 GB): warm-utterance
RTF **27.6 → 22.3** (~19%).

It is **RAM-adaptive**: `trim_decode_buckets_to_budget` evicts the
highest-index bucket (used last / recurring least in a monotonic `past_seq`
sweep) whenever keeping one more would exceed the soft memory budget — a roomy
machine keeps the whole ladder (zero cross-utterance recompiles), a constrained
one keeps as many as fit and recompiles the rest. Scoped to packed (Q4) decode
on non-CUDA backends; CUDA/ROCm D2D-K/V modes keep their per-utterance release.

### Env flags

| Flag | Effect |
| --- | --- |
| `ORPHEUS_KEEP_DECODE_GRAPHS=0` | Disable cross-utterance reuse (full release each `prefill`, the old behavior). |
| `RLX_DECODE_BUCKET_RESIDENT_BYTES` | Per-bucket steady-state resident bytes for the budget math (default ~2 GiB; set for non-3B models). |
| `RLX_SOFT_MEMORY_FRACTION` / `RLX_SOFT_MEMORY_BUDGET_BYTES` | Soft RAM cap (default 80% of physical) the trim respects. |
| `ORPHEUS_METAL_NATIVE_DECODE=1` | Decode on Metal (else CPU-prefill GGUF falls back to CPU decode). Pair with `--metal-prefill packed`. |
| `RLX_LLAMA32_GRAPH_LM_HEAD=1` | Force the in-graph (packed `DequantMatMul`) lm_head instead of host-greedy (both are whisp-correct; host-greedy is faster). |

See [README.md](../../README.md) gotchas and [crates/rlx-minicpm5/README.md](../rlx-minicpm5/README.md).

## See also

- [README.md](../../README.md)
- [AGENTS.md](../../AGENTS.md)
- [docs/cuda-gguf-decode.md](docs/cuda-gguf-decode.md)
