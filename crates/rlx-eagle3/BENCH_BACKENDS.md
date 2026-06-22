# rlx-eagle3 backend micro-bench — `lm_head @ x`

Run via:

```bash
cargo run -p rlx-eagle3 --release --features "mlx metal" \
    --example bench_lm_head_backends -- \
    /Users/Shared/rlx-models/.eagle3-bench/weights/draft
```

## Setup

- Real weight: `lm_head.weight` from `RedHatAI/gemma-4-31B-it-speculator.eagle3`
- Shape: `[V_draft=32000, H_draft=5376]` → 656 MB f32
- Op: `mm(x[1, B, 5376], W[5376, 32000])` → `logits[1, B, 32000]`
- 5 warmup + 100 timed iterations
- Hardware: Apple M4 Pro, 64 GiB unified, macOS

## Per-row µs (lower = faster)

### CPU
| Batch | f32 | f16 | bf16 |
|---|---:|---:|---:|
| b=1 | 30863 | 38715 | 34646 |
| b=16 | 3543 | 2336 | 2141 |

### Metal
| Batch | f32 | f16 | bf16 |
|---|---:|---:|---:|
| **b=1** | **3005** | 3061 | 3079 |
| **b=16** | 205 | 238 | **201** |

### MLX
| Batch | f32 | f16 | bf16 |
|---|---:|---:|---:|
| **b=1** | 10820 | **9510** | 13687 |
| **b=16** | **803** | 977 | 838 |

## What moves the needle — and what doesn't

### What we tried, what changed

| Knob | MLX impact | Verdict |
|---|---|---|
| Larger batch (b=16) | Same ~12 ms per-call wall-clock as b=1 | MLX has a **fixed per-call floor** ~8 ms that doesn't amortize |
| f16 weight+input | b=1: 10.8 → 9.5 ms (1.14×) · b=16: slower | Marginal at most |
| bf16 weight+input | Slower at b=1, neutral at b=16 | No improvement |

### Why MLX hits a per-call floor

Each MLX `run()` does the same kernel-dispatch + sync + readback regardless of dtype or batch:

1. **Process-wide `runtime_guard()` mutex** (uncontended, ~ns).
2. **Build the input/param `Array` leaves** — params go zero-copy (`from_f32_slice_view`), the 21 KB input gets a copy (~µs).
3. **`compiled.invoke(&leaves)`** — submits the compiled trace to MLX. *This is where the wall clock disappears.*
4. **`to_f32()` on the output** — forces MLX eval/sync + read back 128 KB (32000 × 4 B).

Steps 1–2 are sub-µs. Steps 3–4 are MLX runtime work, opaque to us. **No knob in `rlx-eagle3` or `rlx-mlx` shrinks them.**

### Why Metal MPS doesn't have this floor

MPS dispatches synchronously and has highly tuned matrix-vector kernels (special `simdgroup_float8x8` paths for skinny matmul). MLX's matmul is generic GEMM-based and pays a kernel-dispatch latency on Apple Silicon that's larger than MPS's. For *one* op, MPS wins; for fused multi-op graphs, MLX's compile model would catch up.

## Implications for the EAGLE3 draft port

The draft forward has **~10 matmuls per step** (q/k/v/o + gate/up/down + lm_head + fc + maybe verifier_norm). If MLX's per-call floor really is fixed at ~8 ms, then a **10-op forward on MLX would pay 80+ ms in dispatch overhead alone** — worse than CPU.

But that's the *current* result for f32 unfused matmul. The MLX win comes from **kernel fusion via `mx.compile`** — when the full forward is captured as one compiled trace, MLX runs the whole graph in one dispatch. The ~10-op floor collapses to one floor.

The bench above gives the **worst case for MLX** — a single unfused matmul. Until we have the full HIR draft forward (task #22) we won't know how much fusion recovers. **The current data does *not* mean MLX is wrong for EAGLE3 draft work — it means single-op micro-benches penalize MLX disproportionately.**

## What we'd need to actually improve MLX on this op

Concrete next steps, in order of effort:

1. **Capture `propose()` end-to-end as one compiled MLX trace.** Currently every matmul is its own `Session::compile` round-trip. With the full forward in one graph + `MlxExecutable::warm_compile`, the per-call floor pays once.

2. **Verify `Array::from_f32_slice_view` for the param.** It's used (`build_leaf_for` checks `dtype == F32` + length match). Confirm the underlying `rlx_mlx_array_from_data_view` actually pins the buffer for GPU residency without re-uploading.

3. **Profile MLX's matmul kernel for `(1, K) @ (K, N)` shapes.** The Apple-Silicon-tuned kernel paths in MLX may not cover this skinny shape — they're optimized for `(B*S, K) @ (K, N)` with large B*S. An MLX issue / PR to expose a `matvec` op would be the right escalation.

4. **Try `MlxMode::AsyncCommit`** — defers the eval/wait. Only helps if `propose()` chains multiple ops between readbacks.

## Verdict for now

For the EAGLE3 draft port (task #22):
- **Target Metal first** — it gives 10× over CPU on the dominant op, today, no tuning needed.
- **Re-bench MLX once the full forward compiles in one trace.** Single-op MLX is the worst case; multi-op fused MLX may close the gap.
- Don't conclude "MLX is bad" from this micro-bench — conclude "MLX needs multi-op submissions to amortize its per-call floor."

This is what real-weight micro-benches give you: a clear ordering of where to look first.
