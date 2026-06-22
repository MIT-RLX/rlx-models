# rlx-eagle3 end-to-end propose() benchmark

Run via:

```bash
cargo run -p rlx-eagle3 --release --features "mlx metal" \
    --example bench_propose_e2e -- \
    /Users/Shared/rlx-models/.eagle3-bench/weights/draft
```

## Setup

- Real weights: `RedHatAI/gemma-4-31B-it-speculator.eagle3` (4.47 GB safetensors)
- 3 warmup + 20 timed `propose(n=3)` calls
- Compiles **3 graphs** per backend (past_seq=0, 1, 2), each runs once per `propose()`
- KV cache maintained host-side across the 3 steps
- Greedy argmax sampling; d2t offset map applied per step
- Hardware: Apple M4 Pro, 64 GiB unified, macOS

## Results — tokens/second (higher is faster)

| Pipeline | tok/s | vs llama.cpp |
|---|---:|---:|
| llama.cpp b9606 EAGLE3 (full pipeline) | 6.10 | 1.00× |
| rlx-eagle3 scalar reference (`Eagle3DraftReference`) | 2.30 | 0.38× |
| rlx-eagle3 HIR · CPU | 5.32 | 0.87× |
| **rlx-eagle3 HIR · MLX** | **15.58** | **2.55×** |
| **rlx-eagle3 HIR · Metal** | **22.19** | **3.64×** |

## Key takeaways

1. **The HIR draft port is the right move.** 9.6× over the scalar reference on Metal, 6.8× on MLX, 2.3× on CPU. Per-step compile + run amortizes the fixed per-call cost across ~10 ops.
2. **rlx draft beats llama.cpp's full pipeline** on Metal (3.64×) and MLX (2.55×). This doesn't mean rlx beats llama.cpp end-to-end yet — llama.cpp's 6.1 t/s includes the **verifier** (Q4_K_M Gemma 4 31B), which we haven't wired into rlx-eagle3.
3. **MLX got dramatically faster.** Single-op MLX vs CPU was 2.65×; multi-op MLX is now 6.8× over scalar. Confirmed: the fix for MLX is *always submit multi-op graphs*. The host-gather refactor (no `Op::Gather` in the trace) was a 2× MLX speedup on top of that.
4. **Numerical correctness verified on CPU and MLX.** Both produce identical greedy proposals `[808, 1872, 236751]` from the same synthetic verifier-aux. Earlier `hir_parity` test pins HIR-CPU vs scalar at `logits max|Δ| = 1.12e-8` (essentially float associativity only).

## Known issues

### Metal numerical drift

Metal proposes a different sequence `[661, 51505, 43157]` vs CPU/MLX's `[808, 1872, 236751]`. This is the known Apple Silicon `simdgroup_float8x8` reduced-precision matmul issue — same drift that `rlx-gemma` workarounds with `RLX_METAL_PRECISE=1`.

Setting `RLX_METAL_PRECISE=1` on this bench **did not** fix it (still drifts to `[661, ...]`). The flag is read by `rlx-gemma`'s Metal sgemm path, not by the generic `Session::compile(Device::Metal)` route this bench uses. Plumbing the precise flag through rlx-runtime's Metal MIR lowering is the right fix — out of scope for this commit.

### past_seq=0 hits an MPS nil-device crash

Fixed in `build_draft_step_graph`: at past_seq=0 we skip the concat with past_k/past_v and use the new k/v directly as the KV cache. The graph input names stay consistent across all past_seq values so the runner doesn't need to special-case.

## Files

| Path | Purpose |
|---|---|
| `crates/rlx-eagle3/src/hir_draft.rs` | HIR builder for one draft step (~150 LOC) |
| `crates/rlx-eagle3/tests/hir_parity.rs` | CPU parity test: HIR vs scalar reference |
| `crates/rlx-eagle3/examples/bench_propose_e2e.rs` | End-to-end `propose(n=3)` across backends |
| `crates/rlx-eagle3/examples/bench_draft_step_backends.rs` | Single-step backend comparison |
| `crates/rlx-eagle3/examples/bench_lm_head_backends.rs` | Single-op micro-bench (lm_head matmul) |
| `crates/rlx-eagle3/BENCH_BACKENDS.md` | Single-op + multi-op backend analysis |

## Next steps (in order of payoff)

1. **Plumb `RLX_METAL_PRECISE` through `Session::compile`** so Metal numerical parity holds. Unlocks claiming Metal's 22 t/s as a real, parity-clean number.
2. **Wire HIR draft into `Eagle3Speculator::propose`** in place of the scalar reference. Currently `propose()` panics with `unimplemented!()`; the HIR runner from this bench is the replacement.
3. **Build verifier glue** (`rlx-gemma` → `VerifierHiddenSource`) so we can run a true end-to-end EAGLE3 pipeline (verifier + draft) and compare apples-to-apples vs llama.cpp's 6.1 t/s.
4. **f16/bf16 sweep on the full step.** Single-op dtype change didn't help; multi-op may.
5. **MLX-side fix for `Op::Gather` host-eval** — add to `first_host_eval_op` so future graphs that include gather don't panic, just gracefully fall back to Lazy.
