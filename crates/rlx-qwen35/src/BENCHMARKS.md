# Qwen3.5 performance notes

Hardware notes below are from development machines (Apple Silicon and NVIDIA RTX). Re-measure locally with `RLX_QWEN35_BENCH=1`.

## Quick CLI bench

```bash
RLX_QWEN35_BENCH=1 cargo run -p rlx-qwen35 --release --features apple-silicon -- \
  --weights model.gguf --packed --device metal --fast \
  --temperature 0.0 --seed 0 --max-tokens 16 \
  --prompt "What is the capital of France?"
```

`--fast` sets `prefill_seq` to the prompt length and a tight decode `max_seq`.
On short contexts (≤128), Metal / MLX / CUDA keep the prefill arena and warm
one decode bucket automatically.

## Methodology

| Harness | Command |
|---------|---------|
| Real GGUF batch=1 (CPU/Metal/MLX) | `QWEN35_GGUF_PATH=… cargo test -p rlx-models --test qwen35_backend_gguf_bench --features "metal,mlx" --release -- --nocapture` |
| Real GGUF heterogeneous batch=2 | `QWEN35_GGUF_PATH=… cargo test -p rlx-models --test qwen35_batch_gguf_bench --features "metal,mlx" --release -- --nocapture` |
| Real GGUF batch=2 check | `cargo test -p rlx-models --test qwen35_batch_gguf_quick_check --release -- --nocapture` |
| Real GGUF VLM check | `QWEN35_GGUF_PATH=… QWEN35_MMPROJ_PATH=… cargo test -p rlx-models --test qwen35_vlm_gguf_quick_check --features qwen35-vlm --release -- --nocapture` |
| Forward CLI | `cargo run --release -p rlx-models --example qwen35_forward_bench --features "metal,mlx" -- /path/to/model.gguf --device cpu --packed --tokens 16` |
| Synthetic (6-layer toy) | `cargo bench -p rlx-models --bench qwen35_inference` |

Steady-state generate: 16 tokens after 4-token warmup, prompt `[1..=8]`, packed Q4_K_M.

### Heterogeneous batch=2

Two prompts with different lengths (8 vs 7 token ids), batch=2 runner, packed Q4_K_M:

| Metric | Meaning |
|--------|---------|
| `prefill b2` | `predict_logits_batch` steady-state for both rows |
| `decode b2 uniform agg` | 32 total tokens (16×2) / wall time |
| `decode b2 per-row-limits agg` | row0=16 tok, row1=8 tok (24 total) / wall time |
| `eff` | `agg_b2 / (2 × tok/s_b1)` — 1.0 = perfect 2× batch scaling |

Example output line:

```text
qwen35 het-batch bench Cpu: prefill b1=…ms b2=…ms | decode b1=… tok/s/stream | decode b2 uniform agg=… tok/s (…/stream, eff=…x) | …
```

## Tier C optimizations (2026-05-19)

| Item | Change |
|------|--------|
| **C.11** | Fused tiled GGUF dequant+matmul (`rlx-cpu/src/gguf_matmul.rs`) — no full F32 weight cache |
| **C.10** | GDN decode/prefill BLAS path (`sgemv`/`sger`/`sscal`, n≤128) |
| **C.9** | Parallel GDN prefill: time-outer loop, Rayon over heads |
| **C.8** | MLX default `Compiled` + `warm_compile` (override: `RLX_MLX_MODE=lazy`) |
| Metal dequant | Fused CPU path on unified memory (opt-in GPU: `RLX_METAL_DEQUANT_GPU=1`) |

## Baselines (0.8B Q4_K_M, pre-Tier C — replace after re-bench)

| Backend | Prefill steady (3 tok) | Generate tok/s |
|---------|------------------------|----------------|
| CPU | ~310 ms | ~2.35 |
| Metal | ~1225 ms | ~1.08 |
| MLX | ~3521 ms | ~0.99 |

Re-run the env-gated bench test after placing GGUF at `/tmp/rlx-models/Qwen3.5-0.8B-Q4_K_M.gguf`.

## Why it was slow (before Tier C)

1. Full-matrix GGUF dequant on every matmul (~150 packed params × 24 layers).
2. Scalar GDN inner loops (n=128) instead of BLAS.
3. MLX re-lowering every `run()` when GDN forced lazy mode.
4. Metal GPU full dequant scratch + MPS sync on decode-sized matmuls.
5. Per-`past_seq` decode graph compile + weight upload amortization.

Expected post-Tier C: **CPU 5–15 tok/s**, Metal/MLX closer to CPU on decode (still bounded by graph dispatch until fused Metal GDN lands).

## Qwen3.8-27B on Metal — quant choice dominates decode

Apple M4 Pro (14-core, 64 GB, ~273 GB/s), `--device metal --packed`,
`max_seq=32`, 12 tokens. **Steady state = mean of steps 2+** from
`RLX_QWEN35_DECODE_TRACE=1`; step 1 is a ~1.9 s cold outlier that skews any
`tok/s` averaged over a short run.

| GGUF | size | decode | vs Q3_K_S |
|------|------|--------|-----------|
| `Q3_K_S` | 12.57 GB | 232–248 ms/tok (**4.0–4.3 tok/s**) | — |
| `Q4_K_M` | 17.10 GB | 152–162 ms/tok (**6.2–6.5 tok/s**) | **~1.5×** |

**The larger file decodes ~1.5× faster.** `dequant_gguf.msl` documents why:
Q3_K's GEMV is *dequant-ALU bound* at **~48 GB/s** vs Q4_K's **~185 GB/s**
(Q6_K ~161) — its 2-bit-shift + hmask-bit-test + offset per value is ~4× the
arithmetic of a 4-bit nibble, and a "byte-once" rewrite measured *slower*. The
fused and simdgroup Q3_K kernels are already default-on, so **the kernel is at
its floor; the quant is the lever.** Byte mix: `Q3_K_S` is 84% Q3_K, while
`Q4_K_M` is 67% Q4_K / 27% Q6_K / 6% Q5_K.

Same ordering holds on Qwen3.5-0.8B (same arch/kernels), where bigger is again
faster: Q3_K_S 25.8 → Q4_K_M 28.1 → Q6_K **41.7 tok/s**.

### After the quant switch, dispatch count dominates

Solving the two measurements for a quant-independent term gives a floor of
**~97 ms/token** — which matches the ~2 480 GPU dispatches/token in the
`RLX_METAL_THUNK_PROFILE=1` histogram at ~40 µs each. That is ~42% of Q3_K_S
decode but **~63% of Q4_K_M decode**, so op-fusion / dispatch reduction is the
next lever, not matmul tuning. (`RLX_USE_ICB=1` pre-encodes thunks but only
bought ~3% here — it attacks encode, which was never the bottleneck.)

Caveat: `RLX_METAL_THUNK_PROFILE=1` GPU-syncs per thunk and reported matmul as
89% of decode. That is an artifact — the unprofiled split is ~58% matmul /
~42% fixed. **Size the fixed cost from paired measurements, not the profile.**

### TTFT: Q4_K_M pays a paging tax on a RAM-tight box

Prefill for `Q4_K_M` measured 13–38 s across runs vs a stable 5.3–6.1 s for
`Q3_K_S`. This is **not** a kernel regression — both files have byte-identical
tensor *shapes*, so the f32 dequant scratch is the same size, and disabling
either fused prefill GEMM (`RLX_METAL_Q4K_GEMM_DISABLE` /
`RLX_METAL_Q6K_GEMM_DISABLE`) or forcing MPS (`RLX_METAL_Q4K_GEMM_MAX_M=8`)
made prefill *worse*, so those paths are already winning. `/usr/bin/time -l`:

| | peak RSS | page faults | sys |
|---|---|---|---|
| `Q3_K_S` | 24.12 GB | 486 k | 15.9 s |
| `Q4_K_M` | 22.95 GB | **1 395 k** | **30.1 s** |

Peak RSS is *lower* for Q4_K_M; it just takes 2.9× the page faults streaming a
17.1 GB mmap while ~25 GB of swap was already committed. Re-measure TTFT on a
machine with real headroom before treating it as a cost of the quant.

### Other levers measured

- `--aot-cache DIR` — recompile 4.77 s → 1.49 s on re-run (0.8B). The 27B
  spends ~26 s of a ~95 s cold start in compile.
- `--fast` — sets `prefill_seq` to the prompt length instead of padding to
  `max_seq`. Note it changes prefill GEMM shape, so greedy output can differ
  from a padded run at near-tie steps.
- `--spec-decode` — **not usable at 27B.** `build_runner` constructs two
  independent runners from the same path and there is no shared weight store,
  so draft + target duplicate the full weight set (~25 GB at Q3_K_S, ~34 GB at
  Q4_K_M). Making MTP speculation practical at this size needs the two runners
  to share the packed weights.

### Per-kernel GEMV bandwidth (the honest way to target kernels)

Decode is weight-streaming, so a GEMV kernel's achieved GB/s *is* the token
rate. `cargo test -p rlx-metal --release --test gemv_bandwidth -- --nocapture`
measures each K-quant directly at a 27B FFN shape — a whole-model number can't
(at model scale it is confounded by paging; on a small model everything is
launch-bound). 8 matmuls share one graph to amortise the ~0.2 ms per-`run()`
overhead, and the reported figure is the **min** over iterations.

M4 Pro, `m=1`, `K=5120`, peak ~273 GB/s. `Q8_0` (trivial dequant, ~82%) is the
practical ceiling for this access pattern:

| quant | n=17408 | n=1024 |
|-------|---------|--------|
| Q8_0 | 226.0 | 118.5 |
| Q6_K | 223.1 | 133.1 |
| Q4_K | 205.4 | 82.4 |
| **Q5_K (was)** | **90.1** | **19.0** |
| **Q5_K (now)** | **144.6** | **70.4** |
| Q3_K | 57.7 | — |

**Q5_K was the only hot K-quant without a simdgroup GEMV** — one thread per
output row, so it was occupancy-starved: less than half of Q4_K/Q6_K at n=17408
and **6× slower at n=1024**, which is exactly where Qwen3.5/3.8 put it
(`attn_qkv`, `ssm_out`). Added `q5k_mv_f32_sg` (32 threads per row via
`simd_sum`, `NSG=4`/`NR0=2`, modelled on `q3k_mv_f32_sg`): **1.60× at n=17408
and 3.71× at n=1024**. Off-switch `RLX_METAL_Q5K_SG_DISABLE=1`; parity test
`metal_q5k_mv_sg_parity.rs` covers aligned and tail (`n=1030`) shapes against
both the scalar kernel and CPU.

**End-to-end (Qwen3.5-0.8B, 200-token runs, warm-up dropped, quiet machine):
11.67 → 10.02 ms/tok = ~1.15×**, output token-identical with the kernel on or
off. That matches the predicted ~1.13× from Q5_K's ~20% byte share of this
model. Discard interference outliers when reading these: one run in each arm
was hit (scalar 20.41, SG 23.02) while the clean values repeated exactly.

**Reference gap, both arms measured on a quiet machine** (`llama-bench -n 128
-ngl 99`): llama.cpp **170.75 ± 1.31 tok/s** vs rlx **85.7 before / 99.8 after**
— i.e. 1.99× → **1.71× behind**. An earlier "llama is 1.65× faster (48.41 vs
29.4)" figure in this file's history was taken while the box was loaded; both
numbers were depressed ~3×, so only re-measure ratios on an idle machine.

Q3_K stays at its documented ALU-bound floor. **Do not read a single run of this
bench as gospel**: Q4_K measured 168.7 then 199.6/205.4 across runs, and on the
first pass that made Q4_K look 22% slower than Q6_K — it is not, they are equal.
Re-run before optimising anything.

### Decode TPS tuning — what actually helps (measured, M4 Pro)

Everything below was A/B'd with `RLX_QWEN35_DECODE_TRACE=1` (steady state =
mean of steps 2+) **and checked for token-identical output**. A speedup that
changes the tokens is not a speedup.

| Lever | Effect | Verdict |
|-------|--------|---------|
| `Q4_K_M` instead of `Q3_K_S` | **1.51×** (4.30 → 6.49 tok/s, 27B) | **use it** |
| Right-size `--max-seq` | ~5% (4096 → 64: 37.96 → 35.99 ms/tok, 0.8B, 2 reps) | minor |
| `RLX_METAL_SDPA_FLASH_DECODE=1` | 46.4 vs 35.8 ms/tok | slower |
| `RLX_QWEN35_FAST_GREEDY_LM=1` | 145.3 vs 35.8 **and changes tokens** | do not use |

**What the default-on Metal optimizations are worth** (measured by *disabling*
each; mean vs ~34.7 ms/tok defaults, ±7% noise floor, 0.8B @ ctx 64):
`FUSE_SSM=0` → 39.77 (**~+15%, real**); `GPU_KV=0` → 36.27 (+4.5%);
`GQA_NATIVE=0` → 36.12 (+4.1%); `INPLACE_KV=0` → 34.44 (nil). The last three sit
inside the noise band *at ctx 64* — `runner.rs` documents GPU_KV as "+42% @4K,
+68% @8K (grows with ctx)" and GQA_NATIVE as "+17% @8K", which is consistent.
**There is no flag left to flip: the good ones are already on by default.**

> **Do not "test" a default-on flag by setting it to `1`.** On Metal,
> `runner.rs` already force-enables `RLX_QWEN35_GPU_KV`, `INPLACE_KV`,
> `FUSE_SSM` and `GQA_NATIVE` (`if var.is_err() { set_var("1") }`), so setting
> them to `1` is a **no-op** and any delta you measure is pure noise. To value
> one of them, set it to `0`. An earlier revision of this file reported these
> four as "slower" from exactly that mistake; the numbers were noise between
> identical configurations.

**Noise floor on this box (4 runs, identical config):** mean 33.94–38.67 ms/tok
(**±7%**), min 15.65–23.09 (**±20%**). The mean is the stabler estimator — the
min is an extreme order statistic and swings more, not less. Treat anything
under ~15% as unresolved unless you run many pairs.

### Fused ggml `L2_NORM` — implemented, and what it did NOT buy

`builder::l2_norm` expands to six thunks (`mul → reduce → copy → sqrt →
max → div`), run twice per Gated-DeltaNet layer. `fuse_l2_norm` in rlx-metal now
collapses each chain into one `L2NormLastDim` dispatch (off-switch
`RLX_METAL_FUSE_L2NORM=0`, parity test `metal_fused_l2_norm_parity.rs`).

Verified on the 0.8B: **36 chains fired** (2 × 18 GDN layers), output unchanged,
and the profile confirms the exact op deltas — `binary` 102→66, `reduce` 36→**0**,
`copy` 54→18, `activation` 54→18, `binary_broadcast` 72→**0**, plus 36
`l2_norm_lastdim`. That is **216 dispatches → 36, i.e. 766 → 586 per step (−23.5%)**.

**The speedup did not follow.** Six alternating 200-token pairs (180 steps
averaged, warm-up dropped): unfused median 35.62 ms/tok, fused median 33.97 —
**1.048×, with pairwise ratios scattered 0.83–2.61 and an unfused spread of
3.48×**. Fused won 4 of 6 pairs. A ~5% median gain that this machine cannot
resolve.

**So the "dispatch count is the currency" model is wrong**, and the earlier
arithmetic that ~2 480 dispatches × ~40 µs ≈ the 97 ms floor was a coincidence,
not a causal fit: removing 23.5% of dispatches bought ~5%, not ~20%. The ops
this pass removed were cheap in *both* GPU time (~4.5%) and launch cost. The
floor is set by the expensive dispatches (GDN, norms, attention, and the matmul
launches), so the next attempt should target those, not the elementwise tail.

The pass is kept enabled: it is parity-tested, strictly does less work (one
kernel instead of six), never lost badly (fused max 38.53 vs unfused max 86.71),
and showed a tighter run-to-run spread. But its benefit is **unproven**, so
treat it as hygiene rather than a win.

### The 1.5× that is on the table but not reachable yet

Decode is dispatch-bound: **1 764 thunks per token** on the 24-layer 0.8B
(~73 ops/layer), against a ~97 ms/token quant-independent floor on the 27B.
`RLX_METAL_CONCURRENT_NOBARRIER=1` — the deliberate racy ceiling probe — runs at
**22.90 vs 34.92 ms/tok (1.52×)**, so dispatch overlap is worth ~1.5×. The
correct barriered path captures only **1.03×**: `RLX_METAL_CONCURRENT_STATS=1`
reports *1 764 thunks, 25 encoders, 579 barriers* — a third of dispatches are
barrier-gated, so almost nothing overlaps.

**Both opt-in dispatch paths currently produce WRONG OUTPUT on qwen35** (both
are off by default, so this is latent, not live):

- `RLX_USE_ICB=1` → all-zero token ids. This is exactly the silent failure
  `icb.rs` warns about in its own header (buffer bound by a stale index ⇒ `len`
  reads 0 ⇒ every thread early-returns ⇒ command buffer completes having
  written nothing, with no error).
- `RLX_METAL_CONCURRENT=1` → different tokens than serial, so
  `concurrent_barrier_set` is missing a hazard.

**Reference point: llama.cpp is ~1.65× faster on the identical file.**
`llama-bench -m Qwen3.5-0.8B-Q4_K_M.gguf -n 128 -ngl 99` → **48.41 ± 6.65 tok/s**
vs rlx's ~29.4 (33.97 ms/tok median). So the headroom is real and achievable,
and it is close to the 1.52× dispatch-overlap ceiling below — suggestive, but
the mechanism is unproven; llama may win for unrelated reasons.

**ROOT CAUSE FOUND (one instance fixed): multi-dispatch thunks.**
`concurrent_barrier_set` reasons *between* thunks, but several encode helpers
emit **more than one dispatch for a single thunk** with a RAW dependency between
them — invisible to that analysis, and unfixable by any amount of fencing
between thunks (which is why `FENCE_ALL`, `SPLIT_ENC` and `BARRIER_FRESH` all
failed). The confirmed instance is `encode_activation_out`'s fallback:

```rust
// Fallback: copy then in-place (still one schedule node; two dispatches).
encode_copy(enc, …);        // writes dst
encode_activation(enc, …);  // reads AND writes dst, in place
```

Serial ordering makes that correct; a Concurrent encoder lets the in-place
activation read `dst` before the copy lands.

Found by per-thunk bisection: `RLX_METAL_CONCURRENT_IDX_LO/HI` (with
`SPLIT_ENC=1`) select which thunk indices get a Concurrent encoder, so the
window can be binary-searched. `[0,101)` OK → `[0,102)` broke → `[101,102)`
alone broke, and thunk 101 is `activation_out`.

**Fix:** `intra_thunk_barrier()` + a `CONCURRENT_ENCODER` thread-local, called
between the copy and the in-place activation. It is a **no-op on Serial**
encoders (where `memoryBarrierWithScope:` is not even valid), so the default
path is provably unaffected — verified byte-identical output plus all tests
green. After it, `[101,102)` passes and full concurrency improves from wrong at
token 0 to wrong at token 6.

**Not finished:** more instances remain. `[0,102)` is OK and `[0,104)` breaks,
yet neither `[102,103)` nor `[103,104)` breaks alone — so at least one remaining
case involves an interaction rather than a lone thunk. **The failure is racy:
repeat each bisect config (the harness runs 3×) or you will get non-monotonic
nonsense.** Auditing every helper that issues 2+ dispatches for internal RAW
dependencies is the systematic finish.

**Earlier framing, now superseded — the defect is not dispatch ordering:**
`RLX_METAL_CONCURRENT_SPLIT_ENC=1` (new, default off) closes the encoder before
every thunk, so each encoder holds exactly ONE dispatch — concurrency is then
impossible and encoder boundaries give total ordering. Results:

| config | output |
|--------|--------|
| Serial, no split | `[279, 6511, 314, 9338, 369, 279]` ✓ |
| **Serial + SPLIT_ENC** (control) | **identical ✓** — splitting into ~1 764 encoders is sound |
| **Concurrent + SPLIT_ENC** | **wrong ✗** |

Same encoders, same dispatches, same order — the only difference is the
`MTLDispatchType` argument at encoder creation. So the whole hazard-analysis
surface is irrelevant, and the question reduces to: *why does an encoder created
with `dispatchType = Concurrent` compute a different result for a single,
fully-ordered dispatch?* Both fusion parity tests **pass** under
`RLX_METAL_CONCURRENT=1`, so simple graphs are fine — something in the qwen35
graph is sensitive to the encoder's dispatch type itself.

**Seven hypotheses tested and ALL disproven** (do not re-try these):

- *MPS matmuls not closing the concurrent encoder* — `--fast` drops prefill to
  m=5, routing to the fused GEMM instead of MPS. Still wrong.
- *Arena slot reuse creating WAR hazards* — `RLX_ARENA_NO_REUSE=1` is wrong too
  (in fact worse).
- *Deferred `narrow_batch` dispatch* — retested properly with
  `RLX_METAL_NARROW_BATCH=1` (inverted name: `=1` DISABLES batching). The earlier
  attempt fenced the narrow's *original* index, where nothing is dispatched, so
  that test was meaningless. Still wrong with batching off.
- *Missing cross-encoder barrier* — the code assumes a fresh encoder is
  implicitly ordered after the previous one. `RLX_METAL_CONCURRENT_BARRIER_FRESH=1`
  barriers on newly-opened encoders too. Still wrong.
- *Ops that internally issue multiple dispatches racing* (split-K partials +
  reduction) — `RLX_METAL_GEMV_SPLITK=0` and `RLX_METAL_SDPA_SPLITK=0`. Still wrong.

Bisected with `RLX_QWEN35_DEBUG_LAYERS=1`: layers 00 and 01 are **bit-identical**
to serial, and layer_02 diverges by 3.6% (min −1.3454 vs −1.2971) — a sudden
break, not accumulation. Small graphs (both fusion parity tests) pass under
`RLX_METAL_CONCURRENT=1`, so the mechanism works in isolation; something in the
full qwen35 graph from layer 2 onward triggers it.

The two original hypotheses, also disproven:

1. `narrow_batch` deferring `Narrow`/`SplitLastAxis` past the point the in-order
   scan assumes — fencing both variants in `concurrent_barrier_set` did not
   restore correct output.
2. An incomplete read set in some `mlp_io` arm — `RLX_METAL_CONCURRENT_FENCE_ALL=1`
   (added in `concurrent_barrier_set`, default off) fences *every* thunk,
   degenerating Concurrent to serial ordering. `CONCURRENT_STATS` confirms it
   emits 1 608 barriers over 1 764 thunks — and output is **still wrong, and
   still non-deterministic run to run** (`103480…`, `369…`, `483…`).

A barrier before every dispatch orders nothing, so `concurrent_barrier_set` is
exonerated. `concurrent` only changes three things (dispatch type, barrier set,
and a segment skip that is a no-op because ICB segments are only compiled under
`RLX_USE_ICB`), so the defect is in the **Concurrent encoder mechanism itself**:
`memoryBarrierWithScope: MTLBarrierScopeBuffers` is not restoring ordering for
this graph. The `msg_send!` target was checked and is correct (identical to the
crate's own `as_ptr()`, and the `Ref` type `unsafe impl`s `Message`). Prime
suspect: ops that do not dispatch on the shared compute encoder (MPS matmuls,
deferred host ops) not properly closing the concurrent encoder. Debug the
encoder, not the hazard set.

Bisection hooks left in `concurrent_barrier_set` (both default off, so no effect
on the serial path): `RLX_METAL_CONCURRENT_FENCE_ALL=1` and
`RLX_METAL_CONCURRENT_OPAQUE=<thunk_name>,…`.

Also unverified: **`--batch 2` changes the tokens.** On the 0.8B, batch=1 gives
`[279, 6511, 314, …]` (CPU and Metal agree on step 1) while batch=2 gives
`[191973, 401, 1147, …]` for the same prompt. Batched decode would otherwise be
near-free aggregate throughput on a latency-bound decode — worth fixing, but do
not use it for throughput until `qwen35_batch_gguf_quick_check` is green.

## Pestle-27B-Ternary on Metal — the lm-head was the whole story

Decode profile first (`RLX_METAL_THUNK_PROFILE=1`; it serialises thunks, so
absolute times inflate but the shares are right):

| thunk | count | ms | pct |
|---|---:|---:|---:|
| `dequant_matmul_gguf` | 979 | 115.02 | **73.5%** |
| `attention` | 16 | 15.08 | 9.6% |
| `concat` | 144 | 8.28 | 5.3% |
| `sgemm` | 7 | 6.49 | 4.1% |
| `binary` | 1771 | 4.96 | 3.2% |
| `gated_delta_net` | 48 | 1.47 | 0.9% |

Those 979 matmuls are 978 Q2_0 Pestle factors **plus one G8_0 lm-head**, and
the single lm-head was most of the 115 ms: Metal's fused-GEMV gate covered only
`Q1_0`/`Q2_0`, so `output` — `[248320, 5120]` in `G8_0` — fell to the
dequant-to-scratch path. That reserves a ~5 GiB f32 slab *and re-dequantizes
the entire head on every decoded token*. The comment above
`dequant_gguf_scratch_bytes` already warned about this exact shape for
Ternary-Bonsai's `Q1_0` head; `G8_0` simply was not in the list.

### Measure median per-step time, never total ÷ tokens

**The first decode step pays the decode-bucket compile.** On a 200-token run it
was **92.4 s of a 117.1 s decode — 79% of the total — in a single step**, with
the other 199 steps at a median of 121.9 ms (p90 133.7). Dividing total decode
by token count therefore reports *compile amortisation*, not decode speed, and
it is why every end-to-end number in this file's history was noisy, why a
measured 1.5× kernel win moved the "total" by 2.3%, and why throughput appeared
to collapse as generations got longer.

Use `RLX_QWEN35_DECODE_TRACE=1` and take the **median `total_ms` over steps ≥ 2**.
Against `mortar.cpp`'s 11.15 tok/s, steady-state decode is ~0.74×, not the
~0.14× that total ÷ tokens implies.

### Both changes together

| config | median step | steady |
|---|---:|---:|
| baseline (both off) | 196.71 ms | 5.08 tok/s |
| both on | **121.67 ms** | **8.22 tok/s** |

**1.62×**, 75.0 ms/token. Note baseline-both-off (196.71 ms) is within noise of
fused-off-alone (195.78 ms): the `Q2_0` saving is *invisible* while the scratch
lm-head is present, because 9 ms hides behind a 71 ms cost. Order matters when
attributing wins — measure each against the fully-optimised build, not against
the baseline.

Reference: `mortar.cpp` 11.15 tok/s, so this moves 0.46× → **0.74×**.

### The fused `G8_0` lm-head

Same binary, flag toggled, median per-step over 118 steps at `max_seq` 135:

| `RLX_METAL_G8_0_FUSED_DISABLE` | median step | p25 | p90 | steady |
|---|---:|---:|---:|---:|
| `1` (scratch lm-head) | 195.78 ms | 192.73 | 202.47 | 5.11 tok/s |
| unset (fused) | **124.32 ms** | 123.52 | 131.07 | **8.04 tok/s** |

**+57%**, and unlike every earlier attempt the distributions do not overlap
(p90 fused 131 ms < p25 disabled 193 ms). The 71.5 ms/token saved matches the
mechanism arithmetically: the scratch path dequantises `[248320, 5120]` to f32
every token, ~10 GiB of traffic, ≈68 ms at ~150 GB/s. Peak footprint also drops
**48.05 → 37.06 GB**. Output byte-identical.

### The `Q2_0` 16-bit inner loop

Same protocol, 118 steps:

| `RLX_METAL_Q2_0_SCALAR` | median step | p25 | p90 | steady |
|---|---:|---:|---:|---:|
| `1` (byte loads) | 130.88 ms | 130.46 | 132.04 | 7.64 tok/s |
| unset (16-bit) | **121.67 ms** | 121.24 | 123.58 | **8.22 tok/s** |

**+7.6%** (9.21 ms/token), again non-overlapping. Note this is 3× the "+2.3%"
the total ÷ tokens metric reported for the same change — the old figure was
diluted by the 92 s compile step sitting in the denominator.

### What the two kernel changes were

**`Q2_0` inner loop: byte loads → 16-bit loads.** It was
`qs[i / 4]` with a shift, mask, int→float convert and FMA *per element*; the
f16 scale at the block head leaves `qs` at offset `2 mod 4`, which is why it
could not vectorise. But the offset is always *even*, so 16-bit loads are safe.
`q2_0_dot16` now does two `ushort` loads per 16 codes and masks each code **in
place** (`w & 0x000C` *is* the code times 4), applying the `2^-2j` once at the
end — the trick `q4k_mv_f32_sg` already used for nibbles. The `−1` of
`(q−1)·d` folds out through a per-block `Σy`, removing the per-element convert.
Shared by all four `Q2_0` GEMVs (`mv`, `dual_mv`, `swiglu_mv`, `mv_residual`).

**~1.5× in isolation, +7.6% on the model.** Clean A/B at 27B shapes,
min-of-trials, idle box:

| shape (k×n) | scalar | 16-bit |
|---|---:|---:|
| 5120×5120 | 81.0 | **127.2** GB/s |
| 5120×3968 | 78.4 | **119.3** |
| 3968×5120 | 76.5 | **119.0** |
| 4096×5120 | 81.7 | **122.8** |

The gap between 1.5× in isolation and +7.6% on the model is the point: the
factors are a real but minority cost next to the lm-head. **Profile before
optimising** — several hypotheses died here. The three Pestle scale multiplies looked like the obvious
fusion target and are **3.2%**; and rank 3968 = 31 `Q2_0` blocks (odd) looked
like a shape cliff, but every shape sits in one band.

### Benchmarking discipline (learned the hard way here)

- **Idle box, min-of-trials.** Mean-of-N swung 40–120 GB/s on one unchanged
  kernel. Two runs in this session were silently contended by a concurrent
  model run and produced a baseline that was ~7% too slow, which made the first
  `Q2_0` result look 4× better than it was.
- **Amortise the submit.** A 5-node graph on Metal is ~0.5 ms of
  command-buffer submit, which swamps the kernels: the first version of
  `tests/pestle_chain_bench.rs` showed 1, 2 and 5 dispatches as identical.
  Chain ~32 projections per graph.
- **Do not use `Device::Cpu` as the reference for `Q2_0` at `m == 1`.** That
  path quantizes activations to int8 (`q2_0_dot_q8_block`, llama.cpp-style) and
  is deliberately approximate — 2.2% off on a SwiGLU chain. Compute the
  reference on the host in exact f32 instead
  (`metal_q2_0_fused_decode_parity.rs`).

### Also landed on CUDA

`dequant_matmul_gguf_q2_gemv` / `_g8_gemv` (cooperative block-per-row, the
`Q1_0` shape), plus both schemes in `gguf_fused_gemv_m1_supported` — which is
load-bearing twice, selecting the scratch-free kernel *and* telling
`dequant_gguf_scratch_bytes` not to reserve the `[n, k]` slab. Arena request
for the 27B dropped **13.17 → 8.44 GiB**. Still over a 16 GiB card, and
`RLX_QWEN35_HOST_EMBED=1` does not move it further; the remaining 8.4 GiB is
unexplained and wants its own profile.
