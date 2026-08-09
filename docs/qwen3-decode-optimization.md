# Qwen3-0.6B Decode Optimization — Levers, Measurements, Strategy

Goal: **fast end-to-end inference with the best accuracy and the most TPS** for
Qwen3-0.6B on Apple Silicon (M4 Pro, Metal). This documents every lever
investigated, what was measured, what shipped, and what to build next — so the
dead ends are not re-litigated.

Weights used: `/Users/Shared/weights/qwen3-0.6b/` (safetensors + config +
tokenizer) and `/Users/Shared/weights/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf`.

---

## TL;DR — the strategy

1. **Serve Q4_K_M native-packed.** It is **token-identical to F32** (measured,
   not an approximation) at 4× less memory and bandwidth-faster. It is the
   accuracy sweet spot: zero quality loss vs full precision. Do **not** go below
   Q4 (Q3/Q2 lose quality) and do **not** use W8A8/int8-Q attention (an
   approximation that diverges after ~17–32 tokens).
2. **Enable the lossless levers:** F16-resident weights, fused MLP, in-place KV,
   GQA-native (auto-on on Metal), **f16-KV (+9%, the one cheap decode win)`**,
   flash-decode.
3. **Add exact prefix caching** for TTFT (6–7× on long shared prefixes; token
   identical). This is the single highest-ROI end-to-end win and it is shipped.
4. **Reserve the splice / int8-KV / W8A8 / batching for 100k+ context and
   multi-user throughput** — they are capacity/memory/concurrency levers, not
   single-stream decode-speed levers.

Best accuracy and most TPS are **not** in tension here: the lossy shortcuts were
the dead ends; the accurate path (Q4-packed + lossless kernels + exact caching)
is the fast one.

---

## The bottleneck shifts with context

Measured (packed-Q4 decode): **~52 tps @140 tokens → ~12–15 tps @8k.**

| context | dominant cost |
|---|---|
| ≤ ~1k | weight/projection reads + per-dispatch overhead (rlx emits ~500–600 kernels/token, ~4× llama.cpp) |
| 4k–16k | attention grows O(context); KV-cache read grows O(context) |
| 100k–1M | the quadratic — only bounded/retrieval attention survives |

A single lever cannot win at every regime — optimize the *actual* bottleneck at
each. Note the "attention is 65% at 8k" thunk-profiler reading is **partly an
isolation artifact** (per-thunk commit+wait launch latency across 28 attention
dispatches); the real cost is distributed dispatch overhead. See *Attention
occupancy* below.

---

## Lever scorecard

| lever | status | measured impact | accuracy | effort to realize |
|---|---|---|---|---|
| **Prefix caching** | **SHIPPED** | **6–7× TTFT** @2–4k prefix (grows) | **exact (parity PASS)** | done |
| **f16-KV** | flag exists | **+9%** decode @4k | near-lossless (verify) | flip flag |
| Q4-packed decode | shipped | token-identical to F32 | **exact** | default |
| fused MLP, in-place KV, GQA-native, F16 weights | shipped/auto | already on | none | done |
| HNSW selective-KV splice | wired + fixed | flat decode @100k+; **net-slower <100k** | approximate recall | tune per-step overhead |
| int8-KV storage | microbenched | 4× KV memory; ~1.5–1.8× (read-bound only) | 5e-5 (accurate) | integration (memory lever) |
| W8A8 attention | kernel landed | ~2.9× attention @16k; **0.71–1.0× end-to-end** | ~1e-3, diverges greedy | incremental-KV + only long ctx |
| Speculative decoding | engine exists | 75% accept (Q4 draft) but **0.25 tps** | exact math | **blocked — no cheap draft <0.6B** |
| Fused QKV | machinery landed | **no reduction** | exact | **blocked — mixed schemes + re-split** |
| Attention occupancy redesign | investigated | **dead end** (tuner already optimal) | none | — |

---

## Details per lever

### Prefix caching — SHIPPED, exact, the end-to-end win
Precompute a shared prompt prefix's KV once; reuse across generations, paying
only the (short) suffix. **Token-identical** to a cold full prefill.

| prefix | cold TTFT | warm TTFT | speedup | parity |
|---|---|---|---|---|
| 2048 | 9.2 s | 1.2 s | **7.4×** | PASS |
| 4096 | 17.7 s | 2.8 s | **6.3×** | PASS |

API (`Qwen3Runner`): `cache_prefix(prefix) -> SessionSnapshot`,
`generate_with_prefix_stoppable(&snap, full_prompt, n, cb)`. Backed by
`Qwen3Generator::prefill_with_reuse_fast` — replays the suffix through the fast
GPU `feed_continuation` path (not the slow host `decode_get_logits`) using the
`step_cached` pending-token invariant, plus a `restore_cache` GPU-binding reset.
Only pays for long prefixes; the win grows with prefix length. F32 / GGUF-native
path (packed non-GGUF has no persistent cache). Bench:
`examples/prefix_cache_bench.rs`.

### f16-KV — the one cheap lossless decode win (+9%)
`RLX_QWEN3_F16_KV=1`. Halves KV-cache read bytes. Measured baseline ~9.6 →
10.5 tps @4k. Near-lossless (f16 vs f32 KV) — verify token-parity before
defaulting. Everything else below is either done, blocked, or extreme-ctx only.

### HNSW selective-KV splice — the long-context architecture (100k+)
The disk-tiered HNSW KV store (`rlx-runtime`: `kv_context_store.rs`, `hnsw.rs`,
`quantized_kv.rs`) is **already wired end-to-end** into decode: `apply_retention`
(generator.rs) evicts old rows to the store, retrieves top-k relevant blocks, and
rebinds them into the decode graph's `past_k/past_v` every step, RoPE-correct.
It works on the **packed path** too.

Measured (`context_scale_bench`, 16k, packed): retrieval flat ~25 ms (O(log N)),
100% recall, decode tps flat ~59 — the quadratic is eliminated at scale.

**But integrated end-to-end (`examples/retrieval_decode_splice.rs`) it is
net-slower below ~100k** because `apply_retention` runs per step:
- `query_scoring(true)` forces the Q-export *oneshot* decode → per-step recompile
  → 0.49 tps (15× slower). **Fix: `query_scoring` OFF (K·K proxy) → 4.81 tps.**
- A retrieval throttle (`RLX_RETRIEVE_INTERVAL=N`) amortizes the per-step
  retrieve+splice+rebind (retrieve on steps 1,1+N,…; grow the resident set by
  incremental fold between): @2k 4.95→7.03 tps @N=8, near plain 7.55.
- Even fixed, splice is net-slower at 2–7k (5–7 vs plain 12–13 tps packed) —
  attention is a minority of decode until much longer context. **Wins only at
  extreme context (30k–1M)** where plain attention grows O(context) and the
  bounded resident set stays flat, or as a **capacity/memory** lever.

### int8-KV storage — memory, not speed
Per-row int8 K/V is accurate (max Δ ~5e-5) but only ~1.5–1.8× at long ctx
because decode attention is compute-bound, not KV-read-bound. It is a **4× KV
memory / capacity win** (longer context, more concurrent sequences), not a
single-stream speed win.

### W8A8 attention — validated kernel, but a single-stream loss
int8 Q·K integer dot + int8 V (`RLX_METAL_W8A8_ATTN`). Kernel is structurally
correct (all-exact = 0.000000 vs baseline flash). Error decomposition vs baseline
(GQA 16/8): Q 1e-5, K 3.9e-4 (row) / 1.5e-4 (block-32), **V 1.0e-3 (dominant** —
the direct output operand). Sweet spot int8-Q + block-K + f32-V = 1.55e-4 (7×
better) keeps the integer-dot speed; coherent-prompt greedy parity 32/96 tokens
(full W8A8 17/96) — **approximate, eventually diverges**. Speed: naive full
re-quant per step is O(ctx) → **0.71–0.90× (net slower)**; incremental
(`RLX_METAL_W8A8_INCR`, timing-faithful probe) reaches break-even at ~6k, never a
speedup — the baseline flash attention is already optimized and the int8 partial
isn't faster. CPU/AMX quantize offload measured (285 ns / ~1140 cycles/token,
94% hideable) but incremental GPU quantize already reaches break-even, so the
offload adds ~0. **W8A8's home is the 4× KV-density multiplier feeding the HNSW
store**, not single-stream attention.

### Speculative decoding — blocked (no cheap draft)
The engine (`rlx-runtime::spec_decode::SpecDecoder`) + adapter
(`rlx-qwen3::Qwen3Speculator`) + single-pass logits (`sequence_logits`) all
exist and are tested. Accept-rate sweep vs a Q4 target (both native-packed):

| draft | size | accept | tok/round |
|---|---|---|---|
| Q2_K | 296 MB | 35% | 2.4 |
| Q3_K_M | 347 MB | 61% | 3.43 |
| Q4_K_M | 397 MB | 75% | 4.0 |

**Smaller quant = worse**, not better: same-architecture quant is only ~1.3×
cheaper/token while acceptance craters. Net speedup = accepted/(1 + n_draft ×
cost_ratio) < 1 for all. A Q3 draft would only pay against an **F16/F32 target**
(~1.5× ceiling), which this box doesn't serve. There is no Qwen3 smaller than
0.6B, no MTP head on the checkpoint, and the adapter re-prefills the full context
every round (O(ctx)). **Dead end for single 0.6B on-device.**

### Fused QKV — blocked (mixed schemes + compiler re-split)
`FusedProj` + `FlowCtx::resolve_linear_fused` + decode-layer restructure
(gated `RLX_QWEN3_FUSED_QKV`, off by default): concat q/k/v weights → one GEMV →
`narrow_`-split. Two blockers:
1. **Q4_K_M uses mixed K-quant schemes across q/k/v** (q_proj ≠ k_proj scheme) →
   byte-concat of differing block formats impossible → falls back to `Separate`
   (3 GEMVs) = **token-identical no-op** (safe).
2. **Dense F16 fuses but yields no reduction** (sgemm 113 both, 7.23→7.28 tps) —
   the `mm`→`narrow_` is re-split by a narrow-of-matmul compiler pass.

Also: the GGUF `take_packed` **consumes** (contrary to its doc), so the resolver
must fully own q/k/v (Fused-or-Separate), never fall back to per-key. Realizing
fused QKV needs a **mixed-scheme fused DequantMatMul kernel** or suppressing the
narrow-of-matmul re-split — for a ~2-dispatch/layer win of unproven value.
Low priority. The machinery is landed as a foundation.

### Attention occupancy — a dead end
The flash-decode tuner `sdpa_flash_partitions_tuned` already picks **P=8 (128
threadgroups) at 8k**, and the code documents a measured sweep where more
partitions are *slower* (P=8→13.0, P=16→12.3, P=32→11.6 tps — combine overhead >
occupancy gain). Flash engaged gives only +4% and attention stays ~64%. Do not
"force more partitions."

---

## Environment flags

| flag | effect | recommend |
|---|---|---|
| `RLX_QWEN3_F16_WEIGHTS` | F16-resident projection weights | on (auto on Metal) |
| `RLX_QWEN3_F16_KV` | F16 KV cache (halve KV read) | **on** (+9%, verify parity) |
| `RLX_QWEN3_GQA_NATIVE` | skip repeat-KV expand | on (auto on Metal) |
| `RLX_QWEN3_INPLACE_KV` | in-place KV append (no O(ctx) concat) | on |
| `RLX_QWEN3_BAKE_WEIGHTS` | bake weight-only concats once | on (auto on Metal) |
| `RLX_METAL_SDPA_FLASH_DECODE` | split-KV flash decode attention | neutral–+4% |
| `RLX_QWEN3_FUSED_QKV` | fused QKV (blocked; safe no-op on Q4_K_M) | off |
| `RLX_METAL_W8A8_ATTN` | int8 attention (approx; extreme-ctx only) | off |
| `RLX_METAL_W8A8_QMODE=f32` / `_KMODE` / `_VMODE` | W8A8 per-operand precision (diagnostic) | — |
| `RLX_METAL_W8A8_BLOCK` | per-32 block scales for W8A8 | — |
| `RLX_METAL_W8A8_INCR=<tokens>` | incremental-quantize timing probe | — |
| `RLX_RETRIEVE_INTERVAL=N` | throttle KV-store retrieve/splice to every N steps | on for splice |
| `RLX_METAL_THUNK_PROFILE=1` | per-thunk GPU-window timing (isolation — inflates attention) | profiling |

---

## Benches

- `examples/prefix_cache_bench.rs` — cold vs warm TTFT + token parity.
- `examples/retrieval_decode_splice.rs` — integrated store↔attention decode.
  Envs: `PACKED`, `STORE`, `QSCORE`, `SINKS/RECENT/TOPK/BLOCK`,
  `RLX_RETRIEVE_INTERVAL`, `N_GEN`, `PROMPT_LEN`.
- `examples/context_scale_bench.rs` (`--features metal,mmap-kv`) — HNSW store
  retrieval latency / recall + bounded decode at 16k–100k.
- `examples/spec_decode_bench.rs` — draft/target accept rate (`SPEC_DRAFT`,
  `SPEC_TARGET`, `SPEC_N`, `SPEC_TOKENS`).
- `examples/native_packed_generate.rs` — packed decode + token parity
  (`E2E_PACKED_ONLY`, `E2E_PROMPT_LEN`, `E2E_PROMPT_TEXT`, `E2E_PROMPT_VARIED`).
- rlx-metal: `examples/{w8a8_decode_bench,int8_kv_bench,mps_re_bench}.rs`;
  `tests/metal_w8a8_decode_attn_parity.rs`.

---

## What to build next (if pursuing more)

Lossless, in ROI order:
1. **Make f16-KV default** after a token-parity check (+9%, free).
2. **Tune the splice per-step overhead** (already-throttled) so it wins below
   ~30k — the long-context architecture is here and correct.
3. A **mixed-scheme fused DequantMatMul kernel** — the only path to fused QKV on
   the served K-quant model; also unblocks fused gate+up-as-one-weight. Large.
4. A **high-throughput batched-serving path** — where int8-KV, W8A8, and the
   splice finally earn their keep (KV memory / concurrency).

Avoid: sub-Q4 quants, W8A8/int8-Q for single-stream latency, speculative on this
model, and "force more flash partitions."
