# Kimi-K3 MoE Expert Paging & Compute Optimization

Optimizations to the **disaggregated expert-parallel cluster** path (recurrent backbone on
the Mac; stateless MoE experts fanned out to worker nodes over TCP). Three layers, each
measured and bit-accurate:

1. **Parallel cold expert paging** — worker-side disk read + assembly.
2. **Fused CPU MXFP4 kernel** — skip the per-token f32 weight expansion.
3. **Native CUDA/HIP register-decode GEMM** — run the grouped MXFP4 matmul on-device
   (no host round-trip), with an m>1 amortization variant for prefill.

All work is against the packed **MXFP4** expert path (`RLX_KIMI_PACKED_EXPERTS=1`); experts
stay 4-bit end-to-end (no re-quant). Reference cluster: Mac mini M4 Pro (Metal/backbone) +
`msi` RTX 3080Ti (CUDA) + `amd` MI100 gfx908 / 780M gfx1103 (ROCm).

---

## 1. Parallel cold expert paging

**Problem.** The worker's `KimiExpertProvider` paged owned fired experts with a *serial*
per-expert loop (both the packed `compute_packed` and the f32 `compute` path): one
`load_expert*` at a time + serial buffer assembly, leaving the worker's 16–20 cores idle.
Measured floor: ~135 MB/s (msi) / ~77 MB/s (amd) — overhead-bound, not disk-bound.

**Fix** (`src/dist_experts.rs`). Mirror the Mac runner's `moe:paging`:
- resolve each expert's on-disk byte ranges on the calling thread (`expert_ranges`),
- `pread` them concurrently on a rayon pool (`load_expert_ranges[_packed]` — open each
  distinct shard once + `read_exact_at`, no whole-16GB-shard mmap),
- assemble the contiguous GPU buffers in parallel (`par_chunks_mut`),
- serve re-fired experts from the shared byte-budgeted LRU (`io_opt`, `RLX_KIMI_EXPERT_CACHE=<MB>`).

Also switched the f32 loader `load_expert_ranges` (`src/loader.rs`) from mmap-whole-shard
to `pread` (bit-identical bytes).

**Result (release, per-worker cold paging, packed path):**

| engine | before | after |
|---|---|---|
| msi CUDA | 15.6 s (141 MB/s) | **0.31 s (7113 MB/s)** |
| amd MI100 | 17.6 s (59) | **1.16 s (611)** |
| amd 780M | 15.0 s (48) | **1.0 s (495)** |
| CPU ranks | 4–11 s | **0.2–1.1 s** |

The f32 path barely moved from the pread change alone (**it is dequant-bound**, ~132 MB
f32/expert of CPU dequant, not read-bound) — which motivated §2/§3.

---

## 2. Fused CPU MXFP4 kernel — "skip expanding to f32"

**Problem.** The grouped MXFP4 MoE path (`rlx-cpu` `exec_dequant_grouped_mat_mul_mlx_inner`
and the by-value `dequant_grouped_matmul_mxfp4_bt`) called `dequant_matmul_mxfp4` with
`m=1` per row — which **allocated + decoded the entire ~88 MB f32 expert weight for every
token**, then a scalar matmul, parallel over only the ~3 routed rows. Both CUDA and ROCm
**host-delegate** here, so it capped all engines.

**Fix** (`../rlx crates/io/rlx-mlx-io/src/dequant.rs`, `crates/backends/rlx-cpu/.../quant.rs`).
New `grouped_matmul_mxfp4_bt`: decode e2m1 nibbles **inline into the accumulator** (no f32
weight materialized) and parallelize over **all m·n outputs** (saturates cores when m is
tiny — the decode case). Both host-delegate call sites switched off the materializing path
(`_inner` → `dequant_matvec_mxfp4`, `_bt` → `grouped_matmul_mxfp4_bt`).

**Result:** 4–6× packed compute vs the materializing path (dispatch 255 s → 49 s in debug);
bit-accurate (packed-vs-f32 rel-L2 ~5e-7). In release the CPU ranks compute in ~1.5–2.4 s.

---

## 3. Native CUDA/HIP register-decode GEMM + m>1 amortization

**Problem.** Even fused, `Op::DequantGroupedMatMulMlx` (MXFP4) **host-delegated** on both
CUDA and ROCm: dtoh(x,idx) + per-expert dtoh(weights) → CPU fused kernel → htod(out). The
round-trip + CPU compute contends with paging on the worker.

**Fix.** One **shared** kernel file (`../rlx crates/backends/rlx-gpu-kernels/kernels/dequant_matmul_mlx.cu`,
compiled by NVRTC on CUDA and hipRTC on ROCm) with two entry points:

- `dequant_grouped_matmul_mlx_mxfp4` (**V1**, one thread per output `(r,j)`): picks the
  row's expert `e=(uint)idx[r]`, streams `W_e[j,:]` and decodes each e2m1 nibble in a
  register — no f32 weight. Optimal for MoE **decode** (each row → a distinct expert).
- `dequant_grouped_matmul_mlx_mxfp4_amort` (**V2**, m>1 amortization, one thread per output
  **column**): groups the m rows by expert *in-thread* and decodes each fired expert's
  `W_e[col,:]` **once**, reusing it across every row routing to that expert (no host sort,
  no scratch). The launch picks V2 for `1 < m ≤ 16` (prefill), else V1.

Grouped scales are **f32 in the arena** (e8m0 pre-decoded by the loader + widened bf16→f32),
so the kernel reads f32 directly — *not* `mlx_group_scale`. Accumulation order matches the
CPU `grouped_matmul_mxfp4_bt`.

**Wiring** (identical shape per backend, `rlx-cuda` + `rlx-rocm`):
`kernels/mod.rs` (register both kernels), `backend/step.rs` (new `DequantGroupedMatmulMlxNative`
variant + label + offset-dependency arm; CUDA also the active-extent list), `backend/compile.rs`
(branch `MlxMxfp4` → native, affine → host-delegate), `backend/run.rs` (launch: kernel/grid
by `m`; CUDA offsets are `u64` fields, ROCm are `u32` widened to `u64` at launch since the
kernel is shared with CUDA's possibly->4 GiB arenas).

**Result (release, GPU-worker compute, host-delegate → native):**

| engine | host-delegate | native |
|---|---|---|
| msi CUDA | 6.00 s | **4.52 s** (1.33×) |
| amd MI100 | 5.68 s | **4.91 s** (1.16×) |
| amd 780M | 4.37 s | **3.83 s** (1.14×) |

MoE 23.2 s → 20.6 s; token **bit-exact 26088**. The gain is modest because the release
host-delegate was already decent and **per-call graph compilation now co-dominates** the
worker compute.

---

## 4. Per-call recompile → bucketed graph cache

**Problem.** The worker `compile_built` a fresh graph on **every** MoE call
(`compute_packed`), because the graph shape depends on `n` (the fired-expert count, which
varies per call). After §3 this per-call recompile *co-dominated* GPU-worker compute.

**Fix** (`src/dist_experts.rs`). Bucket `n` up to a power of two (`nb`) — the padded
expert slots `[n, nb)` are never referenced by `eidx` (routing indexes `0..n`), so they
contribute 0 and are left zero. Cache the compiled graph by `(nb, rows)`; on a hit only
the codes/scales are re-uploaded and `run`. A handful of buckets recur across layers and
decode steps, so nearly every call reuses a compiled graph. Gated by
`RLX_KIMI_NO_GRAPH_CACHE`; the worker shutdown line reports the `compile`/`run` split and
graph-cache hit count.

**Result (release, 2 forwards = 14 MoE calls/worker):** **12/14 graph-cache hits** — 2
compiles instead of 14. Warm-iter MoE **20.4 s → 17.2 s**, dispatch **10.0 s → 7.5 s**;
token **bit-exact 26088**. The compile/run split then localized the *next* bottleneck to
`run` (the GEMM): CUDA 0.34 s/call, MI100 0.73 s/call, 780M 0.65 s/call.

## On MFMA / WMMA tensor cores (investigated — not wired, by design)

The natural next step for the `run`-dominated GEMM is AMD matrix cores (ROCm ships a
gated `matmul_mfma.cu`, `RLX_ROCM_MFMA=1`). But MFMA is **fundamentally starved by the MoE
grouped-decode shape**, which we confirmed from the op structure:

- MFMA computes 16×16×16 tiles; a tile's 16 rows must **share one weight matrix**.
- The routed op's total `M = rows·top_k` (≥16), but each of those rows picks its **own**
  expert, so MFMA must run **per-expert** — and in decode each expert sees only
  `rows_e ≈ 1–3` tokens. A 16-row MFMA tile is then ~6–19 % full (0 % at `M=1` decode).
- A correct fused grouped-decode-MFMA kernel (in-tile MXFP4→f16 decode + expert grouping)
  is a large from-scratch rocWMMA effort whose payoff exists **only for large-batch
  prefill** (many tokens repeating an expert) — not the cluster's `O(1)/token` decode.

The scalar register-decode kernel (§3) is therefore the right choice for MoE decode. MFMA
is worth building **iff** large-batch prefill throughput becomes a goal; the design is
"host-group rows by expert (CSR) → per-expert decode-to-f16 tiles in shared memory →
rocWMMA GEMM → unpermute", gated behind `RLX_ROCM_MFMA` and selected only when the
measured average `rows_e` clears an MFMA-tile threshold.

## How to run

Workers run a prebuilt `target/release/examples/kimi_k3_cluster`; source is mirrored with
`scripts/matrix/sync_to_remote.sh` (`RLX_REMOTE_HOST=<node>`). Build on a node with a login
shell (cargo is only in the login PATH): `bash -lc 'cargo build --release --example
kimi_k3_cluster --features cluster,<cuda|rocm>'`.

Launch the 5-engine fleet + measure (see `scripts/`):

```sh
BIN=./target/release/examples/kimi_k3_cluster \
  WENV="RLX_KIMI_PACKED_EXPERTS=1" scripts/fleet_launch.sh      # one rank per engine
BODY=cpu LAYERS=8 scripts/fleet_run.sh /tmp/orch.log            # orchestrator + per-engine timing
```

Topology (`--shards "1:0-370,3:370-430,2:466-706,4:706-806,5:806-896"`): msi CUDA(r1)+CPU(r3),
amd MI100(r2)+780M(r4)+CPU(r5). "One rank == one compute engine."

Key env: `RLX_KIMI_PACKED_EXPERTS=1` (packed MXFP4 path → native GPU kernel / fused CPU
kernel), `RLX_KIMI_EXPERT_CACHE=<MB>` (resident LRU across re-fires), `RLX_KIMI_PAGING_DIAG=1`
(per-call paging split), `RLX_KIMI_IO_STATS=1` (paging report). XDNA is **not usable** for
experts (no XRT overlay/toolchain; the AIE array is INT8-only — cannot run FP32/MXFP4
grouped matmul).

---

## Validation

- **`expert-selfcheck --packed`** (per-node, packed-vs-f32 on the same device): rel-L2 ≤ 5e-7
  on CUDA *and* ROCm at rows=1/4/8 (exercises V1 + V2).
- **Cluster token bit-exact 26088** across all three backends.
- Clippy clean on `rlx-mlx-io` / `rlx-cpu`; all three backends build green
  (metal/mlx, cuda, rocm).

---

## Remaining work

- **`run`-side GEMM throughput** is the current worker bottleneck (post graph-cache),
  especially on MI100/780M (~2× CUDA's per-call GEMM). The scalar register-decode kernel is
  correct for decode; MFMA would need large-batch prefill to pay off (see above).
- **Reduce the padded upload** — the pow2 bucket re-uploads up to ~2× the codes/scales; a
  partial `set_param_range` of only the real `n` experts (padded slots stay stale, never
  read) would trim it, though `run` is currently GEMM- not upload-bound.
- **Expert-range rebalancing** by measured per-engine throughput (780M and CPU ranks are
  weaker than CUDA/MI100 — the straggler gates dispatch).
- Build the Mac orchestrator in **release** for true end-to-end numbers (the total is
  currently bounded by the debug backbone ~38–57 s).
