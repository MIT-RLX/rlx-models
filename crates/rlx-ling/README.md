# rlx-ling — Ling 3.0 / BailingMoeV3 for RLX

Native RLX support for [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
(`model_type = "bailing_hybrid"`, `BailingMoeV3ForCausalLM`) — ~7.9 B total
parameters, ~0.6 B active.

## Architecture

Two interleavings run at once.

**Attention** alternates on a `layer_group_size` cycle. For Ling-3.0-tiny
(24 layers, group size 4) that is MLA at layers 3/7/11/15/19/23 and KDA
everywhere else:

| | layers | mechanism |
|---|---|---|
| **KDA** | 18 | Kimi Delta Attention — gated delta-net linear attention. 4-tap causal short conv + silu on q/k/v, L2-normed q/k, sigmoid `beta`, a per-channel log-decay gate, and a gated output RMSNorm. Rides `Op::GatedDeltaNet { gate_per_channel: true }`. |
| **MLA** | 6 | DeepSeek-style multi-head latent attention: rank-256 Q, rank-512 KV, one decoupled interleaved-RoPE head shared across heads, plus a Bailing-specific head-wise sigmoid output gate. |

**FFN** is dense for the first `first_k_dense_replace` layers (just layer 0) and
a fine-grained `noaux_tc` MoE after: 128 routed experts, 8 active, grouped
top-k over 8 groups of 16, one always-on shared expert, `routed_scaling_factor`
2.5. This is shared verbatim with `rlx-deepseek`, which reuses rlx-llada2's
`group_limited_gate` custom op.

```
word_embeddings
  → 24 × ( RMSNorm → (KDA | MLA) → +res → RMSNorm → (MLP | MoE) → +res )
  → RMSNorm → lm_head          (untied)
```

## Usage

```rust
let cfg = LingConfig::from_file(dir.join("config.json"))?;
let mut wm = WeightMap::from_safetensors_dir(&dir)?;
prepare_checkpoint(&cfg, &mut wm)?;               // stacks per-expert MoE tensors

let built = build_ling_text_flow(&cfg, &mut wm, seq, true)?;
let mut compiled = compile_built(built, Device::Cpu)?;

let (cos, sin) = cfg.rope_tables(seq);            // only the MLA layers use these
let logits = compiled.run(&[
    ("input_ids", &ids), ("rope_cos", &cos), ("rope_sin", &sin),
]);
```

## Details worth knowing

Things that differ from what a reading of the config alone would suggest:

* **The KDA gate form flips with `kda_lower_bound`.** `fla/ops/kda/gate.py` uses
  `-exp(A_log)·softplus(g + dt_bias)` only when the bound is unset. Ling 3.0
  ships `kda_lower_bound = -5`, which selects
  `lower_bound · sigmoid(exp(A_log) · (g + dt_bias))` — a different function that
  lands in `[lower_bound, 0)` by construction, not a clamp of the first.
* **`A_log` is per head** (`[num_heads]` in the checkpoint), broadcast across the
  head's channels.
* **`FusedRMSNormGated` gates after normalising**, so the KDA output gate does not
  participate in the variance.
* **MLA's score scale is `qk_head_dim^-0.5`** (192), not `v_head_dim^-0.5`.
* Several config keys are **dead** in the reference modeling code and are ignored
  here: `use_qk_norm` (V3 MLA has no q/k norm), `group_norm_size`, `linear_silu`,
  `up_proj_norm`, `value_norm`, `scale_router_input`, and
  `partial_rotary_factor` / `rotary_dim` (the rotary module overrides both and
  rotates the full `qk_rope_head_dim` slice).
* Bailing names the embedding `model.word_embeddings.weight` and the MLA
  out-projection `dense`; attention lives under `layers.{i}.attention`.

## Tests

```sh
cargo test -p rlx-ling                            # config, checkpoint prep, graph smoke
```

Numerical parity against a PyTorch transcription of `modeling_bailing_moe_v3.py`
plus the FLA kernels it calls (Triton isn't required — the recurrence is written
out sequentially):

```sh
python3 ../../scripts/ling3_reference.py .fixtures/ling3-parity
RLX_LING_PARITY_DIR=.fixtures/ling3-parity cargo test -p rlx-ling --test parity_reference
```

Current result on CPU **and Metal**: cosine `1.00000000`, max |Δ| 1.7e-6, argmax
identical at every position.

Cross-backend consistency (CPU vs `RLX_TEST_DEVICE`, split per block so a failure
names the culprit):

```sh
RLX_TEST_DEVICE=metal cargo test -p rlx-ling --features metal --test backend_consistency
```

| backend | tiny-model correctness | notes |
|---|---|---|
| CPU | ✅ cosine 1.00000000 vs PyTorch | reference |
| Metal | ✅ cosine 1.00000000 | |
| MLX | ✅ cosine 1.00000000 | needed upstream fix #1 |
| CoreML / ANE | ✅ cosine 0.99999297, argmax identical | needed upstream fix #2; fp16 by design |
| wgpu | ✅ cosine 1.00000000 | needed upstream fix #3 |
| CUDA (RTX 3080 Ti) | ✅ cosine 1.00000000 | needed upstream fix #3 |
| ROCm (MI100) | ✅ cosine 1.00000000 | needed upstream fix #3 |
| Vulkan | untested | shares the fixed `rlx-unfuse` path |

All seven tested backends now pass the full suite (28 tests each), including
parity against the PyTorch reference. CoreML runs in fp16 on the ANE, so it takes
an fp16-appropriate tolerance (2e-2 relative) while the rest keep the f32 bound
(1e-3); its top-1 token matches the f32 reference at every position.

### Three upstream bugs, all in `FusedSwiGLU`/MoE lowering

`Op::FusedSwiGLU` carries a `gate_first` flag — `true` means the fused input is
`[gate | up]` rather than the canonical `[up | gate]`. This MoE emits gate before
up, so the flag is set, and dropping it silently computes `gate · silu(up)`:
plausible, finite, ~1e-2-wrong logits.

1. **`rlx-mlx/src/lower/env.rs`** destructured `Op::FusedSwiGLU { cast_to, .. }`,
   discarding the flag.
2. **`rlx-coreml/src/mil/matmul.rs`** hardcoded the slice offsets — and
   separately, `lower_grouped_matmul` emitted a MIL `gather` without the
   iOS17+-required `validate_indices`, so CoreML could not even parse the model
   ("Unable to parse ML Program"). Four `gather` sites were missing it.
3. **`rlx-unfuse/src/lib.rs::expand_swiglu`** — the shared decompose used by
   wgpu, CUDA, ROCm (and Vulkan/TPU/oneapi). Its caller passes
   `node.inputs[0]`, the FusedSwiGLU's *input*, so the flag genuinely cannot be
   recovered inside the function; it now takes `gate_first` as a parameter.

A fourth turned up while building decode:
`rlx-coreml/src/mil/ssm.rs::lower_gated_delta_net` consumed the carried SSM state
but never published the final one, so a decode harness reading the state node back
got the *initial* state and the recurrence silently stopped carrying. It now binds
the final state to the input's name, mirroring rlx-mlx.

**None of this is Ling-specific** — `rlx_deepseek::moe::emit_deepseek_moe` is
shared with rlx-deepseek and rlx-kimi-k3, which emit gate before up too, so
DeepSeek-V3 and Kimi-K3 were wrong on the same backends.

The reproducer that isolated #3 is kept as
`single_layer_moe_after_attention_matches_cpu`: one decoder layer, where the
same layer with a dense MLP is exact and the MoE alone is exact, but both
together fail. It only surfaced with attention present because the swiglu fusion
fires in a second, post-unfuse fusion round that the attention-free graph does
not reach.

## Benchmark (real weights)

`inclusionAI/Ling-3.0-tiny`, 15.8 GB bf16 → f32, seq 64, M4 Pro / 68.7 GB, best of 3:

| device | TTFT | prefill throughput | peak RSS |
|---|---|---|---|
| Metal | **0.390 s** | 164.0 tok/s | 45.6 GB |
| CPU | 0.638 s | 100.4 tok/s | 42.6 GB |
| MLX | 110.7 s | 0.6 tok/s | 48.9 GB |

All three return the same top-5. For `"The capital of France is Paris. The capital
of Japan is"` → `" Tokyo"` at p=0.9711, then `" Kyoto"`, `" Beijing"`.

MLX is slow because the 4735-node graph exceeds `RLX_MLX_COMPILE_MAX_NODES`
(1536) and falls back to `MlxMode::Lazy`; it is correct, just uncompiled.

One-time costs (not in TTFT): ~10 s weight load, ~31 s expert stacking, ~2 s graph
build, 6–18 s compile.

### Generation (decode)

Incremental decode is implemented end to end — `flow_decode::build_ling_decode_flow`
plus `DecodeSession` — and produces coherent text from the real checkpoint:

```
prompt:    "The capital of France is Paris. The capital of Japan is"
generated: " Tokyo. The capital of Germany is Berlin. The capital of Italy is
            Rome. The capital of Spain is Madrid."
```

| device | decode | per token |
|---|---|---|
| Metal | **16.0 tok/s** | 0.062 s |
| CPU | 14.0–15.2 tok/s | 0.066–0.071 s |

`--inplace-scan` keeps the KDA scan state (18.9 MB/token) in a param the GDN
kernel mutates instead of round-tripping it through graph I/O: ~4% on CPU, nil on
Metal. Valid only where that in-place update survives to the next `run()` — CPU,
Metal and wgpu, **not** MLX/CoreML — and `decode_equivalence` covers both modes so
an unsupported backend fails loudly.

Metal overtook CPU only after two Metal kernel fixes (below); before them decode
was 6.0 tok/s there. Per token this model reads roughly:

| | bytes/token |
|---|---|
| routed experts (8 of 128 × 23 layers, f32) | 1.73 GB |
| `lm_head` (157184 × 1536, f32) | 0.97 GB |
| attention + dense projections | ~0.2 GB |
| **total** | **~2.9 GB** |

Against a ~273 GB/s ceiling that is a ~11 ms/token floor; CPU runs at 69 ms
(≈42 GB/s effective) and Metal at 88 ms. With only ~1024 outputs per matmul there
is little parallelism for the GPU to exploit, while the CPU pays no dispatch cost.

Measured profile of a 69 ms CPU token: `Sgemm` 30 ms (dominated by `lm_head`),
`GroupedMatMul` 25 ms (experts), everything else ~14 ms. That points at fewer bytes rather than faster kernels — but the obvious version of
that does **not** pay off, and it is worth recording why.

`--f16-head` stores `lm_head` as F16, halving its 0.97 GB/token. Measured:

| device | f32 | f16 head |
|---|---|---|
| Metal | 0.089 s/token | 0.087 s (~2%) |
| CPU | 0.070 s/token | 0.081 s (**16% worse**) |

CPU regresses because Accelerate's f32 sgemm beats the f16-widen path, and Metal's
`sgemm_f16w` is not faster per byte than its f32 kernel — so halving the bytes did
not halve the time. Worse, it degrades greedy output: f32 continues *"The capital of
Germany is Berlin…"* while f16 drifts to *"Is the capital of France the same capital
as…"*. f16 carries ~5e-4 relative error and 157k near-tied logits flip argmax easily.
Off by default.

The remaining 1.73 GB/token is the expert banks, which would need a bf16/f16
`GroupedMatMul` (no backend has one). Given f16 did not convert bytes into time on
either backend here, that is worth prototyping before committing to it — the
bottleneck may be kernel efficiency rather than raw bandwidth.

*Tried and rejected:* routing the CPU MoE matmul through `par_sgemm`. At `m == 1`
each call is a 1.5 MFLOP matvec and decode issues 368 of them per token, so ~368
Rayon hand-offs cost more than the extra memory parallelism buys — decode went
0.069 → 0.085 s/token. Reverted, with a comment at the call site so it is not
retried.

### Two upstream Metal kernel fixes

Decode on Metal went **6.0 → 16.0 tok/s (2.7×)**, overtaking CPU. Both fixes are
the same story: kernels written for prefill (many rows) starve at decode (`m == 1`).

1. **`grouped_matmul` accumulated into a single `float`** across the whole K
   reduction. That serial FMA chain caps memory-level parallelism, which barely
   matters with many rows but dominates when there is only one. Four independent
   accumulators: **108 → 35 ms/token (3.1×)**. Threadgroup sizing was also wrong —
   an `8×8` group degenerates to 8 threads at `m == 1`, idling 24 of every 32 SIMD
   lanes — worth a further 5%.

2. **New `grouped_gemv_splitk` kernel.** Even fixed, one-thread-per-output gives
   only `n` (~1–1.5k) threads at decode: far too few outstanding loads. The new
   kernel has KSPLIT=32 threads cooperate per column, each striding `k` and summing
   a K/32 slice, then reducing through threadgroup memory — 32× the threads at
   identical DRAM traffic, with `weight[e·K·N + k·N + col..col+31]` still a fully
   coalesced 128-byte line. **45 → 10 ms/token (4.4×)**. Routed only at `m <= 4`,
   so prefill keeps the simple kernel.

   This mirrors the `gemv_f16w_splitk` kernel already in rlx-metal — whose own
   comment diagnoses the same problem for f16 weights ("too few to saturate memory
   bandwidth (~37 GB/s of ~273 peak)"). The f32 MoE twin was simply missing.

Split-K reassociates the K reduction, so results move ~1 ulp; partials are summed
in fixed order, so it stays deterministic. Metal parity is unchanged at cosine
1.00000000 and prefill is untouched (152 tok/s).

3. **New `gemv_f32_splitk` kernel** for plain `sgemm`, which was 57% of a decode
   token after (2) — same one-thread-per-output shape. rlx-metal does have a
   `Simd64SplitK` behind `pick_sgemm`, but it accumulates with float atomics
   (order-nondeterministic); this one reduces through threadgroup memory in fixed
   order instead. Gated `m == 1 && n >= 64`, matching the existing f16 gate, with
   `RLX_METAL_GEMV_SPLITK=0` to opt out. **0.062 → 0.045 s/token (1.37×).**

Metal decode overall: **6.0 → 22.0 tok/s (3.7×)**, versus ~14–15 on CPU. Prefill is
unaffected (the gates are `m <= 4` / `m == 1`).

### TTFT: the remaining lever (not done)

Prefill is **78% `grouped_matmul`** (289 ms of a 369 ms profiled token batch), and
the reason is structural. The kernel maps one thread per `(row, col)` and looks the
expert up per row — so in a MoE, where every row is a token with its own expert,
**each row independently streams its own multi-MB expert slab**. Nothing is reused.

The fix is what rlx-cpu already does: counting-sort tokens by expert, then one dense
GEMM per expert, so each slab is read once and amortised over all its tokens. CUDA
appears to have this (its `grouped_matmul` launch sits behind a `used_sorted` check,
with the naive kernel labelled *"Fallback: per-token expert lookup"*); Metal has no
sorted path for f32.

Two things make this more than a port, both worth knowing before starting:

* Metal *does* have the sorted machinery for **quantized** MoE
  (`encode_dequant_grouped_matmul_gguf` + `rlx_cpu::gguf_matmul::grouped_moe_sort_plan`
  / `grouped_moe_unpermute_out`), so the plan/unpermute helpers are reusable.
* But that path does `cmd_buf.commit(); wait_until_completed()` **per call**. Prefill
  issues 368 GroupedMatMuls, so a naive port buys sorted locality and pays 368 full
  GPU syncs. It needs a batched design (one sorted dispatch per layer, or per-expert
  dispatches accumulated without a host round-trip) to actually win.

Also tried and rejected: reshaping the threadgroup wide-and-shallow (one row per
threadgroup, so every thread shares one expert slab). Locality improved, but 146 →
127 tok/s — too few, too-large threadgroups cost more scheduling flexibility than
the cache reuse gains. The measurement is recorded at the call site.

### The same fix on CUDA

`grouped_matmul.cu` had the identical shape — one thread per output, single
accumulator — and at `m == 1` its `8×8` block wastes 7 of 8 y-threads while the
grid is only `n/8` blocks. Ported `grouped_gemv_splitk` (shared-memory reduction,
same deterministic fixed-order partial sum) and routed it at `m <= 4`.

Verified on an RTX 3080 Ti: all 32 tests pass, including every decode-equivalence
case. **Real weights do not run there**, and both escape routes were tried:

* Plain CUDA OOMs cleanly —
  `device allocation failed for 7909017552 f32 (29.463 GiB)` against 16 GB of VRAM.
  The f32 arena is 31.6 GB of params, 27.8 GB of it expert banks.
* `RLX_CUDA_UNIFIED=1` (the managed, VRAM-oversubscribing arena) is **not a usable
  workaround**: it runs, but returns **all-zero logits** — a silent wrong answer —
  and takes 333 s TTFT (0.2 tok/s) paging ~30 GB over PCIe per forward. On the tiny
  model individual tests pass, but the full suite panics in `arena.rs` as managed
  allocations accumulate. Worth fixing or gating upstream; a silent zero is worse
  than an OOM.

So narrow weights are not a performance nice-to-have for CUDA, they are what makes
the model runnable at all. The arena math (f32 = 29.5 GiB: 25.9 experts + 3.5 rest;
card is 16.0 GiB, ~15.3 usable after context):

| | arena | fits 16 GiB? |
|---|---|---|
| f32 (today) | 29.5 GiB | no |
| bf16 experts only | 16.5 GiB | no |
| bf16 everything | 14.7 GiB | yes, ~0.6 GiB spare |
| **MXFP4 experts** | **6.8 GiB** | yes, comfortably |
| MXFP4 experts + bf16 rest | 5.0 GiB | yes |

MXFP4 is the target, and **the hard half is already built**: rlx-cuda has a native
on-device 4-bit grouped GEMM — `Step::DequantGroupedMatmulMlxNative` for
`QuantScheme::MlxMxfp4 { group_size }`, doing register nibble-decode with no host
round-trip and no f32 weight. The op is `Op::DequantGroupedMatMulMlx` (inputs
`x, w_q:U8, scales, biases, expert_idx`).

What is missing is the **encoder**. Every MXFP4 path in rlx is consume-side, for
checkpoints that arrive pre-quantized (mlx-community 4-bit); there is no f32→MXFP4
packer (e2m1 nibbles + per-group e8m0 scales). Ling ships bf16, so this model needs
one written, plus `emit_deepseek_moe` taught to emit the MLX-grouped op when expert
banks are packed. That is the single piece of work standing between this model and
consumer GPUs — and it is well-scoped, because the kernel it feeds already exists
and is already exercised by the mlx-community models.

The same box runs it fine on **CPU**: prefill 15.7 tok/s (TTFT 4.07 s), decode
4.16 tok/s, peak RSS 57.6 GB of 61 — slower than the M4 Pro's 150 / 15.2 (OpenBLAS
on a mobile x86 vs Accelerate on Apple silicon).

The plain `sgemm` kernel has the identical single-chain shape and is the next
obvious target, but it is the workhorse for *every* Metal matmul: unrolling it
measured only ~3% here while changing the K-reduction order for every model on the
backend, so it was reverted rather than shipped on one model's evidence.

### Decode state must go through graph I/O, not a param

`Op::GatedDeltaNet { carry_state }` documents an in-place state update, and CPU,
Metal and wgpu do mutate the buffer — so binding the state to a persistent param
looks like it works. **MLX does not**: it substitutes the new state into its
evaluation env, which does not survive to the next `run()`, so the state silently
stays at its initial value. Passing state in as a graph input and reading the same
node back as an output is portable across all five backends, and is what surfaced
the CoreML bug below.

### Memory

Two separate problems, one fixed and one not.

**Fixed — the in-graph transpose doubled the arena.** `emit_deepseek_moe`
transposed the expert banks in-graph and constant-folding kept *both* copies
resident: 27.8 GB → 55.6 GB, which overran Metal's 41.75 GB max buffer outright
("failed to allocate a 60.66 GB shared buffer"). `prepare_checkpoint` now
transposes host-side while stacking and sets
`DeepseekMoeDims::experts_pretransposed`. Metal went from not running at all to
running, and **CPU TTFT improved 26.0 s → 0.638 s** — the old figure was memory
thrashing, not compute.

**Improved — the load path.** `--stream` ([`streaming`]) keeps the routed experts
out of the build and uploads them per layer after the arena exists:

| | eager | `--stream` |
|---|---|---|
| load | 21.9 GB resident (peak 32.4), 9.9 s | **5.7 GB (peak 5.8), 0.9 s** |
| params reaching the compiler | 31.6 GB | **3.8 GB** |
| peak RSS | ~48.6 GB | ~49.3 GB |

**Not fixed — peak RSS.** Deferring changes *when* pages become resident, not how
many. On CPU the arena starts at 6.3 GB and climbs linearly to ~48 GB as the f32
experts fault into it. The real fix is for the weights to stop being f32 (bf16 or
packed experts), which needs `Op::GroupedMatMul` to accept a non-f32 weight
dtype, which no backend implements for f32-adjacent widths. **MXFP4 solves it a
different way** — a packed op that never needs an f32 weight at all; see below.
That is what put this model on a 16 GB CUDA card.

## MXFP4 (`--mxfp4`)

The whole model quantized to 4 bits at load time, from the stock bf16
checkpoint. No pre-quantized artifact is needed: `rlx_core::mxfp4_pack` is a new
f32 → MXFP4 *encoder* (E2M1 nibbles + per-group E8M0 scales), the produce-side
counterpart to rlx's existing consume-side MXFP4 kernels.

Two knobs, because the LM head is not like the rest (`quant::QuantPlan`):

| | arena weights | max rel logit dev vs f32 |
|---|---|---|
| f32 | 29.5 GiB | — |
| `--mxfp4 --f32-head` (`QuantPlan::mxfp4_body`) | ~4.9 GiB | 1.9e-3 |
| `--mxfp4` (`QuantPlan::mxfp4_all`) | ~4.0 GiB | 3.1e-2 |

Every other projection feeds a residual stream that dilutes its error; the LM
head's output *is* the logits, so its 4-bit error arrives undiluted — 16× more.
That matters here: the earlier f16-head experiment showed that at only 5e-4
relative error, 157k near-tied logits flipped argmax often enough to derail
greedy generation. `--f32-head` costs 0.85 GiB and avoids that risk.

The token embedding stays f32 either way (0.97 GiB): it is gathered, not
multiplied, and rlx has no MXFP4 gather.

### Measured, real 15.8 GB checkpoint (M4 Pro)

Prefill, seq 64:

| | TTFT | prefill | steady RSS | peak RSS |
|---|---|---|---|---|
| CPU f32 | 5.64 s | 11.3 tok/s | 21.6 GB | 44.1 GB |
| CPU `--mxfp4 --f32-head` | 5.83 s | 11.0 tok/s | 8.9 GB | 24.5 GB |
| CPU `--mxfp4` | 6.27 s | 10.2 tok/s | 8.2 GB | 22.9 GB |
| Metal f32 | 0.43 s | 147.7 tok/s | 15.8 GB | 42.7 GB |
| Metal `--mxfp4` | 1.13 s | 56.7 tok/s | 8.2 GB | 25.0 GB |

Decode, 16 tokens after a 16-token prompt:

| | generation | peak RSS |
|---|---|---|
| Metal f32 | 21.3 tok/s | 32.3 GB |
| Metal `--mxfp4 --f32-head` | 18.9 tok/s | 23.8 GB |
| Metal `--mxfp4` | 18.4 tok/s | 24.1 GB |

CUDA (RTX 3080 Ti Laptop, 16 GB) — **this model could not run here at all
before**; f32 died with `device allocation failed for 7909017552 f32
(29.463 GiB)`:

| | TTFT | prefill |
|---|---|---|
| CUDA f32 | — | *does not fit* |
| CUDA `--mxfp4`, as first written | 63.4 s | 1.0 tok/s |
| CUDA `--mxfp4`, after the three fixes below | **0.266 s** | **240.9 tok/s** |

238x, and 1.6x faster than Metal's f32 path. Logits match CPU and Metal MXFP4 to 3
decimals (+5.433 / +4.682 / +4.604), so all four backends agree on real weights.

Both fixes came from `RLX_CUDA_STEP_PROFILE=1`, and the first one was **not**
where the model said to look:

```text
before                                          after
Llada2GroupLimitedGate      61815 ms (97%)      3.2 ms
DequantGroupedMatmulMlxNative 1429 ms             96 ms
DequantMatmulMlx               165 ms            113 ms
GatedDeltaNet                   31 ms             32 ms   <- now ~12%, next up
```

1. **The MoE router was 97% of prefill.** Its host delegate went through
   `with_whole_arena`, which round-trips the *entire* arena device→host→device
   to compute a top-k over a few thousand floats — 23 layers x 2 x ~6 GB ≈
   276 GB of PCIe traffic, 2.7 s per call. It now stages only the three regions
   it touches (~70 KB). The cost scaled with **arena size, not problem size**,
   so it was invisible on small models and worst on exactly the models big
   enough to need a GPU. Every crate on `group_limited_gate` was paying it
   (rlx-deepseek, rlx-kimi-k3, rlx-glm4moe), on every host-delegating backend.
   Metal is unaffected — its arena is zero-copy, so the staging was already free.

2. **The MXFP4 grouped kernel, 22x** (`gate_up` m=64: 10.33 -> 0.46 ms, 110 GB/s,
   439 GFLOP/s). Two independent problems, both fixed by the new split-K kernel:
   `mlx_rd_byte` issued one 32-bit load per *nibble* (8x more loads than data),
   and one thread per output meant a warp's 32 lanes read weight rows `k/2`
   bytes apart — fully uncoalesced. Now a warp owns one output and its lanes
   split K, so the 32 loads cover 128 contiguous bytes. It is also slightly
   *more* accurate than the old kernel (tree reduction: 1.32e-7 vs 1.77e-7 vs an
   f64 reference).

   Iterate on this with `rlx-models-core/examples/mxfp4_grouped_bench`, which
   runs the op standalone at Ling's real MoE shapes in seconds rather than
   through a whole-model prefill.

3. **The dense MXFP4 GEMM, 1.4x** (`attn_proj` 1.04 -> 0.73 ms; `lm_head` m=64
   79.8 -> 57.7 ms, 536 GFLOP/s). It staged X through shared memory as
   `xs[t*tpg+tid]` — but each thread wrote and read back its *own* slot, so the
   round-trip was a no-op that cost 8 KB of occupancy-limiting shared memory and
   a `__syncthreads()` per K-chunk. X lives in registers now. The transformation
   is provably identity (checksums unchanged), which matters because this kernel
   is shared with the MlxAffine and MXFP8 paths.

**Metal, 45.4 -> 56.7 tok/s (1.25x).** No port was needed: `rlx-metal` already
has native MXFP4 kernels and Ling was using them — the `DeferredHostOp` in that
file is only a fallback. They carried the same no-op staging the CUDA dense
kernel did (`xs[t*tpg+tid]` written and read back by the same thread), worth 8 KB
of occupancy-limiting threadgroup memory and a barrier per K-chunk; removing it
gave 45.4 -> 50.9, and staging the activation as `half` instead gave a further
50.9 -> 56.7. f16 is safe here because its partner in the product is a
3.3-mantissa-bit MXFP4 weight, and the accumulator stays f32 (k runs to 1536
terms). It costs ~1e-4 block-level error against f32's ~2e-7, which is why
`mxfp4_moe_block.rs` takes a device-aware tolerance — still 20x clear of the
smallest layout mistake it must catch.

Metal is still 2.6x off its own f32 path (147.7 tok/s), because its MXFP4 kernels
remain byte-wise and per-thread-per-output. The CUDA split-K rewrite is the
template for closing that.

CPU dequantizes to f32 then BLAS-gemms, so it does strictly more work than f32
and MXFP4 stays a pure memory win there.

Backend support for the packed expert path:

| backend | MXFP4 experts | how |
|---|---|---|
| CUDA | yes | native decode-GEMM kernel |
| CPU | yes | fused nibble-decode accumulate |
| Metal | yes | **native** MSL kernels (`grouped_dequant_matmul_mlx_gemm`) |
| wgpu / ROCm / MLX | yes | not profiled — each has both a native and a host-delegate branch; do not assume which runs, check with that backend's thunk profile |
| CoreML / ANE | **no** | `DequantGroupedMatMulMlx` is not in `rlx-coreml`'s op set. Adding it by the route its sibling `DequantGroupedMatMul` takes would dequantize the bank to f32 and emit it as a MIL const — i.e. it would run, at full f32 memory cost, which defeats the point. Ling on CoreML stays on the f32 path. |

### Packing cost

Packing is per expert, not per bank: MXFP4 needs no transpose (the op contracts
along the last dim, so the stock `[E, N, K]` checkpoint order is already right),
so each expert's three tensors are quantized and appended immediately. The f32
high-water mark is one expert (~3 MB) against the f32 streaming path's
1.2 GB/layer staging buffers. Output is ~169 MB/layer, 3.91 GB total.

The RSS "peak" figures above are dominated by mmap'd checkpoint shards, which
are clean file-backed pages released when the loader drops — steady RSS is the
number that reflects real pressure.

## Status

Runs on the real 15.8 GB checkpoint on CPU, Metal and MLX, producing correct
continuations and parity-exact logits. wgpu and ROCm are correct on everything
except the full-model routed-expert path (above).

`--mxfp4` runs the whole model at 4 bits on CPU, Metal, MLX and wgpu, cutting
steady RSS 21.6 → 8.2 GB (see **MXFP4** above).

`--mxfp4` also makes the model **fit a 16 GB CUDA card for the first time**.

Remaining gaps:

* MXFP4 still costs speed off CUDA: Metal is 56.7 tok/s against its own f32
  path's 147.7. Its kernels are native but still byte-wise and
  one-thread-per-output; the CUDA split-K rewrite is the template and is the
  highest-value follow-up. wgpu/MLX/ROCm are unprofiled.
* CUDA's remaining hot spot is the dense MXFP4 GEMM (113 ms of a 266 ms
  prefill, ~58 ms of it `lm_head`). It still issues one 32-bit load per nibble,
  but **the word-wise fix that gave the grouped kernel 8x was tried here and
  measured worse** (`lm_head` 57.7 -> 69.0 ms) — this kernel splits K across its
  threads, so 8 elements per thread means 8x fewer active threads. The real
  waste is elsewhere: `MLX_TM = 8` means an m=64 GEMM launches 8 row-tiles per
  column and **each re-reads that column's whole packed weight** — 8 x 120 MB
  for `lm_head`'s 120 MB weight, ~2.1 GB/s of useful traffic on a ~400 GB/s
  card. Covering more rows per block (raise `MLX_TM`, or stage the decoded
  column in shared memory) is the lever; see the note at the top of
  `dequant_matmul_mlx_gemm`. After that, `GatedDeltaNet` at 1.8 ms x 18 layers.
* CoreML/ANE cannot use MXFP4 experts at all (op not in its set; see the table).
* An MXFP4 `lm_head` moves logits 16× more than the rest of the model
  (3.1e-2 vs 1.9e-3); `--f32-head` is the safe default for generation quality.
  The greedy-continuation impact has not been measured over a long generation.
* wgpu / ROCm / CUDA full-model divergence not root-caused (above).
* Vulkan untested.
