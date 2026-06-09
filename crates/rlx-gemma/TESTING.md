# rlx-gemma testing matrix

Comprehensive guide to the test suites in this crate: what each
exercises, expected tolerances, when to set knobs, and how to run on
each backend.

## Quick start (Apple Silicon)

```bash
# Full suite on CPU + Metal + MLX + wgpu in one go.
cargo test -p rlx-gemma --features apple-silicon
```

Expected: 54 lib + 8 projector + 28 prefill LM + 4 decode + 8
fixture (3 auto-skip without `RLX_GEMMA4_FIXTURE`) + 4 throughput =
**106 passing, 0 failing**.

## Suites

| File | Tests | What it covers |
|---|---:|---|
| `--lib` | 52 | Config parsing, per-layer accessors, flow construction, multimodal builders, preprocessing, WAV/image decoders, tokenizer placeholders, multimodal runner |
| `tests/gemma4_backend_parity.rs` | 8 | Vision + audio projector graphs on CPU/Metal/MLX/wgpu, max\|Δ\| < 1e-3 vs CPU |
| `tests/gemma4_lm_backend_parity.rs` | 28 | Gemma 4 prefill (causal + sliding) + Gemma 1/2/3 regression + Metal isolation probes |
| `tests/gemma4_decode_backend_parity.rs` | 4 | Decode step with k_eq_v + split RoPE + **full per-layer KV divergence** (`global_head_dim=16` vs sliding 8, `num_global_kv_heads=1` vs sliding 4) |
| `tests/gemma4_reference_fixture.rs` | 8 | Top-5 token id match vs HF transformers reference dump (3 auto-skip without `RLX_GEMMA4_FIXTURE`) + 5 offline-flow validation tests (loader round-trip, top-k logic, error paths) |
| `tests/gemma4_decode_throughput.rs` | 4 | µs/step + tok/s on each backend for regression detection |
| `tests/gguf_parity.rs` | gated | llama.cpp parity for Gemma 1/2 (`--features parity-llama`) |

## Tolerances

Per-backend `max|Δ|` versus CPU on the tiny synthetic graphs in this
test suite:

| Backend | Tolerance | Measured | Notes |
|---|---|---|---|
| CPU | reference | n/a | Reference path |
| MLX | `1e-5` | ~`8e-8` | Essentially bit-exact (Apple's MLX uses fp32 throughout for f32 graphs) |
| wgpu | `1e-3` | ~`1e-6` | fp32-precise on Apple Silicon |
| Metal | `1e-3` | ~`1e-6` | **Requires** `RLX_METAL_PRECISE=1` (see below). Without it, drift is ~1e-1 due to `simdgroup_float8x8` reduced-precision accumulators |
| CUDA / ROCm / Vulkan | `1e-2` | not run on this Mac | Hardware-gated; the tests auto-skip when `is_available()` returns false |

## Runtime knobs

### `RLX_METAL_PRECISE=1`

Forces the scalar fp32 `sgemm` kernel on Metal, bypassing
`simdgroup_float8x8`'s reduced-precision tensor units. Set this for
parity / numerical-debugging work; leave unset for production
inference where the 10–100× throughput of the SIMD path matters.

The parity test files set it automatically via a `std::sync::Once`
guard so concurrent test runs don't race on the env var.

### `RLX_METAL_SGEMM_VARIANT={mps|simd4x4|simd|padded|tiled|naive}`

Lower-level matmul kernel selector. `naive` is equivalent to
`RLX_METAL_PRECISE=1`. See `rlx-metal/src/cost.rs::pick_sgemm` for
the auto-selection rules.

### `RLX_GEMMA4_FIXTURE=<dir>`

Path to a directory with HF Gemma 4 12B reference data
(`config.json`, `tokenizer.json`, `model.safetensors`, and a
`reference.json` produced by `scripts/dump_gemma4_reference.py`).
When set, the fixture parity test runs; when unset it auto-skips.

### Gemma 4 e2e bench env

| Variable | Effect |
|----------|--------|
| `RLX_GEMMA4_BENCH_LITE=1` | `short_chat` + `image_caption` only; smaller decode horizon |
| `RLX_GEMMA4_BENCH_FULL=1` | `hidden=3840` (12B width) |
| `RLX_GEMMA4_BENCH_LAYERS=N` | Layer count (default 6) |
| `RLX_GEMMA4_BENCH_HIDDEN=N` | Hidden size when not full scale (default 1024) |
| `RLX_GEMMA_METAL_UNFUSED_DECODE=1` | Disable Metal tier-1 decode fusion |
| `RLX_GEMMA_METAL_THUNK_DECODE=1` | Thunk decode compile (`RLX_DISABLE_MPSGRAPH=1`; slower, parity/debug) |
| `RLX_GEMMA_DYNAMIC_DECODE=1` | Dynamic decode cache (experimental; usually slower) |
| `RLX_GEMMA_GPU_KV=1` | GPU-resident K/V decode (opt-in; slower on Gemma Metal today) |

```bash
just gemma4-bench-lite          # fast Metal smoke
just gemma4-bench-metal         # full synthetic Metal suite
just gemma4-bench               # CPU → Metal → MLX → wgpu
```

Call `GemmaGenerator::sync_device()` between heavy phases to drain the
global queue and avoid MPS lifecycle warnings. Default decode uses
MPSGraph + fusion on Metal; set `RLX_GEMMA_METAL_THUNK_DECODE=1` only
when debugging bucketed-decode parity.

## Per-backend test invocations

```bash
# CPU only (default)
cargo test -p rlx-gemma

# Apple Silicon (Metal + MLX + wgpu)
cargo test -p rlx-gemma --features apple-silicon

# Compile-only check (cuda/rocm/vulkan don't need a live driver)
cargo test -p rlx-gemma --features cuda --no-run
cargo test -p rlx-gemma --features rocm --no-run
cargo test -p rlx-gemma --features vulkan --no-run

# Everything
cargo test -p rlx-gemma --features all-backends
```

## Remote CUDA execution

Pre-flight from a Mac with `rig.sh` configured to a CUDA host:

```bash
./rig.sh probe                    # confirm GPU + repo
./rig.sh sync                     # push workspace
./rig.sh cargo test -p rlx-gemma --features cuda --release
```

The tests `is_available()`-gated for CUDA will execute on the rig;
everything else self-skips. Tolerance for CUDA is `1e-2`.

## What each Gemma 4 op exercises

| Test | k_eq_v | Split RoPE | Partial RoPE | Per-layer KV | Sliding window | Softcap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `gemma4_lm_prefill_matches_cpu_*` (causal) | ✅ | ✅ | ✅ | (uniform shape — see below) | — | ✅ |
| `gemma4_lm_sliding_matches_cpu_*` | ✅ | ✅ | ✅ | uniform | ✅ | ✅ |
| `gemma4_decode_matches_cpu_*` | ✅ | ✅ | ✅ | **✅ diverging** (16 vs 8 head_dim, 1 vs 4 kv_heads) | — | ✅ |
| `legacy_gemma1_*` | — | — | — | — | — | — |
| `legacy_gemma2_*` | — | — | — | — | ✅ (alternating) | ✅ |
| `legacy_gemma3_*` | — | — | — | — | ✅ (stride-6) | ✅ |

**Notes:**

- The prefill LM parity tests use uniform per-layer KV shape because
  the **prefill** flow's `GemmaKvTapStage` was upgraded to honour
  per-layer rope_table/n_rot in commit landed alongside this doc.
  The diverging-shape variant is exercised end-to-end by the
  **decode** parity tests.
- Sliding window on Metal was a pre-existing kernel gap; landed as
  part of this work via `mask_kind = 4` in `rlx-metal/src/kernels.rs`.

## Adding new tests

When adding a parity test:

1. Add the new arch / feature variant to one of the existing
   `tests/*.rs` files (no need for a new file unless the setup is
   distinctly different).
2. Mirror the per-backend `#[cfg(...)]` + `is_available()` skip
   pattern from the existing tests so it cross-runs cleanly.
3. Set the right tolerance from the table above. If the backend is
   Metal, ensure `RLX_METAL_PRECISE=1` is set via `ensure_metal_precise()`.
4. Update this doc and `README.md` if the test surfaces a new
   coverage axis.
