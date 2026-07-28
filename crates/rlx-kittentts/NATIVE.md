# Native RLX backend (`native` feature)

Runs the [`kitten_tts_mini_rlx`](../kitten_tts_mini_rlx) Rust graph via `rlx-runtime` (no ONNX Runtime).

```bash
cargo build -p rlx-kittentts --features native-fast --release
KITTEN_RLX_INFER=production just kittentts -- --native --ipa "həˈloʊ" --out-wav out.wav
```

Tier-1 compile profile sample: [`kitten.rlx.toml`](../kitten_tts_mini_rlx/kitten.rlx.toml) (loader TBD).

Policy for placement, wave caps, and QMatMul lives in
[`kitten_tts_mini_rlx::device_policy`](../kitten_tts_mini_rlx/src/device_policy.rs)
(`NativeEngine::load` → `prepare`).

## Build

```bash
cargo build -p rlx-kittentts --features native --release
# NVIDIA-style: cuda + wgpu + vulkan
cargo build -p rlx-kittentts --release --features native,cuda,gpu,vulkan
# Apple: metal / mlx / apple-silicon
cargo build -p rlx-kittentts --release --features native,apple-silicon
```

## Weights (recommended)

Primary path: **`model.safetensors`** or **`model.gguf`** — no `graph.json` at runtime.

```bash
just export-kitten-native-weights   # decompose ONNX → graph.rs + model.safetensors
just export-kitten-gguf             # optional GGUF container
```

Deploy under a KittenML checkpoint directory:

```text
checkpoint/
  config.json
  model.safetensors   # or model.gguf
  voices.npz
```

Legacy bundle path (`rlx_bundle/graph.json`) remains available; set `KITTEN_RLX_FORCE_BUNDLE=1` to prefer it.

## Backends

| Device | Duration graph | Wave / vocoder | Notes |
|--------|----------------|----------------|-------|
| `cpu` | on-device | on-device | Reference; native QMatMul on |
| `cuda` / `rocm` | on-device | on-device | Fastest long IPA on NVIDIA; native QMatMul on |
| `metal` / `mlx` | on-device | on-device | Native QMatMul **off** (zeros Metal wave) |
| `gpu` (wgpu) | **CPU** | **CPU** (default) | Upgrades to **Cuda** when available; else CPU wave. `GPU_WAVE=1` → Vulkan |
| `vulkan` | on-device | on-device | Prefer for native GPU when CUDA is unavailable |
| `ane` | CPU | CPU | Pinned off-device |

Force duration/wave placement:

| Variable | Effect |
|----------|--------|
| `KITTEN_RLX_CPU_DURATION=1` | Pin duration to CPU |
| `KITTEN_RLX_CPU_DURATION=0` / `RLX_KITTEN_GPU_DURATION=1` | Keep duration on the request device |
| `KITTEN_RLX_CPU_WAVE=1` | Pin vocoder to CPU |
| `KITTEN_RLX_GPU_WAVE=1` | On-device vocoder; without Cuda, discrete `gpu` uses **Vulkan** (`=wgpu` for slow wgpu) |
| `KITTEN_RLX_FORCE_WGPU=1` | Keep discrete `gpu` on wgpu (disable Cuda upgrade) |

## Discrete NVIDIA (`Gpu` / `Vulkan`)

Act arenas must stay **unsharded** (one storage-buffer bind window). Sharded Vulkan
no longer crashes, but audio collapses (near-silent NSF). Policy is applied once
at load via `device_policy::prepare`.

| Setting | `Device::Gpu` (wgpu) | `Device::Vulkan` | Why |
|------|----------------------|------------------|-----|
| Wave default | **→ Cuda** when available; else CPU / `GPU_WAVE`→Vulkan | on-device | Cuda single-pass long ~0.22 s peak 0.64 |
| Wave compile cap | **80 000** via Vulkan when `GPU_WAVE=1`; **32 000** if `GPU_WAVE=wgpu` | **80 000** | Caps &gt;80 k mush on NVIDIA (peak ~0.05); keep 80 k |
| Mel frames/token | import default (CPU wave); **8** when on-device | **8** | Keep unsharded |
| Stage reserve MiB | 64 (`RLX_WGPU_SHARD_STAGE_MIB`) | 64 (`RLX_VULKAN_SHARD_STAGE_MIB`) | Avoid 2×4 GiB snap from a 576 MiB default |
| `RLX_WGPU_NO_F16_SHADOW` | `1` | — | Skip +2 GiB f16 mirror |
| Native QMatMul | **on** (non-macOS) | **on** | ~3× faster long Vulkan vs QDQ host round-trips |

Long IPA is **wave-aware chunked** so each piece fits the cap when wave is on-device
(`infer_opts::chunk_plan_with_wave`, ~3 duration units/token × 600 samples).
Multi-chunk infer pads every piece to the same compile width so
`SeqCompileCache` hits once (avoids recompiling per chunk length).
Prefer **`cuda`** for a single-pass long waveform; **`vulkan`** or
`gpu` + `KITTEN_RLX_GPU_WAVE=1` for native GPU when CUDA is unavailable
(~1 s hello / ~11 s long, peak ~0.55). Default **`gpu`** keeps the vocoder on CPU.

### Override caps

| Variable | Default | Notes |
|----------|---------|-------|
| `KITTEN_RLX_WGPU_WAVEFORM_CAP` | 32000 | `0` / `off` disables; floor 24000 if set |
| `KITTEN_RLX_VULKAN_WAVEFORM_CAP` | 80000 | same |
| `KITTEN_RLX_MAX_FRAMES_PER_TOKEN` | 8 on Vulkan | Hard-capped by import `MAX_FRAMES_PER_TOKEN`; not auto-set on Gpu |
| `RLX_WGPU_SHARD_STAGE_MIB` | 64 (Kitten default) | Upstream default is 576 |
| `RLX_VULKAN_SHARD_STAGE_MIB` | 64 (Kitten default) | Upstream default is 576 |
| `RLX_WGPU_SHARD_GPU=1` | off | Force GPU kernels on discrete wgpu — **mush** (do not use for Kitten) |
| `RLX_WGPU_FORCE_HOST=1` | off | Force packed host fallbacks |
| `RLX_CUDA_CONV_TF32` | `1` on Cuda | Kitten default; opt out with `0` |
| `RLX_CUDA_CONV_FWD_CUDNN` | `1` on Cuda | Force cuDNN for 1×k / grouped fwd (HiFi-GAN); opt out with `0` |
| `RLX_CUDA_CONV_T_KERNEL` | off | Force naive ConvTranspose kernel (skip cuDNN) |
| `RLX_CUDA_DYN_LSTM_HOST` | off | Force host DynamicQuantizeLSTM (skip on-device path) |
| `RLX_CUDA_LSTM_CUDNN` | **on** | cuDNN LSTM for DynQuant (Kitten); set `0` to use hand kernel |
| `KITTEN_RLX_RNG_BACKEND` | **philox** on Cuda | On-device Philox; set `ort` for ORT-matching host noise |

### Validated peaks (RTX 3080 Ti, Jasper, production)

| Backend | Prompt | Wall (approx) | Peak |
|---------|--------|---------------|------|
| **Gpu → Cuda** (auto) | hello | **~0.17 s** infer | ~0.27 |
| **Gpu → Cuda** (auto) | long (single-pass) | **~0.22 s** infer | **~0.64** |
| Cuda (explicit) | long | ~0.22 s infer | ~0.64 |
| Gpu (no CUDA / `FORCE_WGPU`) | long (CPU wave) | ~4.4 s infer | ~0.63 |
| Gpu + `GPU_WAVE=1` (no CUDA) | hello → Vulkan | ~1.4 s infer | ~0.20 |
| Gpu + `GPU_WAVE=1` (no CUDA) | long → Vulkan 2×80 k | ~5.5 s infer | ~0.55 |
| Gpu + `GPU_WAVE=wgpu` | hello (wgpu host) | ~17 s | ~0.32 |

## Native QMatMul

Rewrites quantized ALBERT `onnx.QMatMul` (+ activation QDQ) into f32 GEMM
(`hir_qdq_fuse::rewrite_qmatmul_to_native_f32`). On f32-uniform GPU arenas the
quantized path is ~200 host round-trips per forward.

| Device | Default |
|--------|---------|
| Cpu, Cuda, Rocm, Vulkan | **on** |
| Gpu (non-macOS wgpu) | **on** |
| Metal, Mlx, Ane, macOS Gpu | **off** (zeros / garbles Metal wave) |

| Variable | Effect |
|----------|--------|
| `KITTEN_RLX_NATIVE_QMATMUL=1` | Force on |
| `KITTEN_RLX_NATIVE_QMATMUL=0` | Force off |

## Environment

| Variable | Purpose |
|----------|---------|
| `KITTEN_RLX_WEIGHTS` | Dir with `model.safetensors` / `model.gguf` |
| `KITTEN_RLX_FORCE_BUNDLE` | Use `rlx_bundle/graph.json` instead of native weights |
| `RLX_ONNX_BUNDLE` | Legacy bundle directory override |
| `KITTEN_RLX_AOT_CACHE` | AOT compile cache |
| `KITTEN_VOICES_NPZ` | Voice style table for parity tests |

## Duration loop

Bundle import uses [`run_with_duration_fixed_point`](../../kitten_tts_mini_rlx/src/bundle_compile.rs).
Native weights compile the full graph in one pass (duration computed in-graph).
The weights path injects a duration carry param and runs the same fixed-point
loop as the bundle importer (`run_with_duration_fixed_point`).

**Optimized infer (production default):** compiles a **waveform-only** graph (vocoder path, no dual-output
monolith). ORT or carry-seeded duration is supplied at infer time — same pattern as ONNX, which never
materializes a full `[seq × max_wave]` activation arena up front. Parity mode compiles split graphs plus
an optional full dual-output fallback (`KITTEN_RLX_COMPILE_FULL_FALLBACK`).

| Variable | Purpose |
|----------|---------|
| `KITTEN_RLX_INFER` | `production` (latency, low RAM) or `parity` (tests / ORT compare) |
| `KITTEN_RLX_FULL_GRAPH` | Force legacy dual-output single graph (high RAM; parity debugging) |
| `KITTEN_RLX_NATIVE_DURATION` | Production: also compile duration-refine graph (native duration loop) |
| `KITTEN_RLX_SINGLE_PASS` | Force ORT-style single graph run (no external duration iteration) |
| `KITTEN_RLX_DURATION_FIXED_POINT` | Force external duration carry iteration (parity) |
| `KITTEN_RLX_COMPILE_FULL_FALLBACK` | Also compile full graph when split mode is on |
| `KITTEN_RLX_SKIP_PREWARM` | Skip seq-bucket prewarm at `NativeEngine::load` |
| `KITTEN_RLX_PREWARM` | Opt in to prewarm in production (parity prewarms by default) |
| `KITTEN_RLX_PREWARM_BUCKETS` | Comma-separated token lengths to precompile (parity default `8,16,32,64,128`; production uses one bucket when prewarm is on) |
| `KITTEN_RLX_SEQ_CACHE_CAPACITY` | Max compiled seq graphs kept resident (production default `1`, parity `4`) |
| `KITTEN_RLX_TIMING` | Print load/prewarm and per-infer timings |
| `KITTEN_RLX_GRAPH_CACHE` | Keep compiled graphs in memory for the process |
| `KITTEN_RLX_QMATMUL_INGRAPH` | Skip per-op GPU sync for `onnx.QMatMul` (Metal default in production/parity) |
| `KITTEN_RLX_QDQ_FUSION` | Fold baked f32 weights into `onnx.QMatMulBaked` at compile (default on) |
| `KITTEN_RLX_NO_QDQ_FUSION` | Disable QMatMul weight bake + HIR fusion |
| `KITTEN_RLX_NO_MIR_QDQ_FUSION` | Disable MIR-level QMatMul bake pass (HIR fuse still runs) |
| `KITTEN_RLX_MIR_QDQ_FUSION` | Force MIR-level QMatMul bake pass |
| `RLX_METAL_PIPELINE_CACHE` | Directory for cached `.metallib` (default under AOT cache or temp) |
| `KITTEN_RLX_PARITY_PROFILE` | Run one `RLX_METAL_THUNK_PROFILE` infer on load in parity mode |
| `KITTEN_RLX_PARITY_ONNX` | Log native vs ONNX waveform alignment metrics in parity tests |
| `RLX_METAL_ONNX_QMATMUL_GPU` | Opt in: in-graph Metal f32 GEMM for large QMatMul (`m*k*n` ≥ min flops) |
| `KITTEN_RLX_QMATMUL_GPU` | Legacy opt-in: standalone sync GPU GEMM per QMatMul (usually slower) |
| `KITTEN_RLX_QMATMUL_GPU_MIN_FLOPS` | Min `m*k*n` for in-graph / standalone GPU GEMM (default 2097152) |
| `KITTEN_RLX_QMATMUL_PARALLEL` | Parallel CPU QMatMul rows when GPU path is off |
| `KITTEN_RLX_NO_ORT_WAVEFORM_FALLBACK` | Pure native: disable ORT waveform rescue when vocoder output is flat or underruns |
| `KITTEN_RLX_NO_ORT_DURATION` | Disable ORT duration oracle (trim/alignment hints); pure native needs oracle today |
| `KITTEN_RLX_ENABLE_NARROW_WAVEFORM_SLICE` | Opt-in alignment-driven `VocoderWaveformSlice` for narrow seq (default: static ONNX `Slice_3` + ORT duration trim) |
| `KITTEN_RLX_SPLIT_GRAPHS` | Split duration refine + waveform-only graphs (slow cold compile; waveform path seeds carry from ORT duration) |
| `KITTEN_RLX_NATIVE_DURATION_LOOP` | Force native duration refine loop instead of ORT carry seed on wide compile slots |
| `KITTEN_RLX_CHUNK_SLOTS` | Override padded-id chunk width for long IPA |

**Pure native** (`KITTEN_RLX_NO_ORT_WAVEFORM_FALLBACK=1`): no ORT waveform rescue. Requires ORT duration oracle for alignment unless `KITTEN_RLX_NO_ORT_DURATION=1`. After graph changes, bump `IMPORT_CACHE_TAG` or use a fresh `KITTEN_RLX_AOT_CACHE`.

**Split graphs** (`KITTEN_RLX_SPLIT_GRAPHS=1` + optional `KITTEN_RLX_WAVEFORM_ONLY_INFER=1`): waveform-only graphs need ORT duration seeded into `DURATION_CARRY`. First compile can take 10–15 minutes; prefer default full graph for development.

**Serve mode** (ORT-like hot latency — load once, many utterances):

```bash
cargo build -p rlx-kittentts --features native-fast,metal --release
KITTEN_RLX_INFER=production KITTEN_RLX_WEIGHTS=crates/kitten_tts_mini_rlx/weights \
  target/release/rlx-kittentts --native --serve
# then: həˈloʊ<TAB>/tmp/out.wav
```

| Variable | Purpose |
|----------|---------|
| `KITTEN_RLX_DURATION_REFINE` | Opt in to duration-refine graph in production (two-pass infer) |
| `KITTEN_RLX_WAVEFORM_ONLY_INFER` | Force single waveform graph (skip duration refine) |
| `KITTEN_RLX_PREFER_METAL` | Set `0` to keep CPU when `--device cpu` on macOS (production auto-picks Metal) |
| `KITTEN_RLX_SERIAL_COMPILE` | Compile dur + wave profiles sequentially (debug) |
| `KITTEN_RLX_ARENA_ALLOW_REUSE` | Force arena reuse on (production default) |
| `KITTEN_RLX_ARENA_NO_REUSE` | Force arena no-reuse |

Graphs compile at **runtime token width** by default (matches ORT `[1, seq]` for short utterances).

Vocoder noise: `NativeEngine::load` sets `KITTEN_RLX_RNG_SEED=42` when unset.
Parity tests set `KITTEN_RLX_PARITY=1` for zero-fill ORT comparison.

## Usage

```rust
use rlx_kittentts::{KittenTTS, Device};

let tts = KittenTTS::load_native(
    "crates/kitten_tts_mini_rlx/weights".as_ref(),
    "path/to/voices.npz".as_ref(),
    Default::default(),
    Default::default(),
    Device::Cpu,
    128,
    24_000,
)?;
let audio = tts.generate_from_ipa("həˈloʊ", "default", 1.0, 6)?;
```

On discrete NVIDIA, pass a large `--max-waveform-samples` if you like; `prepare`
clamps it (and logs) to the safe cap. Wave-aware chunking still covers long IPA.

## Bench / verify

```bash
# All available backends, short + long IPA, RTF
just bench-kittentts-backends
# Filter: KITTEN_TTS_BENCH_DEVICES=cpu,cuda,vulkan,gpu

export KITTEN_RLX_WEIGHTS=crates/kitten_tts_mini_rlx/weights
export KITTEN_VOICES_NPZ=/path/to/voices.npz
just test-kitten-native-compile
cargo test -p rlx-kittentts --features native --release native_infer_smoke -- --nocapture
just test-kittentts-native-parity
just test-kittentts-native-weights-parity
```

## See also

- [`device_policy.rs`](../kitten_tts_mini_rlx/src/device_policy.rs) — placement, caps, QMatMul
- [`kitten_tts_mini_rlx/README.md`](../kitten_tts_mini_rlx/README.md) — bundle / weights layout
- Upstream: `rlx-vulkan` host staging (`host_stage.rs`), `rlx-wgpu` `wgpu_prefer_host_fallback`
