# kitten_tts_mini_rlx

RLX native decomposition of [Kitten TTS mini](https://huggingface.co/KittenML/kitten-tts-mini-0.8) ONNX.

Published on crates.io as **`kitten_tts_mini_rlx`** (`weights/**` excluded from the crate tarball — export or download weights locally).

## Layout

| Path | Purpose |
|------|---------|
| `weights/model.safetensors` | Native weights (decomposed from ONNX) |
| `weights/model.gguf` | Optional GGUF container (`just export-kitten-gguf`) |
| `weights/rlx_bundle/` | Legacy ONNX bundle (`graph.json` + safetensors) |
| `src/native/` | Native feature: config, compile cache, module map |
| `src/bundle_compile.rs` | Bundle / HIR import, infer entry points |
| `src/device_policy.rs` | **Device placement, discrete NVIDIA wave caps, native QMatMul** |
| `src/kernels.rs` | Model-specific CPU kernels (`onnx.QMatMul`, …) |
| `decompose_report.json` | Op coverage from export |

## Native weights (recommended)

No `graph.json` at runtime — architecture is imported from the RLX bundle / native flow.

```bash
just export-kitten-native-weights   # bundle export + rlx-onnx-decompose
just export-kitten-gguf             # optional GGUF copy of safetensors
just test-kitten-native-compile     # compile check (needs weights/)
```

Deploy layout:

```text
checkpoint/
  model.safetensors   # or model.gguf
  voices.npz
  config.json
```

Build with `--features native`. `rlx-kittentts --features native` enables this path automatically when `model.safetensors` is present.

Set `KITTEN_RLX_FORCE_BUNDLE=1` to keep using `rlx_bundle/graph.json` instead.

## Device policy

[`src/device_policy.rs`](src/device_policy.rs) is the single place for:

| Concern | API | Summary |
|---------|-----|---------|
| Duration placement | `duration_device` | CPU on wgpu / ANE; on-device Vulkan/Cuda/Metal |
| Wave placement | `wave_device` | CPU on ANE + discrete wgpu; on-device elsewhere |
| Wave / frame caps | `prepare` / `clamp_waveform` | Cap only when wave is on-device; frames=8 for Vulkan / `GPU_WAVE=1` |
| Native QMatMul | `native_qmatmul` | On for Cpu/Cuda/Rocm/Vulkan/discrete Gpu |

`rlx-kittentts` calls `prepare(device, max_waveform_samples)` at `NativeEngine::load`.
Full tables, env overrides, and MSI numbers: [rlx-kittentts/NATIVE.md](../rlx-kittentts/NATIVE.md).

Constants: `WGPU_WAVEFORM_CAP` (32 000), `VULKAN_WAVEFORM_CAP` (80 000),
`DISCRETE_MAX_FRAMES_PER_TOKEN` (8). Legacy `DISCRETE_WGPU_*` aliases remain.

**Do not** raise wave caps past the unsharded window: sharded arenas produce near-silent audio even when they no longer SIGSEGV.

## Legacy bundle export

```bash
just export-kitten-rlx-bundle
```

The export script bakes `__onnx_import__/duration_carry` into `/Expand_1` and `/Where_1`
so RLX import matches ORT single-pass duration semantics.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `KITTEN_RLX_FORCE_BUNDLE` | Prefer `rlx_bundle/graph.json` over native weights |
| `RLX_ONNX_BUNDLE` | Override bundle directory |
| `RLX_ONNX_SEQUENCE_LENGTH` | Active token count for compile-time shape restoration |
| `KITTEN_RLX_BUNDLE` | Legacy alias for `RLX_ONNX_BUNDLE` |
| `KITTEN_RLX_AOT_CACHE` | AOT compile cache directory |
| `KITTEN_RLX_SKIP_FUSION` | Set `1` to disable fusion (debug / probes) |
| `KITTEN_RLX_WGPU_WAVEFORM_CAP` | Discrete wgpu wave clamp (`0`/`off` disables) |
| `KITTEN_RLX_VULKAN_WAVEFORM_CAP` | Vulkan wave clamp |
| `KITTEN_RLX_MAX_FRAMES_PER_TOKEN` | Mel frames/token (Vulkan / `GPU_WAVE=1` default 8) |
| `KITTEN_RLX_NATIVE_QMATMUL` | `0`/`1` force quantized vs native GEMM rewrite |
| `KITTEN_RLX_CPU_DURATION` / `KITTEN_RLX_CPU_WAVE` | Pin duration / vocoder to CPU |
| `KITTEN_RLX_GPU_WAVE` | Force on-device vocoder; discrete `gpu` routes to **Vulkan** (~1 s hello). Use `=wgpu` for the slow wgpu path |
| `RLX_WGPU_SHARD_STAGE_MIB` / `RLX_VULKAN_SHARD_STAGE_MIB` | Stripe stage reserve (Kitten sets 64) |

## Usage

```rust
use kitten_tts_mini_rlx::{compile, prepare, GraphOptions};
use rlx_runtime::Device;

let device = Device::Vulkan;
let max_wave = prepare(device, 160_000); // clamps on discrete NVIDIA
let graph = compile(
    device,
    weights_dir.as_ref(),
    &GraphOptions {
        sequence_length: 128,
        max_waveform_samples: max_wave,
    },
)?;
```

## Module map

See [`src/native/config.rs`](src/native/config.rs) (`ModuleKind`) and [`src/native/flow/modules.rs`](src/native/flow/modules.rs) for semantic boundaries (bert, text encoder, mel decoder, predictor, duration, vocoder).

## Tests

```bash
cargo test -p kitten_tts_mini_rlx --lib device_policy
cargo test -p kitten_tts_mini_rlx --features native
cargo test -p rlx-kittentts --features native native_infer_smoke
```
