# kitten_tts_mini_rlx

RLX native decomposition of [Kitten TTS mini](https://huggingface.co/KittenML/kitten-tts-mini-0.8) ONNX.

Published on crates.io as **`kitten_tts_mini_rlx`** (`weights/**` excluded from the crate tarball — export or download the ONNX bundle locally).

## Layout

| Path | Purpose |
|------|---------|
| `weights/rlx_bundle/` | Exported ONNX bundle (`manifest.json`, `graph.json`, `weights.safetensors`) |
| `src/bundle_compile.rs` | Bundle → HIR → compile (primary path) |
| `src/kernels.rs` | Model-specific CPU kernels (`onnx.QMatMul`, `onnx.ConcatFromSequence`, …) |
| `src/graph.rs` | Legacy hand-lowered graph (`--features generated-graph` only) |
| `decompose_report.json` | Op coverage from export |

## Export bundle

```bash
# From rlx-models root (needs: pip install onnx onnxshape numpy safetensors)
python3 scripts/export_kitten_rlx_bundle.py \
  /path/to/kitten_tts_mini_v0_8.onnx \
  crates/kitten_tts_mini_rlx/weights/rlx_bundle

# Or via just (set KITTEN_ONNX_PATH if not in HF cache)
just export-kitten-rlx-bundle
```

The export script bakes `__onnx_import__/duration_carry` into `/Expand_1` and `/Where_1`
so RLX import matches ORT single-pass duration semantics.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RLX_ONNX_BUNDLE` | Override bundle directory (default: `weights/rlx_bundle`) |
| `RLX_ONNX_SEQUENCE_LENGTH` | Active token count for compile-time shape restoration |
| `KITTEN_RLX_BUNDLE` | Legacy alias for `RLX_ONNX_BUNDLE` |
| `KITTEN_SEQUENCE_LENGTH` | Legacy alias for `RLX_ONNX_SEQUENCE_LENGTH` |
| `KITTEN_RLX_AOT_CACHE` | AOT compile cache directory |
| `KITTEN_RLX_SKIP_FUSION` | Set `1` to disable fusion (debug / probes) |
| `KITTEN_RLX_ENABLE_FUSION` | Set `1` to opt into fusion (off by default) |
| `RLX_ARENA_NO_REUSE` | Set automatically by bundle compile (arena safety) |

## Usage

```rust
use kitten_tts_mini_rlx::{bundle_compile, GraphOptions};
use rlx_runtime::Device;

let bundle = bundle_compile::bundle_dir_near_weights(weights_dir).unwrap();
let graph = bundle_compile::compile_from_bundle(
    Device::Cpu,
    &bundle,
    &GraphOptions { sequence_length: 8, max_waveform_samples: 24_000 },
)?;
```

## Tests

```bash
cargo test -p kitten_tts_mini_rlx
```

`tests/bundle_hir.rs` — model-specific HIR lowering checks (ported from `rlx-onnx-import`).
`tests/duration_alignment.rs` — `ConcatFromSequence` reference semantics.

## Import options

Bundle lowering uses `rlx_onnx_import::ImportOptions::quant_bundle()` (quant fusion rewrites,
relaxed strict mode for control-flow stubs). Generic ONNX import stays in `rlx-onnx-import`.
