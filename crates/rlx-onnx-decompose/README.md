# rlx-onnx-decompose

Decompose an ONNX (`.onnx`) file into:

1. **Generated RLX Rust crate** — `src/graph.rs` builds an `HirModule` (same ops as `rlx-onnx-import`)
2. **External weights** — `weights/model.safetensors` (optional `model.gguf` via Python helper)

This is the path toward **native RLX** models (like `rlx-ocr`): Rust graph code + safetensors on disk, no ONNX Runtime at inference time.

## CLI

```bash
cargo build -p rlx-onnx-decompose --release

rlx-onnx-decompose model.onnx -o crates/my_model_rlx \
  --weights safetensors \
  --seq-len 128 \
  --crate-name my_model_rlx

# Prefer bundle export (has shape metadata + quant rewrite):
rlx-onnx-decompose --bundle /tmp/kitten-mini-rlx-bundle \
  -o crates/kitten_tts_mini_rlx --crate-name kitten_tts_mini_rlx
```

### Options

| Flag | Description |
|------|-------------|
| `-o DIR` | Output directory (required) |
| `--weights safetensors\|gguf` | Weight format (default: safetensors) |
| `--crate-name NAME` | Rust crate name (default: from ONNX filename) |
| `--seq-len N` | Bind `sequence_length` symbolic dim (default: 128) |
| `--max-samples N` | Bind `num_samples` symbolic dim (default: 24000) |
| `--bundle DIR` | Decompose from RLX bundle (`manifest.json`, `graph.json`, `weights.safetensors`) instead of raw ONNX |
| `--rlx-root PATH` | Path to `rlx` repo for generated `Cargo.toml` deps (default: auto) |

## Output layout

```
my_model_rlx/
  Cargo.toml
  README.md
  decompose_report.json   # op coverage
  weights/
    model.safetensors
    model.gguf            # if --weights gguf
  src/
    lib.rs                # compile(), re-exports
    graph.rs              # build_graph() — AUTO-GENERATED
    weights.rs            # load_weights()
```

Place the output under `rlx-models/crates/<name>/` so path deps to `rlx-ir` / `rlx-runtime` resolve.

## Library API

```rust
use rlx_onnx_decompose::{decompose, DecomposeOptions, WeightsFormat};

decompose(
    "model.onnx".as_ref(),
    "out/my_model_rlx".as_ref(),
    &DecomposeOptions {
        weights_format: WeightsFormat::Safetensors,
        ..Default::default()
    },
)?;
```

## GGUF export

`--weights gguf` writes safetensors first, then runs:

```bash
python3 scripts/onnx_decompose_to_gguf.py weights/model.safetensors weights/model.gguf
```

Requires: `pip install gguf safetensors numpy`

## Extending coverage

Unsupported ONNX ops appear as stubs in `graph.rs` and in `decompose_report.json`. Add lowering in `rlx-onnx-import/src/lower.rs`, then re-run decompose.

## See also

- [kitten_tts_mini_rlx/README.md](../kitten_tts_mini_rlx/README.md) — example decomposed TTS bundle
- Main repo [README](../../README.md#per-crate-readmes)
- [AGENTS.md](../../AGENTS.md)
