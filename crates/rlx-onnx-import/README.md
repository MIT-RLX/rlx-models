# rlx-onnx-import (rlx-models)

Canonical development lives in the **`rlx` workspace**: [`../rlx/rlx-onnx-import`](../../../rlx/rlx-onnx-import).

This repo pins it via `[workspace.dependencies]` in the root `Cargo.toml` and optional `[patch.crates-io]` in `.cargo/config.toml`.

Do not add a duplicate crate here — import via:

```toml
rlx-onnx-import = { workspace = true }
```

## See also

- [rlx-onnx-decompose/README.md](../rlx-onnx-decompose/README.md) — ONNX → generated crate + safetensors
- Main repo [README](../../README.md#per-crate-readmes)
