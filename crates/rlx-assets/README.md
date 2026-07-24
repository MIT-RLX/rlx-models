# rlx-assets

Versatile model-asset loading for RLX crates: directory trees, single packed
files, in-memory maps, or custom providers — without hard-wiring `std::fs`.

## Features

| Feature | What it enables |
|---------|-----------------|
| *(default)* | [`AssetSource`] / [`AssetProvider`] — lean, no compiler deps |
| `rlxp` | Official [`.rlxp`](https://github.com/MIT-RLX/rlx/blob/main/docs/rlxp.md) (RLXPFLAT) open + pack helpers via `rlx-pkg` |
| `native-pack` | Pack-time ONNX → nested `graphs/*.rlxp` + outer Hub pack (**no** `.onnx` in TOC) |

## `AssetSource`

```rust,ignore
use rlx_assets::AssetSource;

let from_dir = AssetSource::dir("weights/tts/tiny-tts-rlx");
let from_pack = AssetSource::pack_file("tiny-tts.rlxp")?;
// Or: AssetSource::from("path/to/dir_or.rlxp") — auto-detects
```

Sub-loaders that need real paths (tokenizers, G2P) use `local_dir()` / materialize.

## Native subgraph packs (`native-pack`)

TTS models that used to ship ONNX on Hub now bake each subgraph into a nested
`.rlxp` (hot tensors + `graph.json`), then wrap those plus frontend/tokenizer
into one outer pack:

```text
model.rlxp
├── graphs/<name>.rlxp    # nested RLXPFLAT (weights + IR sidecars)
├── tokenizer.json        # cold file sidecars
└── frontend/…
```

```rust,ignore
use rlx_assets::native_pack::{
    load_native_subgraph_rlxp, install_native_subgraph_tls, pack_native_from_onnx_dir,
};

// Pack (local ONNX sources only):
pack_native_from_onnx_dir(src_dir, &specs, out_rlxp, "soprano")?;

// Load + lower (same thread):
let g = load_native_subgraph_rlxp(&path)?;
install_native_subgraph_tls(&g);
// → build_hir_from_parts(...)
```

See module docs on [`native_pack`](src/native_pack.rs). Hub publish scripts skip
ONNX for `rlx-native` repos (`scripts/publish_weights_hf.py`).
