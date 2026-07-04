# rlx-vision

NomicVision **encoder graph builder** for RLX — the image tower behind [`nomic-ai/nomic-embed-vision-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) (a ViT: 768-dim, 12 layers, 224px, patch size 16, CLS pooling). It shares an embedding space with [`rlx-nomic`](../rlx-nomic)'s text tower.

Like its siblings, this crate **only builds the graph**. Preprocessing, compiling, running, and pooling live in [`rlx-embed`](../rlx-embed).

## Public API

```rust
use rlx_vision::{NomicVisionFlow, build_nomic_vision_built, VisionPreprocessWeights};
use rlx_core::config::NomicVisionConfig;
use rlx_core::weight_map::WeightMap;

let built = build_nomic_vision_built(&cfg, &mut weights, /*batch*/ 1)?;
// or the configurable builder:
let built = NomicVisionFlow::new(&cfg, 1).build(&mut weights)?;
# anyhow::Ok(())
```

- `NomicVisionFlow` / `build_nomic_vision_built(cfg, w, batch) -> NomicVisionBuilt` — patch embedding → `repeat_vision_layers` (ViT blocks) → `hidden_states`, via native `rlx_flow::ModelFlow`.
- `NomicVisionBuilt` — the `BuiltModel` plus the extracted `VisionPreprocessWeights` (patch-embed / normalization constants pulled from the checkpoint).
- `build_vision_graph_sized` — lowered `Graph` for inspection.

The graph takes preprocessed pixel values and produces per-patch `hidden_states`; the CLS token is pooled downstream into a 768-dim image embedding.

## Build & test

```sh
cargo test -p rlx-vision                    # tiny ViT-graph build
cargo build -p rlx-vision --features metal  # metal | mlx | cuda | rocm | gpu | vulkan
```

To embed images end-to-end (preprocess → compile → CLS pool), use [`rlx-embed`](../rlx-embed).

## License

GPL-3.0-only. © 2026 Eugene Hauptmann, Nataliya Kosmyna.
