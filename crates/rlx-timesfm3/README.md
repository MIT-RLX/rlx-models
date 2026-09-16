# rlx-timesfm3

Native Rust inference for **Google TimesFM-3** — a 330M-parameter zero-shot foundation model for **multivariate** time-series forecasting.

Upstream: [TimesFM-3 blog post](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [GitHub](https://github.com/google-research/timesfm), weights [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch).

## Features

- Stacked **Mixing Transformer** (causal temporal + full variate attention)
- **RevIN** normalization and **contiguous patch masking** (single-pass horizon decode)
- Past-only and past–future **covariates**
- **9 quantiles** (10th–90th percentile) per step
- Host CPU reference forward (loads official `model.safetensors`)
- **RLX compiled core** on all backends: `cpu`, `metal`, `mlx`, `cuda`, `rocm`, `wgpu`, `vulkan`

## Backends

```bash
# Per-backend quick check (synthetic tiny model)
cargo test -p rlx-timesfm3 --test backend_quick_check

# Metal / CUDA / etc. (enable matching feature)
cargo run -p rlx-timesfm3 --release --features metal -- \
  --synth --device metal --horizon 16

# Compare backends
cargo run -p rlx-timesfm3 --release --features all-backends -- \
  --weights .cache/timesfm3 --devices all --horizon 32
```

CPU inference uses the **host reference core** by default (bit-identical with the ndarray forward). GPU backends use the compiled graph. Set `RLX_TIMESFM3_COMPILED_CORE=1` to run the compiled core on CPU (parity debugging).

### Dev parity suite (`--features dev`)

```bash
cargo test -p rlx-timesfm3 --features dev --test core_parity_dev
```

| Test | What it checks |
|------|----------------|
| `resblock_exact_parity_cpu` | Compiled resblock == host (max diff `0.0`) |
| `decode_exact_parity_cpu` | Session CPU decode == host decode (`0.0`) |
| `compiled_core_tracks_host_cpu` | Compiled full core within `1e-2` of host (transformer SDPA gap) |

## License note

TimesFM **3.0 pretrained weights** are under the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch). This crate code is GPL-3.0 like the rest of rlx-models.

## Commands

```bash
# Unit test (tiny synthetic weights, no download)
cargo test -p rlx-timesfm3

# Demo forecast without weights
cargo run -p rlx-timesfm3 --release -- --synth --horizon 32

# Official checkpoint
huggingface-cli download google/timesfm-3.0-pytorch --local-dir .cache/timesfm3
cargo run -p rlx-timesfm3 --release -- --weights .cache/timesfm3 --horizon 128
```

## API

```rust
use rlx_timesfm3::TimesFM3ForecasterBuilder;

let mut forecaster = TimesFM3ForecasterBuilder::new()
    .weights(".cache/timesfm3")
    .device_name("metal")?
    .build()?;
let context: Vec<f32> = vec![/* history */];
let out = forecaster.predict(&context, 96, false)?;
println!("{:?}", out.forecast);
```

## Architecture (TimesFM-3)

| Component | Detail |
|-----------|--------|
| Patch size | 32 (context), 64 (output per patch) |
| Backbone | 20 × MixingTransformer, d=1280, 16 heads |
| Attention | Alternating causal **seq** + full **variate** |
| Decode | Non-autoregressive CPM — full horizon in one forward |
