# rlx-timestral

Native Rust inference for **Mistral Timestral**, a family of zero-shot time-series
foundation models: [5M](https://huggingface.co/mistralai/Timestral-5M),
[28M](https://huggingface.co/mistralai/Timestral-28M),
[123M](https://huggingface.co/mistralai/Timestral-123M) and
[209M](https://huggingface.co/mistralai/Timestral-209M).

Upstream: [blog post](https://mistral.ai/news/timestral/),
[reference code](https://github.com/mistralai/timestral) (PyTorch). Weights are under
Mistral's Modified MIT license; this crate is GPL-3.0 like the rest of rlx-models.

## What it does

- Univariate and **multivariate** forecasting (the variates of a series are forecast jointly)
- **Past-only** and **known-future** covariates, numeric or **categorical** (target-encoded)
- Missing values as NaN, including gaps and late starts
- **21 quantiles** (0.01 … 0.99) per step, non-crossing
- Contexts up to 32,768 steps; horizons of any length, 1,024 steps per decode block
- Upstream's test-time defaults: symmetric averaging and non-negative clipping
- Runs on `cpu`, `metal`, `mlx`, `wgpu`, `ane` (verified), plus `cuda`, `rocm`, `vulkan` (built, not run)

## Commands

```bash
just fetch-timestral 209M                     # or 5M / 28M / 123M
just timestral -- --weights .cache/timestral/209M --context series.csv --horizon 96
just features=macos-gpu timestral -- --weights .cache/timestral/209M --device metal
just features=macos-gpu timestral -- --weights .cache/timestral/5M --devices all

just test-timestral                           # unit + synthetic backend check, no download
just test-timestral-parity 209M metal         # vs the upstream PyTorch package
```

`--context` takes one value per line or comma-separated; empty fields and `nan` are
missing values. The CLI prints the 5% / median / 95% quantiles per step.

## API

```rust
use rlx_timestral::{ForecastOptions, TimestralForecaster};
use rlx_runtime::Device;

let mut fc = TimestralForecaster::load(".cache/timestral/209M".as_ref(), Device::Metal)?;

// One series.
let forecast = fc.forecast_series(&history, 96)?;
let median = forecast.median();          // (1, 1, 96)
let p90 = forecast.quantile(0.9)?;

// (B, V, T) multivariate context with covariates.
let opts = ForecastOptions {
    known_covariates: Some(calendar),    // (B, C, T + H)
    past_covariates: Some(sensors),      // (B, C, T)
    ..ForecastOptions::default()
};
let forecast = fc.forecast(context.view(), 96, &opts)?;  // quantiles (B, V, 96, 21)
```

`ForecastOptions::raw()` turns off symmetric averaging and clipping (one pass instead of two).

## Number formats

Run the network with every weight and intermediate value rounded to a narrower
float format: `--number-format f16|bf16|f8|f6|f4|fNeXmY|fXmYeZ` on the CLI,
`model.set_number_format(Some(fmt))` in the API, or `RLX_NUMBER_FORMAT` for the
whole process. `weights/activations` gives each its own format: `f4/f8`, or
`f4/f32` for 4-bit weights with f32 compute. Formats of 8 bits or fewer scale
each 16-value block NVFP4-style, with a 3-bit-mantissa scale chosen so the
block's largest value lands at the top of the format (`@fb<N>` sets the block,
`@b<N>` uses OCP MX power-of-two scales, `@tensor` / `@none` the rest).
Arithmetic inside each op still accumulates in f32. The host math around the
network (scaling, refine frames, the quantile map) is f64 whatever the format.

For formats of 8 bits or fewer, as in FP8/FP4 inference, some values stay f32:
the residual stream (what the norms read), vectors such as norm scales and
biases (and values computed only from them), and RoPE tables. Held-out
weighted quantile loss on ETTh1, 5M, 7 columns × 20 forecast origins, horizon
96 (`examples/format_report.rs`):

| Format | WQL | vs f32 |
|---|---|---|
| f32 | 0.2051 | |
| bf16 / f16 | 0.2051 | 0.0% |
| f8 | 0.2068 | +0.8% |
| f6 | 0.2072 | +1.0% |
| f4 | 0.2457 | +20% |
| f4@fb8 | 0.2340 | +14% |
| f4/bf16 (4-bit weights) | 0.2132 | +3.9% |
| f4@fb8/f8 | 0.2126 | +3.6% |

With OCP MX scales and every value rounded, f8 cost 8.6%, f6 8.2% and f4 81%;
rounding the residual stream alone made f4 unusable (WQL 37). f4 QAT
(`rlx-timestral qat`, 300 steps, last 2000 steps held out) reaches 0.2319
(+13%). `RLX_NUMBER_FORMAT_ROUND_RESIDUAL=1` and `RLX_NUMBER_FORMAT_KEEP_PARAMS`
change the rules; `format_report` with a reference file instead of a CSV
measures distance from the f32 forecast.

`f64` runs the network natively in f64 on `--device cpu` and in double-single
(f32 pairs, ~48 bits) on Metal, wgpu, MLX, Vulkan and CUDA; `f64x2` is double-single
everywhere. Against upstream run entirely in float64
(`timestral_reference_dump.py --dtype float64`), 5M; CPU, Metal and wgpu give
identical errors and MLX and Vulkan pass the same check (CUDA has only been run on the
transformer-block test in `rlx-runtime/tests/f64_number_format.rs`):

| Case | f32 | f64 |
|---|---|---|
| random walk | 5.3e-7 | 9.8e-8 |
| 1,500-step horizon | 1.2e-5 | 7.4e-7 |
| multivariate with gaps | 1.0e-6 | 1.1e-7 |
| covariates | 1.7e-6 | 7.5e-8 |
| level 1e6 | 3.1e-8 | 3.1e-8 |
| non-negative | 1.2e-6 | 5.2e-8 |

(`raw` forecasts; `full` is the same or lower.) What remains under `f64` is f32
rounding at the network's inputs and outputs and in the f32 `Forecast`; at a level
of 1e6 that last rounding is the whole error. `just test-timestral-parity` checks it.

Values consumed only by an attention softmax (scores and mask) stay f32, as in
low-precision attention kernels — rounding them under block scaling flushes the
real scores next to a `-1e9` mask entry.

### Quantization-aware training

```bash
rlx-timestral qat --weights .cache/timestral/5M --data series.csv --number-format f8 \
  --steps 300 --lr 1e-4 --out /tmp/timestral-5m-f8
```

Trains on every column of the CSV except the last `--holdout` steps (default
64), with the model running in the chosen format (activations rounded in-graph,
weights rounded on each step's upload, gradients and master weights f32), and
exports an upstream-format checkpoint. The loss is the pinball loss over the 21
quantiles in the anchor frame; `forecast_query` stays frozen. Without
`--number-format` it is plain fine-tuning.

Held-out weighted quantile loss (lower is better) on ETTh1, Timestral-5M, last
128 steps of each of the 7 series, 300 steps at lr 1e-4:

| Format | Released model | QAT | f32 fine-tune, then rounded |
|---|---|---|---|
| f32 | 0.190 | — | 0.181 |
| bf16 / f16 | 0.190 | — | — |
| f8 | 0.212 | **0.174** | 0.200 |
| f6 | 0.206 | **0.187** | 0.206 |
| f4 | 17.2 | **0.668** | 4.05 |

## Architecture

| Part | Detail |
|---|---|
| Patching | 32 steps per patch; each patch embedded from `[values ‖ missing flags]` by a residual MLP |
| Normalization | causal per-patch running mean/std (Welford) then `asinh`; known covariates use whole-row stats |
| Sequence | `[Enc ; Dec]`: context patches, then learned forecast-query tokens |
| Layers | 8 (5M) to 24 (209M); every 4th layer adds variate attention after time attention |
| Time attention | causal over valid patches, partial RoPE (half of each head), sigmoid output gate |
| Variate attention | across the rows of a series per patch slot, with firewalls so covariates never read targets |
| MLP | SwiGLU, `gate * silu(up)` |
| Head | residual MLP to `32 × 21` values per decoder token, then a monotone map around the median |
| Decoding | block by block; each later patch of a block is re-denormalized in the advancing refine frame |

### Porting notes

- Upstream packs `wQKV` rows as `(qkv, head_dim, n_heads)`; the loader reorders them to `(qkv, n_heads, head_dim)`.
- Each attention's projections are pre-fused into one `[D, k·D]` matrix: q, k, v; for time attention also `q·P`, `k·P` (RoPE's `rotate_half` folded into the weights); then the gate. The attention scale is folded into q. Separate matmuls let the CPU compiler re-concatenate the weights on every call, which made the 209M model 3× slower.
- Contexts are left-padded with unobserved patches to a few bucket sizes, so only a few shapes ever compile. Padding patches are never valid keys, so this is exact (`bucket_padding_is_exact`).
- Host math is f64, as upstream keeps the causal statistics, refine frames and categorical codes (which is also why upstream cannot run on MPS). Codes arrive as f32 through the API, so they must stay below 2^24 to remain distinct.

## Parity

Against the upstream package (commit `a534bdb`) on the released weights. The cases:
a random walk, a 1,500-step horizon (multi-block refine), 2 series × 3 variates with
gaps and a late start, past + known + categorical covariates, a series at a level of
1e6, and a non-negative series. `raw` = no symmetric averaging or clipping, `full` =
upstream defaults. Errors are the largest absolute difference relative to the largest
reference value; `range` is relative to the reference's max − min, which a large
constant level cannot hide.

Worst case over cpu / metal / mlx / wgpu / ane:

| Model | `raw` | `full` | `full` vs range | hidden state |
|---|---|---|---|---|
| 5M | 9.8e-6 | 7.6e-6 | 1.5e-3 | 2.7e-5 |
| 28M | 9.3e-5 | 8.8e-5 | 1.7e-3 | 2.1e-6 |
| 123M | 2.1e-5 | 1.7e-5 | 1.7e-3 | 4.6e-6 |
| 209M | 3.1e-5 | 1.0e-4 | 1.8e-3 | 2.2e-6 |

The largest `raw`/`full` errors come from the 1,500-step horizon, where small
differences compound through the fed-back medians; every other case is ≤ 3.4e-6. The
`range` column is set by the level-1e6 case, where one f32 step (0.0625) is already
~1.5e-3 of the forecast's range; upstream's output is f32 as well.

The table was measured with f32 host math. Host math is f64 now (as upstream's); 5M on
the CPU is then at 7.4e-6 `raw`, 5.8e-6 `full`, 1.5e-3 vs range, 1.7e-5 hidden state.
When host math was f32, accumulating the causal statistics from raw running totals
moved the level-1e6 forecast by 26% of its range; they are still taken around the
row's first observation (see `scaler.rs`).

`ane` is bit-identical to `metal` in most cases, so CoreML likely delegates much of the graph.

## Speed

1,000-step context, 96-step horizon, defaults (two passes), warm, best of 5,
M-series. PyTorch is the upstream package on CPU (it cannot run on MPS).

| Model | rlx cpu | rlx metal | rlx mlx | PyTorch cpu |
|---|---|---|---|---|
| 5M | 13 ms | 4.7 ms | 8.9 ms | 18.8 ms |
| 209M | 218–252 ms | 26 ms | 28 ms | 169 ms |

Measured before inference packed each MLP's gate and up weights into one matmul
(the CPU compiler had been re-concatenating them on every call) and switched
attention and RoPE to the fused `Op::Attention` / `Op::Rope`, which removed
three quarters of the CPU's transposes and every materialized score tensor.
Not re-timed yet: `examples/bench.rs` and `scripts/timestral_bench_torch.py`
time both on the same input, but need an idle machine.
