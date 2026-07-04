# rlx-narma10

Reference **NARMA-10** timeseries generator and reservoir-computing predictors for the RLX stack.

Unpublished workspace crate (`publish = false`) used to validate RLX recurrence graphs and to benchmark echo-state networks (ESNs) on the standard order-10 nonlinear autoregressive benchmark.

## NARMA-10 recurrence

```text
y[t+1] = α·y[t] + β·y[t]·Σ_{i=0}^{9} y[t−i] + γ·u[t−9]·u[t] + δ
```

| Symbol | Value |
|--------|-------|
| `u[t]` | `Uniform(0, 0.5)` |
| `(α, β, γ, δ)` | `(0.3, 0.05, 1.5, 0.1)` |
| `y[t]` for `t < 0` | `0` |

[`host::generate`](src/host.rs) is the CPU reference. [`rlx::generate_on_device`](src/rlx.rs) runs the same recurrence through a compiled RLX graph on any enabled backend.

## Predictors

Three literature-style models share one training protocol ([`TrainConfig`](src/models/train.rs)) and are compared via [`bench_predictors`](src/models/train.rs):

| Model | Type | Reservoir | Readout |
|-------|------|-----------|---------|
| [`EsnRidge`](src/models/esn_ridge.rs) | Dense ESN | N=300, ρ=0.9 (Nakajima RC-tutorial) | Linear + ridge |
| [`LocalEsn`](src/models/local_esn.rs) | Locally connected ESN | 800 units, 20×40 grid (LCESN) | Linear + ridge |
| [`PolyReadoutEsn`](src/models/poly_readout.rs) | Dense ESN | N=400 | Quadratic features + ridge |

**ESN update** (input-driven, no output feedback): after injecting `u[t]`, state is `x[t+1] = tanh(W_res·x[t] + W_in·u[t])`. The readout predicts `y[t+1]` from `x[t+1]` (one-step-ahead, teacher-forced evaluation on the test segment).

**NRMSE** — `√(MSE / Var(y))` on the held-out target segment (population variance).

## Benchmark protocols

### LCESN paper (`TrainConfig::lcesn`)

Matches Matzner & Mráz (ICLR 2025) / long-sequence RC practice for locally connected reservoirs:

| Setting | Value |
|---------|-------|
| Washout | 1000 |
| Train samples (post-washout) | 12_000 |
| Test samples | ~1_010 |
| Total timesteps | 14_000 |
| `local_esn` reservoir | 800 units (20×40), kernel 7 |

Constants: `LCESN_WASHOUT`, `LCESN_TRAIN_SAMPLES`, `LCESN_TEST_SAMPLES`, `LCESN_TIMESTEPS`.

Typical test NRMSE at seed 42: `esn_ridge` ~0.19, `local_esn` ~0.38, `poly_readout_esn` ~0.11 (beats Kodali et al. dense ESN ~0.18).

### Quick check (`TrainConfig::quick`)

Nakajima-style dense ESN sanity check (fast CI):

| Setting | Value |
|---------|-------|
| Timesteps | 5_000 |
| Washout | 100 |
| Train fraction | 0.75 |

## Usage

```rust
use rlx_narma10::{
    EsnRidge, LocalEsn, Narma10Predictor, TrainConfig, bench_predictors, generate, nrmse,
};

let series = generate(14_000, 42);
let cfg = TrainConfig::lcesn();

let mut esn = EsnRidge::new();
let report = esn.fit(&series, &cfg)?;
let pred = esn.predict_all(&series, &cfg)?;
let test_nrmse = nrmse(&pred[report.split_index..], &series.targets[report.split_index..]);

let rows = bench_predictors(&series, &cfg)?;
```

RLX device generation:

```rust
use rlx_narma10::rlx::generate_on_device;
use rlx_runtime::Device;

let series = generate_on_device(Device::Metal, 1_000, 7)?;
```

## Commands

```bash
# LCESN paper protocol (default in the example)
cargo run -p rlx-narma10 --example bench_predictors --release -- --protocol=lcesn --seed=42

# Quick Nakajima-style check
cargo run -p rlx-narma10 --example bench_predictors --release -- --protocol=quick --seed=42

# Unit + predictor tests
cargo test -p rlx-narma10 --release

# RLX graph parity (CPU / Metal / MLX / CUDA / …)
cargo test -p rlx-narma10 --features all-backends --release --test backend_parity
```

## Backends

| Path | Role |
|------|------|
| `host::*` | CPU reference recurrence and metrics |
| `rlx::*` | Compiled unrolled graph via `rlx-runtime` |
| `models::*` | ESN reservoirs + ridge readout (CPU) |

Enable GPU backends with crate features: `metal`, `mlx`, `cuda`, `rocm`, `vulkan`, or `all-backends` (forwards to `rlx-runtime`).

## References

- Atiya & Parlos (2000) — NARMA benchmark
- Jaeger (2003) — ESN NARMA-10 task
- Nakajima RC-tutorial — dense ESN hyperparameters (N=300, ρ=0.9, washout 100)
- Kodali et al. (2025) — NARMA-10 ESN baseline NRMSE ~0.18
- Matzner & Mráz (ICLR 2025) — locally connected ESN (LCESN)
