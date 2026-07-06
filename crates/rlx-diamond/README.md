# rlx-diamond

**Diamond Maps** inference-time reward alignment for RLX — host-side math for value functions, GLASS stochastic flow maps, and weighted renoising from [arXiv:2602.05993](https://arxiv.org/abs/2602.05993). No retraining of the base generative model is required: GLASS posterior sampling uses the frozen denoiser as a reference. This crate is pure host math (no dependencies); model integration lives in `rlx-flux2::diamond`.

## Public API

The crate is a library of flat-tensor (`&[f32]`) numerical routines grouped by module:

| Module | What it provides |
|---|---|
| `glass` | GLASS reparameterization + velocity: `glass_velocity`, `reparam_input`, `reparam_time`, `sample_inner_state`, `sufficient_stat`, `calc_s`, `early_stop_ddpm` |
| `glass_integrate` | `DenoiserReference` trait + `sample_posterior` (integrate the inner ODE `s = 0..1` → posterior sample) |
| `guidance` | `guided_velocity`, `euler_step_vec`, `flux_guided_euler_step` |
| `interpolant` | `denoiser_from_velocity` / `velocity_from_denoiser`, `guidance_coefficient` |
| `value` | value-function aggregation: `log_mean_exp`, `logaddexp`, `softmax_weights`, `softmax_grad_aggregate`, `LogSumExpGrad` |
| `reward` | reward models + gradients: `LatentReward`, `LinearMeasurementReward`, `BluenessReward`, `grad_xt_via_z`, `spsa_grad` |
| `weighted` | weighted renoising: `renoise`, `renoise_params`, `score`, `guidance_b`, `particle_logit_full`, `t_prime_from_snr` |
| `flux_time` | Flux σ ⇄ paper-`t` conversions: `flux_sigma_to_paper_t`, `paper_t_to_flux_sigma`, `flux_x0_from_velocity`, `flux_guidance_coefficient`, … |

```rust
use rlx_diamond::{DenoiserReference, sample_posterior};

struct FrozenDenoiser;
impl DenoiserReference for FrozenDenoiser {
    fn denoise(&self, t_star: f32, x_star: &[f32], out: &mut [f32]) {
        // call the frozen base denoiser here
    }
}

// GLASS multi-step posterior sample z at t′, driven by the frozen denoiser
let mut z = vec![0.0f32; x_t.len()];
sample_posterior(&FrozenDenoiser, t, t_prime, &x_t, /*inner_steps*/ 4, &noise, &mut z);
```

## How it fits

The numerical primitives here are backend-agnostic host math; a diffusion model (Flux 2) wires them into a sampling loop via `rlx-flux2::diamond`, supplying the frozen denoiser as the `DenoiserReference`.

## Tests

```bash
cargo test -p rlx-diamond --release
```
