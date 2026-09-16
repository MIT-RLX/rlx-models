// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Training the denoiser: loss graph, autodiff, Adam.
//!
//! One graph does both passes. The forward graph ends at a scalar loss;
//! [`rlx_autodiff::grad_with_loss`] rewrites it into a graph whose outputs are
//! `[loss, ∂loss/∂p …]` in the order the parameters were handed to it. Compile
//! that once, then every step is: bind the parameters, run, apply Adam.
//!
//! # The loss
//!
//! Relative L2 — `mean((y - t)² / (t² + ε))` — not plain L2.
//!
//! A render is high dynamic range: a lit highlight can be a thousand times a
//! shadow, and plain L2 in linear radiance is then decided almost entirely by
//! the brightest pixels. A network trained on it learns to polish highlights
//! and leaves visible grain across everything darker, which is the opposite of
//! what the eye notices. Dividing by the target's own magnitude asks how wrong
//! each pixel is *relative to what it should be*.
//!
//! ε keeps near-black pixels — where the relative error is unbounded and means
//! nothing — from deciding the gradient.

use anyhow::{Result, ensure};
use rlx_autodiff::grad_with_loss;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::model::{DenoiseNet, OUT_CHANNELS, init_params};

/// Floor in the relative-L2 denominator, for **reporting**.
///
/// Every number this crate has ever printed uses this value, including the
/// OpenImageDenoise comparisons, so it is fixed. Changing it would move the
/// scoreboard along with the model and make two runs incomparable.
pub const LOSS_EPSILON: f32 = 0.01;

/// Floor in the relative-L2 denominator, for **training**.
///
/// The reporting value turns out to be badly scaled for what a path tracer
/// actually produces. Measured over the CAD test set: the median target pixel
/// is 0.038, so `t²` is 0.0015 and the 0.01 floor is seven times larger. For
/// **70% of every image** the denominator is the constant rather than the
/// pixel, which makes the loss plain L2 exactly where the relative form was
/// introduced to avoid it — and dark regions are where Monte-Carlo noise is
/// worst.
///
/// | floor | crossover `t` | pixels genuinely relative |
/// |---|---|---|
/// | 1e-2 | 0.100 | 30% |
/// | 1e-3 | 0.032 | 55% |
/// | 1e-4 | 0.010 | 81% |
///
/// Not smaller than this: the weight on a pixel goes to `1/ε` as `t → 0`, so a
/// very low floor hands near-black pixels enormous gradients, which is what the
/// generous original value was avoiding.
pub const TRAIN_EPSILON: f32 = 1e-3;

/// Optimiser settings. The defaults are Adam's usual ones; only the learning
/// rate normally wants changing.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// Total optimiser steps the run will take, for the decay below. 0 holds
    /// the learning rate constant.
    pub total_steps: u32,
    /// Fraction of the initial learning rate to end at, following a cosine.
    ///
    /// A constant rate cannot settle: near convergence every step is large
    /// compared with the distance left, so validation error oscillates instead
    /// of flattening — measured here as a swing of 0.0059 between neighbouring
    /// epochs at a point where the run had stopped improving. Decaying to a
    /// twentieth lets the last epochs actually land.
    pub final_lr_fraction: f32,
    /// Clip the global gradient norm to this. 0 disables.
    ///
    /// A render's outliers are real — one path finding a light through a
    /// near-specular chain carries thousands of times the mean — and a single
    /// batch containing one can produce a step large enough to leave the
    /// network permanently at NaN. Clipping bounds that without discarding the
    /// sample.
    pub grad_clip: f32,
    /// Weight on the image-gradient term described in [`GRADIENT_DILATIONS`].
    /// 0 leaves the loss as plain relative L2.
    pub gradient_weight: f32,
}

/// Spacings at which the gradient term compares differences.
///
/// Relative L2 is a per-pixel measure, so a filter can spread error smoothly
/// over a neighbourhood and pay almost nothing for it. What that looks like in
/// a render is low-frequency mottling — patches of invented shading across a
/// surface the reference shows as flat — which is the artifact left on glass
/// and dark glossy surfaces once the per-pixel error is already small.
///
/// Comparing *differences* between neighbouring pixels prices that in. One
/// spacing is not enough: a blotch tens of pixels across barely registers in a
/// difference between adjacent pixels, so the same 3x3 kernel is applied
/// dilated, and the term sees structure at each scale.
pub const GRADIENT_DILATIONS: [usize; 3] = [1, 3, 9];

/// Depthwise `[6, 1, 3, 3]` kernel: per colour channel, a forward difference
/// along x and one along y. Fed as an input rather than a parameter, so
/// autodiff leaves it alone.
fn gradient_kernel() -> Vec<f32> {
    let dx: [f32; 9] = [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0];
    let dy: [f32; 9] = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0];
    let mut k = Vec::with_capacity(6 * 9);
    for _ in 0..OUT_CHANNELS {
        k.extend_from_slice(&dx);
        k.extend_from_slice(&dy);
    }
    k
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            total_steps: 0,
            final_lr_fraction: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            grad_clip: 1.0,
            gradient_weight: 0.0,
        }
    }
}

/// One training batch: `n` tiles of `h × w`.
///
/// `input` is `[n, 9, h, w]` — colour, albedo, normal — and `target` is
/// `[n, 3, h, w]`, the converged render. Both are planar and contiguous.
pub struct Batch<'a> {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub input: &'a [f32],
    pub target: &'a [f32],
}

impl Batch<'_> {
    fn check(&self, inputs: usize) -> Result<()> {
        let pixels = self.n * self.h * self.w;
        ensure!(
            self.input.len() == pixels * inputs,
            "input is {} floats, expected {}",
            self.input.len(),
            pixels * inputs
        );
        ensure!(
            self.target.len() == pixels * OUT_CHANNELS,
            "target is {} floats, expected {}",
            self.target.len(),
            pixels * OUT_CHANNELS
        );
        Ok(())
    }
}

/// A compiled training step, its parameters, and the Adam state behind them.
pub struct Trainer {
    net: DenoiseNet,
    graph: CompiledGraph,
    params: Vec<Vec<f32>>,
    moment1: Vec<Vec<f32>>,
    moment2: Vec<Vec<f32>>,
    config: TrainConfig,
    tile: (usize, usize, usize),
    step: u32,
}

impl Trainer {
    /// Compile a training step for one batch shape.
    ///
    /// The shape is baked in: the graph is compiled once against `[n, 9, h, w]`
    /// and every batch has to match. Training on fixed-size tiles rather than
    /// whole frames is what makes that reasonable — and it is what lets the
    /// batch dimension do any work at all, since renders come in different
    /// sizes.
    pub fn new(
        net: DenoiseNet,
        n: usize,
        h: usize,
        w: usize,
        device: Device,
        config: TrainConfig,
    ) -> Result<Self> {
        ensure!(n > 0 && h > 0 && w > 0, "empty batch shape {n}x{h}x{w}");
        ensure!(
            h.is_multiple_of(DenoiseNet::TILE_MULTIPLE)
                && w.is_multiple_of(DenoiseNet::TILE_MULTIPLE),
            "tile {h}x{w} must be a multiple of {}",
            DenoiseNet::TILE_MULTIPLE
        );

        let mut forward = Graph::new("denoise_loss");
        let input = forward.input("input", Shape::new(&[n, net.inputs(), h, w], DType::F32));
        let target = forward.input("target", Shape::new(&[n, OUT_CHANNELS, h, w], DType::F32));

        let predicted = net.forward(&mut forward, input, n, h, w);

        // mean((y - t)^2 / (t^2 + eps))
        let residual = forward.sub(predicted, target);
        let squared = forward.mul(residual, residual);
        let scale = forward.mul(target, target);
        let eps = forward.input("loss_eps", Shape::new(&[1], DType::F32));
        let denom = forward.add(scale, eps);
        let relative = forward.div(squared, denom);
        let mut loss = forward.mean(relative, vec![0, 1, 2, 3], false);

        if config.gradient_weight > 0.0 {
            let kernel = forward.input(
                "grad_kernel",
                Shape::new(&[2 * OUT_CHANNELS, 1, 3, 3], DType::F32),
            );
            let weight = forward.input("grad_weight", Shape::new(&[1], DType::F32));
            // Padding tracks dilation so a 3x3 kernel keeps the tile's size; the
            // absolute size does not matter here, only that both sides match.
            for d in GRADIENT_DILATIONS {
                let conv = |g: &mut Graph, x| {
                    g.conv2d(x, kernel, [3, 3], [1, 1], [d, d], [d, d], OUT_CHANNELS)
                };
                let gy = conv(&mut forward, predicted);
                let gt = conv(&mut forward, target);
                let gres = forward.sub(gy, gt);
                let gsq = forward.mul(gres, gres);
                let gscale = forward.mul(gt, gt);
                let gdenom = forward.add(gscale, eps);
                let grel = forward.div(gsq, gdenom);
                let gmean = forward.mean(grel, vec![0, 1, 2, 3], false);
                let scaled = forward.mul(gmean, weight);
                loss = forward.add(loss, scaled);
            }
        }
        forward.set_outputs(vec![loss]);

        // `wrt` must be in the same order as `net.params()`, because that is the
        // order the gradients come back in and the order Adam's state is kept.
        let wrt: Vec<_> = net
            .params()
            .iter()
            .map(|s| forward.param(s.name, Shape::new(s.shape.as_ref(), DType::F32)))
            .collect();
        let backward = grad_with_loss(&forward, &wrt);
        let graph = Session::new(device).compile(backward);

        let params = init_params(&net, 0x5eed_1234);
        let moment1 = params.iter().map(|p| vec![0.0; p.len()]).collect();
        let moment2 = params.iter().map(|p| vec![0.0; p.len()]).collect();

        Ok(Self {
            net,
            graph,
            params,
            moment1,
            moment2,
            config,
            tile: (n, h, w),
            step: 0,
        })
    }

    pub fn net(&self) -> &DenoiseNet {
        &self.net
    }

    /// The current weights, in `net.params()` order.
    pub fn params(&self) -> &[Vec<f32>] {
        &self.params
    }

    /// Replace the weights — for resuming, or for evaluating a checkpoint.
    pub fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        ensure!(
            params.len() == self.params.len()
                && params
                    .iter()
                    .zip(&self.params)
                    .all(|(a, b)| a.len() == b.len()),
            "parameter shapes do not match this network"
        );
        self.params = params;
        Ok(())
    }

    pub fn steps_taken(&self) -> u32 {
        self.step
    }

    /// One optimiser step. Returns the loss *before* the update, which is the
    /// loss of the weights that produced the gradient.
    pub fn step(&mut self, batch: &Batch<'_>) -> Result<f32> {
        batch.check(self.net.inputs())?;
        let (n, h, w) = self.tile;
        ensure!(
            batch.n == n && batch.h == h && batch.w == w,
            "batch is {}x{}x{}, the graph was compiled for {n}x{h}x{w}",
            batch.n,
            batch.h,
            batch.w
        );

        for (spec, values) in self.net.params().iter().zip(&self.params) {
            self.graph.set_param(spec.name, values);
        }
        // The seed of the chain rule: ∂loss/∂loss. rlx makes it an input rather
        // than a baked-in 1.0, so leaving it unbound is a silent zero — an
        // unbound input is zeros, and zero times anything is no gradient.
        let seed = [1.0f32];
        let eps = [TRAIN_EPSILON];
        let mut feeds: Vec<(&str, &[f32])> = vec![
            ("input", batch.input),
            ("target", batch.target),
            ("loss_eps", &eps),
            ("d_output", &seed),
        ];
        // Bound outside the `if` so the slices outlive the call.
        let kernel = gradient_kernel();
        let gweight = [self.config.gradient_weight];
        if self.config.gradient_weight > 0.0 {
            feeds.push(("grad_kernel", &kernel));
            feeds.push(("grad_weight", &gweight));
        }
        let out = self.graph.run(&feeds);

        let loss = out
            .first()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(f32::NAN);
        if !loss.is_finite() {
            return Ok(loss);
        }
        ensure!(
            out.len() == self.params.len() + 1,
            "expected {} gradients, got {}",
            self.params.len(),
            out.len().saturating_sub(1)
        );

        let scale = self.gradient_scale(&out[1..]);
        self.step += 1;
        let TrainConfig {
            beta1,
            beta2,
            epsilon,
            ..
        } = self.config;
        let learning_rate = self.current_learning_rate();
        // Bias correction: the moments start at zero, so early steps would
        // otherwise be scaled toward zero and the first few hundred updates
        // wasted.
        let correction1 = 1.0 - beta1.powi(self.step as i32);
        let correction2 = 1.0 - beta2.powi(self.step as i32);

        for i in 0..self.params.len() {
            let grad = &out[i + 1];
            let p = &mut self.params[i];
            let m = &mut self.moment1[i];
            let v = &mut self.moment2[i];
            for j in 0..p.len() {
                let g = grad.get(j).copied().unwrap_or(0.0) * scale;
                m[j] = beta1 * m[j] + (1.0 - beta1) * g;
                v[j] = beta2 * v[j] + (1.0 - beta2) * g * g;
                let m_hat = m[j] / correction1;
                let v_hat = v[j] / correction2;
                p[j] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }
        }
        Ok(loss)
    }

    /// The learning rate for the step about to be taken.
    ///
    /// A half-cosine from the configured rate down to `final_lr_fraction` of
    /// it across `total_steps`. With `total_steps` unset the rate is constant,
    /// which is what every caller got before this existed.
    pub fn current_learning_rate(&self) -> f32 {
        let total = self.config.total_steps;
        if total == 0 {
            return self.config.learning_rate;
        }
        let t = (self.step.saturating_sub(1) as f32 / total as f32).clamp(0.0, 1.0);
        let floor = self.config.final_lr_fraction.clamp(0.0, 1.0);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        self.config.learning_rate * (floor + (1.0 - floor) * cosine)
    }

    /// Factor to scale every gradient by so the global norm is within
    /// `grad_clip`. 1.0 when clipping is off or the norm is already inside it.
    fn gradient_scale(&self, grads: &[Vec<f32>]) -> f32 {
        if self.config.grad_clip <= 0.0 {
            return 1.0;
        }
        let sum_sq: f64 = grads
            .iter()
            .flat_map(|g| g.iter())
            .map(|g| (*g as f64) * (*g as f64))
            .sum();
        let norm = sum_sq.sqrt() as f32;
        if norm > self.config.grad_clip && norm.is_finite() {
            self.config.grad_clip / norm
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::IN_CHANNELS;
    use crate::model::Widths;

    /// A batch whose target is a smoothed version of its input, with the guides
    /// carrying the structure. Learnable, and small enough to train in a test.
    fn synthetic(n: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
        let pixels = h * w;
        let mut input = vec![0.0f32; n * IN_CHANNELS * pixels];
        let mut target = vec![0.0f32; n * OUT_CHANNELS * pixels];
        for b in 0..n {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    // A smooth ramp is the signal; the "noise" is a
                    // deterministic high-frequency check the network can learn
                    // to remove because the guides do not contain it.
                    let clean = 0.25 + 0.5 * (x as f32 / w as f32) + 0.2 * (b as f32);
                    let noise = if (x + y) % 2 == 0 { 0.18 } else { -0.18 };
                    for c in 0..3 {
                        input[((b * IN_CHANNELS + c) * pixels) + i] = clean + noise;
                        // Albedo and normal carry the signal, noise-free.
                        input[((b * IN_CHANNELS + 3 + c) * pixels) + i] = clean;
                        input[((b * IN_CHANNELS + 6 + c) * pixels) + i] = 0.5;
                        target[((b * OUT_CHANNELS + c) * pixels) + i] = clean;
                    }
                }
            }
        }
        (input, target)
    }

    #[test]
    fn an_untrained_network_is_the_identity() {
        let net = DenoiseNet::new(Widths::tiny());
        let (n, h, w) = (1, 16, 16);
        let mut trainer =
            Trainer::new(net, n, h, w, Device::Cpu, TrainConfig::default()).expect("compile");
        let (input, target) = synthetic(n, h, w);
        let loss = trainer
            .step(&Batch {
                n,
                h,
                w,
                input: &input,
                target: &target,
            })
            .expect("step");

        // The residual layer starts at zero, so the first forward pass returns
        // the input colour unchanged and the loss is exactly the render's own
        // relative error. Anything else means the residual path is wrong.
        let mut expected = 0.0f64;
        let pixels = h * w;
        for c in 0..OUT_CHANNELS {
            for i in 0..pixels {
                let y = input[c * pixels + i] as f64;
                let t = target[c * pixels + i] as f64;
                // TRAIN_EPSILON, not LOSS_EPSILON: this checks what `step`
                // optimises, which is a different floor from what is reported.
                expected += (y - t) * (y - t) / (t * t + TRAIN_EPSILON as f64);
            }
        }
        expected /= (OUT_CHANNELS * pixels) as f64;
        assert!(
            (loss as f64 - expected).abs() < 1e-4 * expected.max(1e-6),
            "loss {loss} but an identity network should give {expected}"
        );
    }

    /// The two floors serve different purposes and must not drift back into one
    /// constant: the reporting floor is what every published number and every
    /// OpenImageDenoise comparison was measured with, so it is frozen, while the
    /// training floor is tuned to the data. If a future change sets them equal,
    /// either the scoreboard has moved or the training fix has been reverted.
    #[test]
    fn the_reporting_floor_is_not_the_training_floor() {
        assert_eq!(
            LOSS_EPSILON, 0.01,
            "the reporting floor is frozen: every recorded score and every OIDN \
             comparison used 0.01, and moving it invalidates all of them"
        );
        const {
            assert!(
                TRAIN_EPSILON < LOSS_EPSILON,
                "the training floor should sit below the reporting floor"
            );
        }
        // Below this the weight on a near-black pixel, 1/eps, starts to dominate
        // the batch.
        const {
            assert!(TRAIN_EPSILON >= 1e-4, "training floor is too low");
        }
    }

    /// The gradient term has to *add* to the loss, and it has to add more when
    /// the error is spread smoothly than when it is not. Both networks here are
    /// the untrained identity, so the only difference is the target: one
    /// differs from the input by a checkerboard, the other by a broad ramp
    /// carrying the same total per-pixel error. Relative L2 alone cannot tell
    /// them apart; the gradient term is what does.
    #[test]
    fn the_gradient_term_prices_smooth_error() {
        let (n, h, w) = (1, 16, 16);
        let pixels = h * w;
        let (input, flat_target) = synthetic(n, h, w);

        // Same per-pixel magnitude as `synthetic`'s checkerboard, but low
        // frequency: one broad sign change across the tile rather than one per
        // pixel. This is what mottling looks like.
        let mut smooth_target = flat_target.clone();
        for c in 0..OUT_CHANNELS {
            for y in 0..h {
                for x in 0..w {
                    let blob = if x < w / 2 { 0.18 } else { -0.18 };
                    smooth_target[c * pixels + y * w + x] += blob;
                }
            }
        }

        let measure = |target: &[f32], weight: f32| {
            let mut t = Trainer::new(
                DenoiseNet::new(Widths::tiny()),
                n,
                h,
                w,
                Device::Cpu,
                TrainConfig {
                    gradient_weight: weight,
                    ..Default::default()
                },
            )
            .expect("compile");
            t.step(&Batch {
                n,
                h,
                w,
                input: &input,
                target,
            })
            .expect("step")
        };

        // Off by default: the term must not perturb any previously measured run.
        let plain_flat = measure(&flat_target, 0.0);
        let plain_smooth = measure(&smooth_target, 0.0);

        let with_flat = measure(&flat_target, 1.0);
        let with_smooth = measure(&smooth_target, 1.0);

        assert!(
            with_flat > plain_flat && with_smooth > plain_smooth,
            "the term should add: {plain_flat} -> {with_flat}, {plain_smooth} -> {with_smooth}"
        );

        // The checkerboard target is the one the *prediction* matches poorly at
        // high frequency, so it picks up the larger gradient penalty. What
        // matters is that the two are separated at all — relative L2 alone puts
        // them within a few percent of each other.
        let plain_ratio = (plain_flat / plain_smooth) as f64;
        let with_ratio = (with_flat / with_smooth) as f64;
        assert!(
            (with_ratio - plain_ratio).abs() > 0.05,
            "the term did not separate the two error shapes: \
             {plain_ratio:.4} -> {with_ratio:.4}"
        );
    }

    #[test]
    fn training_reduces_the_loss() {
        let net = DenoiseNet::new(Widths::tiny());
        let (n, h, w) = (2, 16, 16);
        let mut trainer = Trainer::new(
            net,
            n,
            h,
            w,
            Device::Cpu,
            TrainConfig {
                learning_rate: 3e-3,
                ..Default::default()
            },
        )
        .expect("compile");
        let (input, target) = synthetic(n, h, w);
        let batch = Batch {
            n,
            h,
            w,
            input: &input,
            target: &target,
        };

        let first = trainer.step(&batch).expect("step");
        let mut last = first;
        for _ in 0..40 {
            last = trainer.step(&batch).expect("step");
            assert!(last.is_finite(), "diverged to {last}");
        }
        assert!(
            last < 0.5 * first,
            "loss went {first} -> {last}: the gradients are not moving the network"
        );
        assert_eq!(trainer.steps_taken(), 41);
    }

    /// The schedule has to start at the configured rate, end near the floor,
    /// and never increase — a decay that overshoots or comes back up is worse
    /// than none.
    #[test]
    fn the_learning_rate_decays_monotonically_to_its_floor() {
        let net = DenoiseNet::new(Widths::tiny());
        let config = TrainConfig {
            learning_rate: 1e-3,
            total_steps: 100,
            final_lr_fraction: 0.05,
            ..Default::default()
        };
        let mut trainer = Trainer::new(net, 1, 16, 16, Device::Cpu, config).expect("compile");
        let (input, target) = synthetic(1, 16, 16);
        let batch = Batch {
            n: 1,
            h: 16,
            w: 16,
            input: &input,
            target: &target,
        };

        let mut previous = f32::INFINITY;
        let mut first = 0.0;
        for i in 0..100 {
            trainer.step(&batch).expect("step");
            let lr = trainer.current_learning_rate();
            if i == 0 {
                first = lr;
            }
            assert!(
                lr <= previous + 1e-9,
                "rate rose at step {i}: {previous} -> {lr}"
            );
            previous = lr;
        }
        // Step 1 is taken at the full rate; the value read after it is step 2's.
        assert!((first - 1e-3).abs() < 2e-5, "started at {first}");
        assert!(
            (previous - 5e-5).abs() < 5e-6,
            "ended at {previous}, wanted a twentieth of 1e-3"
        );
    }

    /// Without `total_steps` the rate must not move at all.
    #[test]
    fn no_schedule_means_a_constant_rate() {
        let net = DenoiseNet::new(Widths::tiny());
        let mut trainer =
            Trainer::new(net, 1, 16, 16, Device::Cpu, TrainConfig::default()).expect("compile");
        let (input, target) = synthetic(1, 16, 16);
        let batch = Batch {
            n: 1,
            h: 16,
            w: 16,
            input: &input,
            target: &target,
        };
        for _ in 0..5 {
            trainer.step(&batch).expect("step");
            assert_eq!(trainer.current_learning_rate(), 1e-3);
        }
    }

    #[test]
    fn a_mismatched_batch_is_rejected() {
        let net = DenoiseNet::new(Widths::tiny());
        let mut trainer =
            Trainer::new(net, 1, 16, 16, Device::Cpu, TrainConfig::default()).expect("compile");
        let (input, target) = synthetic(1, 16, 16);
        assert!(
            trainer
                .step(&Batch {
                    n: 1,
                    h: 8,
                    w: 8,
                    input: &input,
                    target: &target,
                })
                .is_err(),
            "a batch of the wrong shape must not be run against a graph compiled for another"
        );
    }

    /// The same convergence check, on the GPU.
    ///
    /// Compiling for CUDA is not the same as running there: the tests above all
    /// use `Device::Cpu`, so they would pass on a build whose CUDA lowering was
    /// missing every op in the network. This one trains on the device and
    /// requires the loss to fall, which needs the forward convolutions, the
    /// transposed convolutions, the concatenations and every backward rule
    /// behind them to be present and correct.
    #[cfg(feature = "cuda")]
    #[test]
    fn training_reduces_the_loss_on_cuda() {
        let net = DenoiseNet::new(Widths::tiny());
        let (n, h, w) = (2, 16, 16);
        let mut trainer = Trainer::new(
            net,
            n,
            h,
            w,
            Device::Cuda,
            TrainConfig {
                learning_rate: 3e-3,
                ..Default::default()
            },
        )
        .expect("compile for cuda");
        let (input, target) = synthetic(n, h, w);
        let batch = Batch {
            n,
            h,
            w,
            input: &input,
            target: &target,
        };

        let first = trainer.step(&batch).expect("step");
        assert!(first.is_finite(), "the first loss on cuda was {first}");
        let mut last = first;
        for _ in 0..40 {
            last = trainer.step(&batch).expect("step");
            assert!(last.is_finite(), "diverged to {last}");
        }
        assert!(
            last < 0.5 * first,
            "cuda loss went {first} -> {last}: the gradients are not moving the network"
        );
    }

    #[test]
    fn tiles_must_survive_three_halvings() {
        let net = DenoiseNet::new(Widths::tiny());
        assert!(
            Trainer::new(net, 1, 12, 16, Device::Cpu, TrainConfig::default()).is_err(),
            "12 is not a multiple of 8 and the skip connections would not line up"
        );
    }
}
