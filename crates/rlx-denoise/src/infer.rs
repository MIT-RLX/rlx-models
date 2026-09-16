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

//! Running a trained network.
//!
//! The forward graph alone, without the loss or the autodiff rewrite. Compiled
//! once for one input shape, then run against any weights.
//!
//! This is the path that matters for a renderer — training happens once, on
//! whatever hardware has the memory for it, and every frame afterwards goes
//! through here. Because the network is built from `conv2d`,
//! `conv_transpose2d`, `relu` and `concat`, that is the same definition on
//! CUDA, Metal, MLX, ROCm, Vulkan and wgpu.

use anyhow::{Result, ensure};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::model::{DenoiseNet, OUT_CHANNELS};

/// A compiled forward pass and the weights behind it.
pub struct Denoiser {
    net: DenoiseNet,
    graph: CompiledGraph,
    params: Vec<Vec<f32>>,
    shape: (usize, usize, usize),
}

impl Denoiser {
    /// Compile a forward pass for `[n, 9, h, w]`.
    pub fn new(
        net: DenoiseNet,
        params: Vec<Vec<f32>>,
        n: usize,
        h: usize,
        w: usize,
        device: Device,
    ) -> Result<Self> {
        ensure!(n > 0 && h > 0 && w > 0, "empty shape {n}x{h}x{w}");
        ensure!(
            h.is_multiple_of(DenoiseNet::TILE_MULTIPLE)
                && w.is_multiple_of(DenoiseNet::TILE_MULTIPLE),
            "{h}x{w} must be a multiple of {}",
            DenoiseNet::TILE_MULTIPLE
        );
        ensure!(
            params.len() == net.params().len()
                && params
                    .iter()
                    .zip(net.params())
                    .all(|(v, s)| v.len() == s.elems()),
            "parameter shapes do not match this network"
        );

        let mut graph = Graph::new("denoise_forward");
        let input = graph.input("input", Shape::new(&[n, net.inputs(), h, w], DType::F32));
        let out = net.forward(&mut graph, input, n, h, w);
        graph.set_outputs(vec![out]);
        let graph = Session::new(device).compile(graph);

        Ok(Self {
            net,
            graph,
            params,
            shape: (n, h, w),
        })
    }

    pub fn net(&self) -> &DenoiseNet {
        &self.net
    }

    /// Replace the weights without recompiling — evaluating a checkpoint mid
    /// training, or swapping a network trained for one renderer for another.
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

    /// Denoise one batch. `input` is `[n, 9, h, w]` planar; the result is
    /// `[n, 3, h, w]`.
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let (n, h, w) = self.shape;
        let pixels = n * h * w;
        ensure!(
            input.len() == pixels * self.net.inputs(),
            "input is {} floats, this graph was compiled for {}",
            input.len(),
            pixels * self.net.inputs()
        );
        for (spec, values) in self.net.params().iter().zip(&self.params) {
            self.graph.set_param(spec.name, values);
        }
        let out = self.graph.run(&[("input", input)]);
        let result = out.into_iter().next().unwrap_or_default();
        ensure!(
            result.len() == pixels * OUT_CHANNELS,
            "network returned {} floats, expected {}",
            result.len(),
            pixels * OUT_CHANNELS
        );
        Ok(result)
    }

    /// Running total of relative-L2 against a target: the summed
    /// `(y - t)² / (t² + ε)` and the number of values behind it.
    ///
    /// Callers spanning several tiles must accumulate these and take the root
    /// once at the end. Averaging per-tile roots is a different quantity —
    /// smaller, by Jensen — and comparing one against the other reads as an
    /// improvement that is not there.
    pub fn relative_error_sum(predicted: &[f32], target: &[f32], epsilon: f32) -> (f64, usize) {
        if predicted.len() != target.len() {
            return (f64::NAN, 0);
        }
        let mut sum = 0.0f64;
        for (&y, &t) in predicted.iter().zip(target) {
            let (y, t) = (y as f64, t as f64);
            let d = y - t;
            sum += d * d / (t * t + epsilon as f64);
        }
        (sum, target.len())
    }

    /// Relative-L2 of a prediction against a target, the same measure the
    /// training loss uses, reported as a root so it reads as an error rather
    /// than a squared one.
    pub fn relative_error(predicted: &[f32], target: &[f32], epsilon: f32) -> f32 {
        let (sum, n) = Self::relative_error_sum(predicted, target, epsilon);
        if n == 0 {
            return f32::NAN;
        }
        (sum / n as f64).sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::IN_CHANNELS;
    use crate::model::{Arch, Head, Sampling, Widths, init_params};

    #[test]
    fn an_untrained_network_returns_its_input_colour() {
        let net = DenoiseNet::new(Widths::tiny());
        let params = init_params(&net, 7);
        let (n, h, w) = (1, 16, 16);
        let mut denoiser = Denoiser::new(net, params, n, h, w, Device::Cpu).expect("compile");

        let pixels = h * w;
        let mut input = vec![0.0f32; IN_CHANNELS * pixels];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 37) as f32) / 37.0;
        }
        let out = denoiser.run(&input).expect("run");

        // The output layer starts at zero, so the correction is zero and the
        // residual leaves the first three channels exactly as they came in.
        for c in 0..OUT_CHANNELS {
            for p in 0..pixels {
                let got = out[c * pixels + p];
                let want = input[c * pixels + p];
                assert!(
                    (got - want).abs() < 1e-5,
                    "channel {c} pixel {p}: {got} but the input colour was {want}"
                );
            }
        }
    }

    /// The kernel head starts as a near-delta, so an untrained network returns
    /// its input colour almost unchanged — the same property the residual head
    /// has, and what lets training begin at the render's own error.
    #[test]
    fn an_untrained_kernel_head_is_nearly_the_identity() {
        let net = DenoiseNet::with_arch(Arch {
            widths: Widths::tiny(),
            head: Head::Kernel { radius: 2 },
            sampling: Sampling::Strided,
            ..Arch::default()
        });
        let params = init_params(&net, 11);
        let (n, h, w) = (1, 16, 16);
        let mut denoiser = Denoiser::new(net, params, n, h, w, Device::Cpu).expect("compile");

        let pixels = h * w;
        let mut input = vec![0.0f32; IN_CHANNELS * pixels];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 29) as f32) / 29.0;
        }
        let out = denoiser.run(&input).expect("run");

        // exp(8) / (exp(8) + 24) = 0.9920, so up to 0.8% of each pixel comes
        // from its neighbours. Anything far outside that is a wrong tap order
        // or a softmax on the wrong axis.
        for c in 0..OUT_CHANNELS {
            for p in 0..pixels {
                let got = out[c * pixels + p];
                let want = input[c * pixels + p];
                assert!(
                    (got - want).abs() < 0.05,
                    "channel {c} pixel {p}: {got} but the input colour was {want}"
                );
            }
        }
    }

    /// The defining property of a kernel-predicting head: the output is a
    /// convex combination of the input neighbourhood, so it can never leave the
    /// range of colours the renderer actually produced. Direct prediction has
    /// no such bound, and this is the test that tells the two apart.
    #[test]
    fn a_kernel_head_cannot_leave_its_neighbourhood() {
        const RADIUS: usize = 2;
        let net = DenoiseNet::with_arch(Arch {
            widths: Widths::tiny(),
            head: Head::Kernel { radius: RADIUS },
            sampling: Sampling::Strided,
            ..Arch::default()
        });
        // Random weights, not the zero-initialised output layer: the bound has
        // to hold for any weights, not only for the near-delta it starts at.
        let mut params = init_params(&net, 5);
        let last = params.len() - 1;
        for (i, v) in params[last].iter_mut().enumerate() {
            *v = ((i % 17) as f32 / 17.0 - 0.5) * 4.0;
        }
        let (n, h, w) = (1, 16, 16);
        let mut denoiser = Denoiser::new(net, params, n, h, w, Device::Cpu).expect("compile");

        let pixels = h * w;
        let mut input = vec![0.0f32; IN_CHANNELS * pixels];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i * 37) % 101) as f32 / 101.0;
        }
        let out = denoiser.run(&input).expect("run");

        for c in 0..OUT_CHANNELS {
            for y in 0..h {
                for x in 0..w {
                    // The neighbourhood the filter is allowed to draw from,
                    // clamped at the border because padding replicates.
                    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                    for dy in 0..=(2 * RADIUS) {
                        for dx in 0..=(2 * RADIUS) {
                            let sy = (y + dy).saturating_sub(RADIUS).min(h - 1);
                            let sx = (x + dx).saturating_sub(RADIUS).min(w - 1);
                            let v = input[c * pixels + sy * w + sx];
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                    let got = out[c * pixels + y * w + x];
                    assert!(
                        got >= lo - 1e-4 && got <= hi + 1e-4,
                        "channel {c} at ({x},{y}): {got} escaped [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    /// Pooled resampling has to produce the same shape as strided, since the
    /// two are meant to be interchangeable in a benchmark.
    #[test]
    fn pooled_resampling_preserves_resolution() {
        for head in [Head::Direct, Head::Kernel { radius: 2 }] {
            let net = DenoiseNet::with_arch(Arch {
                widths: Widths::tiny(),
                head,
                sampling: Sampling::Pooled,
                ..Arch::default()
            });
            let params = init_params(&net, 2);
            let (n, h, w) = (1, 16, 24);
            let mut denoiser =
                Denoiser::new(net, params, n, h, w, Device::Cpu).expect("compile pooled");
            let out = denoiser
                .run(&vec![0.25f32; IN_CHANNELS * h * w])
                .expect("run pooled");
            assert_eq!(out.len(), OUT_CHANNELS * h * w, "head {head:?}");
        }
    }

    #[test]
    fn a_mismatched_input_is_rejected() {
        let net = DenoiseNet::new(Widths::tiny());
        let params = init_params(&net, 1);
        let mut denoiser = Denoiser::new(net, params, 1, 16, 16, Device::Cpu).expect("compile");
        assert!(denoiser.run(&[0.0; 16]).is_err());
    }

    #[test]
    fn weights_of_the_wrong_shape_are_rejected() {
        let net = DenoiseNet::new(Widths::tiny());
        let mut params = init_params(&net, 1);
        params[0].push(0.0);
        assert!(Denoiser::new(net, params, 1, 16, 16, Device::Cpu).is_err());
    }

    #[test]
    fn a_summed_error_matches_the_whole_set_at_once() {
        // Splitting a set in two and combining the sums has to give exactly the
        // number computed over the whole set — which is the property that lets
        // validation run one tile at a time.
        let p = [0.3f32, 0.9, 1.4, 0.05];
        let t = [0.2f32, 1.0, 1.2, 0.10];
        let (whole, n) = Denoiser::relative_error_sum(&p, &t, 0.01);
        let (a, na) = Denoiser::relative_error_sum(&p[..2], &t[..2], 0.01);
        let (b, nb) = Denoiser::relative_error_sum(&p[2..], &t[2..], 0.01);
        assert_eq!(n, na + nb);
        assert!((whole - (a + b)).abs() < 1e-12, "{whole} vs {}", a + b);
    }

    #[test]
    fn relative_error_is_zero_on_an_exact_match() {
        let t = [0.2f32, 0.5, 1.5, 0.01];
        assert!(Denoiser::relative_error(&t, &t, 0.01).abs() < 1e-6);
    }
}
