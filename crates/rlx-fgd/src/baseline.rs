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

//! The neural-network baseline of Figures 1 and 2.
//!
//! A Fourier-feature MLP trained with Adam — “optimise the parameters of a
//! fixed representation”, the approach the paper is arguing against. It is here
//! so the comparison can be run rather than quoted: same target, same loss,
//! same machine.
//!
//! Deliberately small and self-contained (explicit forward and backward passes,
//! no autodiff), because its role is to be a fair reference curve rather than a
//! fast one. The architecture follows the authors' baseline: a random Fourier
//! embedding, GELU hidden layers, and a scalar head.

use crate::algorithm::Rng;
use crate::tasks::rkhs::PointwiseLoss;

/// Configuration for the baseline network and its optimiser.
#[derive(Debug, Clone)]
pub struct MlpConfig {
    /// Input dimension.
    pub inputs: usize,
    /// Number of random Fourier features; the embedding has twice this width.
    pub features: usize,
    /// Standard deviation of the Fourier frequency matrix.
    pub feature_scale: f64,
    /// Hidden layer widths.
    pub hidden: Vec<usize>,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Seed for initialisation.
    pub seed: u64,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            inputs: 2,
            features: 128,
            feature_scale: 0.75,
            hidden: vec![256, 256, 256],
            learning_rate: 3e-3 * 0.05,
            seed: 0,
        }
    }
}

struct Dense {
    weight: Vec<f64>,
    bias: Vec<f64>,
    inputs: usize,
    outputs: usize,
    weight_m: Vec<f64>,
    weight_v: Vec<f64>,
    bias_m: Vec<f64>,
    bias_v: Vec<f64>,
}

impl Dense {
    fn new(inputs: usize, outputs: usize, rng: &mut Normal) -> Self {
        // LeCun normal, the default for the reference implementation's layers.
        let sigma = (1.0 / inputs as f64).sqrt();
        Self {
            weight: (0..inputs * outputs)
                .map(|_| rng.sample() * sigma)
                .collect(),
            bias: vec![0.0; outputs],
            inputs,
            outputs,
            weight_m: vec![0.0; inputs * outputs],
            weight_v: vec![0.0; inputs * outputs],
            bias_m: vec![0.0; outputs],
            bias_v: vec![0.0; outputs],
        }
    }

    fn forward(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend_from_slice(&self.bias);
        for i in 0..self.inputs {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            let row = &self.weight[i * self.outputs..(i + 1) * self.outputs];
            for (o, w) in out.iter_mut().zip(row) {
                *o += xi * w;
            }
        }
    }
}

/// The Fourier-feature MLP baseline.
pub struct FourierMlp {
    config: MlpConfig,
    frequencies: Vec<f64>,
    layers: Vec<Dense>,
    step: i32,
}

impl FourierMlp {
    /// Initialise the network.
    pub fn new(config: MlpConfig) -> Self {
        assert!(config.inputs > 0, "input dimension must be positive");
        let mut rng = Normal::new(config.seed);
        let frequencies: Vec<f64> = (0..config.features * config.inputs)
            .map(|_| rng.sample() * config.feature_scale)
            .collect();
        let mut widths = vec![config.features * 2];
        widths.extend_from_slice(&config.hidden);
        widths.push(1);
        let layers = widths
            .windows(2)
            .map(|w| Dense::new(w[0], w[1], &mut rng))
            .collect();
        Self {
            config,
            frequencies,
            layers,
            step: 0,
        }
    }

    /// Input dimension.
    pub fn inputs(&self) -> usize {
        self.config.inputs
    }

    /// Total parameter count, for comparing against a representation's degrees
    /// of freedom.
    pub fn parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weight.len() + l.bias.len())
            .sum()
    }

    /// The random Fourier embedding `[cos(2πFx), sin(2πFx)]`.
    fn embed(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.config.inputs);
        let two_pi = std::f64::consts::TAU;
        let m = self.config.features;
        let d = self.config.inputs;
        let mut out = vec![0.0; 2 * m];
        for k in 0..m {
            let row = &self.frequencies[k * d..(k + 1) * d];
            let phase = two_pi * row.iter().zip(x).map(|(f, v)| f * v).sum::<f64>();
            out[k] = phase.cos();
            out[m + k] = phase.sin();
        }
        out
    }

    /// Evaluate the network at `x`.
    pub fn eval(&self, x: &[f64]) -> f64 {
        let mut activation = self.embed(x);
        let mut buffer = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            layer.forward(&activation, &mut buffer);
            if index + 1 < self.layers.len() {
                for v in buffer.iter_mut() {
                    *v = gelu(*v);
                }
            }
            std::mem::swap(&mut activation, &mut buffer);
        }
        activation[0]
    }

    /// One Adam step on `(1/B) Σ ℓ(net(xᵢ), yᵢ)`.
    ///
    /// `xs` is row-major with [`FourierMlp::inputs`] columns.
    pub fn train_batch(&mut self, xs: &[f64], ys: &[f64], loss: PointwiseLoss) -> f64 {
        let d = self.config.inputs;
        assert_eq!(xs.len(), ys.len() * d, "batch shapes must agree");
        let batch = ys.len();
        assert!(batch > 0, "batch must be non-empty");

        let layer_count = self.layers.len();
        let mut weight_grads: Vec<Vec<f64>> = self
            .layers
            .iter()
            .map(|l| vec![0.0; l.weight.len()])
            .collect();
        let mut bias_grads: Vec<Vec<f64>> = self
            .layers
            .iter()
            .map(|l| vec![0.0; l.bias.len()])
            .collect();
        let inv_batch = 1.0 / batch as f64;
        let mut total = 0.0;

        for b in 0..batch {
            let x = &xs[b * d..(b + 1) * d];

            // Forward, keeping pre-activations and activations for the backward pass.
            let mut activations = vec![self.embed(x)];
            let mut preactivations = Vec::with_capacity(layer_count);
            let mut buffer = Vec::new();
            for (index, layer) in self.layers.iter().enumerate() {
                layer.forward(activations.last().expect("input present"), &mut buffer);
                preactivations.push(buffer.clone());
                if index + 1 < layer_count {
                    for v in buffer.iter_mut() {
                        *v = gelu(*v);
                    }
                }
                activations.push(buffer.clone());
            }

            let prediction = activations[layer_count][0];
            total += loss.value(prediction, ys[b]) * inv_batch;

            // Backward.
            let mut delta = vec![loss.derivative(prediction, ys[b]) * inv_batch];
            for index in (0..layer_count).rev() {
                if index + 1 < layer_count {
                    for (v, z) in delta.iter_mut().zip(&preactivations[index]) {
                        *v *= gelu_derivative(*z);
                    }
                }
                let layer = &self.layers[index];
                let input = &activations[index];
                for (o, v) in delta.iter().enumerate() {
                    bias_grads[index][o] += v;
                }
                for i in 0..layer.inputs {
                    let xi = input[i];
                    if xi == 0.0 {
                        continue;
                    }
                    let row = &mut weight_grads[index][i * layer.outputs..(i + 1) * layer.outputs];
                    for (g, v) in row.iter_mut().zip(&delta) {
                        *g += xi * v;
                    }
                }
                if index > 0 {
                    let mut next = vec![0.0; layer.inputs];
                    for i in 0..layer.inputs {
                        let row = &layer.weight[i * layer.outputs..(i + 1) * layer.outputs];
                        next[i] = row.iter().zip(&delta).map(|(w, v)| w * v).sum();
                    }
                    delta = next;
                }
            }
        }

        self.step += 1;
        let lr = self.config.learning_rate;
        for index in 0..layer_count {
            let layer = &mut self.layers[index];
            adam(
                &mut layer.weight,
                &weight_grads[index],
                &mut layer.weight_m,
                &mut layer.weight_v,
                lr,
                self.step,
            );
            adam(
                &mut layer.bias,
                &bias_grads[index],
                &mut layer.bias_m,
                &mut layer.bias_v,
                lr,
                self.step,
            );
        }
        total
    }

    /// `(1/n) Σ ℓ(net(xᵢ), yᵢ)` over a whole dataset, without an update.
    pub fn dataset_loss(&self, xs: &[f64], ys: &[f64], loss: PointwiseLoss) -> f64 {
        let d = self.config.inputs;
        ys.iter()
            .enumerate()
            .map(|(i, &y)| loss.value(self.eval(&xs[i * d..(i + 1) * d]), y))
            .sum::<f64>()
            / ys.len().max(1) as f64
    }
}

/// Train the baseline on a target sampled uniformly over `[0,1]^d` — the
/// continuous fitting setting of Figure 1, where there is no finite dataset and
/// every step draws fresh points.
pub fn train_on_uniform_samples<F: Fn(&[f64]) -> f64>(
    net: &mut FourierMlp,
    target: F,
    steps: usize,
    batch: usize,
    seed: u64,
) -> Vec<f64> {
    let d = net.inputs();
    let mut rng = Rng::new(seed);
    let mut xs = vec![0.0; batch * d];
    let mut ys = vec![0.0; batch];
    (0..steps)
        .map(|_| {
            for b in 0..batch {
                for k in 0..d {
                    xs[b * d + k] = rng.uniform01();
                }
                ys[b] = target(&xs[b * d..(b + 1) * d]);
            }
            net.train_batch(&xs, &ys, PointwiseLoss::SquaredError)
        })
        .collect()
}

fn adam(values: &mut [f64], grads: &[f64], m: &mut [f64], v: &mut [f64], lr: f64, step: i32) {
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPS: f64 = 1e-8;
    let bias1 = 1.0 - BETA1.powi(step);
    let bias2 = 1.0 - BETA2.powi(step);
    for i in 0..values.len() {
        m[i] = BETA1 * m[i] + (1.0 - BETA1) * grads[i];
        v[i] = BETA2 * v[i] + (1.0 - BETA2) * grads[i] * grads[i];
        values[i] -= lr * (m[i] / bias1) / ((v[i] / bias2).sqrt() + EPS);
    }
}

/// The tanh approximation of GELU, matching the reference implementation's default.
fn gelu(x: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_derivative(x: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    let t = inner.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * x * x)
}

/// Standard normal samples by Box-Muller on the crate's SplitMix64 stream.
struct Normal {
    rng: Rng,
    spare: Option<f64>,
}

impl Normal {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            spare: None,
        }
    }

    fn sample(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = self.rng.uniform01().max(f64::MIN_POSITIVE);
        let u2 = self.rng.uniform01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}
