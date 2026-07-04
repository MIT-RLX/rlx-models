// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Echo-state reservoir dynamics (dense and locally connected).

use crate::host::Rng;

/// Shared hyperparameters for ESN-style reservoirs.
#[derive(Debug, Clone)]
pub struct ReservoirConfig {
    /// Reservoir size (`grid_height * grid_width` when local).
    pub units: usize,
    /// Grid rows for [`Self::local_lcesn`] (0 when dense).
    pub grid_height: usize,
    /// Grid columns for local topology (0 when dense).
    pub grid_width: usize,
    /// Odd kernel edge length for local connectivity (e.g. 7 → 7×7 neighborhood).
    pub kernel: usize,
    /// Leaky integration rate (`1.0` → standard `x = tanh(·)` update).
    pub leak_rate: f32,
    /// Target spectral radius ρ after rescaling `W_res`.
    pub spectral_radius: f32,
    /// Scale on nonzero input weights.
    pub input_scaling: f32,
    /// Scale on output-feedback weights (`0` disables feedback).
    pub feedback_scaling: f32,
    /// Fraction of input weights set to zero (Nakajima-style input sparsity).
    pub input_sparsity: f32,
    /// Fraction of nonzero off-diagonal recurrent weights (dense topology only).
    pub sparsity: f32,
    /// Use toroidal local grid instead of dense sparse `W_res`.
    pub local_topology: bool,
}

impl ReservoirConfig {
    /// Standard dense ESN (Nakajima RC-tutorial style: N=300, ρ=0.9).
    pub fn dense_standard() -> Self {
        Self {
            units: 300,
            grid_height: 0,
            grid_width: 0,
            kernel: 0,
            leak_rate: 1.0,
            spectral_radius: 0.9,
            input_scaling: 0.1,
            feedback_scaling: 0.0,
            input_sparsity: 0.5,
            sparsity: 0.2,
            local_topology: false,
        }
    }

    /// Locally connected grid (LCESN-style, Matzner & Mráz ICLR 2025).
    pub fn local_lcesn() -> Self {
        Self {
            units: 800,
            grid_height: 20,
            grid_width: 40,
            kernel: 7,
            leak_rate: 1.0,
            spectral_radius: 0.9,
            input_scaling: 0.1,
            feedback_scaling: 0.0,
            input_sparsity: 0.5,
            sparsity: 0.0,
            local_topology: true,
        }
    }

    /// Medium reservoir for polynomial readout (HCNN-inspired nonlinear readout).
    pub fn dense_poly() -> Self {
        Self {
            units: 400,
            grid_height: 0,
            grid_width: 0,
            kernel: 0,
            leak_rate: 1.0,
            spectral_radius: 0.95,
            input_scaling: 0.1,
            feedback_scaling: 0.0,
            input_sparsity: 0.5,
            sparsity: 0.2,
            local_topology: false,
        }
    }
}

/// Fixed random reservoir with `tanh` activation and optional leaky integration.
#[derive(Debug, Clone)]
pub struct Reservoir {
    cfg: ReservoirConfig,
    w_in: Vec<f32>,
    w_res: Vec<f32>,
    w_fb: Vec<f32>,
    state: Vec<f32>,
}

impl Reservoir {
    /// Build a reservoir from `cfg` with deterministic weight init from `seed`.
    pub fn new(cfg: ReservoirConfig, seed: u64) -> Self {
        let n = cfg.units;
        let mut rng = Rng::new(seed);
        let w_in: Vec<f32> = (0..n)
            .map(|_| {
                if rng.uniform01() < cfg.input_sparsity as f64 {
                    0.0
                } else {
                    (rng.uniform01() * 2.0 - 1.0) as f32 * cfg.input_scaling
                }
            })
            .collect();

        let w_fb: Vec<f32> = if cfg.feedback_scaling.abs() > f32::EPSILON {
            (0..n)
                .map(|_| (rng.uniform01() * 2.0 - 1.0) as f32 * cfg.feedback_scaling)
                .collect()
        } else {
            vec![0.0; n]
        };

        let mut w_res = if cfg.local_topology {
            local_weights(&cfg, &mut rng)
        } else {
            dense_sparse_weights(n, cfg.sparsity, &mut rng)
        };
        scale_spectral_radius(&mut w_res, n, cfg.spectral_radius);

        Self {
            cfg,
            w_in,
            w_res,
            w_fb,
            state: vec![0.0; n],
        }
    }

    pub fn units(&self) -> usize {
        self.cfg.units
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Leaky update: `x ← (1−λ)x + λ·tanh(W_in·u + W_res·x + W_fb·y_fb)`.
    /// When `leak_rate = 1`, this is the standard `x = tanh(·)` ESN update.
    pub fn step(&mut self, u: f32, y_feedback: f32) {
        let n = self.cfg.units;
        let leak = self.cfg.leak_rate;
        let mut pre = vec![0f32; n];
        for i in 0..n {
            let mut s = self.w_in[i] * u + self.w_fb[i] * y_feedback;
            let row = &self.w_res[i * n..(i + 1) * n];
            for (j, &xj) in self.state.iter().enumerate() {
                s += row[j] * xj;
            }
            pre[i] = s;
        }
        if leak >= 1.0 - f32::EPSILON {
            for i in 0..n {
                self.state[i] = pre[i].tanh();
            }
        } else {
            for i in 0..n {
                let activated = pre[i].tanh();
                self.state[i] = (1.0 - leak) * self.state[i] + leak * activated;
            }
        }
    }
}

fn dense_sparse_weights(n: usize, sparsity: f32, rng: &mut Rng) -> Vec<f32> {
    let mut w = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.uniform01() < sparsity as f64 {
                w[i * n + j] = (rng.uniform01() * 2.0 - 1.0) as f32;
            }
        }
    }
    w
}

fn local_weights(cfg: &ReservoirConfig, rng: &mut Rng) -> Vec<f32> {
    let n = cfg.units;
    let h = cfg.grid_height;
    let w = cfg.grid_width;
    assert_eq!(h * w, n);
    let k = cfg.kernel;
    let half = (k / 2) as i32;
    let mut mat = vec![0f32; n * n];
    for i in 0..n {
        let ri = (i / w) as i32;
        let ci = (i % w) as i32;
        for dr in -half..=half {
            for dc in -half..=half {
                let nr = (ri + dr).rem_euclid(h as i32) as usize;
                let nc = (ci + dc).rem_euclid(w as i32) as usize;
                let j = nr * w + nc;
                if i != j {
                    mat[i * n + j] = (rng.uniform01() * 2.0 - 1.0) as f32;
                }
            }
        }
    }
    mat
}

fn scale_spectral_radius(w: &mut [f32], n: usize, target: f32) {
    let iters = (32 + n / 4).max(64);
    let radius = estimate_spectral_radius(w, n, iters);
    if radius > 1e-8 {
        let scale = target / radius;
        for v in w.iter_mut() {
            *v *= scale;
        }
    }
}

fn estimate_spectral_radius(w: &[f32], n: usize, iters: usize) -> f32 {
    let mut v: Vec<f32> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.1 }).collect();
    for _ in 0..iters {
        let mut out = vec![0f32; n];
        for i in 0..n {
            let row = &w[i * n..(i + 1) * n];
            out[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
        }
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v = out.into_iter().map(|x| x / norm).collect();
    }
    let mut out = vec![0f32; n];
    for i in 0..n {
        let row = &w[i * n..(i + 1) * n];
        out[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
    }
    out.iter().zip(&v).map(|(a, b)| a * b).sum::<f32>().abs()
}
