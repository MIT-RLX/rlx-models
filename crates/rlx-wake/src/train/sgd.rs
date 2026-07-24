// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

#[derive(Debug, Clone)]
pub struct SgdConfig {
    pub lr: f32,
    pub weight_decay: f32,
    pub epochs: usize,
    pub log_every: usize,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 5e-3,
            weight_decay: 1e-4,
            epochs: 40,
            log_every: 5,
        }
    }
}

pub fn sgd_step(w: &mut [f32], dw: &[f32], lr: f32, weight_decay: f32) {
    debug_assert_eq!(w.len(), dw.len());
    for i in 0..w.len() {
        w[i] -= lr * (dw[i] + weight_decay * w[i]);
    }
}

#[inline]
pub fn bce_loss(prob: f32, label: f32) -> f32 {
    let p = prob.clamp(1e-6, 1.0 - 1e-6);
    -(label * p.ln() + (1.0 - label) * (1.0 - p).ln())
}

/// dL/dlogit for sigmoid + BCE.
#[inline]
pub fn bce_dlogit(prob: f32, label: f32) -> f32 {
    prob - label
}
