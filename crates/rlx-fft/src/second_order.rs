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

//! Second-order / adaptive twiddle optimizers (Adam, diagonal preconditioning, HVP).
//!
//! Adaptive steps use **angle parameterization** on the unit circle so re/im are not
//! updated independently (which breaks twiddle magnitude even with projection).

use crate::twiddle_stability::{
    apply_twiddle_update, clip_twiddle_grad, project_twiddles_unit_circle,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TwiddleOptimizer {
    Sgd,
    /// Adam on twiddle angles (unit-circle safe).
    Adam,
    /// Diagonal preconditioner on twiddle angles.
    DiagPrecond,
}

impl TwiddleOptimizer {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sgd => "sgd",
            Self::Adam => "adam",
            Self::DiagPrecond => "diag_precond",
        }
    }

    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "sgd" => Ok(Self::Sgd),
            "adam" => Ok(Self::Adam),
            "diag" | "diag_precond" | "precond" => Ok(Self::DiagPrecond),
            other => anyhow::bail!("unknown optimizer {other} (sgd, adam, diag_precond)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwiddleOptState {
    pub optimizer: TwiddleOptimizer,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Moment buffers — length `n_complex_twiddles` for adaptive opts.
    m_enc: Vec<f32>,
    m_dec: Vec<f32>,
    v_enc: Vec<f32>,
    v_dec: Vec<f32>,
    step: usize,
}

impl TwiddleOptState {
    pub fn new(optimizer: TwiddleOptimizer, enc_len: usize, dec_len: usize) -> Self {
        let enc_state = state_len(optimizer, enc_len);
        let dec_state = state_len(optimizer, dec_len);
        Self {
            optimizer,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m_enc: vec![0f32; enc_state],
            m_dec: vec![0f32; dec_state],
            v_enc: vec![0f32; enc_state],
            v_dec: vec![0f32; dec_state],
            step: 0,
        }
    }

    pub fn step_pair(
        &mut self,
        encoder: &mut [f32],
        decoder: &mut [f32],
        enc_grad: &[f32],
        dec_grad: &[f32],
        lr: f32,
        grad_clip: f32,
        project: bool,
    ) {
        self.step += 1;
        match self.optimizer {
            TwiddleOptimizer::Sgd => {
                apply_twiddle_update(encoder, enc_grad, lr, grad_clip, project);
                apply_twiddle_update(decoder, dec_grad, lr, grad_clip, project);
            }
            TwiddleOptimizer::Adam => {
                adam_angle_update(
                    encoder,
                    enc_grad,
                    &mut self.m_enc,
                    &mut self.v_enc,
                    self.step,
                    lr,
                    grad_clip,
                    self.beta1,
                    self.beta2,
                    self.eps,
                );
                adam_angle_update(
                    decoder,
                    dec_grad,
                    &mut self.m_dec,
                    &mut self.v_dec,
                    self.step,
                    lr,
                    grad_clip,
                    self.beta1,
                    self.beta2,
                    self.eps,
                );
                let _ = project;
            }
            TwiddleOptimizer::DiagPrecond => {
                diag_precond_angle_update(
                    encoder,
                    enc_grad,
                    &mut self.v_enc,
                    lr,
                    grad_clip,
                    self.beta2,
                    self.eps,
                );
                diag_precond_angle_update(
                    decoder,
                    dec_grad,
                    &mut self.v_dec,
                    lr,
                    grad_clip,
                    self.beta2,
                    self.eps,
                );
                let _ = project;
            }
        }
    }
}

fn state_len(optimizer: TwiddleOptimizer, flat_len: usize) -> usize {
    match optimizer {
        TwiddleOptimizer::Sgd => flat_len,
        TwiddleOptimizer::Adam | TwiddleOptimizer::DiagPrecond => flat_len / 2,
    }
}

/// Cartesian (re, im) gradient → scalar dL/dθ for w = e^{iθ} on the unit circle.
fn cartesian_grad_to_angle(tw: &[f32], grad: &[f32]) -> Vec<f32> {
    debug_assert_eq!(tw.len(), grad.len());
    let mut out = Vec::with_capacity(tw.len() / 2);
    for (w, g) in tw.chunks(2).zip(grad.chunks(2)) {
        let re = w[0];
        let im = w[1];
        let mag = (re * re + im * im).sqrt().max(1e-12);
        let ur = re / mag;
        let ui = im / mag;
        out.push(-g[0] * ui + g[1] * ur);
    }
    out
}

fn apply_angle_deltas(tw: &mut [f32], deltas: &[f32]) {
    for (chunk, &delta) in tw.chunks_mut(2).zip(deltas) {
        let re = chunk[0];
        let im = chunk[1];
        let mag = (re * re + im * im).sqrt().max(1e-12);
        let ur = re / mag;
        let ui = im / mag;
        let (s, c) = delta.sin_cos();
        // w_new = w * exp(-i*delta)
        chunk[0] = ur * c + ui * s;
        chunk[1] = ui * c - ur * s;
    }
}

fn adam_angle_update(
    tw: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: usize,
    lr: f32,
    grad_clip: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) {
    let mut angle_grad = cartesian_grad_to_angle(tw, grad);
    clip_twiddle_grad(&mut angle_grad, grad_clip);
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let mut deltas = vec![0f32; angle_grad.len()];
    for i in 0..angle_grad.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * angle_grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * angle_grad[i] * angle_grad[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        deltas[i] = lr * m_hat / (v_hat.sqrt() + eps);
    }
    apply_angle_deltas(tw, &deltas);
}

fn diag_precond_angle_update(
    tw: &mut [f32],
    grad: &[f32],
    v: &mut [f32],
    lr: f32,
    grad_clip: f32,
    beta2: f32,
    eps: f32,
) {
    let mut angle_grad = cartesian_grad_to_angle(tw, grad);
    clip_twiddle_grad(&mut angle_grad, grad_clip);
    let mut deltas = vec![0f32; angle_grad.len()];
    for i in 0..angle_grad.len() {
        v[i] = beta2 * v[i] + (1.0 - beta2) * angle_grad[i] * angle_grad[i];
        deltas[i] = lr * angle_grad[i] / (v[i].sqrt() + eps);
    }
    apply_angle_deltas(tw, &deltas);
}

/// Finite-difference Hessian–vector product on twiddle flat buffer (diagnostic / small n).
pub fn hvp_twiddles_finite_diff<F>(
    tw: &[f32],
    direction: &[f32],
    mut loss_and_grad: F,
    eps: f32,
) -> anyhow::Result<Vec<f32>>
where
    F: FnMut(&[f32]) -> anyhow::Result<(f32, Vec<f32>)>,
{
    anyhow::ensure!(tw.len() == direction.len());
    let mut plus = tw.to_vec();
    let mut minus = tw.to_vec();
    for i in 0..tw.len() {
        plus[i] += eps * direction[i];
        minus[i] -= eps * direction[i];
    }
    let (_, g_plus) = loss_and_grad(&plus)?;
    let (_, g_minus) = loss_and_grad(&minus)?;
    Ok(g_plus
        .iter()
        .zip(g_minus.iter())
        .map(|(a, b)| (a - b) / (2.0 * eps))
        .collect())
}

/// One diagonal Gauss–Newton style step using grad² as curvature proxy (angle space).
pub fn diag_gn_step(tw: &mut [f32], grad: &[f32], lr: f32, damping: f32, project: bool) {
    let angle_grad = cartesian_grad_to_angle(tw, grad);
    let deltas: Vec<f32> = angle_grad
        .iter()
        .map(|g| lr * g / (g * g + damping))
        .collect();
    apply_angle_deltas(tw, &deltas);
    if project {
        project_twiddles_unit_circle(tw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_delta_descent_on_quadratic() {
        let mut tw = vec![0.6f32, 0.8];
        for _ in 0..100 {
            let g_theta = 2.0 * tw[1].atan2(tw[0]);
            apply_angle_deltas(&mut tw, &[0.05 * g_theta]);
        }
        let theta = tw[1].atan2(tw[0]);
        assert!(theta.abs() < 0.2, "theta={theta}");
    }

    #[test]
    fn angle_adam_reduces_quadratic_angle() {
        let mut tw = vec![0.6f32, 0.8];
        let mut m = [0.0];
        let mut v = [0.0];
        for step in 1..=400 {
            let g_theta = 2.0 * tw[1].atan2(tw[0]);
            let mut angle_grad = vec![g_theta];
            clip_twiddle_grad(&mut angle_grad, 0.0);
            let bc1 = 1.0 - 0.9f32.powi(step);
            let bc2 = 1.0 - 0.999f32.powi(step);
            let mut deltas = vec![0.0];
            m[0] = 0.9 * m[0] + 0.1 * angle_grad[0];
            v[0] = 0.999 * v[0] + 0.001 * angle_grad[0] * angle_grad[0];
            deltas[0] = 0.15 * (m[0] / bc1) / ((v[0] / bc2).sqrt() + 1e-8);
            apply_angle_deltas(&mut tw, &deltas);
        }
        let theta = tw[1].atan2(tw[0]);
        assert!(theta.abs() < 0.2, "theta={theta}");
    }

    #[test]
    fn cartesian_to_angle_chain_rule() {
        let tw = vec![1.0, 0.0];
        let grad = vec![0.0, 1.0];
        let dtheta = cartesian_grad_to_angle(&tw, &grad)[0];
        assert!((dtheta - 1.0).abs() < 1e-5);
    }
}
