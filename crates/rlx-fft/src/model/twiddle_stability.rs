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

//! Twiddle update stability — unit-circle projection, LR scaling, gradient clipping.

/// Default global L2 clip for twiddle gradients in butterfly training steps.
pub const DEFAULT_GRAD_CLIP: f32 = 1.0;

/// Scale base learning rate down for larger FFT sizes (depth + amplitude grow with n).
pub fn lr_for_n_fft(base_lr: f64, n_fft: usize) -> f32 {
    let log_n = (n_fft as f64).log2().max(1.0);
    (base_lr / log_n.sqrt()) as f32
}

/// Project each complex twiddle `(re, im)` onto the unit circle.
pub fn project_twiddles_unit_circle(twiddles: &mut [f32]) {
    for chunk in twiddles.chunks_mut(2) {
        let re = chunk[0];
        let im = chunk[1];
        let mag = (re * re + im * im).sqrt();
        if mag > 1e-12 && (mag - 1.0).abs() > 1e-6 {
            chunk[0] = re / mag;
            chunk[1] = im / mag;
        }
    }
}

/// Clip global L2 norm of a twiddle gradient buffer (0 = disabled).
pub fn clip_twiddle_grad(grad: &mut [f32], max_norm: f32) {
    if max_norm <= 0.0 {
        return;
    }
    let norm = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}

/// Cartesian (re, im) gradient → scalar dL/dθ for w = e^{iθ} on the unit circle.
pub fn cartesian_grad_to_angle(tw: &[f32], grad: &[f32]) -> Vec<f32> {
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

/// Apply angle deltas: w_new = w · exp(−i·delta), keeping unit magnitude.
pub fn apply_angle_deltas(tw: &mut [f32], deltas: &[f32]) {
    for (chunk, &delta) in tw.chunks_mut(2).zip(deltas) {
        let re = chunk[0];
        let im = chunk[1];
        let mag = (re * re + im * im).sqrt().max(1e-12);
        let ur = re / mag;
        let ui = im / mag;
        let (s, c) = delta.sin_cos();
        chunk[0] = ur * c + ui * s;
        chunk[1] = ui * c - ur * s;
    }
}

/// Stable SGD: angle-space update with optional gradient clip and unit-circle projection.
pub fn apply_twiddle_update(
    twiddles: &mut [f32],
    grad: &[f32],
    lr: f32,
    grad_clip: f32,
    project: bool,
) {
    let mut angle_grad = cartesian_grad_to_angle(twiddles, grad);
    clip_twiddle_grad(&mut angle_grad, grad_clip);
    let deltas: Vec<f32> = angle_grad.iter().map(|g| lr * g).collect();
    apply_angle_deltas(twiddles, &deltas);
    if project {
        project_twiddles_unit_circle(twiddles);
    }
}

pub fn max_twiddle_magnitude(twiddles: &[f32]) -> f32 {
    twiddles
        .chunks(2)
        .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
        .fold(0f32, f32::max)
}

pub fn twiddle_drift_from_unit(twiddles: &[f32]) -> f32 {
    twiddles
        .chunks(2)
        .map(|c| {
            let mag = (c[0] * c[0] + c[1] * c[1]).sqrt();
            (mag - 1.0).abs()
        })
        .fold(0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_restores_unit_magnitude() {
        let mut tw = vec![2.0, 0.0, 0.0, 3.0];
        project_twiddles_unit_circle(&mut tw);
        assert!((tw[0] - 1.0).abs() < 1e-6);
        assert!((tw[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lr_scales_down_with_n() {
        let lr64 = lr_for_n_fft(1e-3, 64);
        let lr1024 = lr_for_n_fft(1e-3, 1024);
        assert!(lr1024 < lr64);
    }

    #[test]
    fn angle_update_stays_on_unit_circle() {
        let mut tw = vec![0.6, 0.8, -0.3, 0.954];
        let grad = vec![100.0, -50.0, 80.0, 120.0];
        for _ in 0..200 {
            apply_twiddle_update(&mut tw, &grad, 0.002, 1.0, true);
            assert!(tw.iter().all(|v| v.is_finite()));
            assert!(max_twiddle_magnitude(&tw) <= 1.01);
        }
    }
}
