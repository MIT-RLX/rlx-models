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

//! Echo cancellation quality metrics.

/// Mean squared energy of a signal.
pub fn signal_power(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32
}

/// Echo return loss enhancement (ERLE) in dB: 10·log10(P_echo / P_residual).
pub fn erle_db(echo: &[f32], residual: &[f32]) -> f32 {
    let pe = signal_power(echo);
    let pr = signal_power(residual);
    if pe <= 1e-12 || pr <= 1e-12 {
        return 0.0;
    }
    10.0 * (pe / pr).log10()
}

/// Mean squared error vs reference.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        / n as f32
}

/// Improvement vs clean target: 10·log10(MSE(mic,clean) / MSE(out,clean)).
pub fn mse_improvement_db(mic: &[f32], out: &[f32], clean: &[f32]) -> f32 {
    let m_mic = mse(mic, clean);
    let m_out = mse(out, clean);
    if m_mic <= 1e-12 || m_out <= 1e-12 {
        return 0.0;
    }
    10.0 * (m_mic / m_out).log10()
}
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sa = 0.0;
    let mut sb = 0.0;
    let mut sab = 0.0;
    let mut sa2 = 0.0;
    let mut sb2 = 0.0;
    for i in 0..n {
        let x = a[i];
        let y = b[i];
        sa += x;
        sb += y;
        sab += x * y;
        sa2 += x * x;
        sb2 += y * y;
    }
    let nf = n as f32;
    let denom = (nf * sa2 - sa * sa) * (nf * sb2 - sb * sb);
    if denom <= 1e-12 {
        return 0.0;
    }
    (nf * sab - sa * sb) / denom.sqrt()
}

pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
