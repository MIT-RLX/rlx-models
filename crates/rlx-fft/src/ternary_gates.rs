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

//! Ternary butterfly routing — skip (0), forward (+1), reverse (−1).

use crate::butterfly::{bit_reverse_permute, num_stages};
use crate::config::FftLearnConfig;
use crate::learned_model::FastLearnedFftModel;
use crate::pruned::{gate_count, gate_index};
use crate::twiddle::{exact_twiddles, twiddle_index};
use anyhow::{Result, ensure};

/// Discrete gate mode per butterfly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateMode {
    Skip = 0,
    Forward = 1,
    Reverse = -1,
}

impl GateMode {
    pub fn from_i8(v: i8) -> Self {
        match v {
            0 => Self::Skip,
            -1 => Self::Reverse,
            _ => Self::Forward,
        }
    }

    pub fn to_i8(self) -> i8 {
        match self {
            Self::Skip => 0,
            Self::Forward => 1,
            Self::Reverse => -1,
        }
    }

    /// Relative compute cost vs a full butterfly (skip < reverse < forward).
    pub fn compute_cost(self) -> f32 {
        match self {
            Self::Skip => 0.0,
            Self::Reverse => 0.85,
            Self::Forward => 1.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Forward => "forward",
            Self::Reverse => "reverse",
        }
    }
}

#[inline]
fn cmul(re_a: f32, im_a: f32, re_w: f32, im_w: f32) -> (f32, f32) {
    (re_a * re_w - im_a * im_w, re_a * im_w + im_a * re_w)
}

#[inline]
fn cadd(re_a: f32, im_a: f32, re_b: f32, im_b: f32) -> (f32, f32) {
    (re_a + re_b, im_a + im_b)
}

#[inline]
fn csub(re_a: f32, im_a: f32, re_b: f32, im_b: f32) -> (f32, f32) {
    (re_a - re_b, im_a - im_b)
}

pub fn init_ternary_gates(n_fft: usize) -> Vec<i8> {
    vec![GateMode::Forward.to_i8(); gate_count(n_fft)]
}

pub fn init_ternary_logits(n_fft: usize) -> Vec<[f32; 3]> {
    vec![[0.0, 2.0, -2.0]; gate_count(n_fft)]
}

/// Map teacher soft gates `[0,1]` → ternary `{0,1}` (skip vs forward).
pub fn ternary_gates_from_teacher(teacher: &FastLearnedFftModel, threshold: f32) -> Vec<i8> {
    teacher
        .gates
        .iter()
        .map(|&g| {
            if g >= threshold {
                GateMode::Forward.to_i8()
            } else {
                GateMode::Skip.to_i8()
            }
        })
        .collect()
}

/// Softmax logits from teacher float gates — keeps borderline butterflies as forward.
pub fn ternary_logits_from_teacher(
    teacher: &crate::learned_model::FastLearnedFftModel,
) -> Vec<[f32; 3]> {
    teacher
        .gates
        .iter()
        .map(|&g| {
            if g >= 0.75 {
                [-1.5, 3.0, -2.5]
            } else if g >= 0.45 {
                [-0.5, 2.0, -2.0]
            } else if g <= 0.25 {
                [3.0, -1.0, -2.0]
            } else {
                [1.0, 1.0, -2.0]
            }
        })
        .collect()
}

pub fn logits_from_gates(gates: &[i8]) -> Vec<[f32; 3]> {
    gates
        .iter()
        .map(|&g| {
            let mode = GateMode::from_i8(g);
            let mut logits = [0.0f32; 3];
            match mode {
                GateMode::Skip => logits[0] = 2.0,
                GateMode::Forward => logits[1] = 2.0,
                GateMode::Reverse => logits[2] = 2.0,
            }
            logits
        })
        .collect()
}

pub fn hard_gates_from_logits(logits: &[[f32; 3]]) -> Vec<i8> {
    logits
        .iter()
        .map(|l| {
            let (idx, _) = l
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((1, &0.0));
            match idx {
                0 => GateMode::Skip.to_i8(),
                2 => GateMode::Reverse.to_i8(),
                _ => GateMode::Forward.to_i8(),
            }
        })
        .collect()
}

pub fn softmax3(logits: [f32; 3], temp: f32) -> [f32; 3] {
    let t = temp.max(1e-4);
    let exps = [logits[0] / t, logits[1] / t, logits[2] / t].map(f32::exp);
    let sum = exps[0] + exps[1] + exps[2];
    [exps[0] / sum, exps[1] / sum, exps[2] / sum]
}

fn apply_butterfly_mode(
    mode: GateMode,
    in_a_re: f32,
    in_a_im: f32,
    in_b_re: f32,
    in_b_im: f32,
    top_re: f32,
    top_im: f32,
    bot_re: f32,
    bot_im: f32,
) -> (f32, f32, f32, f32) {
    match mode {
        GateMode::Skip => (in_a_re, in_a_im, in_b_re, in_b_im),
        GateMode::Forward => (top_re, top_im, bot_re, bot_im),
        GateMode::Reverse => (bot_re, bot_im, top_re, top_im),
    }
}

fn apply_stage_ternary(
    buf: &[f32],
    next: &mut [f32],
    twiddles: &[f32],
    gates: &[i8],
    n_fft: usize,
    stage: usize,
) {
    let half = n_fft / 2;
    let stride = 1usize << stage;
    for b in 0..half {
        let group = b / stride;
        let k = b % stride;
        let i0 = (group * 2 * stride + k) * 2;
        let i1 = i0 + stride * 2;
        let w_base = twiddle_index(stage, b, half, 0);
        let w_re = twiddles[w_base];
        let w_im = twiddles[w_base + 1];
        let gi = gate_index(stage, b, half);
        let mode = GateMode::from_i8(gates[gi]);
        let in_a_re = buf[i0];
        let in_a_im = buf[i0 + 1];
        let in_b_re = buf[i1];
        let in_b_im = buf[i1 + 1];
        let (b_re, b_im) = cmul(in_b_re, in_b_im, w_re, w_im);
        let (top_re, top_im) = cadd(in_a_re, in_a_im, b_re, b_im);
        let (bot_re, bot_im) = csub(in_a_re, in_a_im, b_re, b_im);
        let (oa_re, oa_im, ob_re, ob_im) = apply_butterfly_mode(
            mode, in_a_re, in_a_im, in_b_re, in_b_im, top_re, top_im, bot_re, bot_im,
        );
        next[i0] = oa_re;
        next[i0 + 1] = oa_im;
        next[i1] = ob_re;
        next[i1 + 1] = ob_im;
    }
}

fn apply_stage_ternary_soft(
    buf: &[f32],
    next: &mut [f32],
    twiddles: &[f32],
    logits: &[[f32; 3]],
    n_fft: usize,
    stage: usize,
    temp: f32,
) {
    let half = n_fft / 2;
    let stride = 1usize << stage;
    for b in 0..half {
        let group = b / stride;
        let k = b % stride;
        let i0 = (group * 2 * stride + k) * 2;
        let i1 = i0 + stride * 2;
        let w_base = twiddle_index(stage, b, half, 0);
        let w_re = twiddles[w_base];
        let w_im = twiddles[w_base + 1];
        let gi = gate_index(stage, b, half);
        let w = softmax3(logits[gi], temp);
        let in_a_re = buf[i0];
        let in_a_im = buf[i0 + 1];
        let in_b_re = buf[i1];
        let in_b_im = buf[i1 + 1];
        let (b_re, b_im) = cmul(in_b_re, in_b_im, w_re, w_im);
        let (top_re, top_im) = cadd(in_a_re, in_a_im, b_re, b_im);
        let (bot_re, bot_im) = csub(in_a_re, in_a_im, b_re, b_im);
        let (sk_a_re, sk_a_im, sk_b_re, sk_b_im) = apply_butterfly_mode(
            GateMode::Skip,
            in_a_re,
            in_a_im,
            in_b_re,
            in_b_im,
            top_re,
            top_im,
            bot_re,
            bot_im,
        );
        let (fw_a_re, fw_a_im, fw_b_re, fw_b_im) = apply_butterfly_mode(
            GateMode::Forward,
            in_a_re,
            in_a_im,
            in_b_re,
            in_b_im,
            top_re,
            top_im,
            bot_re,
            bot_im,
        );
        let (rv_a_re, rv_a_im, rv_b_re, rv_b_im) = apply_butterfly_mode(
            GateMode::Reverse,
            in_a_re,
            in_a_im,
            in_b_re,
            in_b_im,
            top_re,
            top_im,
            bot_re,
            bot_im,
        );
        next[i0] = w[0] * sk_a_re + w[1] * fw_a_re + w[2] * rv_a_re;
        next[i0 + 1] = w[0] * sk_a_im + w[1] * fw_a_im + w[2] * rv_a_im;
        next[i1] = w[0] * sk_b_re + w[1] * fw_b_re + w[2] * rv_b_re;
        next[i1 + 1] = w[0] * sk_b_im + w[1] * fw_b_im + w[2] * rv_b_im;
    }
}

pub fn ternary_forward_complex(
    input: &[f32],
    twiddles: &[f32],
    gates: &[i8],
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(input.len() == n_fft * 2);
    ensure!(gates.len() >= gate_count(n_fft));
    let stages = num_stages(n_fft);
    let half = n_fft / 2;
    ensure!(twiddles.len() >= stages * half * 2);
    let mut buf = input.to_vec();
    bit_reverse_permute(&mut buf, n_fft);
    for s in 0..stages {
        let mut next = vec![0f32; n_fft * 2];
        apply_stage_ternary(&buf, &mut next, twiddles, gates, n_fft, s);
        buf = next;
    }
    Ok(buf)
}

pub fn ternary_forward_real_batch(
    signal: &[f32],
    twiddles: &[f32],
    gates: &[i8],
    batch: usize,
    n_fft: usize,
) -> Result<Vec<f32>> {
    ensure!(signal.len() == batch * n_fft);
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let mut complex = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            complex[i * 2] = signal[b * n_fft + i];
        }
        let y = ternary_forward_complex(&complex, twiddles, gates, n_fft)?;
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&y);
    }
    Ok(out)
}

pub fn ternary_forward_real_batch_soft(
    signal: &[f32],
    twiddles: &[f32],
    logits: &[[f32; 3]],
    batch: usize,
    n_fft: usize,
    temp: f32,
) -> Result<Vec<f32>> {
    ensure!(signal.len() == batch * n_fft);
    let stages = num_stages(n_fft);
    let mut out = vec![0f32; batch * n_fft * 2];
    for b in 0..batch {
        let mut complex = vec![0f32; n_fft * 2];
        for i in 0..n_fft {
            complex[i * 2] = signal[b * n_fft + i];
        }
        bit_reverse_permute(&mut complex, n_fft);
        let mut buf = complex;
        for s in 0..stages {
            let mut next = vec![0f32; n_fft * 2];
            apply_stage_ternary_soft(&buf, &mut next, twiddles, logits, n_fft, s, temp);
            buf = next;
        }
        out[b * n_fft * 2..(b + 1) * n_fft * 2].copy_from_slice(&buf);
    }
    Ok(out)
}

pub fn gate_mode_counts(gates: &[i8]) -> (usize, usize, usize) {
    let mut skip = 0usize;
    let mut forward = 0usize;
    let mut reverse = 0usize;
    for &g in gates {
        match GateMode::from_i8(g) {
            GateMode::Skip => skip += 1,
            GateMode::Forward => forward += 1,
            GateMode::Reverse => reverse += 1,
        }
    }
    (skip, forward, reverse)
}

pub fn compute_fraction(gates: &[i8]) -> f32 {
    if gates.is_empty() {
        return 1.0;
    }
    let cost: f32 = gates
        .iter()
        .map(|&g| GateMode::from_i8(g).compute_cost())
        .sum();
    cost / gates.len() as f32
}

pub fn exact_twiddles_for(cfg: &FftLearnConfig) -> Vec<f32> {
    exact_twiddles(cfg)
}

/// Bake ternary modes into `(active, reverse)` params for compiled graphs.
/// Skip → (0,0), Forward → (1,0), Reverse → (1,1).
pub fn bake_ternary_params(gates: &[i8]) -> (Vec<f32>, Vec<f32>) {
    let mut active = Vec::with_capacity(gates.len());
    let mut reverse = Vec::with_capacity(gates.len());
    for &g in gates {
        match GateMode::from_i8(g) {
            GateMode::Skip => {
                active.push(0.0);
                reverse.push(0.0);
            }
            GateMode::Forward => {
                active.push(1.0);
                reverse.push(0.0);
            }
            GateMode::Reverse => {
                active.push(1.0);
                reverse.push(1.0);
            }
        }
    }
    (active, reverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterfly::butterfly_forward_real_batch;
    use crate::config::FftLearnConfig;
    use crate::reference::max_abs_error;

    #[test]
    fn all_forward_matches_butterfly() {
        let cfg = FftLearnConfig::new(64, 1).unwrap();
        let tw = exact_twiddles(&cfg);
        let gates = init_ternary_gates(64);
        let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let ref_spec = butterfly_forward_real_batch(&signal, &tw, 1, 64).unwrap();
        let pred = ternary_forward_real_batch(&signal, &tw, &gates, 1, 64).unwrap();
        let err = max_abs_error(&pred, &ref_spec);
        assert!(err < 1e-4, "forward ternary err={err}");
    }

    #[test]
    fn skip_increases_sparsity() {
        let mut gates = init_ternary_gates(64);
        gates.fill(GateMode::Skip.to_i8());
        assert!(compute_fraction(&gates) < 0.01);
    }
}
