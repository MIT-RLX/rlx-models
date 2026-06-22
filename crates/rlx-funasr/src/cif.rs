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

//! Host-side **CIF** (Continuous Integrate-and-Fire) — `CifPredictorV2`.
//!
//! The small predictor head (a context `Conv1d`, ReLU, a `Linear→1`, sigmoid)
//! produces a per-frame weight `α`; the sequential integrate-and-fire then
//! emits one acoustic embedding each time the running `α` sum crosses the
//! threshold. Both run on the host (the fire loop is inherently sequential),
//! exactly matching `cif_predictor.py`.

use crate::config::CifConfig;

/// Weights for the CIF predictor head.
pub struct PredictorWeights {
    /// `cif_conv1d.weight`, shape `[d, d, k]` (k = l_order + r_order + 1).
    pub conv_w: Vec<f32>,
    /// `cif_conv1d.bias`, shape `[d]`.
    pub conv_b: Vec<f32>,
    /// `cif_output.weight`, shape `[1, d]`.
    pub out_w: Vec<f32>,
    /// `cif_output.bias`, shape `[1]`.
    pub out_b: f32,
}

/// Per-frame `α ∈ [0, ∞)`, length `t`.
pub fn compute_alphas(
    encoder_out: &[f32],
    t: usize,
    d: usize,
    w: &PredictorWeights,
    cfg: &CifConfig,
) -> Vec<f32> {
    let k = cfg.l_order + cfg.r_order + 1;
    let l = cfg.l_order;
    // padded[ time index p in 0..t+l+r ] ↔ original frame (p - l)
    let at = |p: i64, c: usize| -> f32 {
        let ti = p - l as i64;
        if ti < 0 || ti >= t as i64 {
            0.0
        } else {
            encoder_out[ti as usize * d + c]
        }
    };
    let mut alphas = vec![0.0f32; t];
    let mut conv_out = vec![0.0f32; d];
    for o in 0..t {
        // depthwise-free full conv1d over the d input channels
        for (c, co) in conv_out.iter_mut().enumerate() {
            let mut acc = w.conv_b[c];
            let wbase = c * d * k;
            for i in 0..d {
                let wrow = wbase + i * k;
                for kk in 0..k {
                    acc += w.conv_w[wrow + kk] * at(o as i64 + kk as i64, i);
                }
            }
            *co = acc.max(0.0); // ReLU
        }
        // cif_output: Linear(d, 1)
        let mut a = w.out_b;
        for c in 0..d {
            a += w.out_w[c] * conv_out[c];
        }
        // sigmoid → relu(a*smooth - noise)
        let s = 1.0 / (1.0 + (-a).exp());
        alphas[o] = (s * cfg.smooth_factor - cfg.noise_threshold).max(0.0);
    }
    alphas
}

/// Run tail-processing + integrate-and-fire. Returns row-major acoustic
/// embeddings `[n_tokens, d]` and `n_tokens`.
pub fn integrate_and_fire(
    encoder_out: &[f32],
    t: usize,
    d: usize,
    alphas: &[f32],
    cfg: &CifConfig,
) -> (Vec<f32>, usize) {
    // tail processing (single-utterance, no padding mask): append a zero frame
    // and a `tail_threshold` α so the trailing partial integration fires.
    let mut a = alphas.to_vec();
    let hidden_frames = if cfg.tail_threshold > 0.0 {
        a.push(cfg.tail_threshold);
        t + 1
    } else {
        t
    };
    let hidden = |ti: usize, c: usize| -> f32 {
        if ti < t {
            encoder_out[ti * d + c]
        } else {
            0.0 // appended tail frame
        }
    };
    let token_num = a.iter().sum::<f32>().floor() as usize;

    let fired = fire_loop(&hidden, hidden_frames, d, &a, cfg.threshold);

    // inference truncates to the (floored) predicted token count
    let n = token_num.min(fired.len());
    let mut out = vec![0.0f32; n * d];
    for (i, fr) in fired.iter().take(n).enumerate() {
        out[i * d..(i + 1) * d].copy_from_slice(fr);
    }
    (out, n)
}

/// The sequential integrate-and-fire (`cif` in `cif_predictor.py`): walk the
/// frames, accumulate `α·hidden`, and emit a frame each time the running `α`
/// sum crosses `threshold`, carrying the remainder into the next frame.
fn fire_loop(
    hidden: &dyn Fn(usize, usize) -> f32,
    n: usize,
    d: usize,
    alphas: &[f32],
    threshold: f32,
) -> Vec<Vec<f32>> {
    let mut integrate = 0.0f32;
    let mut frame = vec![0.0f32; d];
    let mut fired: Vec<Vec<f32>> = Vec::new();
    for (ti, &alpha) in alphas.iter().enumerate().take(n) {
        let distribution_completion = 1.0 - integrate;
        integrate += alpha;
        let fire = integrate >= threshold;
        let cur = if fire { distribution_completion } else { alpha };
        let remainds = alpha - cur;
        for c in 0..d {
            frame[c] += cur * hidden(ti, c);
        }
        if fire {
            integrate -= threshold;
            fired.push(frame.clone());
            for c in 0..d {
                frame[c] = remainds * hidden(ti, c);
            }
        }
    }
    fired
}

/// Vectorized reference (`cif_v1`/`cif_wo_hidden_v1` in `cif_predictor.py`,
/// threshold = 1.0): fire wherever the integer part of the α prefix-sum
/// increments; each emitted frame is the difference of cumulative `α·hidden`
/// between consecutive fires, adjusted by the fractional remainder carried
/// across the boundary. Used to cross-check [`fire_loop`].
#[cfg(test)]
fn cif_v1_ref(hidden: &[Vec<f32>], alphas: &[f32]) -> Vec<Vec<f32>> {
    let n = alphas.len();
    let d = if n > 0 { hidden[0].len() } else { 0 };
    let mut prefix = vec![0f64; n];
    let mut s = 0.0;
    for i in 0..n {
        s += alphas[i] as f64;
        prefix[i] = s;
    }
    let fire_pos: Vec<usize> = (0..n)
        .filter(|&i| {
            let prev = if i == 0 { 0.0 } else { prefix[i - 1].floor() };
            prefix[i].floor() - prev > 0.0
        })
        .collect();
    // cumulative α·hidden
    let mut psh = vec![vec![0f64; d]; n];
    let mut acc = vec![0f64; d];
    for i in 0..n {
        for c in 0..d {
            acc[c] += alphas[i] as f64 * hidden[i][c] as f64;
        }
        psh[i].copy_from_slice(&acc);
    }
    let frames_at: Vec<&Vec<f64>> = fire_pos.iter().map(|&i| &psh[i]).collect();
    let remains: Vec<f64> = fire_pos
        .iter()
        .map(|&i| prefix[i] - prefix[i].floor())
        .collect();
    let remain_frames: Vec<Vec<f64>> = fire_pos
        .iter()
        .zip(&remains)
        .map(|(&i, &r)| (0..d).map(|c| r * hidden[i][c] as f64).collect())
        .collect();
    let mut out = Vec::with_capacity(fire_pos.len());
    for k in 0..fire_pos.len() {
        let mut row = vec![0f32; d];
        for c in 0..d {
            let f = frames_at[k][c];
            let sf = if k == 0 { 0.0 } else { frames_at[k - 1][c] };
            let srf = if k == 0 { 0.0 } else { remain_frames[k - 1][c] };
            let rf = remain_frames[k][c];
            row[c] = (f - sf + srf - rf) as f32;
        }
        out.push(row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_fire_loop_matches_cif_v1() {
        // deterministic pseudo-random positive alphas + hidden
        let n = 40usize;
        let d = 6usize;
        let mut alphas = vec![0f32; n];
        let mut hidden = vec![vec![0f32; d]; n];
        let mut seed = 12345u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };
        for i in 0..n {
            alphas[i] = 0.05 + 0.9 * rng(); // in (0.05, 0.95), avoids exact integer boundaries
            for c in 0..d {
                hidden[i][c] = rng() * 2.0 - 1.0;
            }
        }
        let flat: Vec<f32> = hidden.iter().flatten().copied().collect();
        let scalar = fire_loop(&|t, c| flat[t * d + c], n, d, &alphas, 1.0);
        let vectorized = cif_v1_ref(&hidden, &alphas);
        assert_eq!(scalar.len(), vectorized.len(), "fire counts differ");
        let mut maxd = 0f32;
        for (a, b) in scalar.iter().zip(&vectorized) {
            for c in 0..d {
                maxd = maxd.max((a[c] - b[c]).abs());
            }
        }
        assert!(maxd < 1e-4, "scalar vs cif_v1 max diff {maxd}");
    }

    #[test]
    fn fire_count_matches_alpha_sum() {
        let d = 2;
        let t = 5;
        // hidden = ones; alphas summing to ~3 should fire ~3 frames.
        let enc = vec![1.0f32; t * d];
        let alphas = vec![0.6f32; t]; // sum = 3.0
        let cfg = CifConfig {
            idim: d,
            l_order: 1,
            r_order: 1,
            threshold: 1.0,
            tail_threshold: 0.0,
            smooth_factor: 1.0,
            noise_threshold: 0.0,
        };
        let (emb, n) = integrate_and_fire(&enc, t, d, &alphas, &cfg);
        assert_eq!(n, 3);
        assert_eq!(emb.len(), n * d);
        // each fired frame should integrate to ~1.0 of the unit hidden → ~1.0
        for v in &emb {
            assert!((*v - 1.0).abs() < 1e-5, "frame value {v}");
        }
    }
}
