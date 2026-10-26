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

//! Causal per-patch mean/std scaling with an asinh squash (upstream `timestral.scaler`).
//!
//! Every function works on one row, in f64 whatever the network's number format:
//! upstream computes the causal statistics and the refine frames in float64 too, and
//! `--number-format f64` needs the transforms around the network in f64 as well. The
//! running totals are still taken around a pivot value (the row's or patch's first
//! observation), which keeps them small on large-level series.

/// Per-patch `(loc, scale)` of one row: causal running mean/std at each patch's last step.
///
/// `data.len()` must be a multiple of `patch_size`. Unobserved steps have zero weight;
/// the variance uses Bessel's correction and the std is floored at `minimum_scale`.
pub fn causal_patch_loc_scale(
    data: &[f64],
    observed: &[bool],
    patch_size: usize,
    minimum_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(data.len() % patch_size, 0);
    let n = data.len() / patch_size;
    let mut locs = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(n);
    // Accumulate around the row's first observed value: differences of nearby f64 values
    // are exact, so the running sums stay small. Without the shift, a series at a level of
    // 1e6 drifts the f64 mean by whole units and shifts the forecast by a quarter of its
    // range.
    let pivot = data
        .iter()
        .zip(observed)
        .find(|(_, o)| **o)
        .map_or(0.0, |(x, _)| *x);
    let (mut cum_w, mut cum_x, mut m2, mut prev_mean) = (0f64, 0f64, 0f64, 0f64);
    for (t, (&x, &obs)) in data.iter().zip(observed).enumerate() {
        let x = x - pivot;
        let w = if obs { 1.0 } else { 0.0 };
        cum_w += w;
        cum_x += w * x;
        let denominator = cum_w.max(1.0);
        let mean = cum_x / denominator;
        m2 += (x - prev_mean) * (x - mean) * w;
        prev_mean = mean;
        if (t + 1) % patch_size == 0 {
            let variance = m2 / (denominator - 1.0).max(1.0);
            let scale = variance.sqrt().max(minimum_scale);
            let mean = mean + pivot;
            locs.push(mean);
            scales.push(scale);
        }
    }
    (locs, scales)
}

/// `asinh((x - loc) / scale)` on an observed step, 0 on an unobserved one.
#[inline]
pub fn affine_asinh(x: f64, loc: f64, scale: f64, observed: bool, use_asinh: bool) -> f64 {
    if !observed {
        return 0.0;
    }
    let y = (x - loc) / scale.max(f32::EPSILON as f64);
    if use_asinh { y.asinh() } else { y }
}

/// Inverse of [`affine_asinh`]: `sinh(x) * scale + loc`.
#[inline]
pub fn invert_affine_asinh(x: f64, loc: f64, scale: f64, use_asinh: bool) -> f64 {
    let y = if use_asinh { x.sinh() } else { x };
    y * scale + loc
}

/// Map ±inf (a sinh overflow) to the nearest finite value along `values`; NaN stays.
///
/// An all-infinite slice becomes zeros.
pub fn clamp_nonfinite(values: &mut [f64]) {
    let max_finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let min_finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    for v in values.iter_mut() {
        let c = if *v == f64::INFINITY {
            max_finite
        } else if *v == f64::NEG_INFINITY {
            min_finite
        } else {
            *v
        };
        *v = if c.is_infinite() { 0.0 } else { c };
    }
}

/// A Welford state `(weight, mean, m2)`; `(0, 0, 0)` is the identity.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Welford {
    pub w: f64,
    pub mean: f64,
    pub m2: f64,
}

/// Merge two Welford states exactly as upstream `welford_combine`.
pub fn welford_combine(a: Welford, b: Welford) -> Welford {
    let w = a.w + b.w;
    let w_safe = w.max(f32::MIN_POSITIVE as f64);
    let delta = b.mean - a.mean;
    let mean = a.mean + delta * b.w / w_safe;
    let m2 = a.m2 + b.m2 + delta * delta * a.w * b.w / w_safe;
    if w == 0.0 {
        Welford {
            w,
            mean: 0.0,
            m2: 0.0,
        }
    } else {
        Welford { w, mean, m2 }
    }
}

/// Inclusive Hillis-Steele prefix scan of Welford states (upstream `cumulative_welford`).
pub fn cumulative_welford(states: &mut [Welford]) {
    let n = states.len();
    let mut d = 1;
    while d < n {
        let prev = states.to_vec();
        for i in 0..n {
            let left = if i >= d {
                prev[i - d]
            } else {
                Welford::default()
            };
            states[i] = welford_combine(left, prev[i]);
        }
        d *= 2;
    }
}

/// Per-patch `(loc, scale)` of a decoded block in the advancing refine frame.
///
/// Horizon patch `j` is normalized with the statistics of the observed context plus
/// the block's medians of patches `0..j-1`, each median denormalized in its own frame
/// before it is folded in. `sinh_median[j][k]` is the block median in the last context
/// patch's frame with the asinh undone, `(median - loc) / scale`.
pub fn advancing_block_frames(
    context: &[f64],
    observed: &[bool],
    sinh_median: &[Vec<f64>],
    patch_size: usize,
    minimum_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let ps = patch_size;
    let n_ctx = context.len() / ps;
    let n_pred = sinh_median.len();
    let mut states = vec![Welford::default(); n_ctx + n_pred];
    for (p, state) in states.iter_mut().take(n_ctx).enumerate() {
        let vals = &context[p * ps..(p + 1) * ps];
        let obs = &observed[p * ps..(p + 1) * ps];
        let cnt: f64 = obs.iter().map(|&o| if o { 1.0 } else { 0.0 }).sum();
        // Moments around the patch's first observed value keep the sums small.
        let pivot = vals
            .iter()
            .zip(obs)
            .find(|(_, o)| **o)
            .map_or(0.0, |(v, _)| *v);
        let dev: f64 = vals
            .iter()
            .zip(obs)
            .map(|(&v, &o)| if o { v - pivot } else { 0.0 })
            .sum();
        let mean = pivot + dev / cnt.max(1.0);
        let m2: f64 = vals
            .iter()
            .zip(obs)
            .map(|(&v, &o)| if o { (v - mean).powi(2) } else { 0.0 })
            .sum();
        *state = if cnt == 0.0 {
            Welford::default()
        } else {
            Welford { w: cnt, mean, m2 }
        };
    }
    cumulative_welford(&mut states);
    let mut st = states[n_ctx - 1];

    let offset = st.mean;
    // Upstream: `finfo(float32).max ** 0.5 / 1e3`, whatever the compute dtype.
    let clip = (f32::MAX as f64).sqrt() / 1e3;
    let mut locs = Vec::with_capacity(n_pred);
    let mut scales = Vec::with_capacity(n_pred);
    for median in sinh_median {
        let var = (st.m2 / (st.w - 1.0).max(1.0)).max(0.0);
        let scale = var.sqrt().max(minimum_scale);
        locs.push(st.mean);
        scales.push(scale);
        let folded: Vec<f64> = median
            .iter()
            .map(|&m| (m * scale + st.mean - offset).clamp(-clip, clip) + offset)
            .collect();
        let b_mean = folded.iter().sum::<f64>() / ps as f64;
        let b_m2: f64 = folded.iter().map(|f| (f - b_mean).powi(2)).sum();
        st = welford_combine(
            st,
            Welford {
                w: ps as f64,
                mean: b_mean,
                m2: b_m2,
            },
        );
    }
    (locs, scales)
}

/// Non-causal whole-row `(loc, scale, is_binary)` for a known-future covariate row.
///
/// Rows whose observed values are all 0 or 1 (indicators) keep `loc = 0`, `scale = 1`.
pub fn instance_norm_stats(
    values: &[f64],
    observed: &[bool],
    minimum_scale: f64,
) -> (f64, f64, bool) {
    let n = observed.iter().filter(|&&o| o).count() as f64;
    let n = n.max(1.0);
    let loc = values
        .iter()
        .zip(observed)
        .map(|(&v, &o)| if o { v } else { 0.0 })
        .sum::<f64>()
        / n;
    let var = values
        .iter()
        .zip(observed)
        .map(|(&v, &o)| if o { (v - loc).powi(2) } else { 0.0 })
        .sum::<f64>()
        / n;
    let scale = var.sqrt().max(minimum_scale);
    let zero_one = values
        .iter()
        .zip(observed)
        .all(|(&v, &o)| v == 0.0 || v == 1.0 || !o);
    let is_binary = zero_one && observed.iter().any(|&o| o);
    if is_binary {
        (0.0, 1.0, true)
    } else {
        (loc, scale, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_stats_match_closed_form() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let obs = [true; 4];
        let (loc, scale) = causal_patch_loc_scale(&data, &obs, 2, 1e-6);
        assert_eq!(loc, vec![1.5, 2.5]);
        // Sample std of [1, 2] and [1, 2, 3, 4].
        assert!((scale[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((scale[1] - 1.2909944487358056).abs() < 1e-12);
    }

    #[test]
    fn welford_scan_equals_sequential_merge() {
        let mut states: Vec<Welford> = (0..7)
            .map(|i| Welford {
                w: 2.0,
                mean: i as f64,
                m2: 0.5,
            })
            .collect();
        let mut seq = Vec::new();
        let mut acc = Welford::default();
        for s in &states {
            acc = welford_combine(acc, *s);
            seq.push(acc);
        }
        cumulative_welford(&mut states);
        for (a, b) in states.iter().zip(&seq) {
            assert!((a.mean - b.mean).abs() < 1e-12 && (a.m2 - b.m2).abs() < 1e-9);
        }
    }

    #[test]
    fn clamp_nonfinite_keeps_order() {
        let mut v = [f64::NEG_INFINITY, -1.0, 2.0, f64::INFINITY];
        clamp_nonfinite(&mut v);
        assert_eq!(v, [-1.0, -1.0, 2.0, 2.0]);
    }
}
