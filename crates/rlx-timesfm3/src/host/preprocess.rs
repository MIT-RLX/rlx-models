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

//! RevIN, patching, and decode-time preprocessing.

use crate::config::TimesFM3Config;
use crate::host::math::safe_div_sigma;
use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView5,
};

pub struct RunningStats {
    pub n: Array3<f32>,
    pub mu: Array3<f32>,
    pub sigma: Array3<f32>,
}

pub fn update_running_stats(
    n: ArrayView2<f32>,
    mu: ArrayView2<f32>,
    sigma: ArrayView2<f32>,
    x: ArrayView2<f32>,
    mask: ArrayView2<bool>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (b, p) = x.dim();
    let mut new_n = n.to_owned();
    let mut new_mu = mu.to_owned();
    let mut new_sigma = sigma.to_owned();
    for bi in 0..b {
        let mut inc_n = 0.0f32;
        let mut inc_sum = 0.0f32;
        for pi in 0..p {
            if !mask[[bi, pi]] {
                inc_n += 1.0;
                inc_sum += x[[bi, pi]];
            }
        }
        let inc_mu = if inc_n == 0.0 { 0.0 } else { inc_sum / inc_n };
        let mut inc_var = 0.0f32;
        if inc_n > 0.0 {
            for pi in 0..p {
                if !mask[[bi, pi]] {
                    let d = x[[bi, pi]] - inc_mu;
                    inc_var += d * d;
                }
            }
            inc_var /= inc_n;
        }
        let inc_sigma = inc_var.sqrt();
        let old_n = n[[bi, 0]];
        let old_mu = mu[[bi, 0]];
        let old_sigma = sigma[[bi, 0]];
        let nn = old_n + inc_n;
        new_n[[bi, 0]] = nn;
        if nn == 0.0 {
            new_mu[[bi, 0]] = 0.0;
            new_sigma[[bi, 0]] = 0.0;
        } else {
            let nm = (old_n * old_mu + inc_mu * inc_n) / nn;
            new_mu[[bi, 0]] = nm;
            let var = (old_n * old_sigma * old_sigma
                + inc_n * inc_sigma * inc_sigma
                + old_n * (old_mu - nm).powi(2)
                + inc_n * (inc_mu - nm).powi(2))
                / nn;
            new_sigma[[bi, 0]] = var.sqrt();
        }
    }
    (new_n, new_mu, new_sigma)
}

pub fn get_running_stats(values: ArrayView4<f32>, masks: ArrayView4<bool>) -> RunningStats {
    let (b, v, n, _p) = values.dim();
    let mut all_n = Array3::<f32>::zeros((b, v, n));
    let mut all_mu = Array3::<f32>::zeros((b, v, n));
    let mut all_sigma = Array3::<f32>::zeros((b, v, n));
    for bi in 0..b {
        for vi in 0..v {
            let mut cur_n = Array2::<f32>::zeros((1, 1));
            let mut cur_mu = Array2::<f32>::zeros((1, 1));
            let mut cur_sigma = Array2::<f32>::zeros((1, 1));
            for ni in 0..n {
                let x = values
                    .slice(ndarray::s![bi, vi, ni, ..])
                    .insert_axis(ndarray::Axis(0));
                let m = masks
                    .slice(ndarray::s![bi, vi, ni, ..])
                    .insert_axis(ndarray::Axis(0));
                let (cn, cm, cs) =
                    update_running_stats(cur_n.view(), cur_mu.view(), cur_sigma.view(), x, m);
                cur_n = cn;
                cur_mu = cm;
                cur_sigma = cs;
                all_n[[bi, vi, ni]] = cur_n[[0, 0]];
                all_mu[[bi, vi, ni]] = cur_mu[[0, 0]];
                all_sigma[[bi, vi, ni]] = cur_sigma[[0, 0]];
            }
        }
    }
    RunningStats {
        n: all_n,
        mu: all_mu,
        sigma: all_sigma,
    }
}

pub fn revin(
    x: ArrayView4<f32>,
    mu: ArrayView3<f32>,
    sigma: ArrayView3<f32>,
    reverse: bool,
) -> Array4<f32> {
    let (b, v, n, p) = x.dim();
    let mut out = x.to_owned();
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                let m = mu[[bi, vi, ni]];
                let s = safe_div_sigma(sigma[[bi, vi, ni]]);
                for pi in 0..p {
                    if reverse {
                        out[[bi, vi, ni, pi]] = x[[bi, vi, ni, pi]] * s + m;
                    } else {
                        out[[bi, vi, ni, pi]] = (x[[bi, vi, ni, pi]] - m) / s;
                    }
                }
            }
        }
    }
    out
}

pub fn get_output_patch_via_roll(x: ArrayView4<f32>, rolls: usize) -> (Array4<f32>, Array4<bool>) {
    let (b, v, n, p) = x.dim();
    let out_p = p * rolls;
    let mut result = Array4::<f32>::zeros((b, v, n, out_p));
    let mut wrap = Array4::<bool>::from_elem((1, 1, n, out_p), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for oi in 0..out_p {
                    let roll_idx = oi / p + 1;
                    let within = oi % p;
                    let src_patch = ni + roll_idx;
                    if src_patch >= n {
                        wrap[[0, 0, ni, oi]] = true;
                        result[[bi, vi, ni, oi]] = 0.0;
                    } else {
                        result[[bi, vi, ni, oi]] = x[[bi, vi, src_patch, within]];
                    }
                }
            }
        }
    }
    (result, wrap)
}

pub fn stitch_patches_quantile(patch_preds: ArrayView5<f32>, patch_len: usize) -> Array4<f32> {
    let (b, v, num_patches, total_len, q) = patch_preds.dim();
    let overlap = total_len - patch_len;
    if num_patches == 1 {
        return patch_preds.slice(ndarray::s![.., .., 0, .., ..]).to_owned();
    }
    let horizon = num_patches * patch_len + overlap;
    let mut out = Array4::<f32>::zeros((b, v, horizon, q));

    for bi in 0..b {
        for vi in 0..v {
            // First chunk
            for t in 0..patch_len {
                for qi in 0..q {
                    out[[bi, vi, t, qi]] = patch_preds[[bi, vi, 0, t, qi]];
                }
            }
            // Middle stitched overlaps
            let mut pos = patch_len;
            for pi in 1..num_patches {
                let prev = patch_preds.slice(ndarray::s![bi, vi, pi - 1, patch_len.., ..]);
                let next = patch_preds.slice(ndarray::s![bi, vi, pi, ..overlap, ..]);
                for t in 0..overlap {
                    let w = 1.0 - t as f32 / overlap as f32;
                    for qi in 0..q {
                        out[[bi, vi, pos - overlap + t, qi]] =
                            w * prev[[t, qi]] + (1.0 - w) * next[[t, qi]];
                    }
                }
                let end = if pi == num_patches - 1 {
                    total_len
                } else {
                    patch_len
                };
                for t in overlap..end {
                    for qi in 0..q {
                        out[[bi, vi, pos + t - overlap, qi]] = patch_preds[[bi, vi, pi, t, qi]];
                    }
                }
                pos += patch_len;
            }
        }
    }
    out
}

pub fn clamp_values(x: ArrayView4<f32>, clip: f32) -> Array4<f32> {
    x.mapv(|v| v.clamp(-clip, clip))
}

pub fn nan_to_num(x: ArrayView4<f32>) -> Array4<f32> {
    x.mapv(|v| if v.is_nan() { 0.0 } else { v })
}

pub fn apply_linear_detrend(
    ctx_vals: ArrayView3<f32>,
    ctx_masks: ArrayView3<bool>,
    context_len: usize,
    threshold: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>, Array3<bool>) {
    let (b, v, t) = ctx_vals.dim();
    let mut out = ctx_vals.to_owned();
    let mut m_trend = Array3::<f32>::zeros((b, v, 1));
    let mut c_trend = Array3::<f32>::zeros((b, v, 1));
    let mut apply = Array3::<bool>::from_elem((b, v, 1), false);

    for bi in 0..b {
        for vi in 0..v {
            let mut n_v = 0.0f32;
            let mut sum_t = 0.0f32;
            let mut sum_t2 = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_ty = 0.0f32;
            let mut sum_y2 = 0.0f32;
            for ti in 0..t {
                if ctx_masks[[bi, vi, ti]] {
                    continue;
                }
                let tn = (ti as f32 - (context_len - 1) as f32) / context_len as f32;
                n_v += 1.0;
                sum_t += tn;
                sum_t2 += tn * tn;
                sum_y += ctx_vals[[bi, vi, ti]];
                sum_ty += tn * ctx_vals[[bi, vi, ti]];
                sum_y2 += ctx_vals[[bi, vi, ti]].powi(2);
            }
            let det = n_v * sum_t2 - sum_t * sum_t;
            let (m, c) = if det == 0.0 {
                let mean = if n_v > 0.0 { sum_y / n_v } else { 0.0 };
                (0.0, mean)
            } else {
                let m = (n_v * sum_ty - sum_t * sum_y) / det;
                let c = (sum_y - m * sum_t) / n_v;
                (m, c)
            };
            let mean_y = if n_v > 0.0 { sum_y / n_v } else { 0.0 };
            let var_orig = (sum_y2 / n_v.max(1.0) - mean_y.powi(2)).max(0.0);
            let std_orig = var_orig.sqrt();

            let mut sum_yd = 0.0f32;
            let mut sum_yd2 = 0.0f32;
            for ti in 0..t {
                if ctx_masks[[bi, vi, ti]] {
                    continue;
                }
                let tn = (ti as f32 - (context_len - 1) as f32) / context_len as f32;
                let yd = ctx_vals[[bi, vi, ti]] - (m * tn + c);
                sum_yd += yd;
                sum_yd2 += yd * yd;
            }
            let mean_yd = sum_yd / n_v.max(1.0);
            let var_det = (sum_yd2 / n_v.max(1.0) - mean_yd.powi(2)).max(0.0);
            let std_det = var_det.sqrt();
            let do_detrend = std_det < threshold * std_orig;

            m_trend[[bi, vi, 0]] = m;
            c_trend[[bi, vi, 0]] = c;
            apply[[bi, vi, 0]] = do_detrend;

            if do_detrend {
                for ti in 0..t {
                    let tn = (ti as f32 - (context_len - 1) as f32) / context_len as f32;
                    out[[bi, vi, ti]] = ctx_vals[[bi, vi, ti]] - (m * tn + c);
                }
            }
        }
    }
    (out, m_trend, c_trend, apply)
}

pub fn pad_left_1d(x: ArrayView2<f32>, pad: usize) -> Array2<f32> {
    let (b, t) = x.dim();
    let mut out = Array2::<f32>::zeros((b, t + pad));
    for bi in 0..b {
        for ti in 0..t {
            out[[bi, ti + pad]] = x[[bi, ti]];
        }
    }
    out
}

pub fn pad_left_mask(x: ArrayView1<bool>, pad: usize) -> Array1<bool> {
    let t = x.len();
    let mut out = Array1::<bool>::from_elem(t + pad, true);
    for i in 0..t {
        out[i + pad] = x[i];
    }
    out
}

pub fn build_decode_inputs(
    cfg: &TimesFM3Config,
    target: ArrayView3<f32>,
    horizon: usize,
    mask: Option<ArrayView1<bool>>,
    past_only: Option<ArrayView3<f32>>,
    past_future: Option<ArrayView3<f32>>,
) -> (
    Array4<f32>,
    Array4<bool>,
    Array3<bool>,
    Array2<bool>,
    usize,
    usize,
) {
    let b = target.shape()[0];
    let num_target = target.shape()[1];
    let context = target.shape()[2];

    let ctx_padding = (cfg.input_patch_len - context % cfg.input_patch_len) % cfg.input_patch_len;
    let context_padded = context + ctx_padding;
    let num_context_patches = context_padded / cfg.input_patch_len;

    let extract_len = if cfg.use_stitching {
        (2 * cfg.input_patch_len).min(cfg.output_patch_len)
    } else {
        cfg.output_patch_len
    };
    let overlap = extract_len - cfg.input_patch_len;
    let num_forecast_patches = ((horizon as f32 - overlap as f32) / cfg.input_patch_len as f32)
        .ceil()
        .max(1.0) as usize;
    let num_horizon_patches = if cfg.use_stitching {
        num_forecast_patches + cfg.rolls() - 1
    } else {
        horizon.div_ceil(cfg.input_patch_len)
    };
    let padded_horizon = num_horizon_patches * cfg.input_patch_len;

    let num_past_only = past_only.map(|p| p.shape()[1]).unwrap_or(0);
    let num_pf = past_future.map(|p| p.shape()[1]).unwrap_or(0);
    let num_variates = num_target + num_past_only + num_pf;
    let total_len = context_padded + padded_horizon;
    let num_patches = total_len / cfg.input_patch_len;

    let mut all_vals = Array3::<f32>::zeros((b, num_variates, total_len));
    let mut all_masks = Array3::<bool>::from_elem((b, num_variates, total_len), false);

    // Target context
    for bi in 0..b {
        for vi in 0..num_target {
            for ti in 0..context {
                all_vals[[bi, vi, ti + ctx_padding]] = target[[bi, vi, ti]];
            }
        }
        if let Some(m) = mask {
            for ti in 0..context_padded {
                if ti < ctx_padding || m[ti - ctx_padding] {
                    for vi in 0..num_variates {
                        all_masks[[bi, vi, ti]] = true;
                    }
                }
            }
        }
    }

    // Horizon masked for targets
    for bi in 0..b {
        for vi in 0..num_target + num_past_only {
            for ti in context_padded..total_len {
                all_masks[[bi, vi, ti]] = true;
            }
        }
    }

    if let Some(po) = past_only {
        for bi in 0..b {
            for vi in 0..num_past_only {
                for ti in 0..context {
                    all_vals[[bi, num_target + vi, ti + ctx_padding]] = po[[bi, vi, ti]];
                }
            }
        }
    }

    if let Some(pf) = past_future {
        for bi in 0..b {
            for vi in 0..num_pf {
                let vidx = num_target + num_past_only + vi;
                for ti in 0..context {
                    all_vals[[bi, vidx, ti + ctx_padding]] = pf[[bi, vi, ti]];
                }
                for ti in 0..horizon.min(pf.shape()[2] - context) {
                    all_vals[[bi, vidx, context_padded + ti]] = pf[[bi, vi, context + ti]];
                }
            }
        }
    }

    let p = cfg.input_patch_len;
    let mut values = Array4::<f32>::zeros((b, num_variates, num_patches, p));
    let mut masks = Array4::<bool>::from_elem((b, num_variates, num_patches, p), false);
    for bi in 0..b {
        for vi in 0..num_variates {
            for ni in 0..num_patches {
                for pi in 0..p {
                    let ti = ni * p + pi;
                    values[[bi, vi, ni, pi]] = all_vals[[bi, vi, ti]];
                    masks[[bi, vi, ni, pi]] = all_masks[[bi, vi, ti]];
                }
            }
        }
    }

    let mut patch_is_target = Array3::<bool>::from_elem((b, num_variates, num_patches), false);
    for bi in 0..b {
        for vi in 0..num_target + num_past_only {
            for ni in 0..num_patches {
                patch_is_target[[bi, vi, ni]] = true;
            }
        }
    }

    let mut cpm_mask = Array2::<bool>::from_elem((b, num_patches), false);
    for bi in 0..b {
        for ni in num_context_patches..num_patches {
            cpm_mask[[bi, ni]] = true;
        }
    }

    (
        values,
        masks,
        patch_is_target,
        cpm_mask,
        num_context_patches,
        padded_horizon,
    )
}
