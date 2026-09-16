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

//! Host reference TimesFM-3 forward and decode.

use crate::config::TimesFM3Config;
use crate::host::math::{cumprod_bool_mask, linear2, relu_array, rms_norm};
use crate::host::preprocess::{
    RunningStats, build_decode_inputs, get_output_patch_via_roll, get_running_stats, revin,
    stitch_patches_quantile,
};
use crate::host::transformer::MixingStack;
use crate::weights::TimesFM3Weights;
use anyhow::{Result, ensure};
use ndarray::{Array2, Array3, Array4, Array5, ArrayView1, ArrayView2, ArrayView3, ArrayView4, s};

pub struct TimesFM3Model {
    pub cfg: TimesFM3Config,
    pub weights: TimesFM3Weights,
    stack: MixingStack,
}

/// Inputs to the compiled / host core (resblock → transformer → output head).
pub struct CorePrepared {
    pub res_in: Array2<f32>,
    pub effective_mask: Array3<bool>,
    pub stats: RunningStats,
    pub b: usize,
    pub v: usize,
    pub n: usize,
    pub patch_cpm_mask: Option<ndarray::Array1<bool>>,
}

impl TimesFM3Model {
    pub fn load(weights_path: &std::path::Path, cfg: TimesFM3Config) -> Result<Self> {
        let weights = TimesFM3Weights::load(weights_path, &cfg)?;
        Ok(Self::from_weights(cfg, weights))
    }

    pub fn from_weights(cfg: TimesFM3Config, weights: TimesFM3Weights) -> Self {
        let stack = MixingStack::new(cfg.clone(), weights.layers.clone());
        Self {
            cfg,
            weights,
            stack,
        }
    }

    pub fn synth(cfg: TimesFM3Config, seed: u64) -> Self {
        Self::from_weights(cfg.clone(), TimesFM3Weights::synth(&cfg, seed))
    }

    pub fn forward(
        &self,
        values: ArrayView3<f32>,
        masks: ArrayView3<bool>,
        patch_is_target: ArrayView2<bool>,
        patch_cpm_mask: Option<ArrayView1<bool>>,
    ) -> Array5<f32> {
        let v = values.shape()[1];
        let n = values.shape()[2] / self.cfg.input_patch_len;
        let p = self.cfg.input_patch_len;
        let values4 = patchify(values, v, n, p);
        let masks4 = patchify_bool(masks, v, n, p);
        let pit = expand_patch_is_target(patch_is_target, n);
        let prep = self
            .prepare_core(values4.view(), masks4.view(), pit.view(), patch_cpm_mask)
            .expect("prepare_core");
        let raw = self.core_forward_host(&prep);
        self.finalize_core(raw.view(), &prep)
            .expect("finalize_core")
    }

    /// Build resblock input and RevIN stats for the compiled core.
    pub fn prepare_core(
        &self,
        values: ArrayView4<f32>,
        masks: ArrayView4<bool>,
        patch_is_target: ArrayView3<bool>,
        patch_cpm_mask: Option<ArrayView1<bool>>,
    ) -> Result<CorePrepared> {
        let (b, v, n, _) = values.dim();
        let mut masks = masks.to_owned();
        if let Some(cpm) = patch_cpm_mask {
            for bi in 0..b {
                for vi in 0..v {
                    for ni in 0..n {
                        if cpm[ni] && patch_is_target[[bi, vi, ni]] {
                            for pi in 0..self.cfg.input_patch_len {
                                masks[[bi, vi, ni, pi]] = true;
                            }
                        }
                    }
                }
            }
        }

        let stats = get_running_stats(values.view(), masks.view());
        let norm_vals = revin(values.view(), stats.mu.view(), stats.sigma.view(), false);
        let mut norm_vals = norm_vals;
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    for pi in 0..self.cfg.input_patch_len {
                        if masks[[bi, vi, ni, pi]] {
                            norm_vals[[bi, vi, ni, pi]] = 0.0;
                        }
                    }
                }
            }
        }

        let (fcov, wrap) = get_output_patch_via_roll(norm_vals.view(), self.cfg.rolls());
        let fcov_norm = revin(fcov.view(), stats.mu.view(), stats.sigma.view(), false);
        let (masks_fcov_raw, _) = roll_bool_mask(masks.view(), self.cfg.rolls());
        let mut masks_fcov = masks_fcov_raw;
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    if patch_is_target[[bi, vi, ni]] || wrap[[0, 0, ni, 0]] {
                        for pi in 0..self.cfg.output_patch_len {
                            masks_fcov[[bi, vi, ni, pi]] = true;
                        }
                    }
                }
            }
        }
        let mut fcov_norm = fcov_norm;
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    for pi in 0..self.cfg.output_patch_len {
                        if masks_fcov[[bi, vi, ni, pi]] {
                            fcov_norm[[bi, vi, ni, pi]] = 0.0;
                        }
                    }
                }
            }
        }

        let mut cat_vals = Array4::<f32>::zeros((b, v, n, self.cfg.resblock_input_dim() / 2));
        let mut cat_masks = Array4::<f32>::zeros((b, v, n, self.cfg.resblock_input_dim() / 2));
        let half = self.cfg.input_patch_len + self.cfg.output_patch_len;
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    for pi in 0..self.cfg.input_patch_len {
                        cat_vals[[bi, vi, ni, pi]] = norm_vals[[bi, vi, ni, pi]];
                        cat_masks[[bi, vi, ni, pi]] =
                            if masks[[bi, vi, ni, pi]] { 1.0 } else { 0.0 };
                    }
                    for pi in 0..self.cfg.output_patch_len {
                        cat_vals[[bi, vi, ni, self.cfg.input_patch_len + pi]] =
                            fcov_norm[[bi, vi, ni, pi]];
                        cat_masks[[bi, vi, ni, self.cfg.input_patch_len + pi]] =
                            if masks_fcov[[bi, vi, ni, pi]] {
                                1.0
                            } else {
                                0.0
                            };
                    }
                }
            }
        }

        let mut res_in = Array4::<f32>::zeros((b, v, n, self.cfg.resblock_input_dim()));
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    for di in 0..half {
                        res_in[[bi, vi, ni, di]] = cat_vals[[bi, vi, ni, di]];
                        res_in[[bi, vi, ni, half + di]] = cat_masks[[bi, vi, ni, di]];
                    }
                }
            }
        }

        let mut patch_mask = Array3::<bool>::from_elem((b, v, n), false);
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    patch_mask[[bi, vi, ni]] =
                        cat_masks.slice(s![bi, vi, ni, ..]).iter().all(|&m| m > 0.5);
                }
            }
        }
        let effective = cumprod_bool_mask(&patch_mask);
        Ok(CorePrepared {
            res_in: flatten_bvnp(res_in.view(), self.cfg.resblock_input_dim()),
            effective_mask: effective,
            stats,
            b,
            v,
            n,
            patch_cpm_mask: patch_cpm_mask.map(|m| m.to_owned()),
        })
    }

    /// Host reference core: resblock → mixing stack → output projection.
    pub fn core_forward_host(&self, prep: &CorePrepared) -> Array2<f32> {
        let d = self.cfg.model_dims();
        let res4 = unflatten_bvnp(
            prep.res_in.view(),
            prep.b,
            prep.v,
            prep.n,
            self.cfg.resblock_input_dim(),
        );
        let transformer_in = self.resblock(res4.view());
        let transformer_out = self
            .stack
            .forward(transformer_in.view(), prep.effective_mask.view());
        let flat = flatten_bvnp(transformer_out.view(), d);
        let mut raw = linear2(flat.view(), self.weights.output_head.w.view(), None);
        if let Some(ref b_) = self.weights.output_head.b {
            for (mut row, &bias) in raw.rows_mut().into_iter().zip(b_.iter()) {
                row += bias;
            }
        }
        raw
    }

    /// RevIN denorm + clip after the core logits.
    pub fn finalize_core(&self, raw: ArrayView2<f32>, prep: &CorePrepared) -> Result<Array5<f32>> {
        let (b, v, n) = (prep.b, prep.v, prep.n);
        let mut mu = prep.stats.mu.clone();
        let mut sigma = prep.stats.sigma.clone();
        if self.cfg.use_iterative_cpm_revin
            && let Some(ref cpm) = prep.patch_cpm_mask
        {
            let (rm, rs) = cpm_iterative_revin_refine(
                &self.cfg,
                raw,
                prep.stats.n.view(),
                mu.view(),
                sigma.view(),
                cpm.view(),
            );
            mu = rm;
            sigma = rs;
        }

        let raw5 = unflatten_bvn_logits(
            raw,
            b,
            v,
            n,
            self.cfg.output_patch_len,
            self.cfg.num_quantiles(),
        );
        let mut denorm = Array5::<f32>::zeros(raw5.dim());
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    let m = mu[[bi, vi, ni]];
                    let s = crate::host::math::safe_div_sigma(sigma[[bi, vi, ni]]);
                    for oi in 0..self.cfg.output_patch_len {
                        for qi in 0..self.cfg.num_quantiles() {
                            denorm[[bi, vi, ni, oi, qi]] = raw5[[bi, vi, ni, oi, qi]] * s + m;
                        }
                    }
                }
            }
        }
        denorm.mapv_inplace(|x| x.clamp(-self.cfg.value_clip, self.cfg.value_clip));
        Ok(denorm)
    }

    pub fn decode(
        &self,
        target: ArrayView3<f32>,
        horizon: usize,
        past_only: Option<ArrayView3<f32>>,
        past_future: Option<ArrayView3<f32>>,
        mask: Option<ArrayView1<bool>>,
    ) -> Array4<f32> {
        let b = target.shape()[0];
        let (values, masks, pit, cpm, ctx_patches, _padded_hor) =
            build_decode_inputs(&self.cfg, target, horizon, mask, past_only, past_future);
        let prep = self
            .prepare_core(values.view(), masks.view(), pit.view(), Some(cpm.row(0)))
            .expect("prepare_core");
        let raw = self.core_forward_host(&prep);
        let logits = self
            .finalize_core(raw.view(), &prep)
            .expect("finalize_core");

        let extract_len = if self.cfg.use_stitching {
            (2 * self.cfg.input_patch_len).min(self.cfg.output_patch_len)
        } else {
            self.cfg.output_patch_len
        };
        let overlap = extract_len - self.cfg.input_patch_len;
        let num_forecast_patches = ((horizon as f32 - overlap as f32)
            / self.cfg.input_patch_len as f32)
            .ceil()
            .max(1.0) as usize;

        if self.cfg.use_stitching {
            let mut patch_preds = Array5::<f32>::zeros((
                b,
                logits.shape()[1],
                num_forecast_patches,
                extract_len,
                self.cfg.num_quantiles(),
            ));
            for pi in 0..num_forecast_patches {
                let idx = ctx_patches - 1 + pi;
                for t in 0..extract_len {
                    for q in 0..self.cfg.num_quantiles() {
                        patch_preds
                            .slice_mut(s![.., .., pi, t, q])
                            .assign(&logits.slice(s![.., .., idx, t, q]));
                    }
                }
            }
            let stitched = stitch_patches_quantile(patch_preds.view(), self.cfg.input_patch_len);
            stitched.slice(s![.., .., ..horizon, ..]).to_owned()
        } else {
            let idx = ctx_patches - 1;
            logits
                .slice(s![.., .., idx, ..horizon, ..])
                .to_owned()
                .into_shape_with_order((b, logits.shape()[1], horizon, self.cfg.num_quantiles()))
                .unwrap()
        }
    }

    fn resblock(&self, x: ArrayView4<f32>) -> Array4<f32> {
        let w = &self.weights.resblock;
        let (b, v, n, d_in) = x.dim();
        let d_out = self.cfg.model_dims();
        let flat = flatten_bvnp(x.view(), d_in);
        let mut h = flat.clone();
        if let Some(ref ln) = w.pre_norm {
            h = rms_norm(h.view(), ln.view());
        }
        h = linear2(h.view(), w.hidden.view(), None);
        relu_array(&mut h);
        let out = linear2(h.view(), w.output.view(), None);
        let res = linear2(flat.view(), w.residual.view(), None);
        let y = &out + &res;
        unflatten_bvnp(y.view(), b, v, n, d_out)
    }
}

fn flatten_bvnp(x: ArrayView4<f32>, d: usize) -> Array2<f32> {
    let (b, v, n, _) = x.dim();
    let mut out = Array2::zeros((b * v * n, d));
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for di in 0..d {
                    out[[bi * v * n + vi * n + ni, di]] = x[[bi, vi, ni, di]];
                }
            }
        }
    }
    out
}

fn unflatten_bvnp(x: ArrayView2<f32>, b: usize, v: usize, n: usize, d: usize) -> Array4<f32> {
    let mut out = Array4::zeros((b, v, n, d));
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for di in 0..d {
                    out[[bi, vi, ni, di]] = x[[bi * v * n + vi * n + ni, di]];
                }
            }
        }
    }
    out
}

fn roll_bool_mask(x: ArrayView4<bool>, rolls: usize) -> (Array4<bool>, Array4<bool>) {
    let (b, v, n, p) = x.dim();
    let out_p = p * rolls;
    let mut result = Array4::<bool>::from_elem((b, v, n, out_p), false);
    let wrap = Array4::<bool>::from_elem((1, 1, n, out_p), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for oi in 0..out_p {
                    let src_patch = ni + oi / p + 1;
                    let within = oi % p;
                    result[[bi, vi, ni, oi]] = src_patch >= n || x[[bi, vi, src_patch, within]];
                }
            }
        }
    }
    (result, wrap)
}

fn patchify(x: ArrayView3<f32>, v: usize, n: usize, p: usize) -> Array4<f32> {
    let b = x.shape()[0];
    let mut out = Array4::zeros((b, v, n, p));
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for pi in 0..p {
                    out[[bi, vi, ni, pi]] = x[[bi, vi, ni * p + pi]];
                }
            }
        }
    }
    out
}

fn patchify_bool(x: ArrayView3<bool>, v: usize, n: usize, p: usize) -> Array4<bool> {
    let b = x.shape()[0];
    let mut out = Array4::from_elem((b, v, n, p), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for pi in 0..p {
                    out[[bi, vi, ni, pi]] = x[[bi, vi, ni * p + pi]];
                }
            }
        }
    }
    out
}

fn expand_patch_is_target(pit: ArrayView2<bool>, n: usize) -> Array3<bool> {
    let (b, v) = pit.dim();
    let mut out = Array3::from_elem((b, v, n), false);
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                out[[bi, vi, ni]] = pit[[bi, vi]];
            }
        }
    }
    out
}

fn unflatten_bvn_logits(
    x: ArrayView2<f32>,
    b: usize,
    v: usize,
    n: usize,
    o: usize,
    q: usize,
) -> Array5<f32> {
    let mut out = Array5::zeros((b, v, n, o, q));
    for bi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                for oi in 0..o {
                    for qi in 0..q {
                        out[[bi, vi, ni, oi, qi]] = x[[bi * v * n + vi * n + ni, oi * q + qi]];
                    }
                }
            }
        }
    }
    out
}

fn cpm_iterative_revin_refine(
    cfg: &TimesFM3Config,
    raw_logits: ArrayView2<f32>,
    revin_n: ArrayView3<f32>,
    revin_mu: ArrayView3<f32>,
    revin_sigma: ArrayView3<f32>,
    patch_cpm_mask: ArrayView1<bool>,
) -> (Array3<f32>, Array3<f32>) {
    let (b, v, n) = revin_mu.dim();
    let median = cfg.median_quantile_index();
    let mut refined_mu = revin_mu.to_owned();
    let mut refined_sigma = revin_sigma.to_owned();

    let mut carry_n = Array2::<f32>::zeros((b, v));
    let mut carry_mu = Array2::<f32>::zeros((b, v));
    let mut carry_sigma = Array2::<f32>::zeros((b, v));
    let mut block_offset = Array2::<i32>::zeros((b, 1));

    for ni in 0..n {
        let is_cpm = patch_cpm_mask[ni];
        for bi in 0..b {
            for vi in 0..v {
                let actual_n = revin_n[[bi, vi, ni]];
                let actual_mu = revin_mu[[bi, vi, ni]];
                let actual_sigma = revin_sigma[[bi, vi, ni]];
                if is_cpm {
                    refined_mu[[bi, vi, ni]] = carry_mu[[bi, vi]];
                    refined_sigma[[bi, vi, ni]] = carry_sigma[[bi, vi]];
                    carry_n[[bi, vi]] = actual_n;
                    carry_mu[[bi, vi]] = refined_mu[[bi, vi, ni]];
                    carry_sigma[[bi, vi]] = refined_sigma[[bi, vi, ni]];
                } else {
                    carry_n[[bi, vi]] = actual_n;
                    carry_mu[[bi, vi]] = actual_mu;
                    carry_sigma[[bi, vi]] = actual_sigma;
                }
                let _ = raw_logits[[bi * v * n + vi * n + ni, median]];
                block_offset[[bi, 0]] = if is_cpm {
                    (block_offset[[bi, 0]] + 1) % cfg.rolls() as i32
                } else {
                    0
                };
            }
        }
    }
    (refined_mu, refined_sigma)
}

pub fn validate_context(context_len: usize) -> Result<()> {
    ensure!(
        context_len <= crate::config::MAX_CONTEXT_LENGTH,
        "context length {context_len} exceeds max {}",
        crate::config::MAX_CONTEXT_LENGTH
    );
    Ok(())
}
