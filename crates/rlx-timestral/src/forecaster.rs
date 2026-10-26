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

//! Probabilistic forecasting with Timestral (upstream `timestral.forecaster`).
//!
//! [`TimestralForecaster::forecast`] takes a context `(B, V, T)` — series, variates,
//! time; NaN marks a missing value — and returns quantiles `(B, V, H, Q)`. Horizons
//! longer than one decode block are forecast block by block, feeding each block's
//! median back as context.
//!
//! Two test-time options are on by default, as upstream:
//!
//! - **Symmetric averaging** forecasts the context and its negation, reflects the
//!   second forecast (`-Q(1 - p)`) and averages the two. Every covariate row is
//!   negated with the context.
//! - **Non-negative clipping** clamps the forecast at 0 for every series whose
//!   observed context is entirely non-negative.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use ndarray::{Array3, Array4, ArrayView3, Axis, s};

use crate::config::TimestralConfig;
use crate::covariates::target_encode;
use crate::model::{Role, Rows, TimestralModel, decimal};
use crate::scaler::{advancing_block_frames, causal_patch_loc_scale, clamp_nonfinite};

/// Quantile forecast `(B, V, H, Q)`, sorted along the quantile axis.
#[derive(Debug, Clone)]
pub struct Forecast {
    pub quantiles: Array4<f32>,
    pub quantile_levels: Vec<f32>,
}

impl Forecast {
    /// `(B, V, H)` median forecast.
    pub fn median(&self) -> Array3<f32> {
        self.quantile(0.5).expect("median level")
    }

    /// `(B, V, H)` forecast at one of the model's quantile levels.
    pub fn quantile(&self, level: f32) -> Result<Array3<f32>> {
        let Some(i) = self.quantile_levels.iter().position(|&q| q == level) else {
            bail!("level {level} is not one of {:?}", self.quantile_levels);
        };
        Ok(self.quantiles.index_axis(Axis(3), i).to_owned())
    }
}

/// Covariates and test-time options of one [`TimestralForecaster::forecast`] call.
#[derive(Debug, Clone)]
pub struct ForecastOptions {
    /// Optional mask shaped like the context; `false` marks extra missing values.
    pub observation_mask: Option<Array3<bool>>,
    /// `(B, C, T)` covariates known over the context only; NaN = missing.
    pub past_covariates: Option<Array3<f32>>,
    /// `(B, C, T + H)` covariates known over the context and the horizon.
    pub known_covariates: Option<Array3<f32>>,
    /// `(B, C, T)` category codes (NaN = missing) over the context. Codes are f32, so
    /// integer codes must stay below 2^24 to remain distinct.
    pub categorical_past_covariates: Option<Array3<f32>>,
    /// `(B, C, T + H)` category codes over the context and the horizon.
    pub categorical_known_covariates: Option<Array3<f32>>,
    pub symmetric_averaging: bool,
    pub clip_nonnegative: bool,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            observation_mask: None,
            past_covariates: None,
            known_covariates: None,
            categorical_past_covariates: None,
            categorical_known_covariates: None,
            symmetric_averaging: true,
            clip_nonnegative: true,
        }
    }
}

impl ForecastOptions {
    /// No symmetric averaging and no clipping: the network's own forecast.
    pub fn raw() -> Self {
        Self {
            symmetric_averaging: false,
            clip_nonnegative: false,
            ..Self::default()
        }
    }
}

/// Block-autoregressive quantile forecaster around a [`TimestralModel`].
pub struct TimestralForecaster {
    model: TimestralModel,
    /// Context cap in time steps (most recent kept).
    pub max_context: usize,
    /// Patches decoded per block, capped at the trained block size.
    pub per_shot_horizon_patches: usize,
    /// Clamp forecasts to `[min - k*s, max + k*s]` of the observed context; 0 disables.
    pub quantile_real_cap_k: f32,
    /// Fed-back medians at or above this magnitude become 0.
    pub extreme_value_threshold: f32,
}

impl TimestralForecaster {
    pub fn new(model: TimestralModel) -> Self {
        let cfg = model.config();
        let max_context = cfg.context_length;
        let per_shot = cfg.block_size;
        Self {
            model,
            max_context,
            per_shot_horizon_patches: per_shot,
            quantile_real_cap_k: 1e4,
            extreme_value_threshold: 1e30,
        }
    }

    pub fn config(&self) -> &TimestralConfig {
        self.model.config()
    }

    pub fn model(&self) -> &TimestralModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut TimestralModel {
        &mut self.model
    }

    pub fn quantile_levels(&self) -> &[f32] {
        &self.model.config().quantile_knots
    }

    /// Forecast one series `(T,)` with the default options; returns `(H, Q)` quantiles.
    pub fn forecast_series(
        &mut self,
        context: &[f32],
        prediction_length: usize,
    ) -> Result<Forecast> {
        let ctx = Array3::from_shape_vec((1, 1, context.len()), context.to_vec())?;
        self.forecast(ctx.view(), prediction_length, &ForecastOptions::default())
    }

    /// Forecast `prediction_length` steps after `context` `(B, V, T)`.
    ///
    /// The variates of a series are forecast jointly.
    pub fn forecast(
        &mut self,
        context: ArrayView3<f32>,
        prediction_length: usize,
        opts: &ForecastOptions,
    ) -> Result<Forecast> {
        ensure!(prediction_length >= 1, "prediction_length must be >= 1");
        let (b, _, t) = context.dim();
        ensure!(t >= 1, "context must have at least one step");
        // Host math runs in f64 (see `crate::scaler`); only the result returns to f32.
        let context = context.mapv(f64::from);
        let mut mask = context.mapv(f64::is_finite);
        if let Some(user) = &opts.observation_mask {
            ensure!(
                user.dim() == context.dim(),
                "observation_mask must match the context"
            );
            mask.zip_mut_with(user, |m, &u| *m &= u);
        }
        let inputs = ndarray::Zip::from(&context)
            .and(&mask)
            .map_collect(|&x, &m| if m { x } else { 0.0 });

        let check = |name: &str, len: usize, v: Option<(usize, usize, usize)>| -> Result<()> {
            if let Some(dim) = v {
                ensure!(
                    self.config().covariate_enabled,
                    "this checkpoint was not trained with covariates; got {name}"
                );
                ensure!(
                    dim.0 == b && dim.2 == len,
                    "{name} must be (B={b}, C, {len}), got {dim:?}"
                );
            }
            Ok(())
        };
        let h_len = t + prediction_length;
        check(
            "past_covariates",
            t,
            opts.past_covariates.as_ref().map(|a| a.dim()),
        )?;
        check(
            "known_covariates",
            h_len,
            opts.known_covariates.as_ref().map(|a| a.dim()),
        )?;
        check(
            "categorical_past_covariates",
            t,
            opts.categorical_past_covariates.as_ref().map(|a| a.dim()),
        )?;
        check(
            "categorical_known_covariates",
            h_len,
            opts.categorical_known_covariates.as_ref().map(|a| a.dim()),
        )?;

        let wide = |a: &Array3<f32>| a.mapv(f64::from);
        let mut past = opts.past_covariates.as_ref().map(wide);
        let mut known = opts.known_covariates.as_ref().map(wide);
        let cat_past = opts.categorical_past_covariates.as_ref();
        let cat_known = opts.categorical_known_covariates.as_ref();
        if cat_past.is_some() || cat_known.is_some() {
            let target = self.encoding_target(&inputs, &mask);
            past = cat_rows(
                past,
                cat_past.map(|c| encode_categorical(c, target.as_ref())),
            );
            known = cat_rows(
                known,
                cat_known.map(|c| encode_categorical(c, target.as_ref())),
            );
        }
        if known.is_some() && !self.config().supports_known_covariates() {
            bail!("this checkpoint cannot condition on known-future covariates");
        }

        let mut quantiles = self.forecast_raw(
            &inputs,
            &mask,
            prediction_length,
            past.as_ref(),
            known.as_ref(),
        )?;
        if opts.symmetric_averaging {
            let levels = self.quantile_levels();
            ensure!(
                levels
                    .iter()
                    .zip(levels.iter().rev())
                    .all(|(lo, hi)| (lo + hi - 1.0).abs() < 1e-6),
                "symmetric averaging needs quantile levels symmetric about 0.5"
            );
            let neg = |a: &Array3<f64>| a.mapv(|v| -v);
            let mirrored = self.forecast_raw(
                &neg(&inputs),
                &mask,
                prediction_length,
                past.as_ref().map(neg).as_ref(),
                known.as_ref().map(neg).as_ref(),
            )?;
            let mut reflected = mirrored.mapv(|v| -v);
            reflected.invert_axis(Axis(3));
            quantiles.zip_mut_with(&reflected, |q, &m| *q = 0.5 * (*q + m));
        }
        if opts.clip_nonnegative {
            for bi in 0..b {
                for vi in 0..context.dim().1 {
                    let obs = mask.slice(s![bi, vi, ..]);
                    let vals = inputs.slice(s![bi, vi, ..]);
                    let any = obs.iter().any(|&o| o);
                    let nonneg = vals.iter().zip(obs).all(|(&x, &o)| !o || x >= 0.0);
                    if any && nonneg {
                        quantiles
                            .slice_mut(s![bi, vi, .., ..])
                            .mapv_inplace(|q| q.max(0.0));
                    }
                }
            }
        }
        Ok(Forecast {
            quantiles: quantiles.mapv(|v| v as f32),
            quantile_levels: self.quantile_levels().to_vec(),
        })
    }

    /// `(B, T)` target to encode categorical covariates against, or `None` when a series
    /// has several target variates. The forward-filled target from its first observation
    /// on, over the last `max_context` steps; NaN elsewhere.
    fn encoding_target(&self, inputs: &Array3<f64>, mask: &Array3<bool>) -> Option<Vec<Vec<f64>>> {
        let (b, v, t) = inputs.dim();
        if v != 1 {
            return None;
        }
        let window_start = t.saturating_sub(self.max_context);
        Some(
            (0..b)
                .map(|bi| {
                    let row: Vec<f64> = inputs.slice(s![bi, 0, ..]).to_vec();
                    let obs: Vec<bool> = mask.slice(s![bi, 0, ..]).to_vec();
                    let filled = ffill(&row, &obs);
                    let mut started = false;
                    (0..t)
                        .map(|ti| {
                            started |= obs[ti];
                            if started && ti >= window_start {
                                filled[ti]
                            } else {
                                f64::NAN
                            }
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// One forecast pass: `(B, V, T)` inputs to `(B, V, H, Q)` raw-space quantiles.
    fn forecast_raw(
        &mut self,
        inputs: &Array3<f64>,
        mask: &Array3<bool>,
        prediction_length: usize,
        past: Option<&Array3<f64>>,
        known: Option<&Array3<f64>>,
    ) -> Result<Array4<f64>> {
        let ps = self.config().patch_size;
        let (b, n_tgt, t) = inputs.dim();
        let horizon = prediction_length.div_ceil(ps) * ps;

        // Stack target and covariate rows per series: (B, V_all, T).
        let mut rows_v: Vec<Vec<f64>> = Vec::new();
        let mut rows_m: Vec<Vec<bool>> = Vec::new();
        let mut roles = Vec::new();
        let mut kf_rows: Vec<Option<(Vec<f64>, Vec<bool>)>> = Vec::new();
        let n_past = past.map_or(0, |p| p.dim().1);
        let n_known = known.map_or(0, |k| k.dim().1);
        for bi in 0..b {
            for vi in 0..n_tgt {
                let v: Vec<f64> = inputs.slice(s![bi, vi, ..]).to_vec();
                let m: Vec<bool> = mask.slice(s![bi, vi, ..]).to_vec();
                rows_v.push(ffill(&v, &m));
                rows_m.push(m);
                roles.push(Role::Target);
                kf_rows.push(None);
            }
            for ci in 0..n_past {
                let p = past.unwrap().slice(s![bi, ci, ..]).to_vec();
                rows_m.push(p.iter().map(|x| !x.is_nan()).collect());
                rows_v.push(
                    p.iter()
                        .map(|&x| if x.is_nan() { 0.0 } else { x })
                        .collect(),
                );
                roles.push(Role::PastCovariate);
                kf_rows.push(None);
            }
            for ci in 0..n_known {
                let k = known.unwrap().slice(s![bi, ci, ..]).to_vec();
                let (ctx, fut) = k.split_at(t);
                rows_m.push(ctx.iter().map(|x| !x.is_nan()).collect());
                rows_v.push(
                    ctx.iter()
                        .map(|&x| if x.is_nan() { 0.0 } else { x })
                        .collect(),
                );
                roles.push(Role::KnownCovariate);
                let mut fv = vec![0f64; horizon];
                let mut fm = vec![false; horizon];
                for (j, &x) in fut.iter().enumerate() {
                    fm[j] = !x.is_nan();
                    fv[j] = if x.is_nan() { 0.0 } else { x };
                }
                kf_rows.push(Some((fv, fm)));
            }
        }
        let v_all = n_tgt + n_past + n_known;
        let r = b * v_all;
        let rows = Rows::new(r, t, rows_v.concat(), rows_m.concat()).keep_last(self.max_context);
        let pad = (ps - rows.t % ps) % ps;
        let rows = rows.pad_left(pad);

        let has_cov = n_past + n_known > 0;
        let known_future = (n_known > 0).then(|| {
            let mut v = Vec::with_capacity(r * horizon);
            let mut m = Vec::with_capacity(r * horizon);
            for kf in &kf_rows {
                match kf {
                    Some((fv, fm)) => {
                        v.extend_from_slice(fv);
                        m.extend_from_slice(fm);
                    }
                    None => {
                        v.extend(std::iter::repeat_n(0.0, horizon));
                        m.extend(std::iter::repeat_n(false, horizon));
                    }
                }
            }
            Rows::new(r, horizon, v, m)
        });
        let roles = has_cov.then_some(roles);

        let q = self.block_ar_decode(
            &rows,
            b,
            v_all,
            prediction_length,
            roles.as_deref(),
            known_future.as_ref(),
        )?;
        let nq = self.config().num_quantiles();
        let mut out = Array4::<f64>::zeros((b, n_tgt, prediction_length, nq));
        for bi in 0..b {
            for vi in 0..n_tgt {
                let ri = bi * v_all + vi;
                let src = &q[ri * prediction_length * nq..(ri + 1) * prediction_length * nq];
                let mut dst = out.slice_mut(s![bi, vi, .., ..]);
                for (d, &v) in dst.iter_mut().zip(src) {
                    *d = v;
                }
                if self.quantile_real_cap_k > 0.0 {
                    let (vals, obs) = rows.row(ri);
                    self.cap_quantiles(&mut dst, vals, obs);
                }
            }
        }
        let nan = out.iter().filter(|v| v.is_nan()).count();
        ensure!(
            nan == 0,
            "forecast produced {nan} NaN values; the input is degenerate"
        );
        Ok(out)
    }

    fn cap_quantiles(&self, q: &mut ndarray::ArrayViewMut2<f64>, vals: &[f64], obs: &[bool]) {
        let c = self.config();
        let observed = || vals.iter().zip(obs).filter(|(_, o)| **o).map(|(v, _)| *v);
        let ctx_max = observed().fold(f64::NEG_INFINITY, f64::max);
        let ctx_min = observed().fold(f64::INFINITY, f64::min);
        let ctx_max = if ctx_max.is_finite() { ctx_max } else { 0.0 };
        let ctx_min = if ctx_min.is_finite() { ctx_min } else { 0.0 };
        let (_, scales) = causal_patch_loc_scale(vals, obs, c.patch_size, decimal(c.minimum_scale));
        let s = scales[scales.len() - 1] * decimal(self.quantile_real_cap_k);
        let (lo, hi) = (ctx_min - s, ctx_max + s);
        // `torch.clamp` keeps NaN, so the degenerate-input check below still sees it.
        q.mapv_inplace(|v| if v.is_nan() { v } else { v.max(lo).min(hi) });
    }

    /// Decode block by block in the packed layout; returns `(R, H, Q)` row-major.
    ///
    /// After each block the context grows by the block's feedback: targets carry their
    /// median, known-future covariates their real future, past-only covariates a blank.
    fn block_ar_decode(
        &mut self,
        inputs: &Rows,
        b: usize,
        v_all: usize,
        prediction_length: usize,
        roles: Option<&[Role]>,
        known_future: Option<&Rows>,
    ) -> Result<Vec<f64>> {
        let c = self.config().clone();
        let ps = c.patch_size;
        let nq = c.num_quantiles();
        let m = c.median_index();
        let r = inputs.r;
        let group_ids: Vec<usize> = (0..b).flat_map(|g| std::iter::repeat_n(g, v_all)).collect();
        let per_shot = self.per_shot_horizon_patches.min(c.block_size).max(1);

        let mut context = inputs.clone();
        let mut blocks: Vec<(usize, Vec<f64>)> = Vec::new();
        let extreme = decimal(self.extreme_value_threshold);
        let mut remaining = prediction_length.div_ceil(ps);
        let mut offset = 0;
        while remaining > 0 {
            let block = per_shot.min(remaining);
            let blen = block * ps;
            let kf_block = known_future.map(|kf| slice_time(kf, offset, blen));
            let mut q =
                self.model
                    .predict_block(&context, &group_ids, block, roles, kf_block.as_ref())?;
            self.refine_block(&mut q, &context, block, roles)?;
            remaining -= block;
            offset += blen;
            if remaining > 0 {
                let mut fv = Vec::with_capacity(r * blen);
                let mut fm = Vec::with_capacity(r * blen);
                for i in 0..r {
                    let role = roles.map_or(Role::Target, |rs| rs[i]);
                    for step in 0..blen {
                        let (v, o) = match role {
                            Role::Target => {
                                let med = q[(i * blen + step) * nq + m];
                                let ok = med.is_finite() && med.abs() < extreme;
                                (if ok { med } else { 0.0 }, true)
                            }
                            Role::PastCovariate => (0.0, false),
                            Role::KnownCovariate => {
                                let kf = kf_block.as_ref().unwrap();
                                (kf.values[i * blen + step], kf.observed[i * blen + step])
                            }
                        };
                        fv.push(v);
                        fm.push(o);
                    }
                }
                context = context.concat_time(&Rows::new(r, blen, fv, fm));
                if context.t > self.max_context {
                    let keep = ps.max((self.max_context / ps) * ps);
                    context = context.keep_last(keep);
                }
            }
            blocks.push((blen, q));
        }

        let mut out = Vec::with_capacity(r * prediction_length * nq);
        for i in 0..r {
            let mut row = Vec::new();
            for (blen, q) in &blocks {
                row.extend_from_slice(&q[i * blen * nq..(i + 1) * blen * nq]);
            }
            out.extend_from_slice(&row[..prediction_length * nq]);
        }
        Ok(out)
    }

    /// Move a decoded block from the anchor frame to the advancing refine frame.
    ///
    /// The model denormalizes the whole block with the frame of the last context
    /// patch. The first horizon patch keeps that frame; each later one is
    /// re-denormalized with [`advancing_block_frames`]. Covariate rows are unchanged.
    fn refine_block(
        &self,
        q: &mut [f64],
        context: &Rows,
        n_pred: usize,
        roles: Option<&[Role]>,
    ) -> Result<()> {
        if n_pred <= 1 {
            return Ok(());
        }
        let c = self.config();
        let ps = c.patch_size;
        let nq = c.num_quantiles();
        let m = c.median_index();
        let (_, locs, scales) = self.model.scale(context);
        let blen = n_pred * ps;
        for i in 0..context.r {
            if roles.is_some_and(|rs| rs[i] != Role::Target) {
                continue;
            }
            let loc0 = *locs[i].last().unwrap();
            let scale0 = *scales[i].last().unwrap();
            let rq = &mut q[i * blen * nq..(i + 1) * blen * nq];
            let sinh_median: Vec<Vec<f64>> = (0..n_pred)
                .map(|p| {
                    (0..ps)
                        .map(|k| {
                            let med = rq[(p * ps + k) * nq + m];
                            (med - loc0) / scale0
                        })
                        .collect()
                })
                .collect();
            let (v, o) = context.row(i);
            let (loc_j, scale_j) =
                advancing_block_frames(v, o, &sinh_median, ps, decimal(c.minimum_scale));
            for p in 0..n_pred {
                let (lj, sj) = if p == 0 {
                    (loc0, scale0)
                } else {
                    (loc_j[p], scale_j[p])
                };
                for k in 0..ps {
                    let qs = &mut rq[(p * ps + k) * nq..(p * ps + k + 1) * nq];
                    for v in qs.iter_mut() {
                        *v = (*v - loc0) / scale0 * sj + lj;
                    }
                    clamp_nonfinite(qs);
                }
            }
        }
        Ok(())
    }
}

/// Forward-fill unobserved steps; leading unobserved steps take `values[0]` (0 here,
/// since callers zero unobserved values first).
fn ffill(values: &[f64], observed: &[bool]) -> Vec<f64> {
    if observed.iter().all(|&o| o) {
        return values.to_vec();
    }
    let mut last = 0usize;
    values
        .iter()
        .zip(observed)
        .enumerate()
        .map(|(i, (_, &o))| {
            if o {
                last = i;
            }
            let v = values[last];
            if v.is_nan() { 0.0 } else { v }
        })
        .collect()
}

fn slice_time(rows: &Rows, offset: usize, len: usize) -> Rows {
    let mut v = Vec::with_capacity(rows.r * len);
    let mut m = Vec::with_capacity(rows.r * len);
    for i in 0..rows.r {
        let (rv, ro) = rows.row(i);
        v.extend_from_slice(&rv[offset..offset + len]);
        m.extend_from_slice(&ro[offset..offset + len]);
    }
    Rows::new(rows.r, len, v, m)
}

/// Target-encode `(B, C, L)` codes against per-series targets; raw codes without a target.
fn encode_categorical(codes: &Array3<f32>, target: Option<&Vec<Vec<f64>>>) -> Array3<f64> {
    let Some(target) = target else {
        return codes.mapv(f64::from);
    };
    let (b, nc, len) = codes.dim();
    let mut out = Array3::<f64>::zeros((b, nc, len));
    for bi in 0..b {
        let ctx = &target[bi];
        let t = ctx.len();
        for ci in 0..nc {
            let row: Vec<f64> = codes
                .slice(s![bi, ci, ..])
                .iter()
                .map(|&c| c as f64)
                .collect();
            let future = (len > t).then(|| &row[t..]);
            let (past_enc, future_enc) = target_encode(ctx, &row[..t], future);
            for ti in 0..t {
                // A missing code before the target starts is padding, not the missing category.
                out[[bi, ci, ti]] = if ctx[ti].is_nan() && row[ti].is_nan() {
                    f64::NAN
                } else {
                    past_enc[ti]
                };
            }
            if let Some(fe) = future_enc {
                for (j, v) in fe.into_iter().enumerate() {
                    out[[bi, ci, t + j]] = v;
                }
            }
        }
    }
    out
}

fn cat_rows(a: Option<Array3<f64>>, b: Option<Array3<f64>>) -> Option<Array3<f64>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(ndarray::concatenate(Axis(1), &[a.view(), b.view()]).unwrap()),
        (a, None) => a,
        (None, b) => b,
    }
}

/// A local checkpoint directory, or a Hub repo id fetched into the HF cache.
pub fn resolve_checkpoint(path_or_repo: &str) -> Result<std::path::PathBuf> {
    let path = Path::new(path_or_repo);
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    ensure!(
        !path.exists(),
        "{path_or_repo} is a file; pass the directory holding config.json and model.safetensors"
    );
    #[cfg(feature = "hf-download")]
    {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(path_or_repo.to_string());
        let weights = repo.get("model.safetensors")?;
        repo.get("config.json")?;
        Ok(weights.parent().unwrap().to_path_buf())
    }
    #[cfg(not(feature = "hf-download"))]
    bail!(
        "{path_or_repo} is not a local directory; download it first \
         (`huggingface-cli download {path_or_repo} --local-dir DIR`) or build with `--features hf-download`"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffill_carries_last_observation() {
        let v = [0.0, 2.0, 0.0, 0.0, 5.0];
        let o = [false, true, false, false, true];
        assert_eq!(ffill(&v, &o), vec![0.0, 2.0, 2.0, 2.0, 5.0]);
    }
}
