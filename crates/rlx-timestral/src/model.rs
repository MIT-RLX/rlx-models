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

//! The Timestral network and its single-block forecast primitive (upstream
//! `timestral.model`): host scaling and masks around the compiled graph.

use anyhow::{Context, Result, ensure};

use crate::config::TimestralConfig;
use crate::masks;
use crate::net::Net;
use crate::scaler::{
    affine_asinh, causal_patch_loc_scale, clamp_nonfinite, instance_norm_stats, invert_affine_asinh,
};

/// Role of a packed row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Target,
    PastCovariate,
    KnownCovariate,
}

/// Packed `(R, T)` rows with their observation mask, row-major. Values are f64, like
/// all host math around the network (see [`crate::scaler`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    pub r: usize,
    pub t: usize,
    pub values: Vec<f64>,
    pub observed: Vec<bool>,
}

impl Rows {
    pub fn new(r: usize, t: usize, values: Vec<f64>, observed: Vec<bool>) -> Self {
        assert_eq!(values.len(), r * t);
        assert_eq!(observed.len(), r * t);
        Self {
            r,
            t,
            values,
            observed,
        }
    }

    pub fn row(&self, i: usize) -> (&[f64], &[bool]) {
        let span = i * self.t..(i + 1) * self.t;
        (&self.values[span.clone()], &self.observed[span])
    }

    /// Prepend `n` unobserved zero steps to every row.
    pub fn pad_left(&self, n: usize) -> Rows {
        if n == 0 {
            return self.clone();
        }
        let t = self.t + n;
        let mut values = vec![0.0; self.r * t];
        let mut observed = vec![false; self.r * t];
        for i in 0..self.r {
            let (v, o) = self.row(i);
            values[i * t + n..(i + 1) * t].copy_from_slice(v);
            observed[i * t + n..(i + 1) * t].copy_from_slice(o);
        }
        Rows::new(self.r, t, values, observed)
    }

    /// Append `other`'s steps to each row.
    pub fn concat_time(&self, other: &Rows) -> Rows {
        assert_eq!(self.r, other.r);
        let t = self.t + other.t;
        let mut values = Vec::with_capacity(self.r * t);
        let mut observed = Vec::with_capacity(self.r * t);
        for i in 0..self.r {
            let (v, o) = self.row(i);
            let (v2, o2) = other.row(i);
            values.extend_from_slice(v);
            values.extend_from_slice(v2);
            observed.extend_from_slice(o);
            observed.extend_from_slice(o2);
        }
        Rows::new(self.r, t, values, observed)
    }

    /// Keep the last `t` steps of each row.
    pub fn keep_last(&self, t: usize) -> Rows {
        if t >= self.t {
            return self.clone();
        }
        let skip = self.t - t;
        let mut values = Vec::with_capacity(self.r * t);
        let mut observed = Vec::with_capacity(self.r * t);
        for i in 0..self.r {
            let (v, o) = self.row(i);
            values.extend_from_slice(&v[skip..]);
            observed.extend_from_slice(&o[skip..]);
        }
        Rows::new(self.r, t, values, observed)
    }
}

/// Patches a context of `l` patches is left-padded to, so a few compiled shapes serve
/// every length. Padding patches are unobserved, so they are never valid keys and the
/// forecast is unchanged (upstream already pads this way to a patch multiple).
pub fn context_bucket(l: usize) -> usize {
    let step = (l.next_power_of_two() / 8).max(4);
    l.div_ceil(step) * step
}

/// Static sizes baked into one compiled graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphDims {
    /// Packed rows (series × variates).
    pub r: usize,
    /// Encoder (context) patches.
    pub l: usize,
    /// Decoder (forecast-query) tokens.
    pub n_dec: usize,
    /// Also output the transformer hidden state.
    pub hidden: bool,
    /// Training layout: attention projections are built in-graph from
    /// per-part weights (see [`crate::graph::graph_params_train`]) so RoPE's rotated copies
    /// stay tied to the query and key weights they derive from.
    pub train: bool,
}

impl GraphDims {
    pub fn s(&self) -> usize {
        self.l + self.n_dec
    }

    pub fn key(&self) -> u64 {
        ((self.r as u64) << 40)
            | ((self.l as u64) << 20)
            | ((self.n_dec as u64) << 2)
            | ((self.train as u64) << 1)
            | self.hidden as u64
    }
}

/// `[R*S, head_dim / 2]` RoPE tables for integer `positions`, as `Op::Rope`
/// reads them (NeoX pairing over the whole head; pairs past `rope_dim / 2` get
/// `cos = 1`, `sin = 0`).
pub fn rope_half_tables(cfg: &TimestralConfig, positions: &[i64]) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.head_dim() / 2;
    let rd = cfg.rope_dim();
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &p in positions {
        for i in 0..half {
            let f = if i < rd / 2 {
                1.0 / cfg.rope_theta.powf((2 * i) as f32 / rd as f32)
            } else {
                0.0
            } * p as f32;
            cos.push(f.cos());
            sin.push(f.sin());
        }
    }
    (cos, sin)
}

/// `[R*S, D]` RoPE tables for integer `positions` (`[R*S]`), repeated over heads.
///
/// Only the first `rope_dim` dims of each head rotate: the frequency table is zero
/// past `rope_dim / 2`, giving `cos = 1`, `sin = 0`.
pub fn rope_tables(cfg: &TimestralConfig, positions: &[i64]) -> (Vec<f32>, Vec<f32>) {
    let hd = cfg.head_dim();
    let half = hd / 2;
    let rd = cfg.rope_dim();
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            if i < rd / 2 {
                1.0 / cfg.rope_theta.powf((2 * i) as f32 / rd as f32)
            } else {
                0.0
            }
        })
        .collect();
    let d = cfg.d_model;
    let mut cos = Vec::with_capacity(positions.len() * d);
    let mut sin = Vec::with_capacity(positions.len() * d);
    let mut c_head = vec![0f32; hd];
    let mut s_head = vec![0f32; hd];
    for &p in positions {
        for k in 0..hd {
            let f = inv_freq[k % half] * p as f32;
            c_head[k] = f.cos();
            s_head[k] = f.sin();
        }
        for _ in 0..cfg.num_heads {
            cos.extend_from_slice(&c_head);
            sin.extend_from_slice(&s_head);
        }
    }
    (cos, sin)
}

/// Whether any layer runs variate attention (and so the graph takes `var_bias`).
pub fn has_variate_attention(cfg: &TimestralConfig) -> bool {
    cfg.block_schedule
        .iter()
        .any(|k| k.axes().iter().any(|&t| !t))
}

/// Named inputs for one compiled graph shape.
#[derive(Debug, Clone)]
pub struct GraphInputs {
    pub dims: GraphDims,
    pub inputs: Vec<(String, Vec<f32>)>,
}

/// The Timestral network and the host math around it: the packed rows'
/// scaling, masks and positions in, the monotone quantile map out. What runs
/// the graph is [`Net`] (`crate::net`), the one part that needs a device.
pub struct TimestralModel {
    pub(crate) cfg: TimestralConfig,
    pub(crate) forecast_query: Vec<f32>,
    pub(crate) net: Net,
    /// Left-pad contexts to [`context_bucket`] sizes (exact; on by default).
    pub bucketing: bool,
    /// Keep the transformer output of the last block in [`Self::last_hidden`].
    pub capture_hidden: bool,
    pub(crate) last_hidden: Option<Vec<f32>>,
}

impl TimestralModel {
    pub fn config(&self) -> &TimestralConfig {
        &self.cfg
    }

    /// Transformer output `[R*S, D]` of the last block run with [`Self::capture_hidden`].
    pub fn last_hidden(&self) -> Option<&[f32]> {
        self.last_hidden.as_deref()
    }

    /// Causal per-patch normalization of `rows`: `(scaled, loc, scale)`, the last two per patch.
    pub fn scale(&self, rows: &Rows) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let c = &self.cfg;
        let ps = c.patch_size;
        let min_scale = decimal(c.minimum_scale);
        let mut scaled = vec![0f64; rows.r * rows.t];
        let mut locs = Vec::with_capacity(rows.r);
        let mut scales = Vec::with_capacity(rows.r);
        for i in 0..rows.r {
            let (v, o) = rows.row(i);
            let (loc, scale) = causal_patch_loc_scale(v, o, ps, min_scale);
            for (tt, out) in scaled[i * rows.t..(i + 1) * rows.t].iter_mut().enumerate() {
                let p = tt / ps;
                *out = affine_asinh(v[tt], loc[p], scale[p], o[tt], c.use_asinh);
            }
            locs.push(loc);
            scales.push(scale);
        }
        (scaled, locs, scales)
    }

    /// Forecast one block of `horizon_patches` patches after packed `context`.
    ///
    /// `group_ids` gives the series of each row (rows of a series attend each other).
    /// `roles` is `None` when every row is a target; `known_future` holds the
    /// known-covariate values over this block (`(R, horizon_patches * patch_size)`,
    /// unobserved zeros elsewhere).
    ///
    /// Returns raw-space quantiles `(R, horizon_patches * patch_size, Q)`, row-major,
    /// non-decreasing along `Q`.
    pub fn predict_block(
        &mut self,
        context: &Rows,
        group_ids: &[usize],
        horizon_patches: usize,
        roles: Option<&[Role]>,
        known_future: Option<&Rows>,
    ) -> Result<Vec<f64>> {
        ensure!(
            group_ids.len() == context.r,
            "group_ids must have one id per row"
        );
        ensure!(horizon_patches >= 1, "horizon_patches must be >= 1");
        ensure!(
            context.t.is_multiple_of(self.cfg.patch_size),
            "context length {} must be a multiple of patch_size {}",
            context.t,
            self.cfg.patch_size
        );
        let all_targets = roles.is_none_or(|r| r.iter().all(|&x| x == Role::Target));
        if all_targets || !self.cfg.covariate_enabled {
            return self.predict_context_only(context, group_ids, horizon_patches);
        }
        let roles = roles.unwrap();
        let ps = self.cfg.patch_size;
        let horizon = horizon_patches * ps;
        let blank;
        let kf = match known_future {
            Some(kf) => kf,
            None => {
                blank = Rows::new(
                    context.r,
                    horizon,
                    vec![0.0; context.r * horizon],
                    vec![false; context.r * horizon],
                );
                &blank
            }
        };
        ensure!(
            kf.r == context.r && kf.t == horizon,
            "known_future must be ({}, {horizon}), got ({}, {})",
            context.r,
            kf.r,
            kf.t
        );
        self.predict_window(&context.concat_time(kf), group_ids, horizon_patches, roles)
    }

    fn pad_for_bucket(&self, context_patches: usize) -> usize {
        if self.bucketing {
            context_bucket(context_patches) - context_patches
        } else {
            0
        }
    }

    /// No covariates: encoder over the context, `h` forecast queries after it.
    fn predict_context_only(
        &mut self,
        context: &Rows,
        group_ids: &[usize],
        h: usize,
    ) -> Result<Vec<f64>> {
        let ps = self.cfg.patch_size;
        let pad = self.pad_for_bucket(context.t / ps);
        // Extra trailing decoder tokens are invisible to earlier ones (the chain is causal).
        let hb = if self.bucketing {
            h.next_power_of_two().max(h)
        } else {
            h
        };
        let (block, anchors) = self.context_only_block(context, group_ids, hb, pad, false);
        let raw = self.run(block)?;
        Ok(self.quantiles_from_head(&raw, context.r, hb, 0, h, &anchors))
    }

    /// Graph inputs for `n_dec` forecast tokens after `context` (left-padded by
    /// `pad` patches), plus each row's anchor frame `(loc, scale)` — the
    /// normalization its forecasts are read in.
    pub fn context_only_block(
        &self,
        context: &Rows,
        group_ids: &[usize],
        n_dec: usize,
        pad: usize,
        train: bool,
    ) -> (GraphInputs, Vec<(f64, f64)>) {
        let ps = self.cfg.patch_size;
        let rows = context.pad_left(pad * ps);
        let (r, l) = (rows.r, rows.t / ps);
        let (scaled, locs, scales) = self.scale(&rows);
        let obs_per_patch = per_patch_any(&rows.observed, ps);
        let key_valid = masks::key_validity(&obs_per_patch, l, self.cfg.attend_missing_patches);
        let time_mask = masks::forecast_time_mask(&key_valid, l, n_dec);
        let var_mask = masks::variate_mask(group_ids, &key_valid, l, n_dec, None, None);
        let first = first_observed(&obs_per_patch, l, group_ids, pad);
        let positions = masks::positions(&first, l, n_dec, false);
        let dims = GraphDims {
            r,
            l,
            n_dec,
            hidden: self.capture_hidden && !train,
            train,
        };
        let block = self.graph_inputs(dims, &rows, &scaled, &positions, &time_mask, &var_mask);
        let anchors = (0..r).map(|i| (locs[i][l - 1], scales[i][l - 1])).collect();
        (block, anchors)
    }

    /// With covariates: the encoder spans context plus horizon, so known-future values
    /// are keys. Target and past-only rows keep a blank horizon.
    fn predict_window(
        &mut self,
        full: &Rows,
        group_ids: &[usize],
        h: usize,
        roles: &[Role],
    ) -> Result<Vec<f64>> {
        let c = self.cfg.clone();
        let ps = c.patch_size;
        ensure!(
            h <= c.block_size,
            "decode block {h} must be in [1, {}] with covariates",
            c.block_size
        );
        let ll0 = full.t / ps - h;
        ensure!(ll0 >= 1, "context must have at least one patch");
        let pad = self.pad_for_bucket(ll0);
        let rows = full.pad_left(pad * ps);
        let (r, n) = (rows.r, rows.t / ps);
        let ll = n - h;
        let targets: Vec<bool> = roles.iter().map(|&x| x == Role::Target).collect();
        let known: Vec<bool> = roles.iter().map(|&x| x == Role::KnownCovariate).collect();

        let (mut scaled, mut locs, mut scales) = self.scale(&rows);
        for i in (0..r).filter(|&i| known[i]) {
            let (v, o) = rows.row(i);
            let (loc, scale, binary) = instance_norm_stats(v, o, decimal(c.minimum_scale));
            for tt in 0..rows.t {
                let y = affine_asinh(v[tt], loc, scale, o[tt], false);
                scaled[i * rows.t + tt] = if c.use_asinh && o[tt] && !binary {
                    y.asinh()
                } else {
                    y
                };
            }
            locs[i] = vec![loc; n];
            scales[i] = vec![scale; n];
        }

        let obs_per_patch = per_patch_any(&rows.observed, ps);
        let key_valid = masks::key_validity(&obs_per_patch, n, c.attend_missing_patches);
        let time_mask =
            masks::two_stream_time_mask(&vec![ll; r], n, c.block_size, &key_valid, &known);
        let var_mask =
            masks::variate_mask(group_ids, &key_valid, n, n, Some(&targets), Some(&known));
        let first = first_observed(&obs_per_patch, n, group_ids, pad);
        let positions = masks::positions(&first, n, n, true);

        let dims = GraphDims {
            r,
            l: n,
            n_dec: n,
            hidden: self.capture_hidden,
            train: false,
        };
        let raw =
            self.run(self.graph_inputs(dims, &rows, &scaled, &positions, &time_mask, &var_mask))?;
        let anchors: Vec<(f64, f64)> = (0..r)
            .map(|i| (locs[i][ll - 1], scales[i][ll - 1]))
            .collect();
        Ok(self.quantiles_from_head(&raw, r, n, ll, h, &anchors))
    }

    /// Every named graph input for one block, in the graph's `dims`.
    pub fn graph_inputs(
        &self,
        dims: GraphDims,
        rows: &Rows,
        scaled: &[f64],
        positions: &[i64],
        time_mask: &[bool],
        var_mask: &[bool],
    ) -> GraphInputs {
        let c = &self.cfg;
        let ps = c.patch_size;
        let GraphDims { r, l, n_dec, .. } = dims;
        let mut patches = Vec::with_capacity(r * l * 2 * ps);
        for i in 0..r {
            for p in 0..l {
                let span = i * rows.t + p * ps..i * rows.t + (p + 1) * ps;
                patches.extend(scaled[span.clone()].iter().map(|&x| x as f32));
                patches.extend(
                    rows.observed[span]
                        .iter()
                        .map(|&o| if o { 0.0 } else { 1.0 }),
                );
            }
        }
        let dec: Vec<f32> = (0..r * n_dec)
            .flat_map(|_| self.forecast_query.iter().copied())
            .collect();
        let (cos, sin) = if dims.train {
            rope_tables(c, positions)
        } else {
            rope_half_tables(c, positions)
        };
        let mut inputs = vec![
            ("patches".to_string(), patches),
            ("dec".to_string(), dec),
            ("rope_cos".to_string(), cos),
            ("rope_sin".to_string(), sin),
            ("time_bias".to_string(), masks::to_bias(time_mask)),
        ];
        if has_variate_attention(c) {
            inputs.push(("var_bias".to_string(), masks::to_bias(var_mask)));
        }
        GraphInputs { dims, inputs }
    }

    /// Run the compiled graph on prepared inputs; returns raw head values
    /// `[R*n_dec, ps*Q]` (before the monotone map).
    pub fn run_raw(&mut self, block: GraphInputs) -> Result<Vec<f32>> {
        self.run(block)
    }

    /// Run the graph; returns raw head values `[R*n_dec, ps*Q]`.
    fn run(&mut self, block: GraphInputs) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let ps = c.patch_size;
        let d = c.d_model;
        let GraphDims { r, l, n_dec, .. } = block.dims;
        let outs = self.net.run(c, block)?;
        let mut outs = outs.into_iter();
        let head = outs.next().context("graph produced no output")?;
        ensure!(
            head.len() == r * n_dec * ps * c.num_quantiles(),
            "head output has {} values, expected {}",
            head.len(),
            r * n_dec * ps * c.num_quantiles()
        );
        if self.capture_hidden {
            let hidden = outs.next().context("graph produced no hidden output")?;
            ensure!(hidden.len() == r * (l + n_dec) * d, "hidden output size");
            self.last_hidden = Some(hidden);
        }
        Ok(head)
    }

    /// Monotone quantiles of decoder tokens `from..from+h` of each row, denormalized with
    /// the row's anchor `(loc, scale)`: `(R, h*ps, Q)`.
    fn quantiles_from_head(
        &self,
        raw: &[f32],
        r: usize,
        n_dec: usize,
        from: usize,
        h: usize,
        anchors: &[(f64, f64)],
    ) -> Vec<f64> {
        let c = &self.cfg;
        let ps = c.patch_size;
        let nq = c.num_quantiles();
        let widths = quantile_widths(&c.quantile_knots, c.quantile_width_factor);
        let m = c.median_index();
        let mut out = Vec::with_capacity(r * h * ps * nq);
        let mut q = vec![0f64; nq];
        for (i, &(loc, scale)) in anchors.iter().enumerate() {
            for tok in from..from + h {
                let base = (i * n_dec + tok) * ps * nq;
                for step in 0..ps {
                    let raw_q = &raw[base + step * nq..base + (step + 1) * nq];
                    monotone_quantiles(raw_q, &widths, m, &mut q);
                    for v in q.iter_mut() {
                        *v = invert_affine_asinh(*v, loc, scale, c.use_asinh);
                    }
                    clamp_nonfinite(&mut q);
                    out.extend_from_slice(&q);
                }
            }
        }
        out
    }
}

/// `width_factor * dp` per knot, `dp` being the probability gap to the next level inward.
pub fn quantile_widths(levels: &[f32], width_factor: f32) -> Vec<f64> {
    let width_factor = decimal(width_factor);
    let m = levels.iter().position(|&q| q == 0.5).expect("median knot");
    (0..levels.len())
        .map(|i| {
            let dp = if i > m {
                decimal(levels[i]) - decimal(levels[i - 1])
            } else if i < m {
                decimal(levels[i + 1]) - decimal(levels[i])
            } else {
                0.0
            };
            width_factor * dp
        })
        .collect()
}

/// Non-crossing quantiles: the median channel is free, the others fan out from it as
/// `median ∓ cumsum(widths * softplus(raw))`, clamped to the f32 range as upstream does.
pub fn monotone_quantiles(raw: &[f32], widths: &[f64], m: usize, out: &mut [f64]) {
    const BIG: f64 = f32::MAX as f64;
    let median = raw[m] as f64;
    out[m] = median;
    let mut up = 0f64;
    for i in m + 1..raw.len() {
        up += widths[i] * softplus(raw[i] as f64);
        out[i] = (median + up).clamp(-BIG, BIG);
    }
    let mut down = 0f64;
    for i in (0..m).rev() {
        down += widths[i] * softplus(raw[i] as f64);
        out[i] = (median - down).clamp(-BIG, BIG);
    }
}

/// `torch.nn.functional.softplus` (beta 1, threshold 20).
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// A config value as the decimal it was written as: `1e-5f32 as f64` is not `1e-5`.
pub(crate) fn decimal(x: f32) -> f64 {
    x.to_string().parse().unwrap_or(x as f64)
}

fn per_patch_any(observed: &[bool], ps: usize) -> Vec<bool> {
    observed.chunks(ps).map(|c| c.iter().any(|&o| o)).collect()
}

/// Per-row first observed patch of the row's series. A series with no observation
/// anchors at the first unpadded patch, as it would without bucket padding.
fn first_observed(obs_per_patch: &[bool], n: usize, group_ids: &[usize], pad: usize) -> Vec<usize> {
    let groups = masks::group_first_observed_patch(obs_per_patch, n, group_ids);
    let mut any = vec![false; groups.len()];
    for (row, &g) in group_ids.iter().enumerate() {
        any[g] |= obs_per_patch[row * n..(row + 1) * n].iter().any(|&o| o);
    }
    group_ids
        .iter()
        .map(|&g| if any[g] { groups[g] } else { pad })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_quantiles_do_not_cross() {
        let levels = [0.1, 0.25, 0.5, 0.75, 0.9];
        let w = quantile_widths(&levels, 3.0);
        let mut out = [0f64; 5];
        monotone_quantiles(&[-4.0, 9.0, 0.3, -30.0, 2.0], &w, 2, &mut out);
        assert!(out.windows(2).all(|p| p[0] <= p[1]), "{out:?}");
        assert_eq!(out[2], 0.3f32 as f64);
    }

    #[test]
    fn buckets_cover_every_length_with_few_shapes() {
        assert_eq!(context_bucket(1), 4);
        assert_eq!(context_bucket(32), 32);
        assert_eq!(context_bucket(33), 40);
        assert_eq!(context_bucket(1000), 1024);
        let shapes: std::collections::BTreeSet<_> = (1..=1024).map(context_bucket).collect();
        assert!(shapes.len() < 64, "{} shapes", shapes.len());
    }
}
