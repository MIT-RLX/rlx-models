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

//! Fine-tuning and quantization-aware training (QAT) of a Timestral checkpoint.
//!
//! The loss is the pinball (quantile) loss over the model's quantile levels, in
//! the normalized frame each forecast is read in (the anchor frame of the last
//! context patch, `asinh((y - loc) / scale)`). Upstream has not published its
//! training loss; pinball over the quantile knots is the standard choice for a
//! quantile head and what the monotone head is built for.
//!
//! With a [`NumberFormat`], the model part of the graph runs in that format —
//! activations rounded in-graph, weights rounded as each step uploads the f32
//! masters — while the loss head and every gradient stay f32 (the
//! straight-through estimator). Without one this is plain fine-tuning.
//!
//! Training windows are context-only (no covariates); `forecast_query` stays
//! frozen (it enters the graph as an input).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, NumberFormat};
use rlx_tune::{DpConfig, ParamSlot, Trainer};

use crate::config::TimestralConfig;
use crate::graph::{GraphDims, TRAIN_CONSTANTS, build_graph, graph_params_train};
use crate::model::{Rows, TimestralModel};
use crate::scaler::affine_asinh;
use crate::weights::TimestralWeights;

/// How to train.
#[derive(Debug, Clone)]
pub struct TrainOptions {
    /// Run the model part in this format (QAT); `None` fine-tunes in f32.
    pub format: Option<NumberFormat>,
    pub steps: usize,
    pub lr: f32,
    /// Windows per step.
    pub batch: usize,
    pub context_patches: usize,
    pub horizon_patches: usize,
    pub device: Device,
    pub seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            format: None,
            steps: 200,
            lr: 1e-4,
            batch: 8,
            context_patches: 16,
            horizon_patches: 4,
            device: Device::Cpu,
            seed: 0,
        }
    }
}

/// What training produced.
pub struct Trained {
    /// Trained weights in the training layout (see [`graph_params_train`]).
    pub params: HashMap<String, Vec<f32>>,
    /// Loss per step.
    pub losses: Vec<f32>,
}

/// Train `weights` on windows drawn from `series` (NaN = missing).
pub fn train(
    cfg: &TimestralConfig,
    weights: &TimestralWeights,
    series: &[Vec<f32>],
    opts: &TrainOptions,
) -> Result<Trained> {
    let ps = cfg.patch_size;
    let (cl, hl) = (opts.context_patches * ps, opts.horizon_patches * ps);
    ensure!(
        opts.horizon_patches >= 1 && opts.context_patches >= 1,
        "empty window"
    );
    let usable: Vec<&Vec<f32>> = series.iter().filter(|s| s.len() >= cl + hl).collect();
    ensure!(
        !usable.is_empty(),
        "no series is long enough for a {cl}-step context plus a {hl}-step horizon"
    );
    if let Some(f @ (NumberFormat::F64 | NumberFormat::F64x2)) = opts.format {
        bail!("training in {f} is not available yet");
    }

    let dims = GraphDims {
        r: opts.batch,
        l: opts.context_patches,
        n_dec: opts.horizon_patches,
        hidden: false,
        train: true,
    };
    let (graph, slots) = train_graph(cfg, dims, opts.format)?;
    let params = graph_params_train(cfg, weights);

    // Host-side input preparation is the inference path's, on CPU.
    let prep = TimestralModel::new(cfg.clone(), weights, Device::Cpu);
    let mut rng = Lcg(opts.seed.wrapping_mul(2654435761).wrapping_add(1));
    let mut next_batch = |_: usize, _: usize| -> Vec<(String, Vec<f32>)> {
        sample_batch(cfg, &prep, &usable, opts, &mut rng)
    };

    let mut dp = DpConfig::new(opts.lr).device(opts.device);
    if let Some(f) = opts.format.filter(|f| *f != NumberFormat::F32) {
        dp.qat = Some(f);
        dp.qat_forward_rewritten = true;
    }
    let mut trainer = Trainer::new(graph, &slots, &params, opts.steps, None, &dp)?;
    let mut losses = Vec::with_capacity(opts.steps);
    for _ in 0..opts.steps {
        losses.push(trainer.step(&mut next_batch)?.loss);
    }
    Ok(Trained {
        params: trainer.params(),
        losses,
    })
}

/// The model in its training layout, the monotone quantile map and the pinball
/// loss, as one graph with a scalar output. With a `format`, only the model
/// part is rewritten to run in it. Returns the trainable params.
///
/// Extra inputs: `target` `[N, 1]` (normalized targets) and `weight` `[N, 1]`
/// (`1 / observed count` on observed steps, 0 elsewhere), `N = r·n_dec·ps`.
pub fn train_graph(
    cfg: &TimestralConfig,
    dims: GraphDims,
    format: Option<NumberFormat>,
) -> Result<(Graph, Vec<ParamSlot>)> {
    let mut g = build_graph(cfg, dims)
        .lower_to_mir()
        .context("lower Timestral graph")?
        .into_graph();
    if let Some(NumberFormat::Float { format, scaling }) = format {
        g = rlx_runtime::emulate_format::emulate_float_format(
            g,
            format,
            scaling,
            &Default::default(),
        );
    }
    let head = g.outputs[0];
    let ps = cfg.patch_size;
    let q = cfg.num_quantiles();
    let n = dims.r * dims.n_dec * ps;
    let f32s = |dims: &[usize]| Shape::new(dims, DType::F32);

    let raw = g.add_node(
        Op::Reshape {
            new_shape: vec![n as i64, q as i64],
        },
        vec![head],
        f32s(&[n, q]),
    );
    let sp = g.activation(Activation::Softplus, raw, f32s(&[n, q]));
    let (median_m, cum_m) = monotone_matrices(cfg);
    let median_m = constant(&mut g, &median_m, &[q, q]);
    let cum_m = constant(&mut g, &cum_m, &[q, q]);
    let qm = g.add_node(Op::MatMul, vec![raw, median_m], f32s(&[n, q]));
    let qc = g.add_node(Op::MatMul, vec![sp, cum_m], f32s(&[n, q]));
    let quant = g.binary(BinaryOp::Add, qm, qc, f32s(&[n, q]));

    let target = g.input("target", f32s(&[n, 1]));
    let weight = g.input("weight", f32s(&[n, 1]));
    let err = g.binary(BinaryOp::Sub, target, quant, f32s(&[n, q]));
    let tau: Vec<f32> = cfg.quantile_knots.clone();
    let tau_m1: Vec<f32> = tau.iter().map(|t| t - 1.0).collect();
    let tau = constant(&mut g, &tau, &[1, q]);
    let tau_m1 = constant(&mut g, &tau_m1, &[1, q]);
    let a = g.binary(BinaryOp::Mul, err, tau, f32s(&[n, q]));
    let b = g.binary(BinaryOp::Mul, err, tau_m1, f32s(&[n, q]));
    let pin = g.binary(BinaryOp::Max, a, b, f32s(&[n, q]));
    let weighted = g.binary(BinaryOp::Mul, pin, weight, f32s(&[n, q]));
    let total = g.reduce(weighted, ReduceOp::Sum, vec![0, 1], false, f32s(&[1]));
    let inv_q = constant(&mut g, &[1.0 / q as f32], &[1]);
    let loss = g.binary(BinaryOp::Mul, total, inv_q, f32s(&[1]));
    g.set_outputs(vec![loss]);

    let slots = g
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Param { name } if !TRAIN_CONSTANTS.contains(&name.as_str()) => Some(ParamSlot {
                name: name.clone(),
                node: node.id,
            }),
            _ => None,
        })
        .collect();
    Ok((g, slots))
}

/// `[Q, Q]` matrices so that `raw · M + softplus(raw) · C` is
/// [`crate::model::monotone_quantiles`]: `M` copies the median column to every
/// level, `C` accumulates the scaled increments outward from it.
fn monotone_matrices(cfg: &TimestralConfig) -> (Vec<f32>, Vec<f32>) {
    let q = cfg.num_quantiles();
    let m = cfg.median_index();
    let widths = crate::model::quantile_widths(&cfg.quantile_knots, cfg.quantile_width_factor);
    let mut median_m = vec![0f32; q * q];
    let mut cum_m = vec![0f32; q * q];
    for i in 0..q {
        median_m[m * q + i] = 1.0;
        if i > m {
            for j in m + 1..=i {
                cum_m[j * q + i] = widths[j] as f32;
            }
        } else if i < m {
            for j in i..m {
                cum_m[j * q + i] = -widths[j] as f32;
            }
        }
    }
    (median_m, cum_m)
}

fn constant(g: &mut Graph, values: &[f32], dims: &[usize]) -> NodeId {
    let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], Shape::new(dims, DType::F32))
}

/// One batch of random context/horizon windows as graph inputs.
fn sample_batch(
    cfg: &TimestralConfig,
    prep: &TimestralModel,
    series: &[&Vec<f32>],
    opts: &TrainOptions,
    rng: &mut Lcg,
) -> Vec<(String, Vec<f32>)> {
    let ps = cfg.patch_size;
    let (cl, hl) = (opts.context_patches * ps, opts.horizon_patches * ps);
    let r = opts.batch;
    let mut values = Vec::with_capacity(r * cl);
    let mut observed = Vec::with_capacity(r * cl);
    let mut futures = Vec::with_capacity(r);
    for _ in 0..r {
        let s = series[rng.below(series.len())];
        let start = rng.below(s.len() - cl - hl + 1);
        let ctx = &s[start..start + cl];
        let obs: Vec<bool> = ctx.iter().map(|v| v.is_finite()).collect();
        // Forward-filled context with the original mask, as the forecaster feeds it.
        let mut last = 0.0;
        for (v, o) in ctx.iter().zip(&obs) {
            if *o {
                last = *v;
            }
            values.push(f64::from(if *o { *v } else { last }));
        }
        observed.extend(obs);
        futures.push(&s[start + cl..start + cl + hl]);
    }
    let rows = Rows::new(r, cl, values, observed);
    let groups: Vec<usize> = (0..r).collect();
    let (block, anchors) = prep.context_only_block(&rows, &groups, opts.horizon_patches, 0, true);

    let count = futures
        .iter()
        .flat_map(|f| f.iter())
        .filter(|v| v.is_finite())
        .count()
        .max(1);
    let mut target = Vec::with_capacity(r * hl);
    let mut weight = Vec::with_capacity(r * hl);
    for (fut, &(loc, scale)) in futures.iter().zip(&anchors) {
        for &y in fut.iter() {
            let ok = y.is_finite();
            target.push(if ok {
                affine_asinh(y as f64, loc, scale, true, cfg.use_asinh) as f32
            } else {
                0.0
            });
            weight.push(if ok { 1.0 / count as f32 } else { 0.0 });
        }
    }
    let mut inputs = block.inputs;
    inputs.push(("target".into(), target));
    inputs.push(("weight".into(), weight));
    inputs
}

/// Write trained params as an upstream checkpoint directory (`config.json` +
/// `model.safetensors` under Mistral's tensor names), loadable by both this
/// crate and the reference package.
pub fn export(
    cfg_path: &Path,
    cfg: &TimestralConfig,
    weights: &TimestralWeights,
    params: &HashMap<String, Vec<f32>>,
    out_dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    std::fs::copy(cfg_path, out_dir.join("config.json")).context("copy config.json")?;
    let tensors = upstream_tensors(cfg, weights, params)?;
    rlx_tune::lm::save_f32_safetensors(&out_dir.join("model.safetensors"), &tensors)
}

/// Training-layout params back to upstream `(name, shape, data)`.
pub fn upstream_tensors(
    cfg: &TimestralConfig,
    weights: &TimestralWeights,
    p: &HashMap<String, Vec<f32>>,
) -> Result<Tensors> {
    let get = |k: &str| {
        p.get(k)
            .with_context(|| format!("missing trained param {k}"))
    };
    let (d, hd, nh, ps) = (cfg.d_model, cfg.head_dim(), cfg.num_heads, cfg.patch_size);
    let hq = ps * cfg.num_quantiles();
    let mut out = Vec::new();
    let pp = "representation.patch_proj";
    lin(
        &mut out,
        p,
        format!("{pp}.hidden_layer.weight"),
        "patch.hidden",
        2 * ps,
        4 * d,
    )?;
    lin(
        &mut out,
        p,
        format!("{pp}.output_layer.weight"),
        "patch.output",
        4 * d,
        d,
    )?;
    lin(
        &mut out,
        p,
        format!("{pp}.residual_layer.weight"),
        "patch.residual",
        2 * ps,
        d,
    )?;
    lin(
        &mut out,
        p,
        "head.projection.hidden_layer.weight".into(),
        "head.hidden",
        d,
        4 * d,
    )?;
    lin(
        &mut out,
        p,
        "head.projection.output_layer.weight".into(),
        "head.output",
        4 * d,
        hq,
    )?;
    lin(
        &mut out,
        p,
        "head.projection.residual_layer.weight".into(),
        "head.residual",
        d,
        hq,
    )?;
    let mut vecs: Vec<(String, String, usize)> = vec![
        (
            format!("{pp}.hidden_layer.bias"),
            "patch.hidden_b".into(),
            4 * d,
        ),
        (
            format!("{pp}.output_layer.bias"),
            "patch.output_b".into(),
            d,
        ),
        (
            format!("{pp}.residual_layer.bias"),
            "patch.residual_b".into(),
            d,
        ),
        ("transformer.out_norm.scale".into(), "out_norm".into(), d),
    ];
    for (i, kind) in cfg.block_schedule.iter().enumerate() {
        let pre = format!("transformer.layers.{i}");
        for (j, &is_time) in kind.axes().iter().enumerate() {
            let t = format!("l{i}.a{j}");
            vecs.push((
                format!("{pre}.attn_norms.{j}.scale"),
                format!("{t}.norm"),
                d,
            ));
            // q (unscaled), k, v back to upstream `(qkv, head_dim, n_heads)` rows.
            let q_scale = 1.0 / (hd as f32).sqrt();
            let mut qkv = vec![0f32; 3 * d * d];
            for (part, key) in ["q", "k", "v"].iter().enumerate() {
                let w = transpose(get(&format!("{t}.{key}"))?, d, d); // [out (h, k), in]
                for h in 0..nh {
                    for k in 0..hd {
                        let src = (h * hd + k) * d;
                        let dst = (part * d + k * nh + h) * d;
                        for c in 0..d {
                            let v = w[src + c];
                            qkv[dst + c] = if part == 0 { v / q_scale } else { v };
                        }
                    }
                }
            }
            out.push((format!("{pre}.attns.{j}.wQKV.weight"), vec![3 * d, d], qkv));
            if cfg.gated(is_time) {
                lin(
                    &mut out,
                    p,
                    format!("{pre}.attns.{j}._gate.weight"),
                    &format!("{t}.gate"),
                    d,
                    d,
                )?;
            }
            lin(
                &mut out,
                p,
                format!("{pre}.attns.{j}.wO.weight"),
                &format!("{t}.o"),
                d,
                d,
            )?;
        }
        let m = format!("l{i}.mlp");
        let h = cfg.mlp_hidden_dim;
        let mut w12 = transpose(get(&format!("{m}.gate"))?, d, h);
        w12.extend(transpose(get(&format!("{m}.up"))?, d, h));
        out.push((format!("{pre}.mlp.w12.weight"), vec![2 * h, d], w12));
        lin(
            &mut out,
            p,
            format!("{pre}.mlp.w3.weight"),
            &format!("{m}.down"),
            h,
            d,
        )?;
        vecs.push((format!("{pre}.mlp_norm.scale"), format!("{m}.norm"), d));
    }
    for (name, key, n) in vecs {
        let v = get(&key)?;
        ensure!(v.len() == n, "{key}: {} values, expected {n}", v.len());
        out.push((name, vec![n], v.clone()));
    }
    out.push((
        "representation.forecast_query".into(),
        vec![d],
        weights.forecast_query.to_vec(),
    ));
    Ok(out)
}

type Tensors = Vec<(String, Vec<usize>, Vec<f32>)>;

/// Push trained `[in, out]` param `key` as upstream `[out, in]` tensor `name`.
fn lin(
    out: &mut Tensors,
    p: &HashMap<String, Vec<f32>>,
    name: String,
    key: &str,
    inp: usize,
    outd: usize,
) -> Result<()> {
    let v = p
        .get(key)
        .with_context(|| format!("missing trained param {key}"))?;
    out.push((name, vec![outd, inp], transpose(v, inp, outd)));
    Ok(())
}

/// Row-major `[rows, cols]` → `[cols, rows]`.
fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = v[r * cols + c];
        }
    }
    t
}

/// Small deterministic generator for window sampling.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n.max(1)
    }
}

/// Weighted quantile loss (as fev-bench reports it) of forecasting the last
/// `horizon` steps of each series from up to `context` steps before them:
/// `2 · Σ pinball / Σ |y|`, averaged over the quantile levels. Lower is better.
pub fn holdout_wql(
    fc: &mut crate::TimestralForecaster,
    series: &[Vec<f32>],
    context: usize,
    horizon: usize,
) -> Result<f32> {
    let levels = fc.quantile_levels().to_vec();
    let (mut loss, mut scale) = (0f64, 0f64);
    for s in series.iter().filter(|s| s.len() > horizon + 8) {
        let split = s.len() - horizon;
        let ctx = &s[split.saturating_sub(context)..split];
        let f = fc.forecast(
            ndarray::ArrayView3::from_shape((1, 1, ctx.len()), ctx)?,
            horizon,
            &crate::ForecastOptions::default(),
        )?;
        for (t, &y) in s[split..].iter().enumerate() {
            if !y.is_finite() {
                continue;
            }
            scale += y.abs() as f64;
            for (k, &tau) in levels.iter().enumerate() {
                let e = (y - f.quantiles[[0, 0, t, k]]) as f64;
                loss += (tau as f64 * e).max((tau as f64 - 1.0) * e);
            }
        }
    }
    ensure!(scale > 0.0, "no held-out values to score");
    Ok((2.0 * loss / scale / levels.len() as f64) as f32)
}

/// [`holdout_wql`] over `windows` forecast origins per series, `horizon` apart
/// and ending at the series' end: one window says little about a format, since
/// a forecast's errors are correlated across its steps.
pub fn rolling_wql(
    fc: &mut crate::TimestralForecaster,
    series: &[Vec<f32>],
    context: usize,
    horizon: usize,
    windows: usize,
) -> Result<f32> {
    let levels = fc.quantile_levels().to_vec();
    let (mut loss, mut scale) = (0f64, 0f64);
    for s in series {
        for w in 0..windows {
            let Some(split) = s.len().checked_sub(horizon * (w + 1)) else {
                break;
            };
            if split < 8 {
                break;
            }
            let ctx = &s[split.saturating_sub(context)..split];
            let f = fc.forecast(
                ndarray::ArrayView3::from_shape((1, 1, ctx.len()), ctx)?,
                horizon,
                &crate::ForecastOptions::default(),
            )?;
            for (t, &y) in s[split..split + horizon].iter().enumerate() {
                if !y.is_finite() {
                    continue;
                }
                scale += y.abs() as f64;
                for (k, &tau) in levels.iter().enumerate() {
                    let e = (y - f.quantiles[[0, 0, t, k]]) as f64;
                    loss += (tau as f64 * e).max((tau as f64 - 1.0) * e);
                }
            }
        }
    }
    ensure!(scale > 0.0, "no held-out values to score");
    Ok((2.0 * loss / scale / levels.len() as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph_params;
    use rlx_runtime::{CompileOptions, Session};

    fn setup() -> (TimestralConfig, TimestralWeights, Vec<Vec<f32>>) {
        let cfg = TimestralConfig::synth_tiny();
        let w = TimestralWeights::synth(&cfg, 21);
        let series = (0..3)
            .map(|s| {
                (0..400)
                    .map(|t| {
                        let t = t as f32;
                        (t * 0.07 + s as f32).sin() * (2.0 + s as f32) + 0.01 * t
                    })
                    .collect()
            })
            .collect();
        (cfg, w, series)
    }

    fn opts() -> TrainOptions {
        TrainOptions {
            batch: 3,
            context_patches: 8,
            horizon_patches: 2,
            ..Default::default()
        }
    }

    /// The training graph's loss equals the pinball loss of the inference
    /// path's own quantiles, computed on the host.
    #[test]
    fn loss_matches_inference_quantiles() {
        let (cfg, w, series) = setup();
        let o = opts();
        let dims = GraphDims {
            r: o.batch,
            l: o.context_patches,
            n_dec: o.horizon_patches,
            hidden: false,
            train: true,
        };
        let (graph, _) = train_graph(&cfg, dims, None).unwrap();
        let mut cg = Session::new(Device::Cpu).compile_with(graph, &CompileOptions::baseline());
        for (k, v) in graph_params_train(&cfg, &w) {
            cg.set_param(&k, &v);
        }
        let prep = TimestralModel::new(cfg.clone(), &w, Device::Cpu);
        let usable: Vec<&Vec<f32>> = series.iter().collect();
        let batch = sample_batch(&cfg, &prep, &usable, &o, &mut Lcg(7));
        let inputs: Vec<(&str, &[f32])> = batch
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let got = cg.run(&inputs).remove(0)[0];

        // Host: inference graph head → monotone quantiles → pinball.
        let mut infer = TimestralModel::new(cfg.clone(), &w, Device::Cpu);
        infer.bucketing = false;
        // The inference graph reads `head_dim / 2`-wide RoPE rows: the first
        // half-head of each training row (which repeats one head's values).
        let (d, half) = (cfg.d_model, cfg.head_dim() / 2);
        let head_inputs: Vec<(String, Vec<f32>)> = batch
            .iter()
            .filter(|(n, _)| n != "target" && n != "weight")
            .map(|(n, v)| match n.as_str() {
                "rope_cos" | "rope_sin" => (
                    n.clone(),
                    v.chunks(d).flat_map(|row| row[..half].to_vec()).collect(),
                ),
                _ => (n.clone(), v.clone()),
            })
            .collect();
        let infer_dims = GraphDims {
            train: false,
            ..dims
        };
        let raw = infer
            .run_raw(crate::model::GraphInputs {
                dims: infer_dims,
                inputs: head_inputs,
            })
            .unwrap();
        let target = &batch.iter().find(|(n, _)| n == "target").unwrap().1;
        let weight = &batch.iter().find(|(n, _)| n == "weight").unwrap().1;
        let q = cfg.num_quantiles();
        let widths = crate::model::quantile_widths(&cfg.quantile_knots, cfg.quantile_width_factor);
        let mut qv = vec![0f64; q];
        let mut want = 0f64;
        for (i, (&y, &wt)) in target.iter().zip(weight).enumerate() {
            crate::model::monotone_quantiles(
                &raw[i * q..(i + 1) * q],
                &widths,
                cfg.median_index(),
                &mut qv,
            );
            for (k, &tau) in cfg.quantile_knots.iter().enumerate() {
                let e = y - qv[k] as f32;
                want += (wt * (tau * e).max((tau - 1.0) * e)) as f64;
            }
        }
        let want = (want / q as f64) as f32;
        assert!(
            (got - want).abs() <= 1e-4 * want.abs().max(1e-3),
            "graph {got} vs host {want}"
        );
    }

    #[test]
    fn export_round_trips_to_the_same_weights() {
        let (cfg, w, _) = setup();
        let tensors = upstream_tensors(&cfg, &w, &graph_params_train(&cfg, &w)).unwrap();
        let map = rlx_core::weight_map::WeightMap::from_tensors(
            tensors.into_iter().map(|(n, s, d)| (n, (d, s))).collect(),
        );
        let back = TimestralWeights::from_map(map, &cfg).unwrap();
        let (a, b) = (graph_params(&cfg, &w), graph_params(&cfg, &back));
        for (k, v) in &a {
            let err = v
                .iter()
                .zip(&b[k])
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                err <= 1e-6 * v.iter().fold(1e-6f32, |m, x| m.max(x.abs())),
                "{k}: off by {err}"
            );
        }
    }

    #[test]
    fn training_and_qat_lower_the_loss() {
        let (cfg, w, series) = setup();
        for format in [None, Some("f8".parse().unwrap())] {
            let o = TrainOptions {
                format,
                steps: 60,
                lr: 3e-3,
                ..opts()
            };
            let t = train(&cfg, &w, &series, &o).unwrap();
            let head: f32 = t.losses[..10].iter().sum::<f32>() / 10.0;
            let tail: f32 = t.losses[50..].iter().sum::<f32>() / 10.0;
            assert!(tail < 0.8 * head, "{format:?}: loss {head} -> {tail}");
        }
    }
}
