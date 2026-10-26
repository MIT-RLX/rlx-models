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

//! One FlowLM step as a graph: a single input row through the backbone, then
//! the EOS head and the sampler head.
//!
//! The KV cache is held on the host. Each step reads `past_k{l}` / `past_v{l}`
//! of a bucketed capacity `C` (rows past the real length are zeros and are
//! masked out) and returns the new row's key and value per layer, which the
//! caller appends. Prompt rows — text tokens, voice frames — go through the same
//! graph one row at a time and ignore the head outputs; attention is causal, so
//! this equals upstream's one-shot prompt pass.
//!
//! The sampler steps are unrolled: their time embeddings depend only on the
//! step index, so they are computed on the host once
//! (`time_conditioning`) and enter as constants.

use anyhow::{Result, bail};
use rlx_ir::NodeId;
use rlx_ir::infer::GraphExt;

use super::{Builder, Tensor, get};
use crate::checkpoint::Checkpoint;
use crate::spec::{FlowType, ModelSpec};

/// Graph output order.
pub const OUT_LATENT: usize = 0;
pub const OUT_EOS: usize = 1;
pub const OUT_HIDDEN: usize = 2;
/// First per-layer `(k, v)` output.
pub const OUT_KV: usize = 3;

const LN_EPS: f32 = 1e-5;
const HEAD_LN_EPS: f32 = 1e-6;
const TIME_RMS_EPS: f32 = 1e-5;

/// Shape facts the step graph and its caller agree on.
#[derive(Debug, Clone)]
pub struct FlowLmDims {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub latent_dim: usize,
    pub cond_dim: usize,
    pub max_period: f32,
}

impl FlowLmDims {
    pub fn new(spec: &ModelSpec) -> Self {
        let t = &spec.flow_lm.transformer;
        Self {
            d_model: t.d_model,
            num_heads: t.num_heads,
            num_kv_heads: t.num_kv_heads(),
            head_dim: t.head_dim(),
            num_layers: t.num_layers,
            latent_dim: spec.latent_dim(),
            cond_dim: spec.cond_dim(),
            max_period: t.max_period,
        }
    }

    /// Width of one K (or V) row: `num_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// Input names, so callers don't spell them.
pub fn past_k_name(layer: usize) -> String {
    format!("past_k{layer}")
}
pub fn past_v_name(layer: usize) -> String {
    format!("past_v{layer}")
}

/// Build the step graph for KV capacity `cap` and `sampler_steps` head steps.
pub(crate) fn build_step(
    ck: &Checkpoint,
    spec: &ModelSpec,
    cap: usize,
    sampler_steps: usize,
) -> Result<Builder> {
    let dims = FlowLmDims::new(spec);
    let t = &spec.flow_lm.transformer;
    let d = dims.d_model;
    let (nh, nkv, hd) = (dims.num_heads, dims.num_kv_heads, dims.head_dim);
    let attn_dim = t.attn_dim();
    let kv_dim = dims.kv_dim();
    let half = hd / 2;
    let ldim = dims.latent_dim;

    let mut b = Builder::new(&format!("pocket_flow_lm_step_c{cap}"));
    let mut x = b.input("x", &[1, 1, d]);
    let noise = b.input("noise", &[1, ldim]);
    let mask = b.input("mask", &[1, cap + 1]);
    let cos = b.input("rope_cos", &[1, 1, 1, half]);
    let sin = b.input("rope_sin", &[1, 1, 1, half]);

    let mut kv_out = Vec::with_capacity(2 * dims.num_layers);
    for l in 0..dims.num_layers {
        let pre = format!("flow_lm.transformer.layers.{l}");
        let in_proj = get(ck, &format!("{pre}.self_attn.in_proj.weight"))?;
        let (rows, cols) = in_proj.dims2()?;
        if rows != attn_dim + 2 * kv_dim || cols != d {
            bail!(
                "{pre}.self_attn.in_proj.weight is {:?}, expected [{}, {d}]",
                in_proj.shape,
                attn_dim + 2 * kv_dim
            );
        }
        let split = |lo: usize, n: usize| Tensor {
            data: in_proj.data[lo * d..(lo + n) * d].to_vec(),
            shape: vec![n, d],
        };
        let (wq, wk, wv) = (
            split(0, attn_dim),
            split(attn_dim, kv_dim),
            split(attn_dim + kv_dim, kv_dim),
        );

        let n1w = get(ck, &format!("{pre}.norm1.weight"))?;
        let n1b = get(ck, &format!("{pre}.norm1.bias"))?;
        let n1 = b.layer_norm(x, Some((&n1w, &n1b)), LN_EPS);
        let q = b.linear(n1, &wq, None)?;
        let k = b.linear(n1, &wk, None)?;
        let v = b.linear(n1, &wv, None)?;
        let q = b.g.reshape_(q, vec![1, 1, nh as i64, hd as i64]);
        let k = b.g.reshape_(k, vec![1, 1, nkv as i64, hd as i64]);
        let v = b.g.reshape_(v, vec![1, 1, nkv as i64, hd as i64]);
        let q = b.rope_interleaved(q, cos, sin);
        let k = b.rope_interleaved(k, cos, sin);
        let new_k = b.own(k);
        let new_v = b.own(v);
        kv_out.push(new_k);
        kv_out.push(new_v);

        let past_k = b.input(&past_k_name(l), &[1, cap, nkv, hd]);
        let past_v = b.input(&past_v_name(l), &[1, cap, nkv, hd]);
        let mut k_full = b.g.concat_(vec![past_k, k], 1);
        let mut v_full = b.g.concat_(vec![past_v, v], 1);
        if nkv != nh {
            // Grouped-query attention: query head h reads kv head h / (nh/nkv).
            let rep = nh / nkv;
            for kv in [&mut k_full, &mut v_full] {
                let r =
                    b.g.reshape_(*kv, vec![1, (cap + 1) as i64, nkv as i64, 1, hd as i64]);
                let e = b.expand(r, &[1, cap + 1, nkv, rep, hd]);
                *kv =
                    b.g.reshape_(e, vec![1, (cap + 1) as i64, nh as i64, hd as i64]);
            }
        }
        let attn = b.g.attention_(q, k_full, v_full, mask, nh, hd);
        let attn = b.g.reshape_(attn, vec![1, 1, attn_dim as i64]);
        let out_proj = get(ck, &format!("{pre}.self_attn.out_proj.weight"))?;
        let attn = b.linear(attn, &out_proj, None)?;
        x = b.g.add(x, attn);

        let n2w = get(ck, &format!("{pre}.norm2.weight"))?;
        let n2b = get(ck, &format!("{pre}.norm2.bias"))?;
        let n2 = b.layer_norm(x, Some((&n2w, &n2b)), LN_EPS);
        let l1 = get(ck, &format!("{pre}.linear1.weight"))?;
        let l2 = get(ck, &format!("{pre}.linear2.weight"))?;
        let h = b.linear(n2, &l1, None)?;
        let h = b.g.gelu_approx(h);
        let h = b.linear(h, &l2, None)?;
        x = b.g.add(x, h);
    }

    let onw = get(ck, "flow_lm.out_norm.weight")?;
    let onb = get(ck, "flow_lm.out_norm.bias")?;
    let mut hidden = b.layer_norm(x, Some((&onw, &onb)), LN_EPS);
    if let Some(z) = ck.opt("flow_lm.z_proj.weight")? {
        hidden = b.linear(hidden, &z, None)?;
    } else if dims.cond_dim != d {
        bail!("config sets z_dim but the checkpoint has no flow_lm.z_proj.weight");
    }
    let eos_w = get(ck, "flow_lm.out_eos.weight")?;
    let eos_b = get(ck, "flow_lm.out_eos.bias")?;
    let eos = b.linear(hidden, &eos_w, Some(&eos_b))?;
    let eos = b.g.reshape_(eos, vec![1, 1]);
    let c = b.g.reshape_(hidden, vec![1, dims.cond_dim as i64]);

    let latent = sampler_head(&mut b, ck, spec, c, noise, sampler_steps)?;
    let latent = b.own(latent);
    let hidden_out = b.own(c);
    let mut outs = vec![latent, eos, hidden_out];
    outs.extend(kv_out);
    b.g.set_outputs(outs);
    Ok(b)
}

/// The sampler head (`SimpleMLPAdaLN`) and the decode rule of `spec`'s flow
/// type, unrolled over `steps` steps. `c` is `[1, cond_dim]`, `noise` `[1, ldim]`.
fn sampler_head(
    b: &mut Builder,
    ck: &Checkpoint,
    spec: &ModelSpec,
    c: NodeId,
    noise: NodeId,
    steps: usize,
) -> Result<NodeId> {
    let kind = spec.flow_lm.flow.kind;
    let steps = match kind {
        FlowType::Lsd | FlowType::FlowMatching => steps.max(1),
        FlowType::Drifting | FlowType::Pointwise => 1,
    };
    let pre = "flow_lm.flow_net";
    let cond_w = get(ck, &format!("{pre}.cond_embed.weight"))?;
    let cond_b = get(ck, &format!("{pre}.cond_embed.bias"))?;
    let y0 = b.linear(c, &cond_w, Some(&cond_b))?;
    let time = time_conditioning(ck, spec, steps)?;
    let fd = spec.flow_lm.flow.dim;
    let inv_steps = b.scalar(1.0 / steps as f32);

    let mut current = noise;
    for step in 0..steps {
        let y = match &time {
            Some(per_step) => {
                let t = b.param(per_step[step].clone(), &[1, fd]);
                b.g.add(y0, t)
            }
            None => y0,
        };
        let v = flow_net(b, ck, y, current)?;
        current = match kind {
            FlowType::Lsd | FlowType::FlowMatching => {
                let dv = b.g.mul(v, inv_steps);
                b.g.add(current, dv)
            }
            FlowType::Drifting => v,
            FlowType::Pointwise => b.g.add(noise, v),
        };
    }
    Ok(current)
}

/// `SimpleMLPAdaLN.forward` after the conditioning sum: `y` `[1, fd]`, `x` `[1, ldim]`.
fn flow_net(b: &mut Builder, ck: &Checkpoint, y: NodeId, x: NodeId) -> Result<NodeId> {
    let pre = "flow_lm.flow_net";
    let fd = b.dims(y)[1];
    let ip_w = get(ck, &format!("{pre}.input_proj.weight"))?;
    let ip_b = get(ck, &format!("{pre}.input_proj.bias"))?;
    let mut h = b.linear(x, &ip_w, Some(&ip_b))?;
    let ys = b.g.silu(y);

    let mut i = 0;
    while ck.has(&format!("{pre}.res_blocks.{i}.in_ln.weight")) {
        let rb = format!("{pre}.res_blocks.{i}");
        let mods = adaln(b, ck, ys, &format!("{rb}.adaLN_modulation.1"), 3, fd)?;
        let (shift, scale1, gate) = (mods[0], mods[1], mods[2]);
        let lw = get(ck, &format!("{rb}.in_ln.weight"))?;
        let lb = get(ck, &format!("{rb}.in_ln.bias"))?;
        let n = b.layer_norm(h, Some((&lw, &lb)), HEAD_LN_EPS);
        let n = b.g.mul(n, scale1);
        let n = b.g.add(n, shift);
        let w0 = get(ck, &format!("{rb}.mlp.0.weight"))?;
        let b0 = get(ck, &format!("{rb}.mlp.0.bias"))?;
        let w2 = get(ck, &format!("{rb}.mlp.2.weight"))?;
        let b2 = get(ck, &format!("{rb}.mlp.2.bias"))?;
        let m = b.linear(n, &w0, Some(&b0))?;
        let m = b.g.silu(m);
        let m = b.linear(m, &w2, Some(&b2))?;
        let gm = b.g.mul(gate, m);
        h = b.g.add(h, gm);
        i += 1;
    }
    if i == 0 {
        bail!("{pre} has no res_blocks");
    }

    let fl = format!("{pre}.final_layer");
    let mods = adaln(b, ck, ys, &format!("{fl}.adaLN_modulation.1"), 2, fd)?;
    let n = b.layer_norm(h, None, HEAD_LN_EPS);
    let n = b.g.mul(n, mods[1]);
    let n = b.g.add(n, mods[0]);
    let lw = get(ck, &format!("{fl}.linear.weight"))?;
    let lb = get(ck, &format!("{fl}.linear.bias"))?;
    b.linear(n, &lw, Some(&lb))
}

/// `adaLN_modulation(y).chunk(n)` with `silu(y)` precomputed. Chunk 1 is a
/// `scale` that is only ever used as `1 + scale`, so the `1` is folded into
/// its bias and the node returned is `1 + scale`.
fn adaln(
    b: &mut Builder,
    ck: &Checkpoint,
    ys: NodeId,
    name: &str,
    n: usize,
    fd: usize,
) -> Result<Vec<NodeId>> {
    let w = get(ck, &format!("{name}.weight"))?;
    let bias = get(ck, &format!("{name}.bias"))?;
    if w.shape != [n * fd, fd] || bias.data.len() != n * fd {
        bail!("{name}: weight {:?}, expected [{}, {fd}]", w.shape, n * fd);
    }
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let wj = Tensor {
            data: w.data[j * fd * fd..(j + 1) * fd * fd].to_vec(),
            shape: vec![fd, fd],
        };
        let mut bj = bias.data[j * fd..(j + 1) * fd].to_vec();
        if j == 1 {
            bj.iter_mut().for_each(|v| *v += 1.0);
        }
        let bj = Tensor {
            data: bj,
            shape: vec![fd],
        };
        out.push(b.linear(ys, &wj, Some(&bj))?);
    }
    Ok(out)
}

/// Per sampler step, the mean of the head's time embeddings (`None` for heads
/// without time conditions). LSD embeds `(s, t) = (i/N, (i+1)/N)`, flow
/// matching `t = i/N`.
pub(crate) fn time_conditioning(
    ck: &Checkpoint,
    spec: &ModelSpec,
    steps: usize,
) -> Result<Option<Vec<Vec<f32>>>> {
    let kind = spec.flow_lm.flow.kind;
    let nconds = kind.num_time_conds();
    if nconds == 0 {
        return Ok(None);
    }
    let mut per_step = Vec::with_capacity(steps);
    for i in 0..steps {
        let times: Vec<f32> = match kind {
            FlowType::Lsd => vec![i as f32 / steps as f32, (i + 1) as f32 / steps as f32],
            _ => vec![i as f32 / steps as f32],
        };
        let mut acc: Option<Vec<f32>> = None;
        for (j, &t) in times.iter().enumerate() {
            let e = timestep_embed(ck, j, t)?;
            acc = Some(match acc {
                None => e,
                Some(a) => a.iter().zip(&e).map(|(x, y)| x + y).collect(),
            });
        }
        let n = nconds as f32;
        per_step.push(acc.unwrap().into_iter().map(|v| v / n).collect());
    }
    Ok(Some(per_step))
}

/// `TimestepEmbedder(t)` on the host, in f32.
fn timestep_embed(ck: &Checkpoint, idx: usize, t: f32) -> Result<Vec<f32>> {
    let pre = format!("flow_lm.flow_net.time_embed.{idx}");
    let freqs = get(ck, &format!("{pre}.freqs"))?.data;
    let mut emb: Vec<f32> = freqs.iter().map(|&f| (t * f).cos()).collect();
    emb.extend(freqs.iter().map(|&f| (t * f).sin()));
    let w0 = get(ck, &format!("{pre}.mlp.0.weight"))?;
    let b0 = get(ck, &format!("{pre}.mlp.0.bias"))?;
    let w2 = get(ck, &format!("{pre}.mlp.2.weight"))?;
    let b2 = get(ck, &format!("{pre}.mlp.2.bias"))?;
    let alpha = get(ck, &format!("{pre}.mlp.3.alpha"))?.data;
    let h: Vec<f32> = host_linear(&emb, &w0, &b0)?
        .into_iter()
        .map(|v| v / (1.0 + (-v).exp()))
        .collect();
    let h = host_linear(&h, &w2, &b2)?;
    // pocket_tts RMSNorm: `x * alpha * rsqrt(eps + x.var(unbiased=True))`.
    let n = h.len() as f32;
    let mean = h.iter().sum::<f32>() / n;
    let var = h.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n - 1.0);
    let r = 1.0 / (var + TIME_RMS_EPS).sqrt();
    Ok(h.iter().zip(&alpha).map(|(v, a)| v * a * r).collect())
}

fn host_linear(x: &[f32], w: &Tensor, b: &Tensor) -> Result<Vec<f32>> {
    let (out_d, in_d) = w.dims2()?;
    if x.len() != in_d {
        bail!("host linear: input {} vs weight {:?}", x.len(), w.shape);
    }
    Ok((0..out_d)
        .map(|o| {
            let row = &w.data[o * in_d..(o + 1) * in_d];
            row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + b.data[o]
        })
        .collect())
}
