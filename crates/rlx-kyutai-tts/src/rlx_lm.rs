//! Native RLX graphs for the Kyutai TTS temporal backbone (Helium + per-layer cross-attn).

use crate::config::KyutaiTtsConfig;
use crate::nn::sin_pos_embed;
use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

const RMS_EPS: f32 = 1e-8;

/// Static dims of the Kyutai TTS temporal transformer.
#[derive(Debug, Clone, Copy)]
pub struct TtsDims {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub ffn: usize,
    pub vocab_out: usize,
    pub rope_theta: f32,
    /// Cross-attention context length (speaker sequence frames).
    pub t_cross: usize,
}

impl TtsDims {
    pub fn from_cfg(cfg: &KyutaiTtsConfig, t_cross: usize) -> Self {
        let t = cfg.backbone_runtime();
        Self {
            d_model: t.d_model,
            n_heads: t.num_heads,
            head_dim: t.d_model / t.num_heads,
            n_layers: t.num_layers,
            ffn: t.dim_feedforward / 2,
            vocab_out: cfg.text_card,
            rope_theta: t.max_period as f32,
            t_cross,
        }
    }
}

fn p(li: usize, name: &str) -> String {
    format!("L{li}.{name}")
}

/// Interleaved RoPE — matches eager [`crate::nn::apply_rope_interleaved`] / Moshi `interleave=True`.
fn apply_rope_interleaved_g(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    nh: i64,
    half: usize,
    rotr: rlx_ir::NodeId,
    roti: rlx_ir::NodeId,
) -> rlx_ir::NodeId {
    let hd = (half * 2) as i64;
    let x5 = g.reshape_(x, vec![1, 1, nh, half as i64, 2]);
    let xr_n = g.narrow_(x5, 4, 0, 1);
    let xi_n = g.narrow_(x5, 4, 1, 1);
    let xr = g.reshape_(xr_n, vec![1, 1, nh, half as i64]);
    let xi = g.reshape_(xi_n, vec![1, 1, nh, half as i64]);
    let xrc = g.mul(xr, rotr);
    let xis = g.mul(xi, roti);
    let xro = g.sub(xrc, xis);
    let xrs = g.mul(xr, roti);
    let xic = g.mul(xi, rotr);
    let xio = g.add(xrs, xic);
    let xro5 = g.reshape_(xro, vec![1, 1, nh, half as i64, 1]);
    let xio5 = g.reshape_(xio, vec![1, 1, nh, half as i64, 1]);
    let stacked = g.concat_(vec![xro5, xio5], 4);
    g.reshape_(stacked, vec![1, 1, nh, hd])
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    out.par_chunks_mut(rows)
        .enumerate()
        .for_each(|(c, out_row)| {
            for (r, slot) in out_row.iter_mut().enumerate() {
                *slot = data[r * cols + c];
            }
        });
    out
}

fn squeeze_norm_alpha(data: &[f32], shape: &[usize], d: usize) -> Vec<f32> {
    if shape.len() == 3 && shape[0] == 1 && shape[1] == 1 && shape[2] == d {
        data[..d].to_vec()
    } else {
        data.to_vec()
    }
}

fn rms(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    name: &str,
    d: usize,
    shape: Shape,
    zero_beta: rlx_ir::NodeId,
) -> rlx_ir::NodeId {
    let w = g.param(name, Shape::new(&[d], DType::F32));
    g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: RMS_EPS,
        },
        vec![x, w, zero_beta],
        shape,
    )
}

fn qkv_proj(
    g: &mut Graph,
    n1: rlx_ir::NodeId,
    li: usize,
    d: usize,
    heads: &[i64],
) -> (rlx_ir::NodeId, rlx_ir::NodeId, rlx_ir::NodeId) {
    let qw = g.param(p(li, "q"), Shape::new(&[d, d], DType::F32));
    let kw = g.param(p(li, "k"), Shape::new(&[d, d], DType::F32));
    let vw = g.param(p(li, "v"), Shape::new(&[d, d], DType::F32));
    let q = g.mm(n1, qw);
    let k = g.mm(n1, kw);
    let v = g.mm(n1, vw);
    let q = g.reshape_(q, heads.to_vec());
    let k = g.reshape_(k, heads.to_vec());
    let v = g.reshape_(v, heads.to_vec());
    (q, k, v)
}

fn cross_attn_block(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    cross_ctx: rlx_ir::NodeId,
    li: usize,
    dims: &TtsDims,
    _p1d: &Shape,
    heads1: &[i64],
) -> rlx_ir::NodeId {
    let TtsDims {
        d_model: d,
        n_heads: nh,
        head_dim: hd,
        t_cross,
        ..
    } = *dims;
    let cx_nw = g.param(p(li, "cx_nw"), Shape::new(&[d], DType::F32));
    let cx_nb = g.param(p(li, "cx_nb"), Shape::new(&[d], DType::F32));
    let n_cx = g.ln(x, cx_nw, cx_nb, RMS_EPS);

    let cx_qw = g.param(p(li, "cx_q"), Shape::new(&[d, d], DType::F32));
    let cx_kw = g.param(p(li, "cx_k"), Shape::new(&[d, d], DType::F32));
    let cx_vw = g.param(p(li, "cx_v"), Shape::new(&[d, d], DType::F32));

    let ncx2d = g.reshape_(n_cx, vec![1, d as i64]);
    let q2d = g.mm(ncx2d, cx_qw);
    let q = g.reshape_(q2d, heads1.to_vec());

    let ctx2d = g.reshape_(cross_ctx, vec![t_cross as i64, d as i64]);
    let k2d = g.mm(ctx2d, cx_kw);
    let v2d = g.mm(ctx2d, cx_vw);
    let cx_heads = [1i64, t_cross as i64, nh as i64, hd as i64];
    let k = g.reshape_(k2d, cx_heads.to_vec());
    let v = g.reshape_(v2d, cx_heads.to_vec());

    let attn = g.attention_kind(
        q,
        k,
        v,
        nh,
        hd,
        MaskKind::None,
        Shape::new(&[1, 1, nh, hd], DType::F32),
    );
    let attn = g.reshape_(attn, vec![1i64, 1, d as i64]);
    let cx_ow = g.param(p(li, "cx_o"), Shape::new(&[d, d], DType::F32));
    let cx_out = g.mm(attn, cx_ow);
    g.add(x, cx_out)
}

fn swiglu_block(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    li: usize,
    d: usize,
    ffn: usize,
    shape: Shape,
    zero_beta: rlx_ir::NodeId,
) -> rlx_ir::NodeId {
    let n2 = rms(g, x, &p(li, "n2"), d, shape, zero_beta);
    let gw = g.param(p(li, "gate"), Shape::new(&[d, ffn], DType::F32));
    let uw = g.param(p(li, "up"), Shape::new(&[d, ffn], DType::F32));
    let gate = g.mm(n2, gw);
    let up = g.mm(n2, uw);
    let gate = g.silu(gate);
    let h = g.mul(gate, up);
    let dw = g.param(p(li, "down"), Shape::new(&[ffn, d], DType::F32));
    let mlp = g.mm(h, dw);
    g.add(x, mlp)
}

fn for_each_transformer_param(
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    d: usize,
    ffn: usize,
    n_layers: usize,
    mut emit: impl FnMut(String, Vec<f32>),
) -> Result<()> {
    let get = |k: &str| -> Result<&Vec<f32>> {
        weights
            .get(k)
            .map(|(v, _)| v)
            .with_context(|| format!("missing weight {k}"))
    };
    let shape_of = |k: &str| -> Result<&Vec<usize>> {
        weights
            .get(k)
            .map(|(_, s)| s)
            .with_context(|| format!("missing weight {k}"))
    };
    for li in 0..n_layers {
        let pre = format!("transformer.layers.{li}");
        let inproj = get(&format!("{pre}.self_attn.in_proj_weight"))?;
        emit(p(li, "q"), transpose(&inproj[0..d * d], d, d));
        emit(p(li, "k"), transpose(&inproj[d * d..2 * d * d], d, d));
        emit(p(li, "v"), transpose(&inproj[2 * d * d..3 * d * d], d, d));
        emit(
            p(li, "o"),
            transpose(get(&format!("{pre}.self_attn.out_proj.weight"))?, d, d),
        );
        emit(
            p(li, "n1"),
            squeeze_norm_alpha(
                get(&format!("{pre}.norm1.alpha"))?,
                shape_of(&format!("{pre}.norm1.alpha"))?,
                d,
            ),
        );
        emit(
            p(li, "n2"),
            squeeze_norm_alpha(
                get(&format!("{pre}.norm2.alpha"))?,
                shape_of(&format!("{pre}.norm2.alpha"))?,
                d,
            ),
        );
        let gate_up = get(&format!("{pre}.gating.linear_in.weight"))?;
        emit(p(li, "gate"), transpose(&gate_up[0..ffn * d], ffn, d));
        emit(
            p(li, "up"),
            transpose(&gate_up[ffn * d..2 * ffn * d], ffn, d),
        );
        emit(
            p(li, "down"),
            transpose(get(&format!("{pre}.gating.linear_out.weight"))?, d, ffn),
        );

        let cx_in = get(&format!("{pre}.cross_attention.in_proj_weight"))?;
        emit(p(li, "cx_q"), transpose(&cx_in[0..d * d], d, d));
        emit(p(li, "cx_k"), transpose(&cx_in[d * d..2 * d * d], d, d));
        emit(p(li, "cx_v"), transpose(&cx_in[2 * d * d..3 * d * d], d, d));
        emit(
            p(li, "cx_o"),
            transpose(
                get(&format!("{pre}.cross_attention.out_proj.weight"))?,
                d,
                d,
            ),
        );
        emit(
            p(li, "cx_nw"),
            get(&format!("{pre}.norm_cross.weight"))?.clone(),
        );
        emit(
            p(li, "cx_nb"),
            get(&format!("{pre}.norm_cross.bias"))?.clone(),
        );
    }
    Ok(())
}

/// Bucketed single-token decode graph with per-layer cross-attention.
pub fn build_temporal_decode_graph_bucketed(dims: &TtsDims, upper: usize) -> Graph {
    let TtsDims {
        d_model: d,
        n_heads: nh,
        head_dim: hd,
        n_layers,
        ffn,
        vocab_out,
        t_cross,
        ..
    } = *dims;
    let half = hd / 2;
    let p1d = Shape::new(&[1, 1, d], DType::F32);
    let mut g = Graph::new("kyutai_tts_temporal_decode_bucketed");

    let mut x = g.input("inputs_embeds", p1d.clone());
    let cos = g.input("rope_cos", Shape::new(&[1, half], DType::F32));
    let sin = g.input("rope_sin", Shape::new(&[1, half], DType::F32));
    let rotr = g.reshape_(cos, vec![1, 1, 1, half as i64]);
    let roti = g.reshape_(sin, vec![1, 1, 1, half as i64]);
    let mask = g.input("attn_mask", Shape::new(&[1, upper + 1], DType::F32));
    let cross_ctx = g.input("cross_ctx", Shape::new(&[1, t_cross, d], DType::F32));
    let zero_beta = g.param("zero_beta", Shape::new(&[d], DType::F32));
    let one = g.param("kv_one", Shape::new(&[1], DType::F32));
    let kv_shape = Shape::new(&[1, upper, nh, hd], DType::F32);
    let heads1 = vec![1i64, 1, nh as i64, hd as i64];
    let mut kv_outputs = Vec::with_capacity(2 * n_layers);

    for li in 0..n_layers {
        let n1 = rms(&mut g, x, &p(li, "n1"), d, p1d.clone(), zero_beta);
        let (q, k, v) = qkv_proj(&mut g, n1, li, d, &heads1);
        let q = apply_rope_interleaved_g(&mut g, q, nh as i64, half, rotr, roti);
        let k = apply_rope_interleaved_g(&mut g, k, nh as i64, half, rotr, roti);

        let new_k = g.mul(k, one);
        let new_v = g.mul(v, one);
        kv_outputs.push(new_k);
        kv_outputs.push(new_v);

        let past_k = g.input(format!("past_k_{li}"), kv_shape.clone());
        let past_v = g.input(format!("past_v_{li}"), kv_shape.clone());
        let k_full = g.concat_(vec![past_k, k], 1);
        let v_full = g.concat_(vec![past_v, v], 1);
        let attn = g.attention_(q, k_full, v_full, mask, nh, hd);
        let attn = g.reshape_(attn, vec![1i64, 1, d as i64]);
        let ow = g.param(p(li, "o"), Shape::new(&[d, d], DType::F32));
        let attn = g.mm(attn, ow);
        x = g.add(x, attn);

        x = cross_attn_block(&mut g, x, cross_ctx, li, dims, &p1d, &heads1);
        x = swiglu_block(&mut g, x, li, d, ffn, p1d.clone(), zero_beta);
    }

    let xn = rms(&mut g, x, "out_norm", d, p1d, zero_beta);
    let tl = g.param("text_linear", Shape::new(&[d, vocab_out], DType::F32));
    let logits = g.mm(xn, tl);
    let mut outs = vec![logits, xn];
    outs.extend(kv_outputs);
    g.set_outputs(outs);
    g
}

pub fn bucket_decode_mask(past_seq: usize, upper: usize) -> Vec<f32> {
    (0..=upper)
        .map(|i| if i < past_seq || i == upper { 1.0 } else { 0.0 })
        .collect()
}

pub fn set_temporal_params(
    compiled: &mut CompiledGraph,
    dims: &TtsDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    let TtsDims {
        d_model: d,
        ffn,
        n_layers,
        vocab_out,
        ..
    } = *dims;
    let get = |k: &str| -> Result<&Vec<f32>> {
        weights
            .get(k)
            .map(|(v, _)| v)
            .with_context(|| format!("missing weight {k}"))
    };
    let shape_of = |k: &str| -> Result<&Vec<usize>> {
        weights
            .get(k)
            .map(|(_, s)| s)
            .with_context(|| format!("missing weight {k}"))
    };
    compiled.set_param("zero_beta", &vec![0.0f32; d]);
    compiled.set_param("kv_one", &[1.0]);
    for_each_transformer_param(weights, d, ffn, n_layers, |name, data| {
        compiled.set_param(&name, &data);
    })?;
    compiled.set_param(
        "out_norm",
        &squeeze_norm_alpha(get("out_norm.alpha")?, shape_of("out_norm.alpha")?, d),
    );
    compiled.set_param(
        "text_linear",
        &transpose(get("text_linear.weight")?, vocab_out, d),
    );
    Ok(())
}

/// Apply sinusoidal positional embedding to the cross context (CPU), matching eager `prepare_kv`.
pub fn prepare_cross_ctx(
    ctx: &Array2<f32>,
    pos_emb: bool,
    pos_emb_scale: f32,
    max_period: f32,
) -> Vec<f32> {
    let t = ctx.nrows();
    let d = ctx.ncols();
    let mut out = if pos_emb && t > 0 {
        let pe = sin_pos_embed(t, d, max_period);
        let mut c = ctx.to_owned();
        for ti in 0..t {
            for di in 0..d {
                c[[ti, di]] += pos_emb_scale * pe[[ti, di]];
            }
        }
        c
    } else {
        ctx.clone()
    };
    out.as_slice_mut().unwrap().to_vec()
}

pub fn decode_bucketed_run(
    compiled: &mut CompiledGraph,
    dims: &TtsDims,
    inputs_embeds: &[f32],
    cross_ctx: &[f32],
    real_past_kv: &[(Vec<f32>, Vec<f32>)],
    past_seq: usize,
    upper: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
    let half = dims.head_dim / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let inv_freq = 1.0f32 / dims.rope_theta.powf(i as f32 / half as f32);
        let f = past_seq as f32 * inv_freq;
        cos[i] = f.cos();
        sin[i] = f.sin();
    }
    let mask = bucket_decode_mask(past_seq, upper);
    let kvw = dims.n_heads * dims.head_dim;
    let pad_len = upper * kvw;
    let padded: Vec<(Vec<f32>, Vec<f32>)> = (0..dims.n_layers)
        .map(|li| {
            let (rk, rv) = real_past_kv
                .get(li)
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .unwrap_or((&[], &[]));
            let mut pk = rk.to_vec();
            pk.resize(pad_len, 0.0);
            let mut pv = rv.to_vec();
            pv.resize(pad_len, 0.0);
            (pk, pv)
        })
        .collect();

    let mut inputs: Vec<(String, &[f32])> = vec![
        ("inputs_embeds".to_string(), inputs_embeds),
        ("rope_cos".to_string(), cos.as_slice()),
        ("rope_sin".to_string(), sin.as_slice()),
        ("attn_mask".to_string(), mask.as_slice()),
        ("cross_ctx".to_string(), cross_ctx),
    ];
    for (li, (k, v)) in padded.iter().enumerate() {
        inputs.push((format!("past_k_{li}"), k.as_slice()));
        inputs.push((format!("past_v_{li}"), v.as_slice()));
    }
    let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    let mut it = compiled.run(&refs).into_iter();
    let logits = it.next().context("bucketed decode produced no logits")?;
    let hidden = it.next().context("bucketed decode produced no hidden")?;
    let mut new_kv = Vec::with_capacity(dims.n_layers);
    for _ in 0..dims.n_layers {
        let k = it.next().context("missing new_k")?;
        let v = it.next().context("missing new_v")?;
        new_kv.push((k, v));
    }
    Ok((logits, hidden, new_kv))
}

pub fn temporal_decode_bucketed_rlx(
    dims: &TtsDims,
    weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    inputs_embeds: &[f32],
    cross_ctx: &[f32],
    real_past_kv: &[(Vec<f32>, Vec<f32>)],
    past_seq: usize,
    upper: usize,
    device: Device,
) -> Result<(Vec<f32>, Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
    let mut compiled =
        Session::new(device).compile(build_temporal_decode_graph_bucketed(dims, upper));
    set_temporal_params(&mut compiled, dims, weights)?;
    decode_bucketed_run(
        &mut compiled,
        dims,
        inputs_embeds,
        cross_ctx,
        real_past_kv,
        past_seq,
        upper,
    )
}
