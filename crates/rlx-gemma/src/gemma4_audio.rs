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

//! Gemma 4 audio tower (`model.audio_tower`) — 12-layer conformer encoder.
//!
//! Ground truth (HF `Gemma4AudioModel`): hidden 1024, 12 layers, 8 heads,
//! head_dim 128, silu, eps 1e-6, conformer block = FF → attn → light-conv →
//! FF with sandwich RMSNorms and `residual_weight=0.5` on each FF. Input is a
//! 128-bin mel spectrogram; a 2× Conv2d sub-sampler (stride 2, kernel 3,
//! channels [128, 32]) reduces it 4× in time + frequency, then an
//! `input_proj_linear` maps `(128/4)*32 = 1024 → 1024`.
//!
//! Attention is **chunked local** (chunk 12, left-context 13, right 0) with a
//! Transformer-XL **relative-position bias** (sinusoidal, `attention_logit_cap`
//! 50 tanh-softcap). The blocked-attention machinery (`_convert_to_block`,
//! `_extract_block_context`, `_rel_shift`) is reproduced with concat-padding +
//! reshape/narrow; the per-block additive mask + `softplus(per_dim_scale)` are
//! data-independent and precomputed on the host.
//!
//! Quantization: audio linears are 2-bit (`...{proj}.linear.weight` + scale),
//! handled by [`crate::qat_loader::GemmaQatLoader`]; convs, norms,
//! `relative_k_proj`, `per_dim_scale` and `output_proj` are unquantized.

use anyhow::Result;
use rlx_core::weight_loader::WeightLoader;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use std::collections::HashMap;

/// Fixed Gemma 4 audio-tower hyperparameters (from `audio_config`).
#[derive(Debug, Clone, Copy)]
pub struct AudioConfig {
    pub hidden: usize,        // 1024
    pub layers: usize,        // 12
    pub heads: usize,         // 8
    pub head_dim: usize,      // 128
    pub eps: f32,             // 1e-6
    pub chunk: usize,         // attention_chunk_size 12
    pub ctx_left: usize,      // attention_context_left 13
    pub ctx_right: usize,     // attention_context_right 0
    pub conv_kernel: usize,   // 5
    pub logit_cap: f32,       // 50.0
    pub residual_weight: f32, // 0.5
    pub out_dims: usize,      // output_proj_dims 1536
    pub sub_ch: [usize; 2],   // subsampling_conv_channels [128, 32]
    pub feature_size: usize,  // 128 mel bins
    pub invalid_logit: f32,   // attention_invalid_logits_value -1e9
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            hidden: 1024,
            layers: 12,
            heads: 8,
            head_dim: 128,
            eps: 1e-6,
            chunk: 12,
            ctx_left: 13,
            ctx_right: 0,
            conv_kernel: 5,
            logit_cap: 50.0,
            residual_weight: 0.5,
            out_dims: 1536,
            sub_ch: [128, 32],
            feature_size: 128,
            invalid_logit: -1.0e9,
        }
    }
}

impl AudioConfig {
    /// Attention context window per block: `chunk + (ctx_left-1) + ctx_right`.
    pub fn context_size(&self) -> usize {
        self.chunk + (self.ctx_left - 1) + self.ctx_right
    }
}

/// Sinusoidal relative-position table `[context_size/2 + 1, hidden]`
/// (HF `Gemma4AudioRelPositionalEncoding`): `position_ids = [ctx/2 .. 0]`
/// (descending), `inv_timescale[i] = exp(-i · ln(10000)/(H/2-1))`,
/// `emb = cat(sin(pos·inv), cos(pos·inv))`. Host-computed (model-fixed).
pub fn audio_rel_pos(cfg: &AudioConfig) -> Vec<f32> {
    let h = cfg.hidden;
    let nts = h / 2; // 512
    let ctx = cfg.context_size(); // 24
    let npos = ctx / 2 + 1; // 13
    let log_inc = (10000.0f64).ln() / ((nts - 1).max(1) as f64);
    let inv: Vec<f64> = (0..nts).map(|i| (-(i as f64) * log_inc).exp()).collect();
    let mut out = vec![0f32; npos * h];
    for p in 0..npos {
        let pos = (ctx / 2 - p) as f64; // descending: ctx/2, ..., 0
        for i in 0..nts {
            let a = pos * inv[i];
            out[p * h + i] = a.sin() as f32; // sin half
            out[p * h + nts + i] = a.cos() as f32; // cos half
        }
    }
    out
}

/// Per-block additive attention mask `[num_blocks, chunk, context]`
/// (0 = allowed, `invalid_logit` = blocked). Context index `j` maps to real
/// sequence position `block*chunk + (j - max_past)`; allowed iff that position
/// is in `[0, seq)`, lies within `[query - (ctx_left-1), query + ctx_right]`,
/// and (causal) not in the future beyond `ctx_right`. Data-independent.
pub fn audio_block_mask(cfg: &AudioConfig, seq: usize) -> (Vec<f32>, usize) {
    let chunk = cfg.chunk;
    let ctx = cfg.context_size();
    let max_past = cfg.ctx_left - 1;
    let nb = seq.div_ceil(chunk);
    let mut m = vec![cfg.invalid_logit; nb * chunk * ctx];
    for b in 0..nb {
        for qi in 0..chunk {
            let qpos = b * chunk + qi; // query sequence position
            for j in 0..ctx {
                let kpos_i = (b * chunk) as isize + (j as isize - max_past as isize);
                if kpos_i < 0 || kpos_i as usize >= seq {
                    continue; // padding / out of range
                }
                let kpos = kpos_i as usize;
                if qpos >= seq {
                    continue; // padded query row (kept blocked)
                }
                let lo = qpos as isize - max_past as isize;
                let hi = qpos as isize + cfg.ctx_right as isize;
                if (kpos as isize) >= lo && (kpos as isize) <= hi {
                    m[(b * chunk + qi) * ctx + j] = 0.0;
                }
            }
        }
    }
    (m, nb)
}

// ── weight helpers (shared with the vision tower's conventions) ─────
fn load_t(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut dyn WeightLoader,
    key: &str,
) -> Result<NodeId> {
    let (data, shape) = w.take_transposed(key)?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok(id)
}
fn load_raw(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut dyn WeightLoader,
    key: &str,
) -> Result<(NodeId, Vec<usize>)> {
    let (data, shape) = w.take(key)?;
    let id = g.param(key, Shape::new(&shape, DType::F32));
    params.insert(key.to_string(), data);
    Ok((id, shape))
}
fn synth(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: &[usize],
) -> NodeId {
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), data);
    id
}

/// Norm gamma = the actual learned weight `w` (Gemma4 RMSNorm/LayerNorm scale
/// by `w` directly). `GemmaQatLoader::take_float` only subtracts 1 from keys
/// ending in `norm.weight` (its delta-gamma bridge), so for those we re-add 1
/// (`gamma = 1 + (w-1) = w`); for the layer-level norms whose keys end in
/// `_attn.weight` / `_out.weight` the loader returns raw `w`, so use it as-is.
fn norm_gamma(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut dyn WeightLoader,
    key: &str,
    dim: usize,
) -> Result<(NodeId, NodeId)> {
    let (wv, _) = load_raw(g, params, w, key)?;
    let gamma = if key.ends_with("norm.weight") {
        let ones = synth(g, params, &format!("{key}.ones"), vec![1.0f32; dim], &[dim]);
        g.add(ones, wv)
    } else {
        wv
    };
    let beta = synth(g, params, &format!("{key}.beta"), vec![0.0f32; dim], &[dim]);
    Ok((gamma, beta))
}

/// Sub-sample conv projection: mel `[B,T,F]` → `[B, T/4, hidden]`.
/// Two Conv2d(stride 2,kernel 3,pad 1) + channel-LayerNorm(no bias) + ReLU,
/// then permute/reshape `[B, T/4, (F/4)*ch1] → input_proj_linear → hidden`.
fn build_subsample(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &AudioConfig,
    w: &mut dyn WeightLoader,
    batch: usize,
    t: usize,
) -> Result<(NodeId, usize)> {
    let f = DType::F32;
    let feats = g.input("audio_feats", Shape::new(&[batch, t, cfg.feature_size], f));
    // [B,T,F] → [B,1,T,F] (NCHW).
    let mut hs = g.reshape_(
        feats,
        vec![batch as i64, 1, t as i64, cfg.feature_size as i64],
    );
    let pfx = "model.audio_tower.subsample_conv_projection";
    let mut cur_t = t;
    let mut cur_f = cfg.feature_size;
    for (li, &out_ch) in cfg.sub_ch.iter().enumerate() {
        let (cw, _) = load_raw(g, params, w, &format!("{pfx}.layer{li}.conv.weight"))?;
        hs = g.conv2d(hs, cw, [3, 3], [2, 2], [1, 1], [1, 1], 1);
        // out spatial: floor((n+2*1-3)/2)+1 = floor((n-1)/2)+1
        cur_t = (cur_t - 1) / 2 + 1;
        cur_f = (cur_f - 1) / 2 + 1;
        // channel LayerNorm (no bias): [B,C,H,W] → [B,H,W,C] → ln(C) → back.
        let (gamma, beta) = norm_gamma(
            g,
            params,
            w,
            &format!("{pfx}.layer{li}.norm.weight"),
            out_ch,
        )?;
        let perm = g.transpose_(hs, vec![0, 2, 3, 1]); // [B,H,W,C]
        let normed = g.ln(perm, gamma, beta, cfg.eps);
        let back = g.transpose_(normed, vec![0, 3, 1, 2]); // [B,C,H,W]
        hs = g.relu(back);
    }
    // [B,C,H,W] → [B,H,W,C] → [B,H, W*C]; HF reshape is W-major, C-minor.
    let perm = g.transpose_(hs, vec![0, 2, 3, 1]); // [B,T/4,F/4,ch1]
    let proj_in = cur_f * cfg.sub_ch[1];
    let flat = g.reshape_(perm, vec![batch as i64, cur_t as i64, proj_in as i64]);
    let ipw = load_t(g, params, w, &format!("{pfx}.input_proj_linear.weight"))?;
    let out = g.mm(flat, ipw); // [B, T/4, hidden]
    Ok((out, cur_t))
}

/// RMSNorm by builder-side key (`gamma = 1 + (w-1)`), normalizing last dim.
fn rms(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut dyn WeightLoader,
    x: NodeId,
    key: &str,
    dim: usize,
    eps: f32,
) -> Result<NodeId> {
    let (gamma, beta) = norm_gamma(g, params, w, key, dim)?;
    Ok(g.rms_norm(x, gamma, beta, eps))
}

/// Conformer feed-forward: residual + 0.5·post_norm(ffw2(silu(ffw1(pre_norm(x))))).
fn feed_forward(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &AudioConfig,
    w: &mut dyn WeightLoader,
    x: NodeId,
    pfx: &str,
) -> Result<NodeId> {
    let h = cfg.hidden;
    let normed = rms(
        g,
        params,
        w,
        x,
        &format!("{pfx}.pre_layer_norm.weight"),
        h,
        cfg.eps,
    )?;
    let w1 = load_t(g, params, w, &format!("{pfx}.ffw_layer_1.linear.weight"))?;
    let a = g.mm(normed, w1);
    let a = g.silu(a);
    let w2 = load_t(g, params, w, &format!("{pfx}.ffw_layer_2.linear.weight"))?;
    let b = g.mm(a, w2);
    let b = rms(
        g,
        params,
        w,
        b,
        &format!("{pfx}.post_layer_norm.weight"),
        h,
        cfg.eps,
    )?;
    let scale = synth(
        g,
        params,
        &format!("{pfx}.rw"),
        vec![cfg.residual_weight],
        &[1],
    );
    let bs = g.mul(b, scale);
    Ok(g.add(x, bs))
}

/// Light depthwise-conv module: residual + linear_end(silu(conv_norm(
/// depthwise_causal_conv(glu(linear_start(pre_norm(x))))))).
fn light_conv(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &AudioConfig,
    w: &mut dyn WeightLoader,
    x: NodeId,
    pfx: &str,
    batch: usize,
    seq: usize,
) -> Result<NodeId> {
    let h = cfg.hidden;
    let normed = rms(
        g,
        params,
        w,
        x,
        &format!("{pfx}.pre_layer_norm.weight"),
        h,
        cfg.eps,
    )?;
    let ws = load_t(g, params, w, &format!("{pfx}.linear_start.linear.weight"))?;
    let st = g.mm(normed, ws); // [B,seq,2h]
    // GLU(dim=-1): a · sigmoid(b)
    let a = g.narrow_(st, 2, 0, h);
    let bgate = g.narrow_(st, 2, h, h);
    let sig = g.add_node(
        rlx_ir::op::Op::Activation(rlx_ir::op::Activation::Sigmoid),
        vec![bgate],
        Shape::new(&[batch, seq, h], DType::F32),
    );
    let glu = g.mul(a, sig); // [B,seq,h]
    // depthwise causal conv1d (kernel K, groups=h, left-pad K-1) via conv2d H=1.
    let k = cfg.conv_kernel;
    let chw = g.transpose_(glu, vec![0, 2, 1]); // [B,h,seq]
    let c4 = g.reshape_(chw, vec![batch as i64, h as i64, 1, seq as i64]); // [B,h,1,seq]
    let pad = synth(
        g,
        params,
        &format!("{pfx}.lpad"),
        vec![0.0; batch * h * (k - 1)],
        &[batch, h, 1, k - 1],
    );
    let padded = g.concat_(vec![pad, c4], 3); // [B,h,1,seq+k-1]
    let (cw, cwsh) = load_raw(g, params, w, &format!("{pfx}.depthwise_conv1d.weight"))?; // [h,1,k]
    debug_assert_eq!(cwsh, vec![h, 1, k]);
    let cw4 = g.reshape_(cw, vec![h as i64, 1, 1, k as i64]); // [h,1,1,k]
    let conv = g.conv2d(padded, cw4, [1, k], [1, 1], [0, 0], [1, 1], h); // [B,h,1,seq]
    let conv = g.reshape_(conv, vec![batch as i64, h as i64, seq as i64]);
    let conv = g.transpose_(conv, vec![0, 2, 1]); // [B,seq,h]
    let conv = rms(
        g,
        params,
        w,
        conv,
        &format!("{pfx}.conv_norm.weight"),
        h,
        cfg.eps,
    )?;
    let conv = g.silu(conv);
    let we = load_t(g, params, w, &format!("{pfx}.linear_end.linear.weight"))?;
    let end = g.mm(conv, we); // [B,seq,h]
    Ok(g.add(x, end))
}

/// `softplus(per_dim_scale) · q_scale` folded host-side into a `[head_dim]`
/// param (q_scale = `head_dim^-0.5 / ln 2`).
fn qscale_vec(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &AudioConfig,
    w: &mut dyn WeightLoader,
    pfx: &str,
) -> Result<NodeId> {
    let (pds, sh) = w.take(&format!("{pfx}.per_dim_scale"))?;
    debug_assert_eq!(sh, vec![cfg.head_dim]);
    let q_scale = (cfg.head_dim as f32).powf(-0.5) / std::f32::consts::LN_2;
    let v: Vec<f32> = pds
        .iter()
        .map(|&x| {
            // softplus(x) = ln(1 + e^x), numerically stable.
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            sp * q_scale
        })
        .collect();
    Ok(synth(
        g,
        params,
        &format!("{pfx}.qscale"),
        v,
        &[cfg.head_dim],
    ))
}

/// Chunked local attention with Transformer-XL relative-position bias.
/// `rel_pos` `[npos, h]`, `mask` `[nb, chunk, ctx]` (additive) are graph inputs.
#[allow(clippy::too_many_arguments)]
fn audio_attention(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &AudioConfig,
    w: &mut dyn WeightLoader,
    x: NodeId,
    pfx: &str,
    rel_pos: NodeId,
    mask: NodeId,
    batch: usize,
    seq: usize,
) -> Result<NodeId> {
    let (h, nh, hd) = (cfg.hidden, cfg.heads, cfg.head_dim);
    let chunk = cfg.chunk;
    let ctx = cfg.context_size();
    let max_past = cfg.ctx_left - 1;
    let right_pad = cfg.ctx_right + chunk - 1;
    let nb = seq.div_ceil(chunk);
    let npos = ctx / 2 + 1;
    let f = DType::F32;

    let qw = load_t(g, params, w, &format!("{pfx}.q_proj.linear.weight"))?;
    let kw = load_t(g, params, w, &format!("{pfx}.k_proj.linear.weight"))?;
    let vw = load_t(g, params, w, &format!("{pfx}.v_proj.linear.weight"))?;
    let q = g.mm(x, qw);
    let k = g.mm(x, kw);
    let v = g.mm(x, vw);
    let q = g.reshape_(q, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
    let k = g.reshape_(k, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
    let v = g.reshape_(v, vec![batch as i64, seq as i64, nh as i64, hd as i64]);
    // q *= q_scale · softplus(per_dim_scale)  [hd];  k *= k_scale (scalar)
    let qsc = qscale_vec(g, params, cfg, w, pfx)?;
    let q = g.mul(q, qsc);
    let k_scale = (1.0f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
    let ksc = synth(g, params, &format!("{pfx}.kscale"), vec![k_scale], &[1]);
    let k = g.mul(k, ksc);

    // block q: pad seq → nb*chunk, reshape [B,nb,chunk,nh,hd]
    let qpad_n = nb * chunk - seq;
    let qb = if qpad_n > 0 {
        let z = synth(
            g,
            params,
            &format!("{pfx}.qpad"),
            vec![0.0; batch * qpad_n * nh * hd],
            &[batch, qpad_n, nh, hd],
        );
        g.concat_(vec![q, z], 1)
    } else {
        q
    };
    let qb = g.reshape_(
        qb,
        vec![batch as i64, nb as i64, chunk as i64, nh as i64, hd as i64],
    );

    // context for k,v: pad left max_past, right right_pad; per-block window of ctx.
    let zl = synth(
        g,
        params,
        &format!("{pfx}.kl"),
        vec![0.0; batch * max_past * nh * hd],
        &[batch, max_past, nh, hd],
    );
    let zr = synth(
        g,
        params,
        &format!("{pfx}.kr"),
        vec![0.0; batch * right_pad * nh * hd],
        &[batch, right_pad, nh, hd],
    );
    let kpad = g.concat_(vec![zl, k, zr], 1);
    let zl2 = synth(
        g,
        params,
        &format!("{pfx}.vl"),
        vec![0.0; batch * max_past * nh * hd],
        &[batch, max_past, nh, hd],
    );
    let zr2 = synth(
        g,
        params,
        &format!("{pfx}.vr"),
        vec![0.0; batch * right_pad * nh * hd],
        &[batch, right_pad, nh, hd],
    );
    let vpad = g.concat_(vec![zl2, v, zr2], 1);
    let mut kblocks = Vec::with_capacity(nb);
    let mut vblocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let kb = g.narrow_(kpad, 1, b * chunk, ctx);
        let kb = g.reshape_(kb, vec![batch as i64, 1, ctx as i64, nh as i64, hd as i64]);
        kblocks.push(kb);
        let vb = g.narrow_(vpad, 1, b * chunk, ctx);
        let vb = g.reshape_(vb, vec![batch as i64, 1, ctx as i64, nh as i64, hd as i64]);
        vblocks.push(vb);
    }
    let kctx = if nb > 1 {
        g.concat_(kblocks, 1)
    } else {
        kblocks.pop().unwrap()
    };
    let vctx = if nb > 1 {
        g.concat_(vblocks, 1)
    } else {
        vblocks.pop().unwrap()
    };
    // [B,nb,ctx,nh,hd]

    // queries [B,nh,nb,chunk,hd]
    let queries = g.transpose_(qb, vec![0, 3, 1, 2, 4]);
    // matrix_ac = queries @ kctx.permute(0,3,1,4,2) → [B,nh,nb,chunk,ctx]
    let kp = g.transpose_(kctx, vec![0, 3, 1, 4, 2]); // [B,nh,nb,hd,ctx]
    let ac = g.mm(queries, kp);

    // relative key: relative_k_proj(rel_pos) → [npos,nh,hd]
    let relw = load_t(g, params, w, &format!("{pfx}.relative_k_proj.weight"))?;
    let relk = g.mm(rel_pos, relw); // [npos, h]
    let relk = g.reshape_(relk, vec![npos as i64, nh as i64, hd as i64]);
    let relk = g.transpose_(relk, vec![1, 2, 0]); // [nh,hd,npos]
    let qflat = g.reshape_(
        queries,
        vec![batch as i64, nh as i64, (nb * chunk) as i64, hd as i64],
    );
    let bd = g.mm(qflat, relk); // [B,nh,nb*chunk,npos]
    let bd = g.reshape_(
        bd,
        vec![
            batch as i64,
            nh as i64,
            nb as i64,
            chunk as i64,
            npos as i64,
        ],
    );
    // rel_shift: pad last → ctx+1, reshape, drop, reshape → [B,nh,nb,chunk,ctx]
    let padn = ctx + 1 - npos;
    let zpad = synth(
        g,
        params,
        &format!("{pfx}.relshift"),
        vec![0.0; batch * nh * nb * chunk * padn],
        &[batch, nh, nb, chunk, padn],
    );
    let bdp = g.concat_(vec![bd, zpad], 4); // [.,chunk,ctx+1]
    let bdp = g.reshape_(
        bdp,
        vec![
            batch as i64,
            nh as i64,
            nb as i64,
            (chunk * (ctx + 1)) as i64,
        ],
    );
    let bdp = g.narrow_(bdp, 3, 0, chunk * ctx);
    let bd = g.reshape_(
        bdp,
        vec![batch as i64, nh as i64, nb as i64, chunk as i64, ctx as i64],
    );

    let mut attn = g.add(ac, bd);
    // tanh softcap
    let cap = synth(g, params, &format!("{pfx}.cap"), vec![cfg.logit_cap], &[1]);
    let icap = synth(
        g,
        params,
        &format!("{pfx}.icap"),
        vec![1.0 / cfg.logit_cap],
        &[1],
    );
    attn = g.mul(attn, icap);
    attn = g.tanh(attn);
    attn = g.mul(attn, cap);
    // + additive mask [nb,chunk,ctx] → [1,1,nb,chunk,ctx]
    let mask5 = g.reshape_(mask, vec![1, 1, nb as i64, chunk as i64, ctx as i64]);
    attn = g.add(attn, mask5);
    let _ = f;
    attn = g.sm(attn, -1);

    // attn_output = attn @ vctx.permute(0,3,1,2,4) → [B,nh,nb,chunk,hd]
    let vp = g.transpose_(vctx, vec![0, 3, 1, 2, 4]); // [B,nh,nb,ctx,hd]
    let o = g.mm(attn, vp); // [B,nh,nb,chunk,hd]
    let o = g.transpose_(o, vec![0, 2, 3, 1, 4]); // [B,nb,chunk,nh,hd]
    let o = g.reshape_(o, vec![batch as i64, (nb * chunk) as i64, h as i64]);
    let o = g.narrow_(o, 1, 0, seq); // drop block padding
    let pw = load_t(g, params, w, &format!("{pfx}.post.linear.weight"))?;
    Ok(g.mm(o, pw))
}

/// One conformer block: FF1 → (norm_pre_attn → attn → norm_post_attn + res) →
/// light_conv → FF2 → norm_out.
#[allow(clippy::too_many_arguments)]
fn conformer_layer(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &AudioConfig,
    w: &mut dyn WeightLoader,
    x: NodeId,
    li: usize,
    rel_pos: NodeId,
    mask: NodeId,
    batch: usize,
    seq: usize,
) -> Result<NodeId> {
    let h = cfg.hidden;
    let lp = format!("model.audio_tower.layers.{li}");
    let hs = feed_forward(g, params, cfg, w, x, &format!("{lp}.feed_forward1"))?;
    let residual = hs;
    let normed = rms(
        g,
        params,
        w,
        hs,
        &format!("{lp}.norm_pre_attn.weight"),
        h,
        cfg.eps,
    )?;
    let att = audio_attention(
        g,
        params,
        cfg,
        w,
        normed,
        &format!("{lp}.self_attn"),
        rel_pos,
        mask,
        batch,
        seq,
    )?;
    let att = rms(
        g,
        params,
        w,
        att,
        &format!("{lp}.norm_post_attn.weight"),
        h,
        cfg.eps,
    )?;
    let hs = g.add(residual, att);
    let hs = light_conv(g, params, cfg, w, hs, &format!("{lp}.lconv1d"), batch, seq)?;
    let hs = feed_forward(g, params, cfg, w, hs, &format!("{lp}.feed_forward2"))?;
    rms(
        g,
        params,
        w,
        hs,
        &format!("{lp}.norm_out.weight"),
        h,
        cfg.eps,
    )
}

/// Build the audio tower: subsample → `n_layers` conformer blocks →
/// (optional) `output_proj`. Inputs: `audio_feats` `[B,T,128]`,
/// `audio_rel_pos` `[npos,h]`, `audio_mask` `[nb,chunk,ctx]`.
/// Output: `[B, T/4, hidden]` (or `out_dims` with `output_proj`).
pub fn build_audio_tower(
    cfg: &AudioConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    t: usize,
    n_layers: usize,
    with_output_proj: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, usize)> {
    let mut g = Graph::new("gemma4_audio");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let (mut hs, seq) = build_subsample(&mut g, &mut params, cfg, weights, batch, t)?;
    let npos = cfg.context_size() / 2 + 1;
    let rel_pos = g.input("audio_rel_pos", Shape::new(&[npos, cfg.hidden], DType::F32));
    let nb = seq.div_ceil(cfg.chunk);
    let mask = g.input(
        "audio_mask",
        Shape::new(&[nb, cfg.chunk, cfg.context_size()], DType::F32),
    );
    for li in 0..n_layers {
        hs = conformer_layer(
            &mut g,
            &mut params,
            cfg,
            weights,
            hs,
            li,
            rel_pos,
            mask,
            batch,
            seq,
        )?;
    }
    if with_output_proj {
        let ow = load_t(
            &mut g,
            &mut params,
            weights,
            "model.audio_tower.output_proj.weight",
        )?;
        let mut o = g.mm(hs, ow);
        let (ob, _) = load_raw(
            &mut g,
            &mut params,
            weights,
            "model.audio_tower.output_proj.bias",
        )?;
        o = g.add(o, ob);
        hs = o;
    }
    g.set_outputs(vec![hs]);
    Ok((g, params, seq))
}

/// Debug: isolate one layer's attention. Input `attn_in` `[B,seq,h]` (already
/// `norm_pre_attn`-normed, as HF feeds `self_attn`); plus `audio_rel_pos` +
/// `audio_mask`. Output: attention post-proj `[B,seq,h]`.
pub fn build_attention_test(
    cfg: &AudioConfig,
    weights: &mut dyn WeightLoader,
    layer: usize,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("gemma4_audio_attntest");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let x = g.input("attn_in", Shape::new(&[batch, seq, cfg.hidden], DType::F32));
    let npos = cfg.context_size() / 2 + 1;
    let rel_pos = g.input("audio_rel_pos", Shape::new(&[npos, cfg.hidden], DType::F32));
    let nb = seq.div_ceil(cfg.chunk);
    let mask = g.input(
        "audio_mask",
        Shape::new(&[nb, cfg.chunk, cfg.context_size()], DType::F32),
    );
    let pfx = format!("model.audio_tower.layers.{layer}.self_attn");
    let o = audio_attention(
        &mut g,
        &mut params,
        cfg,
        weights,
        x,
        &pfx,
        rel_pos,
        mask,
        batch,
        seq,
    )?;
    g.set_outputs(vec![o]);
    Ok((g, params))
}

/// Debug: isolate one layer's depthwise causal conv. Input `dw_in` `[B,h,seq]`
/// (already transposed, as HF feeds `Conv1d`); output `[B,h,seq]`.
pub fn build_depthwise_test(
    cfg: &AudioConfig,
    weights: &mut dyn WeightLoader,
    layer: usize,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("gemma4_audio_dwtest");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let h = cfg.hidden;
    let k = cfg.conv_kernel;
    let chw = g.input("dw_in", Shape::new(&[batch, h, seq], DType::F32)); // [B,h,seq]
    let c4 = g.reshape_(chw, vec![batch as i64, h as i64, 1, seq as i64]);
    let pad = synth(
        &mut g,
        &mut params,
        "lpad",
        vec![0.0; batch * h * (k - 1)],
        &[batch, h, 1, k - 1],
    );
    let padded = g.concat_(vec![pad, c4], 3);
    let pfx = format!("model.audio_tower.layers.{layer}.lconv1d");
    let (cw, _) = load_raw(
        &mut g,
        &mut params,
        weights,
        &format!("{pfx}.depthwise_conv1d.weight"),
    )?;
    let cw4 = g.reshape_(cw, vec![h as i64, 1, 1, k as i64]);
    let conv = g.conv2d(padded, cw4, [1, k], [1, 1], [0, 0], [1, 1], h);
    let out = g.reshape_(conv, vec![batch as i64, h as i64, seq as i64]);
    g.set_outputs(vec![out]);
    Ok((g, params))
}

/// Debug: build subsample → layer-0 components, exposing intermediate taps as
/// outputs `[ff1, attn(post-proj), lconv, ff2, norm_out]` (each `[B,seq,hidden]`).
pub fn build_audio_layer0_debug(
    cfg: &AudioConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    t: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, usize)> {
    let mut g = Graph::new("gemma4_audio_l0dbg");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let (x, seq) = build_subsample(&mut g, &mut params, cfg, weights, batch, t)?;
    let npos = cfg.context_size() / 2 + 1;
    let rel_pos = g.input("audio_rel_pos", Shape::new(&[npos, cfg.hidden], DType::F32));
    let nb = seq.div_ceil(cfg.chunk);
    let mask = g.input(
        "audio_mask",
        Shape::new(&[nb, cfg.chunk, cfg.context_size()], DType::F32),
    );
    let h = cfg.eps;
    let _ = h;
    let lp = "model.audio_tower.layers.0";
    let ff1 = feed_forward(
        &mut g,
        &mut params,
        cfg,
        weights,
        x,
        &format!("{lp}.feed_forward1"),
    )?;
    let normed = rms(
        &mut g,
        &mut params,
        weights,
        ff1,
        &format!("{lp}.norm_pre_attn.weight"),
        cfg.hidden,
        cfg.eps,
    )?;
    let att = audio_attention(
        &mut g,
        &mut params,
        cfg,
        weights,
        normed,
        &format!("{lp}.self_attn"),
        rel_pos,
        mask,
        batch,
        seq,
    )?;
    let attn_norm = rms(
        &mut g,
        &mut params,
        weights,
        att,
        &format!("{lp}.norm_post_attn.weight"),
        cfg.hidden,
        cfg.eps,
    )?;
    let hs = g.add(ff1, attn_norm);
    let lc = light_conv(
        &mut g,
        &mut params,
        cfg,
        weights,
        hs,
        &format!("{lp}.lconv1d"),
        batch,
        seq,
    )?;
    let ff2 = feed_forward(
        &mut g,
        &mut params,
        cfg,
        weights,
        lc,
        &format!("{lp}.feed_forward2"),
    )?;
    let nout = rms(
        &mut g,
        &mut params,
        weights,
        ff2,
        &format!("{lp}.norm_out.weight"),
        cfg.hidden,
        cfg.eps,
    )?;
    g.set_outputs(vec![ff1, att, lc, ff2, nout, normed]);
    Ok((g, params, seq))
}

/// Full audio feature path: audio tower (→ `[B,seq,out_dims]`) → `embed_audio`
/// (pre-projection RMSNorm no-scale → `embedding_projection` `out_dims→out_dims`)
/// → soft tokens `[B, seq, out_dims]` that splice into the LM at audio
/// placeholders. Mirrors `Gemma4Model.embed_audio(audio_tower(...))`.
pub fn build_audio_features(
    cfg: &AudioConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    t: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, usize)> {
    let mut g = Graph::new("gemma4_audio_features");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let (mut hs, seq) = build_subsample(&mut g, &mut params, cfg, weights, batch, t)?;
    let npos = cfg.context_size() / 2 + 1;
    let rel_pos = g.input("audio_rel_pos", Shape::new(&[npos, cfg.hidden], DType::F32));
    let nb = seq.div_ceil(cfg.chunk);
    let mask = g.input(
        "audio_mask",
        Shape::new(&[nb, cfg.chunk, cfg.context_size()], DType::F32),
    );
    for li in 0..cfg.layers {
        hs = conformer_layer(
            &mut g,
            &mut params,
            cfg,
            weights,
            hs,
            li,
            rel_pos,
            mask,
            batch,
            seq,
        )?;
    }
    // output_proj (hidden → out_dims) + bias
    let ow = load_t(
        &mut g,
        &mut params,
        weights,
        "model.audio_tower.output_proj.weight",
    )?;
    let mut o = g.mm(hs, ow);
    let (ob, _) = load_raw(
        &mut g,
        &mut params,
        weights,
        "model.audio_tower.output_proj.bias",
    )?;
    o = g.add(o, ob);
    // embed_audio: pre-projection RMSNorm (no scale) → linear out_dims→out_dims.
    let gamma = synth(
        &mut g,
        &mut params,
        "embed_audio.pre.ones",
        vec![1.0f32; cfg.out_dims],
        &[cfg.out_dims],
    );
    let beta = synth(
        &mut g,
        &mut params,
        "embed_audio.pre.beta",
        vec![0.0f32; cfg.out_dims],
        &[cfg.out_dims],
    );
    let normed = g.rms_norm(o, gamma, beta, cfg.eps);
    let pw = load_t(
        &mut g,
        &mut params,
        weights,
        "model.embed_audio.embedding_projection.weight",
    )?;
    let feats = g.mm(normed, pw);
    g.set_outputs(vec![feats]);
    Ok((g, params, seq))
}

/// Build just the sub-sample projection graph (parity gate for conv+LN path).
/// Output: `[B, T/4, hidden]`.
pub fn build_audio_subsample(
    cfg: &AudioConfig,
    weights: &mut dyn WeightLoader,
    batch: usize,
    t: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, usize)> {
    let mut g = Graph::new("gemma4_audio_subsample");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let (out, seq) = build_subsample(&mut g, &mut params, cfg, weights, batch, t)?;
    g.set_outputs(vec![out]);
    Ok((g, params, seq))
}
