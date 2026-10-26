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

//! The SALA forward as one compiled graph over a chunk of `T` tokens.
//!
//! Prefill and decode are the same graph: a prompt goes through in chunks of
//! `T`, and a decode step is a chunk with one real token. One graph means one
//! copy of the weights, which at 9B parameters is the constraint that matters.
//!
//! ```text
//!   x = embeds · scale_emb                              (host-side gather)
//!   per layer:  x += r · mixer(RMSNorm(x));  x += r · SwiGLU(RMSNorm(x))
//!               r = scale_depth / sqrt(num_layers)
//!   logits = lm_head(RMSNorm(x)[last] / (hidden / dim_model_base))
//! ```
//!
//! **Attention** (`minicpm4`, 25% of layers): GQA over `[cache | chunk]`, no
//! RoPE in the release, output × σ(o_gate(x)). An additive bias input masks
//! empty cache rows and makes the chunk causal. Below `dense_len` keys this
//! is exactly what InfLLM-v2 computes.
//!
//! **Lightning** (75%): simple-GLA linear attention with a fixed per-head
//! log decay `g` (`-alibi_slope`):
//!
//! ```text
//!   S_t = e^g S_{t-1} + k_tᵀ v_t        o_t = scale · q_t S_t
//! ```
//!
//! evaluated per chunk in closed form (`i`, `j` positions in the chunk, `n`
//! real tokens, `S` the state entering the chunk):
//!
//! ```text
//!   o_i   = scale·q_i · ( Σ_{j≤i} e^{g(i-j)} k_jᵀ v_j  +  e^{g(i+1)} S )
//!   S'    = e^{g n} S + Σ_{j<n} e^{g(n-1-j)} k_jᵀ v_j
//! ```
//!
//! The decay factors depend only on `(T, n, g)` and arrive as inputs
//! ([`DecayFeeds`]); q/k get a per-head RMSNorm and RoPE first, and the output
//! an RMSNorm over all heads and × σ(z_proj(x)).

use rlx_fish::nn::{Builder, ParamSpec};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, RopeStyle, Shape};

use crate::config::{Mixer, SalaConfig};
use crate::sparse::{SparseNodes, sparse_bias};

const F32: DType = DType::F32;

pub const IN_EMBEDS: &str = "embeds";
pub const IN_ROPE_COS: &str = "rope_cos";
pub const IN_ROPE_SIN: &str = "rope_sin";
pub const IN_LAST: &str = "last_idx";
pub const IN_ATTN_BIAS: &str = "attn_bias";
pub const IN_DECAY_INTRA: &str = "decay_intra";
pub const IN_DECAY_Q: &str = "decay_q";
pub const IN_DECAY_K: &str = "decay_k";
pub const IN_DECAY_S: &str = "decay_s";

pub fn past_k_input(layer: usize) -> String {
    format!("past_k.{layer}")
}
pub fn past_v_input(layer: usize) -> String {
    format!("past_v.{layer}")
}
pub fn state_input(layer: usize) -> String {
    format!("state.{layer}")
}

/// Shape of one compiled step graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepShape {
    /// Tokens per run.
    pub chunk: usize,
    /// Cache capacity of the attention layers, in tokens.
    pub cap: usize,
    /// Build InfLLM-v2's block selection into the attention layers
    /// (capacities past `dense_len`); its choice in the first attention layer
    /// is the graph's last output.
    pub sparse: bool,
}

/// What the outputs of a step graph are, in order: `logits [vocab]`, then per
/// attention layer `k_new [T·kv·hd]`, `v_new`, then per Lightning layer
/// `state_new [h·hd·hd]`, all in layer order.
#[derive(Debug, Clone)]
pub struct StepLayout {
    pub attn_layers: Vec<usize>,
    pub lightning_layers: Vec<usize>,
}

/// Build the step graph.
pub fn build_step(
    cfg: &SalaConfig,
    shape: StepShape,
    matrix_dtype: DType,
) -> (Graph, Vec<ParamSpec>, StepLayout) {
    let StepShape {
        chunk: t, cap: p, ..
    } = shape;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let eps = cfg.rms_norm_eps;
    let hd = cfg.head_dim;
    let mut b = Builder::with_matrix_dtype("minicpm_sala_step", matrix_dtype);

    let mut x = b.input(IN_EMBEDS, &[1, t, h]);
    let cos = b.input(IN_ROPE_COS, &[t, hd / 2]);
    let sin = b.input(IN_ROPE_SIN, &[t, hd / 2]);
    let last = b.input(IN_LAST, &[1]);
    let has_attn = cfg.mixers.contains(&Mixer::Attention);
    let has_lightning = cfg.mixers.contains(&Mixer::Lightning);
    let bias = has_attn.then(|| b.input(IN_ATTN_BIAS, &[1, nh, t, p + t]));
    let sparse = match (shape.sparse, cfg.sparse) {
        (true, Some(sp)) if has_attn => Some((sp, SparseNodes::declare(&mut b, &sp, t, p))),
        (true, _) => panic!("minicpm-sala: a sparse step needs sparse_config and attention layers"),
        _ => None,
    };
    let mut selection = None;
    let decay = has_lightning.then(|| Decay {
        intra: b.input(IN_DECAY_INTRA, &[nh, t, t]),
        q: b.input(IN_DECAY_Q, &[nh, t, 1]),
        k: b.input(IN_DECAY_K, &[nh, t, 1]),
        s: b.input(IN_DECAY_S, &[nh, 1, 1]),
    });
    let rscale = b.data("sala.residual_scale", vec![cfg.residual_scale()], &[1]);

    let mut layout = StepLayout {
        attn_layers: Vec::new(),
        lightning_layers: Vec::new(),
    };
    let mut kv_outs = Vec::new();
    let mut state_outs = Vec::new();
    for (l, mixer) in cfg.mixers.iter().enumerate() {
        let pre = cfg.weight(&format!("model.layers.{l}"));
        let n1 = b.rms_norm(x, &format!("{pre}.input_layernorm.weight"), h, eps, false);
        let y = match mixer {
            Mixer::Attention => {
                let (y, k, v, sel) = attention(
                    &mut b,
                    cfg,
                    &pre,
                    n1,
                    l,
                    shape,
                    bias.unwrap(),
                    sparse.as_ref(),
                    cos,
                    sin,
                );
                if selection.is_none() {
                    selection = sel;
                }
                layout.attn_layers.push(l);
                kv_outs.push(k);
                kv_outs.push(v);
                y
            }
            Mixer::Lightning => {
                let (y, s) = lightning(
                    &mut b,
                    cfg,
                    &pre,
                    n1,
                    l,
                    t,
                    decay.as_ref().unwrap(),
                    cos,
                    sin,
                );
                layout.lightning_layers.push(l);
                state_outs.push(s);
                y
            }
        };
        let y = b.g.mul(y, rscale);
        x = b.g.add(x, y);
        let n2 = b.rms_norm(
            x,
            &format!("{pre}.post_attention_layernorm.weight"),
            h,
            eps,
            false,
        );
        let ffn = cfg.intermediate_size;
        let gate = b.linear(n2, &format!("{pre}.mlp.gate_proj.weight"), h, ffn);
        let up = b.linear(n2, &format!("{pre}.mlp.up_proj.weight"), h, ffn);
        let act = b.g.silu(gate);
        let gu = b.g.mul(act, up);
        let down = b.linear(gu, &format!("{pre}.mlp.down_proj.weight"), ffn, h);
        let down = b.g.mul(down, rscale);
        x = b.g.add(x, down);
    }

    let xn = b.rms_norm(x, &cfg.weight("model.norm.weight"), h, eps, false);
    let flat = b.g.reshape_(xn, vec![t as i64, h as i64]);
    let row = b.g.gather_(flat, last, 0);
    let row = b.g.reshape_(row, vec![1, h as i64]);
    let head_scale = b.data(
        "sala.lm_head_scale",
        vec![cfg.dim_model_base as f32 / h as f32],
        &[1],
    );
    let row = b.g.mul(row, head_scale);
    let logits = b.linear(row, &cfg.weight("lm_head.weight"), h, cfg.vocab_size);

    let mut outs = vec![logits];
    outs.extend(kv_outs);
    outs.extend(state_outs);
    outs.extend(selection);
    b.g.set_outputs(outs);
    (b.g, b.params, layout)
}

struct Decay {
    intra: NodeId,
    q: NodeId,
    k: NodeId,
    s: NodeId,
}

/// `minicpm4` attention over `[cache | chunk]`. Returns `(y, k_new, v_new)`.
#[allow(clippy::too_many_arguments)]
fn attention(
    b: &mut Builder,
    cfg: &SalaConfig,
    pre: &str,
    x: NodeId,
    layer: usize,
    shape: StepShape,
    bias: NodeId,
    sparse: Option<&(crate::sparse::SparseParams, SparseNodes)>,
    cos: NodeId,
    sin: NodeId,
) -> (NodeId, NodeId, NodeId, Option<NodeId>) {
    let StepShape {
        chunk: t, cap: p, ..
    } = shape;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.attn_head_dim();
    let a = format!("{pre}.self_attn");
    let (mut q, mut k, v) = if cfg.attn_qkv_bias {
        (
            b.linear_bias(x, &format!("{a}.q_proj"), h, nh * hd),
            b.linear_bias(x, &format!("{a}.k_proj"), h, nkv * hd),
            b.linear_bias(x, &format!("{a}.v_proj"), h, nkv * hd),
        )
    } else {
        (
            b.linear(x, &format!("{a}.q_proj.weight"), h, nh * hd),
            b.linear(x, &format!("{a}.k_proj.weight"), h, nkv * hd),
            b.linear(x, &format!("{a}.v_proj.weight"), h, nkv * hd),
        )
    };
    if cfg.attn_qk_norm {
        let eps = cfg.rms_norm_eps;
        q = b.per_head_rms(q, &format!("{a}.q_norm.weight"), t, nh, hd, eps, false);
        k = b.per_head_rms(k, &format!("{a}.k_norm.weight"), t, nkv, hd, eps, false);
    }
    if cfg.attn_rope {
        q = b.g.rope_styled(q, cos, sin, hd, RopeStyle::NeoX);
        k = b.g.rope_styled(k, cos, sin, hd, RopeStyle::NeoX);
    }
    let past_k = b.input(&past_k_input(layer), &[1, p, nkv * hd]);
    let past_v = b.input(&past_v_input(layer), &[1, p, nkv * hd]);
    let kk = b.g.concat_(vec![past_k, k], 1);
    let vv = b.g.concat_(vec![past_v, v], 1);
    let (bias, selection) = match sparse {
        Some((sp, nodes)) => {
            let (extra, kept) = sparse_bias(b, sp, nodes, q, kk, t, p, nh, nkv, hd);
            (b.g.add(bias, extra), Some(kept))
        }
        None => (bias, None),
    };
    let group = nh / nkv;
    let kk = b.repeat_kv(kk, p + t, nkv, hd, group);
    let vv = b.repeat_kv(vv, p + t, nkv, hd, group);
    let mut y =
        b.g.attention_bias(q, kk, vv, bias, nh, hd, Shape::new(&[1, t, nh * hd], F32));
    if cfg.attn_output_gate {
        let g = b.linear(x, &format!("{a}.o_gate.weight"), h, nh * hd);
        let g = b.g.sigmoid(g);
        y = b.g.mul(y, g);
    }
    let y = b.linear(y, &format!("{a}.o_proj.weight"), nh * hd, h);
    (y, k, v, selection)
}

/// Lightning attention for one chunk. Returns `(y, state_new)`.
#[allow(clippy::too_many_arguments)]
fn lightning(
    b: &mut Builder,
    cfg: &SalaConfig,
    pre: &str,
    x: NodeId,
    layer: usize,
    t: usize,
    d: &Decay,
    cos: NodeId,
    sin: NodeId,
) -> (NodeId, NodeId) {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.lightning_kv_heads;
    let hd = cfg.head_dim;
    let eps = cfg.rms_norm_eps;
    let a = format!("{pre}.self_attn");
    let mut q = b.linear(x, &format!("{a}.q_proj.weight"), h, nh * hd);
    let mut k = b.linear(x, &format!("{a}.k_proj.weight"), h, nkv * hd);
    let mut v = b.linear(x, &format!("{a}.v_proj.weight"), h, nkv * hd);
    if cfg.lightning_qk_norm {
        q = b.per_head_rms(q, &format!("{a}.q_norm.weight"), t, nh, hd, eps, false);
        k = b.per_head_rms(k, &format!("{a}.k_norm.weight"), t, nkv, hd, eps, false);
    }
    if cfg.lightning_rope {
        q = b.g.rope_styled(q, cos, sin, hd, RopeStyle::NeoX);
        k = b.g.rope_styled(k, cos, sin, hd, RopeStyle::NeoX);
    }
    if nkv != nh {
        k = b.repeat_kv(k, t, nkv, hd, nh / nkv);
        v = b.repeat_kv(v, t, nkv, hd, nh / nkv);
    }
    // [1, T, nh·hd] → [nh, T, hd]
    let heads = |b: &mut Builder, z: NodeId| {
        let r = b.g.reshape_(z, vec![t as i64, nh as i64, hd as i64]);
        b.g.transpose_(r, vec![1, 0, 2])
    };
    let scale = b.data("sala.lightning_scale", vec![cfg.lightning_scale], &[1]);
    let q = heads(b, q);
    let q = b.g.mul(q, scale);
    let k = heads(b, k);
    let v = heads(b, v);
    let state = b.input(&state_input(layer), &[nh, hd, hd]);

    let kt = b.g.transpose_(k, vec![0, 2, 1]);
    let scores = b.g.mm(q, kt); // [nh, T, T]
    let scores = b.g.mul(scores, d.intra);
    let intra = b.g.mm(scores, v); // [nh, T, hd]
    let qd = b.g.mul(q, d.q);
    let inter = b.g.mm(qd, state); // [nh, T, hd]
    let o = b.g.add(intra, inter);

    let kd = b.g.mul(k, d.k);
    let kdt = b.g.transpose_(kd, vec![0, 2, 1]);
    let fresh = b.g.mm(kdt, v); // [nh, hd, hd]
    let carried = b.g.mul(state, d.s);
    let state_new = b.g.add(carried, fresh);

    let o = b.g.transpose_(o, vec![1, 0, 2]);
    let mut o = b.g.reshape_(o, vec![1, t as i64, (nh * hd) as i64]);
    if cfg.lightning_output_norm {
        o = b.rms_norm(o, &format!("{a}.o_norm.weight"), nh * hd, eps, false);
    }
    if cfg.lightning_output_gate {
        let z = b.linear(x, &format!("{a}.z_proj.weight"), h, nh * hd);
        let z = b.g.sigmoid(z);
        o = b.g.mul(o, z);
    }
    let y = b.linear(o, &format!("{a}.o_proj.weight"), nh * hd, h);
    (y, state_new)
}

/// Host-side decay factors for one chunk of `n` real tokens out of `t`.
#[derive(Debug, Clone)]
pub struct DecayFeeds {
    pub intra: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub s: Vec<f32>,
}

impl DecayFeeds {
    pub fn new(log_decay: &[f32], t: usize, n: usize) -> Self {
        let nh = log_decay.len();
        let mut intra = vec![0f32; nh * t * t];
        let mut q = vec![0f32; nh * t];
        let mut k = vec![0f32; nh * t];
        let mut s = vec![0f32; nh];
        for (hi, &g) in log_decay.iter().enumerate() {
            let g = g as f64;
            for i in 0..t {
                for j in 0..=i {
                    intra[(hi * t + i) * t + j] = (g * (i - j) as f64).exp() as f32;
                }
                q[hi * t + i] = (g * (i + 1) as f64).exp() as f32;
                if i < n {
                    k[hi * t + i] = (g * (n - 1 - i) as f64).exp() as f32;
                }
            }
            s[hi] = (g * n as f64).exp() as f32;
        }
        Self { intra, q, k, s }
    }
}

/// Additive attention bias `[1, nh, T, P + T]`: cache rows below `past` and
/// chunk rows up to the query (and below `n`) are open, the rest closed.
pub fn attention_bias(nh: usize, t: usize, cap: usize, past: usize, n: usize) -> Vec<f32> {
    const CLOSED: f32 = -1e30;
    let w = cap + t;
    let mut row = vec![CLOSED; t * w];
    for i in 0..t {
        let r = &mut row[i * w..(i + 1) * w];
        r[..past].fill(0.0);
        for j in 0..=i.min(n.saturating_sub(1)) {
            r[cap + j] = 0.0;
        }
    }
    row.repeat(nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed-form chunk equals the recurrence `S = e^g S + kᵀv`,
    /// `o = q S`, run token by token — across a chunk boundary and with a
    /// padded tail.
    #[test]
    fn chunk_closed_form_matches_recurrence() {
        let (d, t) = (3usize, 4usize);
        let g = [-0.3f32];
        let total = 6usize; // chunks of 4 (full) then 2 real of 4
        let qs: Vec<Vec<f32>> = (0..total)
            .map(|i| (0..d).map(|c| ((i * 7 + c * 3) % 5) as f32 - 2.0).collect())
            .collect();
        let ks: Vec<Vec<f32>> = (0..total)
            .map(|i| (0..d).map(|c| ((i * 5 + c) % 4) as f32 - 1.5).collect())
            .collect();
        let vs: Vec<Vec<f32>> = (0..total)
            .map(|i| (0..d).map(|c| ((i + c * 2) % 3) as f32 - 1.0).collect())
            .collect();

        // Recurrence.
        let mut s = vec![0f64; d * d];
        let mut want = Vec::new();
        for i in 0..total {
            for a in 0..d {
                for c in 0..d {
                    s[a * d + c] =
                        s[a * d + c] * (g[0] as f64).exp() + (ks[i][a] * vs[i][c]) as f64;
                }
            }
            want.push(
                (0..d)
                    .map(|c| (0..d).map(|a| qs[i][a] as f64 * s[a * d + c]).sum::<f64>())
                    .collect::<Vec<_>>(),
            );
        }

        // Chunked closed form on the host.
        let mut st = vec![0f64; d * d];
        let mut got = Vec::new();
        for (c0, n) in [(0usize, 4usize), (4, 2)] {
            let f = DecayFeeds::new(&g, t, n);
            for i in 0..n {
                let mut o = vec![0f64; d];
                for j in 0..=i {
                    let qk: f64 = (0..d).map(|a| (qs[c0 + i][a] * ks[c0 + j][a]) as f64).sum();
                    for c in 0..d {
                        o[c] += qk * f.intra[i * t + j] as f64 * vs[c0 + j][c] as f64;
                    }
                }
                for c in 0..d {
                    o[c] += (0..d)
                        .map(|a| qs[c0 + i][a] as f64 * f.q[i] as f64 * st[a * d + c])
                        .sum::<f64>();
                }
                got.push(o);
            }
            let mut ns = vec![0f64; d * d];
            for a in 0..d {
                for c in 0..d {
                    ns[a * d + c] = st[a * d + c] * f.s[0] as f64
                        + (0..n)
                            .map(|j| (ks[c0 + j][a] * f.k[j]) as f64 * vs[c0 + j][c] as f64)
                            .sum::<f64>();
                }
            }
            st = ns;
        }
        for (w, g) in want.iter().zip(&got) {
            for (a, b) in w.iter().zip(g) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
        for (a, b) in s.iter().zip(&st) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn bias_opens_past_and_causal_chunk() {
        let b = attention_bias(1, 3, 4, 2, 2);
        let w = 7;
        // Query 0: past 0..2 open, cache 2..4 closed, chunk key 0 open.
        assert_eq!(
            &b[..w].iter().map(|&x| x == 0.0).collect::<Vec<_>>(),
            &[true, true, false, false, true, false, false]
        );
        // Query 2 (padding): sees only the real chunk keys 0..2.
        assert_eq!(
            &b[2 * w..3 * w]
                .iter()
                .map(|&x| x == 0.0)
                .collect::<Vec<_>>(),
            &[true, true, false, false, true, true, false]
        );
    }
}
