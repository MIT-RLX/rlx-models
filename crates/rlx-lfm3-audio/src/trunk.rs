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

//! The LFM2 / LFM3 backbone, as one builder for all four stacks in the model.
//!
//! A layer is a gated short convolution or attention (full or sliding
//! window), then a SwiGLU FFN, each behind a pre-RMSNorm and a residual. LFM3
//! adds a sigmoid gate per head on the attention output, optional "sandwich"
//! norms on both branch outputs, and partial RoPE that keeps the full head's
//! frequency progression and zeroes the tail (`inv_freq[i] = 0` past
//! `partial * head_dim / 2`, i.e. those dimensions are not rotated).
//!
//! | stack | layers | attention | gate | sandwich | conv |
//! |---|---|---|---|---|---|
//! | LM (`blk.`) | 24 | full (θ 1e6, half rotary) / sliding 256 (θ 1e4) | yes | yes | 7 |
//! | speech encoder (`a.blk.`) | 17 | sliding 32, θ 1e6 | no | no | 3 |
//! | depth decoder (`a.gen.code.blk.`) | 3 | bidirectional, θ 1e6 | yes | no | — |
//! | codec decoder (`a.gen.wav.tfm.blk.`) | 8 | sliding 30, θ 1e6 | yes | yes | 7 |
//!
//! Two graph shapes: `sequence` runs a whole sequence from a zero state
//! (encoder, depth decoder, codec), `step` advances the LM by `rows`
//! positions against a fixed-size cache it reads and returns.

use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, NodeId, Op, Shape};

use crate::graph::Gb;

/// What a layer mixes tokens with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Gated short convolution.
    Conv,
    /// Attention with RoPE table `rope`, seeing `window` keys back including
    /// itself (`None` = unbounded).
    Attn { rope: usize, window: Option<usize> },
}

/// One RoPE flavour (`partial < 1` zeroes the tail frequencies).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rope {
    pub theta: f64,
    pub partial: f64,
}

/// Per-stack tensor names (after `{prefix}{layer}.`).
#[derive(Debug, Clone, Copy)]
pub struct Names {
    pub op_norm: &'static str,
    pub ffn_norm: &'static str,
    pub post_op_norm: &'static str,
    pub post_ffn_norm: &'static str,
    pub conv_dw: &'static str,
    pub conv_in: &'static str,
    pub conv_out: &'static str,
    pub attn_out: &'static str,
}

/// llama.cpp's `lfm3` text-model names.
pub const LM_NAMES: Names = Names {
    op_norm: "attn_norm",
    ffn_norm: "ffn_norm",
    post_op_norm: "post_attention_norm",
    post_ffn_norm: "post_ffw_norm",
    conv_dw: "shortconv.conv",
    conv_in: "shortconv.in_proj",
    conv_out: "shortconv.out_proj",
    attn_out: "attn_output",
};

/// The mmproj (`clip`) names of the speech encoder, depth decoder and codec.
pub const MMPROJ_NAMES: Names = Names {
    op_norm: "ln1",
    ffn_norm: "ln2",
    post_op_norm: "attn_post_norm",
    post_ffn_norm: "ffn_post_norm",
    conv_dw: "conv_dw",
    conv_in: "conv_pw1",
    conv_out: "conv_pw2",
    attn_out: "attn_out",
};

/// One LFM2 / LFM3 stack.
#[derive(Debug, Clone)]
pub struct TrunkSpec {
    /// Layer tensor prefix, e.g. `blk.` (layer `i` is `{prefix}{i}.`).
    pub prefix: String,
    /// Final RMSNorm gain.
    pub final_norm: String,
    pub names: Names,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    pub layers: Vec<Mixer>,
    pub ropes: Vec<Rope>,
    /// Attention looks only backwards (`false`: every query sees every key).
    pub causal: bool,
    /// Sigmoid output gate per head (`attn_gate`; LFM3 only).
    pub gate: bool,
    /// RMSNorm on both branch outputs before the residual (LFM3 option).
    pub sandwich: bool,
}

impl TrunkSpec {
    fn layer(&self, i: usize, name: &str) -> String {
        format!("{}{i}.{name}.weight", self.prefix)
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The distinct attention windows, in first-use order.
    pub fn windows(&self) -> Vec<Option<usize>> {
        let mut out = Vec::new();
        for m in &self.layers {
            if let Mixer::Attn { window, .. } = *m
                && !out.contains(&window)
            {
                out.push(window);
            }
        }
        out
    }
}

/// `inv_freq` of `rope` over a head of `head_dim` (zero past the rotated part).
pub fn inv_freq(rope: Rope, head_dim: usize) -> Vec<f64> {
    let angles = (rope.partial * head_dim as f64) as usize / 2;
    (0..head_dim / 2)
        .map(|i| {
            if i < angles {
                rope.theta.powf(-(2.0 * i as f64) / head_dim as f64)
            } else {
                0.0
            }
        })
        .collect()
}

/// `cos`/`sin` tables `[positions.len(), head_dim/2]`.
pub fn rope_tables(
    rope: Rope,
    head_dim: usize,
    positions: impl Iterator<Item = usize>,
) -> (Vec<f32>, Vec<f32>) {
    let inv = inv_freq(rope, head_dim);
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for p in positions {
        for &f in &inv {
            let (s, c) = (p as f64 * f).sin_cos();
            cos.push(c as f32);
            sin.push(s as f32);
        }
    }
    (cos, sin)
}

/// The name of the mask input of a window class.
pub fn mask_name(window: Option<usize>) -> String {
    match window {
        Some(w) => format!("mask_w{w}"),
        None => "mask_full".into(),
    }
}

/// How attention gets its keys and masks.
enum Ctx {
    /// Whole sequence of `seq` positions, keys = the sequence itself.
    Sequence,
    /// `rows` new positions against `max_kv`-slot caches.
    Step {
        max_kv: usize,
        write: NodeId,
        keep: NodeId,
        masks: Vec<(Option<usize>, NodeId)>,
    },
}

/// Cache tensors a step graph reads (`*_in_{layer}`) and returns
/// (`*_out_{layer}`), in output order.
pub struct StepState {
    pub names: Vec<String>,
    pub sizes: Vec<usize>,
}

struct Layerer<'s> {
    spec: &'s TrunkSpec,
    rows: usize,
    ropes: Vec<(NodeId, NodeId)>,
    ctx: Ctx,
    state_out: Vec<(String, usize, NodeId)>,
}

impl Layerer<'_> {
    fn per_head_rms(&self, gb: &mut Gb, x: NodeId, heads: usize, key: &str) -> Result<NodeId> {
        let dh = self.spec.head_dim;
        let r =
            gb.g.reshape_(x, vec![(self.rows * heads) as i64, dh as i64]);
        let n = gb.rms(r, key, self.spec.eps)?;
        Ok(gb
            .g
            .reshape_(n, vec![1, self.rows as i64, (heads * dh) as i64]))
    }

    fn repeat_kv(&self, gb: &mut Gb, x: NodeId) -> NodeId {
        let s = self.spec;
        let group = s.heads / s.kv_heads;
        if group == 1 {
            return x;
        }
        let ax = gb.g.shape(x).rank() - 1;
        let mut pieces = Vec::with_capacity(s.heads);
        for h in 0..s.kv_heads {
            let slice = gb.g.narrow_(x, ax, h * s.head_dim, s.head_dim);
            pieces.extend(std::iter::repeat_n(slice, group));
        }
        gb.g.concat_(pieces, ax)
    }

    fn attention(
        &mut self,
        gb: &mut Gb,
        i: usize,
        x: NodeId,
        rope: usize,
        window: Option<usize>,
    ) -> Result<NodeId> {
        let s = self.spec;
        let (rows, nh, nkv, dh) = (self.rows, s.heads, s.kv_heads, s.head_dim);
        let kvd = s.kv_dim();
        let q = gb.linear(x, &s.layer(i, "attn_q"), rows)?;
        let k = gb.linear(x, &s.layer(i, "attn_k"), rows)?;
        let v = gb.linear(x, &s.layer(i, "attn_v"), rows)?;
        let q = self.per_head_rms(gb, q, nh, &s.layer(i, "attn_q_norm"))?;
        let k = self.per_head_rms(gb, k, nkv, &s.layer(i, "attn_k_norm"))?;
        let (cos, sin) = self.ropes[rope];
        let q = gb.g.rope(q, cos, sin, dh);
        let k = gb.g.rope(k, cos, sin, dh);

        let attn = match &self.ctx {
            Ctx::Sequence => {
                let v3 = gb.g.reshape_(v, vec![1, rows as i64, kvd as i64]);
                let k_rep = self.repeat_kv(gb, k);
                let v_rep = self.repeat_kv(gb, v3);
                let mask_kind = match (s.causal, window) {
                    (false, _) => MaskKind::None,
                    (true, None) => MaskKind::Causal,
                    (true, Some(w)) => MaskKind::SlidingWindow(w - 1),
                };
                let a = gb.g.add_node(
                    Op::Attention {
                        num_heads: nh,
                        head_dim: dh,
                        v_head_dim: None,
                        mask_kind,
                        score_scale: None,
                        attn_logit_softcap: None,
                    },
                    vec![q, k_rep, v_rep],
                    Shape::new(&[1, rows, nh * dh], DType::F32),
                );
                gb.g.reshape_(a, vec![rows as i64, (nh * dh) as i64])
            }
            Ctx::Step {
                max_kv,
                write,
                keep,
                masks,
            } => {
                let (max_kv, write, keep) = (*max_kv, *write, *keep);
                let mask = masks
                    .iter()
                    .find(|(w, _)| *w == window)
                    .map(|(_, m)| *m)
                    .expect("mask input per window class");
                // Write the new rows into the caches: c' = c ⊙ keep + Wᵀ·new.
                let k2 = gb.g.reshape_(k, vec![rows as i64, kvd as i64]);
                let mut caches = Vec::new();
                for (tag, new) in [("k", k2), ("v", v)] {
                    let name = format!("{tag}_in_{i}");
                    let cache = gb.input(&name, &[max_kv, kvd]);
                    let kept = gb.g.mul(cache, keep);
                    let written = gb.g.mm(write, new);
                    let next = gb.g.add(kept, written);
                    self.state_out.push((name, max_kv * kvd, next));
                    caches.push(next);
                }
                let (kc, vc) = (caches[0], caches[1]);
                if rows == 1 {
                    let k3 = gb.g.reshape_(kc, vec![1, max_kv as i64, kvd as i64]);
                    let v3 = gb.g.reshape_(vc, vec![1, max_kv as i64, kvd as i64]);
                    let k_rep = self.repeat_kv(gb, k3);
                    let v_rep = self.repeat_kv(gb, v3);
                    let a = gb.g.add_node(
                        Op::Attention {
                            num_heads: nh,
                            head_dim: dh,
                            v_head_dim: None,
                            mask_kind: MaskKind::Custom,
                            score_scale: None,
                            attn_logit_softcap: None,
                        },
                        vec![q, k_rep, v_rep, mask],
                        Shape::new(&[1, 1, nh * dh], DType::F32),
                    );
                    gb.g.reshape_(a, vec![1, (nh * dh) as i64])
                } else {
                    self.manual_attention(gb, q, kc, vc, mask, max_kv)
                }
            }
        };

        let attn = if s.gate {
            let gate = gb.linear(x, &s.layer(i, "attn_gate"), rows)?;
            let gate = gb.g.sigmoid(gate);
            let gate = gb.g.reshape_(gate, vec![rows as i64, nh as i64, 1]);
            let a3 = gb.g.reshape_(attn, vec![rows as i64, nh as i64, dh as i64]);
            let gated = gb.g.mul(a3, gate);
            gb.g.reshape_(gated, vec![rows as i64, (nh * dh) as i64])
        } else {
            attn
        };
        gb.linear(attn, &s.layer(i, s.names.attn_out), rows)
    }

    /// Multi-row attention against the caches with an additive `[rows,
    /// max_kv]` mask — plain matmul + softmax, so every backend runs it.
    fn manual_attention(
        &self,
        gb: &mut Gb,
        q: NodeId,
        kc: NodeId,
        vc: NodeId,
        bias: NodeId,
        max_kv: usize,
    ) -> NodeId {
        let s = self.spec;
        let (rows, nh, nkv, dh) = (
            self.rows as i64,
            s.heads as i64,
            s.kv_heads as i64,
            s.head_dim as i64,
        );
        let group = nh / nkv;
        let q = gb.g.reshape_(q, vec![rows, nh, dh]);
        let q = gb.g.transpose_(q, vec![1, 0, 2]); // [nh, rows, dh]
        let q = gb.g.reshape_(q, vec![nkv, group * rows, dh]);
        let k = gb.g.reshape_(kc, vec![max_kv as i64, nkv, dh]);
        let k = gb.g.transpose_(k, vec![1, 2, 0]); // [nkv, dh, max_kv]
        let scores = gb.g.mm(q, k); // [nkv, group*rows, max_kv]
        let scale = gb.shared(&format!("lfm3a.attn_scale.{dh}"), &[1], || {
            vec![(dh as f32).powf(-0.5)]
        });
        let scores = gb.g.mul(scores, scale);
        let scores = gb.g.reshape_(scores, vec![nh, rows, max_kv as i64]);
        let scores = gb.g.add(scores, bias);
        let probs = gb.g.sm(scores, -1);
        let probs = gb.g.reshape_(probs, vec![nkv, group * rows, max_kv as i64]);
        let v = gb.g.reshape_(vc, vec![max_kv as i64, nkv, dh]);
        let v = gb.g.transpose_(v, vec![1, 0, 2]); // [nkv, max_kv, dh]
        let out = gb.g.mm(probs, v); // [nkv, group*rows, dh]
        let out = gb.g.reshape_(out, vec![nh, rows, dh]);
        let out = gb.g.transpose_(out, vec![1, 0, 2]);
        gb.g.reshape_(out, vec![rows, nh * dh])
    }

    fn short_conv(&mut self, gb: &mut Gb, i: usize, x: NodeId) -> Result<NodeId> {
        let s = self.spec;
        let (rows, h, k) = (self.rows, s.hidden, s.conv_kernel);
        let bcx = gb.linear(x, &s.layer(i, s.names.conv_in), rows)?;
        let b = gb.g.narrow_(bcx, 1, 0, h);
        let c = gb.g.narrow_(bcx, 1, h, h);
        let xx = gb.g.narrow_(bcx, 1, 2 * h, h);
        let bx = gb.g.mul(b, xx); // [rows, h]

        let history = match self.ctx {
            Ctx::Sequence => gb.shared(&format!("lfm3a.conv_pad.{k}x{h}"), &[k - 1, h], || {
                vec![0.0; (k - 1) * h]
            }),
            Ctx::Step { .. } => {
                let name = format!("conv_in_{i}");
                gb.input(&name, &[k - 1, h])
            }
        };
        let window = gb.g.concat_(vec![history, bx], 0); // [k-1+rows, h]
        if let Ctx::Step { .. } = self.ctx {
            let next = gb.g.narrow_(window, 0, rows, k - 1);
            self.state_out
                .push((format!("conv_in_{i}"), (k - 1) * h, next));
        }
        let width = k - 1 + rows;
        let t = gb.g.transpose_(window, vec![1, 0]);
        let nchw = gb.g.reshape_(t, vec![1, h as i64, width as i64, 1]);
        let (wdata, _) = gb.take(&s.layer(i, s.names.conv_dw))?; // [h, k]
        let w = gb.constant(&s.layer(i, s.names.conv_dw), wdata, &[h, 1, k, 1]);
        let conv = gb.g.add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: h,
            },
            vec![nchw, w],
            Shape::new(&[1, h, rows, 1], DType::F32),
        );
        let conv = gb.g.reshape_(conv, vec![h as i64, rows as i64]);
        let conv = gb.g.transpose_(conv, vec![1, 0]);
        let y = gb.g.mul(c, conv);
        gb.linear(y, &s.layer(i, s.names.conv_out), rows)
    }

    fn ffn(&self, gb: &mut Gb, i: usize, x: NodeId) -> Result<NodeId> {
        let s = self.spec;
        let gate = gb.linear(x, &s.layer(i, "ffn_gate"), self.rows)?;
        let up = gb.linear(x, &s.layer(i, "ffn_up"), self.rows)?;
        let act = gb.g.silu(gate);
        let glu = gb.g.mul(act, up);
        gb.linear(glu, &s.layer(i, "ffn_down"), self.rows)
    }

    fn run(&mut self, gb: &mut Gb, mut x: NodeId) -> Result<NodeId> {
        let s = self.spec;
        for (i, mixer) in s.layers.iter().enumerate() {
            let n = gb.rms(x, &s.layer(i, s.names.op_norm), s.eps)?;
            let mut m = match *mixer {
                Mixer::Conv => self.short_conv(gb, i, n)?,
                Mixer::Attn { rope, window } => self.attention(gb, i, n, rope, window)?,
            };
            if s.sandwich {
                m = gb.rms(m, &s.layer(i, s.names.post_op_norm), s.eps)?;
            }
            x = gb.g.add(x, m);
            let n = gb.rms(x, &s.layer(i, s.names.ffn_norm), s.eps)?;
            let mut f = self.ffn(gb, i, n)?;
            if s.sandwich {
                f = gb.rms(f, &s.layer(i, s.names.post_ffn_norm), s.eps)?;
            }
            x = gb.g.add(x, f);
        }
        gb.rms(x, &s.final_norm, s.eps)
    }
}

/// The stack over a whole sequence `x` (`[seq, hidden]`) from a zero state.
/// Returns the final-normed hidden states `[seq, hidden]`.
pub fn sequence(gb: &mut Gb, spec: &TrunkSpec, x: NodeId, seq: usize) -> Result<NodeId> {
    let half = spec.head_dim / 2;
    let mut ropes = Vec::new();
    for (r, rope) in spec.ropes.iter().enumerate() {
        let (cos, sin) = rope_tables(*rope, spec.head_dim, 0..seq);
        let c = gb.constant(
            &format!("{}rope{r}.cos.{seq}", spec.prefix),
            cos,
            &[seq, half],
        );
        let s = gb.constant(
            &format!("{}rope{r}.sin.{seq}", spec.prefix),
            sin,
            &[seq, half],
        );
        ropes.push((c, s));
    }
    let mut l = Layerer {
        spec,
        rows: seq,
        ropes,
        ctx: Ctx::Sequence,
        state_out: Vec::new(),
    };
    l.run(gb, x)
}

/// The stack advanced by `rows` positions (`x`: `[rows, hidden]`) against
/// `max_kv`-slot caches.
///
/// Inputs besides `x`: `rope{r}_cos` / `rope{r}_sin` `[rows, head_dim/2]`;
/// `kv_write` `[max_kv, rows]` (one-hot: row `j` goes to slot `pos + j`) and
/// `kv_keep` `[max_kv, 1]` (0 at the written slots); one mask per window class
/// ([`mask_name`]) — a `[1, max_kv]` key mask (1 = visible) when `rows == 1`,
/// else an additive `[rows, max_kv]` bias; and the caches of [`StepState`].
///
/// Returns the final-normed hidden states and the cache outputs, which the
/// caller appends to the graph outputs in [`StepState`] order.
pub fn step(
    gb: &mut Gb,
    spec: &TrunkSpec,
    x: NodeId,
    rows: usize,
    max_kv: usize,
) -> Result<(NodeId, Vec<NodeId>, StepState)> {
    let half = spec.head_dim / 2;
    let mut ropes = Vec::new();
    for r in 0..spec.ropes.len() {
        let c = gb.input(&format!("rope{r}_cos"), &[rows, half]);
        let s = gb.input(&format!("rope{r}_sin"), &[rows, half]);
        ropes.push((c, s));
    }
    let write = gb.input("kv_write", &[max_kv, rows]);
    let keep = gb.input("kv_keep", &[max_kv, 1]);
    let masks = spec
        .windows()
        .into_iter()
        .map(|w| {
            let shape: &[usize] = if rows == 1 {
                &[1, max_kv]
            } else {
                &[rows, max_kv]
            };
            (w, gb.input(&mask_name(w), shape))
        })
        .collect();
    let mut l = Layerer {
        spec,
        rows,
        ropes,
        ctx: Ctx::Step {
            max_kv,
            write,
            keep,
            masks,
        },
        state_out: Vec::new(),
    };
    let hidden = l.run(gb, x)?;
    let mut names = Vec::new();
    let mut sizes = Vec::new();
    let mut outs = Vec::new();
    for (name, size, node) in l.state_out {
        names.push(name);
        sizes.push(size);
        outs.push(node);
    }
    Ok((hidden, outs, StepState { names, sizes }))
}
