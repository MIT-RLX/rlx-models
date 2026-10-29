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

//! Line recognizer (`rec*.onnx`): a patch stem over the line band, an
//! optional context branch over a tall window (the base recognizer), then
//! masked depthwise-separable 1-D blocks with squeeze-excite and a CTC head.
//!
//! Inputs: `x [B, 1, H, W]` (ink, `W % 4 == 0`), `mask [B, 1, 1, W/4]` (1 on the
//! line's timesteps) and `inv_count [B, 1, 1, 1]` (1 / timesteps, for the
//! masked squeeze-excite mean). Outputs, per timestep: `best [B, T]` the
//! greedy class (f32-encoded, first of ties), `lse [B, T]` the head's
//! log-sum-exp and `feats [B, T, C]` the head input — the host recomputes the
//! few log-probabilities it needs ([`Head`]) instead of reading `V` per step.

use super::{Builder, Weights};
use anyhow::{Result, bail};
use rlx_ir::hir::HirNodeId;
use rlx_ir::op::{Activation, Op, ReduceOp};
use rlx_ir::{DType, Graph, HirGraphExt, Shape};
use std::collections::HashMap;

/// Height of the line band every recognizer reads.
pub const BAND: usize = 32;
/// Input columns per CTC timestep.
pub const STRIDE: usize = 4;
/// Classes per step the beam search tries.
pub const TOP_K: usize = 8;
/// Groups the context branch splits its channels into (`rec.context.score`).
const CONTEXT_GROUPS: usize = 4;

/// Static facts of one recognizer, read from its weights.
#[derive(Clone, Copy, Debug)]
pub struct RecSpec {
    /// Input height: the band, or a tall window `window × BAND` (context branch).
    pub height: usize,
    pub blocks: usize,
    pub classes: usize,
    pub has_context: bool,
}

impl RecSpec {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let blocks = (0..)
            .take_while(|i| w.has(&format!("rec.blocks.{i}.dw.weight")))
            .count();
        let classes = w.shape("rec.head.weight")?[0];
        let has_context = w.has("rec.context.patch.weight");
        let height = if has_context {
            // The score bias holds one logit per context row: rows × patch = window height.
            let rows = w.shape("rec.context.add")?[2];
            rows * w.shape("rec.context.patch.weight")?[2]
        } else {
            BAND
        };
        if blocks == 0 {
            bail!("recognizer has no blocks");
        }
        Ok(Self {
            height,
            blocks,
            classes,
            has_context,
        })
    }
}

/// Graph for one `[batch, width]` shape.
pub fn build(
    w: &Weights,
    batch: usize,
    width: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let spec = RecSpec::from_weights(w)?;
    if !width.is_multiple_of(STRIDE) {
        bail!("recognizer width {width} is not a multiple of {STRIDE}");
    }
    let t = width / STRIDE;
    let mut b = Builder::new("indra_rec", w);
    let x = b.input("x", &[batch, 1, spec.height, width]);
    let mask = b.input("mask", &[batch, 1, 1, t]);
    let inv_count = b.input("inv_count", &[batch, 1, 1, 1]);

    // Stem: one patch per timestep over the middle band of the window.
    let top = (spec.height - BAND) / 2;
    let band = if spec.height == BAND {
        x
    } else {
        b.m().narrow_(x, 2, top, BAND)
    };
    let mut h = b.conv(band, "rec.stem", [BAND, STRIDE], [0, 0])?;
    if spec.has_context {
        let c = context(&mut b, x, batch, t)?;
        h = b.m().add(h, c);
    }
    h = b.batch_norm(h, "rec.stem_bn")?;
    h = b.act(h, Activation::HardSwish);
    h = b.mul_b(h, mask);
    for i in 0..spec.blocks {
        h = block(&mut b, h, mask, inv_count, i)?;
    }
    let c = b.dims(h)[1];
    let feats = b.m().reshape_(h, vec![batch as i64, c as i64, t as i64]);
    let feats = b.m().transpose_(feats, vec![0, 2, 1]); // [B, T, C]
    // Head as one GEMM with the class axis contiguous: [B·T, C] @ [C, V].
    let rows = b.m().reshape_(feats, vec![(batch * t) as i64, c as i64]);
    let wt = {
        let (d, s) = b.w.get("rec.head.weight")?;
        let (v, ci) = (s[0], s[1]);
        let mut t = vec![0f32; v * ci];
        for r in 0..v {
            for k in 0..ci {
                t[k * v + r] = d[r * ci + k];
            }
        }
        b.constant("rec.head.weight_t", &[ci, v], t)
    };
    let logits = b.m().mm(rows, wt);
    let bias = b.weight_as("rec.head.bias", &[1, spec.classes])?;
    let logits = b.add_b(logits, bias); // [B·T, V]
    let best = b.m().add_node(
        Op::ArgMax {
            axis: 1,
            keep_dim: false,
        },
        vec![logits],
        Shape::new(&[batch * t], DType::F32),
    );
    // log Σ exp(logits) = max + log Σ exp(logits - max)
    let mx = b.reduce(logits, ReduceOp::Max, vec![1], true);
    let mxe = b.expand_to(mx, logits);
    let z = b.m().sub(logits, mxe);
    let e = b.m().exp(z);
    let s = b.m().sum(e, vec![1], true);
    let l = b.act(s, Activation::Log);
    let lse = b.m().add(mx, l);
    let lse = b.m().reshape_(lse, vec![batch as i64, t as i64]);
    let best = b.m().reshape_(best, vec![batch as i64, t as i64]);
    b.finish(vec![best, lse, feats])
}

/// The CTC head on the host: log-probability of one class at one step from
/// the graph's `feats` and `lse`.
pub struct Head<'w> {
    w: &'w [f32],
    b: &'w [f32],
    c: usize,
}

impl<'w> Head<'w> {
    pub fn new(w: &'w Weights) -> Result<Self> {
        let (wd, ws) = w.get("rec.head.weight")?;
        let (bd, _) = w.get("rec.head.bias")?;
        Ok(Self {
            w: wd,
            b: bd,
            c: ws[1],
        })
    }

    pub fn channels(&self) -> usize {
        self.c
    }

    #[inline]
    pub fn logp(&self, feats: &[f32], lse: f32, class: usize) -> f32 {
        let row = &self.w[class * self.c..(class + 1) * self.c];
        let dot: f32 = row.iter().zip(feats).map(|(a, b)| a * b).sum();
        dot + self.b[class] - lse
    }

    /// The `TOP_K` best classes of one step and their log-probabilities,
    /// best first, ties to the smaller class (as `torch.topk`).
    pub fn top_k(&self, feats: &[f32], lse: f32) -> [(u32, f32); TOP_K] {
        let mut top = [(0u32, f32::NEG_INFINITY); TOP_K];
        for class in 0..self.b.len() {
            let v = self.logp(feats, lse, class);
            if v > top[TOP_K - 1].1 {
                let mut i = TOP_K - 1;
                while i > 0 && v > top[i - 1].1 {
                    top[i] = top[i - 1];
                    i -= 1;
                }
                top[i] = (class as u32, v);
            }
        }
        top
    }
}

/// The context branch: features of the whole window, pooled over its rows by
/// a learned per-group attention, added to the stem.
fn context(b: &mut Builder<'_>, x: HirNodeId, batch: usize, t: usize) -> Result<HirNodeId> {
    let mut f = b.conv(x, "rec.context.patch", [STRIDE, STRIDE], [0, 0])?;
    f = b.act(f, Activation::HardSwish);
    f = b.conv_act(f, "rec.context.dw", 1, None)?;
    f = b.conv_act(f, "rec.context.pw", 1, Some(Activation::HardSwish))?; // [B, C, R, T]
    let [_, c, rows, _] = b.dims(f)[..] else {
        unreachable!()
    };
    let s = b.conv(f, "rec.context.score", [1, 1], [0, 0])?;
    let bias = b.weight("rec.context.add")?; // [1, G, R, 1]
    let s = b.add_b(s, bias);
    let s = b.m().sm(s, 2); // over rows
    let g = CONTEXT_GROUPS;
    let (bi, gi, ci, ri, ti) = (
        batch as i64,
        g as i64,
        (c / g) as i64,
        rows as i64,
        t as i64,
    );
    let f5 = b.m().reshape_(f, vec![bi, gi, ci, ri, ti]);
    let s5 = b.m().reshape_(s, vec![bi, gi, 1, ri, ti]);
    let p = b.mul_b(f5, s5);
    let p = b.m().sum(p, vec![3], false); // [B, G, C/G, T]
    let p = b.m().reshape_(p, vec![bi, c as i64, 1, ti]);
    b.conv(p, "rec.context.proj", [1, 1], [0, 0])
}

/// Depthwise 1-D conv → expand → HardSwish → masked squeeze-excite → project,
/// residual, masked.
fn block(
    b: &mut Builder<'_>,
    x: HirNodeId,
    mask: HirNodeId,
    inv_count: HirNodeId,
    i: usize,
) -> Result<HirNodeId> {
    let p = format!("rec.blocks.{i}");
    let k = b.w.shape(&format!("{p}.dw.weight"))?[2];
    let mut h = b.conv(x, &format!("{p}.dw"), [1, 1], [0, k / 2])?;
    h = b.conv(h, &format!("{p}.expand"), [1, 1], [0, 0])?;
    h = b.act(h, Activation::HardSwish);
    let hm = b.mul_b(h, mask);
    let s = b.m().sum(hm, vec![3], true);
    let s = b.mul_b(s, inv_count);
    let s = b.conv(s, &format!("{p}.se.fc1"), [1, 1], [0, 0])?;
    let s = b.act(s, Activation::Relu);
    let s = b.conv(s, &format!("{p}.se.fc2"), [1, 1], [0, 0])?;
    let s = b.act(s, Activation::Sigmoid);
    h = b.mul_b(h, s);
    h = b.conv(h, &format!("{p}.project"), [1, 1], [0, 0])?;
    let y = b.m().add(x, h);
    Ok(b.mul_b(y, mask))
}

/// `mask [B, 1, 1, T]` and `inv_count [B, 1, 1, 1]` for lines of `widths`
/// input columns: a line covers `ceil(width / 4)` timesteps (at most `t`).
pub fn line_masks(widths: &[usize], t: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mask = vec![0.0; widths.len() * t];
    let mut inv = Vec::with_capacity(widths.len());
    for (i, &w) in widths.iter().enumerate() {
        let len = w.div_ceil(STRIDE).min(t);
        mask[i * t..i * t + len].fill(1.0);
        inv.push(1.0 / len.max(1) as f32);
    }
    (mask, inv)
}
