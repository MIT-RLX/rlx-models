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

//! A VGSL network as one RLX graph, for a fixed input size.
//!
//! Every block maps onto the layer Kraken builds for it
//! (`kraken/lib/vgsl/layers.py`), on an NCHW tensor with batch 1:
//! `ActConv2D` (same padding `d·(k−1)/2`), `MaxPool` (floor), `Reshape`
//! (split one axis, rotate a part next to another, merge), the
//! `TransposedSummarizingRNN` LSTMs and the `LinSoftmax` output. The graph
//! stops before the output softmax: its arg-max is all greedy CTC needs,
//! and the logits are what the parity check compares.

use crate::mlmodel::MlModel;
use crate::vgsl::{Block, ConvAct, InputSpec, Vgsl};
use anyhow::{Result, anyhow, bail, ensure};
use rlx_core::vision_ops_ir::{add_bias_nchw, nchw_shape};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, HirModule, Shape};
use std::collections::HashMap;

/// Graph input: the preprocessed line `[1, C, H, W]`.
pub const INPUT: &str = "line";

/// A built graph, its parameters, and the output layout.
pub struct KrakenGraph {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
    /// Output `[1, W', H', classes]`: frames × rows × classes.
    pub frames: usize,
    pub rows: usize,
    pub classes: usize,
}

struct Builder {
    hir: HirModule,
    params: HashMap<String, Vec<f32>>,
}

impl Builder {
    fn m(&mut self) -> HirMut<'_> {
        HirMut::new(&mut self.hir)
    }

    fn param(&mut self, name: &str, shape: &[usize], data: Vec<f32>) -> HirNodeId {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
        let id = self.m().param(name, Shape::new(shape, DType::F32));
        self.params.insert(name.to_string(), data);
        id
    }
}

/// VGSL axis (0 batch, 1 height, 2 width, 3 depth) → NCHW axis.
const VGSL_TO_NCHW: [usize; 4] = [0, 2, 3, 1];

/// Shape after each block, without building anything: lets the runner
/// size inputs and report frames before compiling.
pub fn output_shape(spec: &Vgsl, mut s: [usize; 4]) -> Result<[usize; 4]> {
    for b in &spec.blocks {
        s = block_shape(b, s)?;
    }
    Ok(s)
}

fn conv_out(n: usize, k: usize, st: usize, d: usize) -> usize {
    let p = d * (k - 1) / 2;
    ((n + 2 * p).saturating_sub(d * (k - 1) + 1) / st + 1).max(1)
}

fn block_shape(b: &Block, [n, c, h, w]: [usize; 4]) -> Result<[usize; 4]> {
    Ok(match b {
        Block::Conv {
            kernel,
            out,
            stride,
            dilation,
            ..
        } => [
            n,
            *out,
            conv_out(h, kernel[0], stride[0], dilation[0]),
            conv_out(w, kernel[1], stride[1], dilation[1]),
        ],
        Block::MaxPool { kernel, stride, .. } => {
            ensure!(
                h >= kernel[0] && w >= kernel[1],
                "{}: a {h}x{w} map is smaller than the {}x{} pool",
                b.name(),
                kernel[0],
                kernel[1]
            );
            [
                n,
                c,
                (h - kernel[0]) / stride[0] + 1,
                (w - kernel[1]) / stride[1] + 1,
            ]
        }
        Block::Dropout { .. } => [n, c, h, w],
        Block::Reshape { .. } => reshape_plan(b, [n, c, h, w])?.2,
        Block::Lstm {
            bidirectional,
            along_y,
            summarize,
            hidden,
            ..
        } => {
            let o = hidden * if *bidirectional { 2 } else { 1 };
            match (along_y, summarize) {
                (false, false) | (true, false) => [n, o, h, w],
                (false, true) => [n, o, h, 1],
                (true, true) => [n, o, 1, w],
            }
        }
        Block::Output { out, .. } => [n, *out, h, w],
    })
}

/// Kraken's `Reshape.forward` as (split shape, permutation, merged shape).
fn reshape_plan(b: &Block, s: [usize; 4]) -> Result<(Vec<usize>, Vec<usize>, [usize; 4])> {
    let Block::Reshape {
        name,
        dim,
        part_a,
        part_b,
        high,
        low,
    } = b
    else {
        unreachable!()
    };
    let src = VGSL_TO_NCHW[*dim];
    let (high, low) = (VGSL_TO_NCHW[*high], VGSL_TO_NCHW[*low]);
    let len = s[src];
    let (a, bb) = match (part_a, part_b) {
        (0, b) => (len / b, *b),
        (a, 0) => (*a, len / a),
        (a, b) => (*a, *b),
    };
    ensure!(
        a * bb == len,
        "{name}: cannot split an axis of {len} into {a}x{bb}"
    );
    let mut split: Vec<usize> = s[..src].to_vec();
    split.extend([a, bb]);
    split.extend_from_slice(&s[src + 1..]);
    let (mut src_dim, dest) = if high != src {
        (src, high)
    } else {
        (src + 1, low)
    };
    let mut perm: Vec<usize> = (0..5).collect();
    if dest > src_dim {
        while src_dim < dest {
            perm.swap(src_dim, src_dim + 1);
            src_dim += 1;
        }
    } else {
        while src_dim > dest {
            perm.swap(src_dim, src_dim - 1);
            src_dim -= 1;
        }
    }
    let p: Vec<usize> = perm.iter().map(|&i| split[i]).collect();
    let mut merged = Vec::with_capacity(4);
    merged.extend_from_slice(&p[..dest]);
    merged.push(p[dest] * p[dest + 1]);
    merged.extend_from_slice(&p[dest + 2..]);
    Ok((split, perm, [merged[0], merged[1], merged[2], merged[3]]))
}

fn dims(v: &[usize]) -> Vec<i64> {
    v.iter().map(|&d| d as i64).collect()
}

/// Emit the network for a `[height, width]` line.
pub fn build(spec: &Vgsl, model: &MlModel, height: usize, width: usize) -> Result<KrakenGraph> {
    let InputSpec { channels, .. } = spec.input;
    ensure!(
        spec.input.batch <= 1,
        "batched VGSL inputs ({}) are not supported",
        spec.input.batch
    );
    let mut b = Builder {
        hir: HirModule::new("kraken_vgsl"),
        params: HashMap::new(),
    };
    let mut s = [1, channels, height, width];
    let mut x = b.m().input(INPUT, Shape::new(&s, DType::F32));
    let mut out_classes = None;

    for blk in &spec.blocks {
        let next = block_shape(blk, s)?;
        match blk {
            Block::Conv {
                name,
                act,
                kernel,
                out,
                stride,
                dilation,
            } => {
                let key = format!("{name}_conv");
                let cl = model
                    .convs
                    .get(&key)
                    .ok_or_else(|| anyhow!("model has no layer {key}"))?;
                ensure!(
                    cl.in_channels == s[1] && cl.out_channels == *out && cl.kernel == *kernel,
                    "{key}: weights [{}, {}, {:?}] do not match the spec on a {}-channel input",
                    cl.out_channels,
                    cl.in_channels,
                    cl.kernel,
                    s[1]
                );
                let wt = b.param(
                    &format!("{name}.w"),
                    &[*out, s[1], kernel[0], kernel[1]],
                    cl.weight.clone(),
                );
                let bias = b.param(&format!("{name}.b"), &[*out], cl.bias.clone());
                let pad = [0, 1].map(|d| dilation[d] * (kernel[d] - 1) / 2);
                let y = b.m().add_node(
                    Op::Conv {
                        kernel_size: kernel.to_vec(),
                        stride: stride.to_vec(),
                        padding: pad.to_vec(),
                        dilation: dilation.to_vec(),
                        groups: 1,
                    },
                    vec![x, wt],
                    nchw_shape(1, *out, next[2], next[3], DType::F32),
                );
                let y = add_bias_nchw(&mut b.m(), y, bias, 1, *out, next[2], next[3]);
                x = match act {
                    // Kraken keeps sigmoid conv outputs as logits.
                    ConvAct::Sigmoid | ConvAct::Linear => y,
                    ConvAct::Relu => b.m().relu(y),
                    ConvAct::Tanh => b.m().tanh(y),
                    ConvAct::Softmax => b.m().sm(y, 1),
                    ConvAct::LeakyRelu => {
                        // leaky(x) = a·x + (1 − a)·relu(x), a = 0.01.
                        let ka = b.param(&format!("{name}.slope"), &[1], vec![0.01]);
                        let kr = b.param(&format!("{name}.keep"), &[1], vec![0.99]);
                        let r = b.m().relu(y);
                        let r = b.m().mul(r, kr);
                        let l = b.m().mul(y, ka);
                        b.m().add(l, r)
                    }
                };
            }
            Block::MaxPool { kernel, stride, .. } => {
                x = b.m().add_node(
                    Op::Pool {
                        kind: ReduceOp::Max,
                        kernel_size: kernel.to_vec(),
                        stride: stride.to_vec(),
                        padding: vec![0, 0],
                    },
                    vec![x],
                    Shape::new(&next, DType::F32),
                );
            }
            Block::Dropout { .. } => {}
            Block::Reshape { .. } => {
                let (split, perm, merged) = reshape_plan(blk, s)?;
                let t = b.m().reshape_(x, dims(&split));
                let t = b.m().transpose_(t, perm);
                x = b.m().reshape_(t, dims(&merged));
                debug_assert_eq!(merged, next);
            }
            Block::Lstm {
                name,
                bidirectional,
                along_y,
                summarize,
                hidden,
            } => {
                let layer = model
                    .lstms
                    .get(name)
                    .or_else(|| model.lstms.get(&format!("{name}_transposed")))
                    .ok_or_else(|| anyhow!("model has no LSTM {name}"))?;
                let dirs = if *bidirectional { 2 } else { 1 };
                ensure!(
                    layer.directions.len() == dirs
                        && layer.hidden == *hidden
                        && layer.input == s[1],
                    "{name}: weights ({} directions, {}→{}) do not match the spec",
                    layer.directions.len(),
                    layer.input,
                    layer.hidden
                );
                let (c, h, w) = (s[1], s[2], s[3]);
                // Batch over the other spatial axis, sequence along this one.
                let (batch, seq, to_seq, from_seq) = if *along_y {
                    // NCHW → W·N·H·C ; back from W·N·H'·O → N·O·H'·W.
                    (w, h, vec![3, 0, 2, 1], vec![1, 3, 2, 0])
                } else {
                    // NCHW → H·N·W·C ; back from H·N·W'·O → N·O·H·W'.
                    (h, w, vec![2, 0, 3, 1], vec![1, 3, 0, 2])
                };
                let rows = dirs * 4 * hidden;
                let cat = |f: fn(&crate::mlmodel::LstmDirection) -> &Vec<f32>| {
                    layer
                        .directions
                        .iter()
                        .flat_map(|d| f(d).iter().copied())
                        .collect::<Vec<f32>>()
                };
                let w_ih = b.param(&format!("{name}.w_ih"), &[rows, c], cat(|d| &d.w_ih));
                let w_hh = b.param(&format!("{name}.w_hh"), &[rows, *hidden], cat(|d| &d.w_hh));
                let bias = b.param(&format!("{name}.bias"), &[rows], cat(|d| &d.bias));
                let o = dirs * hidden;
                let t = b.m().transpose_(x, to_seq);
                let t = b.m().reshape_(t, dims(&[batch, seq, c]));
                let mut y = b.m().0.lstm(
                    t,
                    w_ih,
                    w_hh,
                    bias,
                    *hidden,
                    1,
                    *bidirectional,
                    Shape::new(&[batch, seq, o], DType::F32),
                );
                let steps = if *summarize {
                    y = b.m().narrow_(y, 1, seq - 1, 1);
                    1
                } else {
                    seq
                };
                let y = b.m().reshape_(y, dims(&[batch, 1, steps, o]));
                x = b.m().transpose_(y, from_seq);
            }
            Block::Output {
                name, augment, out, ..
            } => {
                let key = format!("{name}_lin");
                let lin = model
                    .linears
                    .get(&key)
                    .ok_or_else(|| anyhow!("model has no layer {key}"))?;
                let input = s[1] + usize::from(*augment);
                ensure!(
                    lin.input == input && lin.output == *out,
                    "{key}: weights [{}, {}] do not match the spec ({out}, {input})",
                    lin.output,
                    lin.input
                );
                let (h, w) = (s[2], s[3]);
                // NCHW → NWHC, features last.
                let mut t = b.m().transpose_(x, vec![0, 3, 2, 1]);
                if *augment {
                    let ones = b.param(&format!("{name}.ones"), &[1, w, h, 1], vec![1.0; w * h]);
                    t = b.m().concat_(vec![ones, t], 3);
                }
                let wt = b.param(
                    &format!("{name}.w"),
                    &[input, *out],
                    transpose(&lin.weight, *out, input),
                );
                let bias = b.param(&format!("{name}.b"), &[*out], lin.bias.clone());
                let y = b.m().mm(t, wt);
                x = b.m().add(y, bias);
                out_classes = Some(*out);
            }
        }
        s = next;
    }
    let Some(classes) = out_classes else {
        bail!("VGSL spec has no output block");
    };
    if !matches!(spec.blocks.last(), Some(Block::Output { .. })) {
        bail!("the output block must come last");
    }
    b.m().set_outputs(vec![x]);
    Ok(KrakenGraph {
        hir: b.hir,
        params: b.params,
        frames: s[3],
        rows: s[2],
        classes,
    })
}

/// Row-major `[rows, cols]` → `[cols, rows]`.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catmus_reshape_folds_height_into_channels() {
        // S1(1x0)1,3 on [1, 64, 15, W]: height moves next to depth,
        // height-major — feature index h·C + c.
        let b = Block::Reshape {
            name: "S".into(),
            dim: 1,
            part_a: 1,
            part_b: 0,
            high: 1,
            low: 3,
        };
        let (split, perm, merged) = reshape_plan(&b, [1, 64, 15, 100]).unwrap();
        assert_eq!(split, vec![1, 64, 1, 15, 100]);
        assert_eq!(perm, vec![0, 3, 1, 2, 4]);
        assert_eq!(merged, [1, 960, 1, 100]);
    }

    #[test]
    fn catmus_geometry() {
        let v = Vgsl::parse(
            "[1,120,0,1 Cr3,13,32 Mp2,2 Cr3,13,32 Mp2,2 Cr3,9,64 Mp2,2 Cr3,9,64 \
             S1(1x0)1,3 Lbx200 Lbx200 Lbx200 O1c225]",
        )
        .unwrap();
        assert_eq!(
            output_shape(&v, [1, 1, 120, 1030]).unwrap(),
            [1, 225, 1, 128]
        );
    }
}
