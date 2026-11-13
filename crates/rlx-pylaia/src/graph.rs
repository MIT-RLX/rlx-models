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

//! The `LaiaCRNN` forward as one RLX graph, for a fixed image size.
//!
//! ```text
//!   image [1, C, H, W]
//!   → per block: conv (BatchNorm folded in) + bias → activation → max-pool
//!   → sequencer: [1, C', H', W'] → [1, W', C'·N]   (channel-major, like
//!     PyLaia's `permute(3, 0, 1, 2).view(W, N, C·H)`)
//!   → bidirectional RNN layers → linear → logits [1, W', labels]
//! ```
//!
//! The width varies per text line, so the runner compiles one graph per
//! input size and caches it.

use crate::config::{Activation, PylaiaConfig, RnnKind, Sequencer};
use crate::weights::PylaiaWeights;
use anyhow::{Result, anyhow};
use rlx_core::vision_ops_ir::{add_bias_nchw, nchw_shape};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{Op, ReduceOp};
use rlx_ir::{DType, HirGraphExt, HirModule, Shape};
use std::collections::HashMap;

/// Graph input name: the preprocessed image `[1, C, H, W]`.
pub const INPUT: &str = "image";

/// A built graph and its parameter tensors.
pub struct PylaiaGraph {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
    /// Output frames (`W'`).
    pub frames: usize,
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

/// Emit the network for an `[height, width]` image.
pub fn build(
    cfg: &PylaiaConfig,
    w: &PylaiaWeights,
    height: usize,
    width: usize,
) -> Result<PylaiaGraph> {
    let [oh, ow] = cfg.conv_output_size([height, width]).ok_or_else(|| {
        anyhow!(
            "a {height}x{width} image is too small for this model (min width {})",
            cfg.min_width(height)
        )
    })?;
    if let Sequencer::Fixed(n) = cfg.sequencer
        && oh != n
    {
        anyhow::bail!(
            "CNN output is {oh} rows high but the `none-{n}` sequencer needs {n}: \
             resize the line to the height the model was trained on"
        );
    }

    let mut b = Builder {
        hir: HirModule::new("pylaia_crnn"),
        params: HashMap::new(),
    };
    let mut x = b.m().input(
        INPUT,
        nchw_shape(1, cfg.in_channels, height, width, DType::F32),
    );

    let (mut h, mut wd) = (height, width);
    for (i, (blk, cw)) in cfg.blocks.iter().zip(&w.conv).enumerate() {
        let c = blk.out_channels;
        let wt = b.param(&format!("conv{i}.w"), &cw.shape, cw.weight.clone());
        let bias = b.param(&format!("conv{i}.b"), &[c], cw.bias.clone());
        let pad = blk.padding();
        let ch = (h + 2 * pad[0] - blk.dilation[0] * (blk.kernel[0] - 1) - 1) / blk.stride[0] + 1;
        let cwid =
            (wd + 2 * pad[1] - blk.dilation[1] * (blk.kernel[1] - 1) - 1) / blk.stride[1] + 1;
        let y = b.m().add_node(
            Op::Conv {
                kernel_size: blk.kernel.to_vec(),
                stride: blk.stride.to_vec(),
                padding: pad.to_vec(),
                dilation: blk.dilation.to_vec(),
                groups: 1,
            },
            vec![x, wt],
            nchw_shape(1, c, ch, cwid, DType::F32),
        );
        let y = add_bias_nchw(&mut b.m(), y, bias, 1, c, ch, cwid);
        x = activate(&mut b, y, blk.activation, &format!("conv{i}"));
        (h, wd) = (ch, cwid);
        if let Some(p) = blk.pool {
            (h, wd) = (h / p[0], wd / p[1]);
            x = b.m().add_node(
                Op::Pool {
                    kind: ReduceOp::Max,
                    kernel_size: p.to_vec(),
                    stride: p.to_vec(),
                    padding: vec![0, 0],
                },
                vec![x],
                nchw_shape(1, c, h, wd, DType::F32),
            );
        }
    }
    debug_assert_eq!((h, wd), (oh, ow));

    let c_last = cfg
        .blocks
        .last()
        .map_or(cfg.in_channels, |b| b.out_channels);
    let n = cfg.sequencer.height();
    let feats = c_last * n;
    let seq = match cfg.sequencer {
        Sequencer::Fixed(_) => {
            let t = b.m().transpose_(x, vec![0, 3, 1, 2]);
            b.m().reshape_(t, vec![1, ow as i64, feats as i64])
        }
        Sequencer::AvgPool(_) => {
            let pool = b.param("seq.pool", &[oh, n], adaptive_avg_matrix(oh, n));
            let t = b.m().transpose_(x, vec![0, 1, 3, 2]); // [1, C, W, H]
            let p = b.m().mm(t, pool); // [1, C, W, N]
            let t = b.m().transpose_(p, vec![0, 2, 1, 3]); // [1, W, C, N]
            b.m().reshape_(t, vec![1, ow as i64, feats as i64])
        }
    };

    let u = cfg.rnn_units;
    let mut y = seq;
    for (l, rw) in w.rnn.iter().enumerate() {
        let rows = 2 * cfg.rnn.gates() * u;
        let w_ih = b.param(&format!("rnn{l}.w_ih"), &[rows, rw.input], rw.w_ih.clone());
        let w_hh = b.param(&format!("rnn{l}.w_hh"), &[rows, u], rw.w_hh.clone());
        let out = Shape::new(&[1, ow, 2 * u], DType::F32);
        y = match cfg.rnn {
            RnnKind::Lstm => {
                let bias: Vec<f32> = rw.b_ih.iter().zip(&rw.b_hh).map(|(a, c)| a + c).collect();
                let bias = b.param(&format!("rnn{l}.bias"), &[rows], bias);
                b.m().0.lstm(y, w_ih, w_hh, bias, u, 1, true, out)
            }
            RnnKind::Gru => {
                let b_ih = b.param(&format!("rnn{l}.b_ih"), &[rows], rw.b_ih.clone());
                let b_hh = b.param(&format!("rnn{l}.b_hh"), &[rows], rw.b_hh.clone());
                b.m().0.gru(y, w_ih, w_hh, b_ih, b_hh, u, 1, true, out)
            }
        };
    }

    let labels = cfg.num_labels;
    let lw = b.param(
        "linear.w",
        &[2 * u, labels],
        transpose(&w.linear_w, labels, 2 * u),
    );
    let lb = b.param("linear.b", &[labels], w.linear_b.clone());
    let logits = b.m().mm(y, lw);
    let logits = b.m().add(logits, lb);
    b.m().set_outputs(vec![logits]);
    Ok(PylaiaGraph {
        hir: b.hir,
        params: b.params,
        frames: ow,
    })
}

fn activate(b: &mut Builder, x: HirNodeId, act: Activation, name: &str) -> HirNodeId {
    match act {
        Activation::Identity => x,
        Activation::Relu => b.m().relu(x),
        Activation::Tanh => b.m().tanh(x),
        // leaky(x) = a·x + (1 − a)·relu(x): two elementwise ops every
        // backend has, no `max` or select needed.
        Activation::LeakyRelu(a) => {
            let ka = b.param(&format!("{name}.slope"), &[1], vec![a]);
            let kr = b.param(&format!("{name}.keep"), &[1], vec![1.0 - a]);
            let r = b.m().relu(x);
            let r = b.m().mul(r, kr);
            let l = b.m().mul(x, ka);
            b.m().add(l, r)
        }
    }
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

/// `[input, output]` matrix of `adaptive_avg_pool` along one axis: output
/// bin `i` averages rows `floor(i·in/out) .. ceil((i+1)·in/out)`.
pub fn adaptive_avg_matrix(input: usize, output: usize) -> Vec<f32> {
    let mut m = vec![0f32; input * output];
    for i in 0..output {
        let start = i * input / output;
        let end = ((i + 1) * input).div_ceil(output);
        let k = 1.0 / (end - start) as f32;
        for r in start..end {
            m[r * output + i] = k;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_avg_is_identity_at_equal_size() {
        let m = adaptive_avg_matrix(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(m[r * 4 + c], f32::from(r == c));
            }
        }
    }

    #[test]
    fn adaptive_avg_overlapping_bins() {
        // 5 → 3: bins [0,2), [1,4), [3,5) as torch computes them.
        let m = adaptive_avg_matrix(5, 3);
        let col = |c: usize| (0..5).map(|r| m[r * 3 + c]).collect::<Vec<_>>();
        assert_eq!(col(0), vec![0.5, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(col(1), vec![0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]);
        assert_eq!(col(2), vec![0.0, 0.0, 0.0, 0.5, 0.5]);
    }
}
