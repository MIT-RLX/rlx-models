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

//! `weights.ckpt` → graph-ready tensors.
//!
//! The checkpoint is a PyTorch Lightning file; the network sits under
//! `state_dict.model.` with the module names of `LaiaCRNN`
//! (`conv.{i}.conv`, `conv.{i}.batchnorm`, `rnn`, `linear`). Inference-mode
//! BatchNorm is an affine map per channel, so it is folded into the conv
//! that feeds it here, once, rather than run as a separate op.

use crate::config::PylaiaConfig;
use anyhow::{Context, Result, anyhow, ensure};
use std::collections::BTreeMap;
use std::path::Path;

/// `nn.BatchNorm2d` default.
const BN_EPS: f32 = 1e-5;

/// One conv block after BatchNorm folding: `[out, in, kh, kw]` + `[out]`.
#[derive(Debug, Clone)]
pub struct ConvWeights {
    pub weight: Vec<f32>,
    pub shape: [usize; 4],
    pub bias: Vec<f32>,
}

/// One bidirectional RNN layer, forward then reverse direction stacked on
/// the gate-row axis — the packing `Op::Lstm` / `Op::Gru` take.
#[derive(Debug, Clone)]
pub struct RnnLayerWeights {
    /// `[2 · gates · units, in]`.
    pub w_ih: Vec<f32>,
    /// `[2 · gates · units, units]`.
    pub w_hh: Vec<f32>,
    /// `[2 · gates · units]`.
    pub b_ih: Vec<f32>,
    /// `[2 · gates · units]`.
    pub b_hh: Vec<f32>,
    pub input: usize,
}

/// Every tensor the graph needs.
#[derive(Debug, Clone)]
pub struct PylaiaWeights {
    pub conv: Vec<ConvWeights>,
    pub rnn: Vec<RnnLayerWeights>,
    /// `[labels, 2 · units]`.
    pub linear_w: Vec<f32>,
    pub linear_b: Vec<f32>,
}

/// Name → `(data, shape)` for the network tensors of a checkpoint, with
/// the Lightning prefix stripped.
pub type TensorTable = BTreeMap<String, (Vec<f32>, Vec<usize>)>;

/// Read the `LaiaCRNN` tensors from a Lightning `weights.ckpt` (or a bare
/// state dict saved with `torch.save(model.state_dict())`).
pub fn read_checkpoint(path: &Path) -> Result<TensorTable> {
    let pt = rlx_torch_ckpt::PtModel::open(path)
        .with_context(|| format!("opening PyLaia checkpoint {}", path.display()))?;
    let names = pt.names();
    let prefix = ["state_dict.model.", "model.", ""]
        .into_iter()
        .find(|p| names.iter().any(|n| n == &format!("{p}linear.weight")))
        .ok_or_else(|| anyhow!("{}: no `linear.weight` — not a LaiaCRNN", path.display()))?;
    let mut out = TensorTable::new();
    for name in names {
        let Some(short) = name.strip_prefix(prefix) else {
            continue;
        };
        if short.ends_with("num_batches_tracked") {
            continue;
        }
        let t = pt.tensor(&name)?;
        out.insert(short.to_string(), (t.data, t.shape));
    }
    Ok(out)
}

impl PylaiaWeights {
    /// Pull and fold every tensor `cfg` needs, checking shapes as it goes.
    /// Fails if the checkpoint holds a network tensor nothing reads.
    pub fn from_table(cfg: &PylaiaConfig, mut t: TensorTable) -> Result<Self> {
        let mut take = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let (data, got) = t
                .remove(name)
                .ok_or_else(|| anyhow!("checkpoint is missing {name}"))?;
            ensure!(got == shape, "{name}: shape {got:?}, expected {shape:?}");
            Ok(data)
        };

        let mut conv = Vec::with_capacity(cfg.blocks.len());
        let mut cin = cfg.in_channels;
        for (i, b) in cfg.blocks.iter().enumerate() {
            let shape = [b.out_channels, cin, b.kernel[0], b.kernel[1]];
            let mut weight = take(&format!("conv.{i}.conv.weight"), &shape)?;
            let c = b.out_channels;
            let bias = if b.batchnorm {
                let p = format!("conv.{i}.batchnorm");
                let gamma = take(&format!("{p}.weight"), &[c])?;
                let beta = take(&format!("{p}.bias"), &[c])?;
                let mean = take(&format!("{p}.running_mean"), &[c])?;
                let var = take(&format!("{p}.running_var"), &[c])?;
                let per = weight.len() / c;
                (0..c)
                    .map(|o| {
                        let s = gamma[o] / (var[o] + BN_EPS).sqrt();
                        for w in &mut weight[o * per..(o + 1) * per] {
                            *w *= s;
                        }
                        beta[o] - mean[o] * s
                    })
                    .collect()
            } else {
                take(&format!("conv.{i}.conv.bias"), &[c])?
            };
            conv.push(ConvWeights {
                weight,
                shape,
                bias,
            });
            cin = c;
        }

        let u = cfg.rnn_units;
        let rows = cfg.rnn.gates() * u;
        let mut rnn = Vec::with_capacity(cfg.rnn_layers);
        for l in 0..cfg.rnn_layers {
            let input = if l == 0 { cfg.frame_features() } else { 2 * u };
            let mut layer = RnnLayerWeights {
                w_ih: Vec::with_capacity(2 * rows * input),
                w_hh: Vec::with_capacity(2 * rows * u),
                b_ih: Vec::with_capacity(2 * rows),
                b_hh: Vec::with_capacity(2 * rows),
                input,
            };
            for suffix in ["", "_reverse"] {
                layer.w_ih.extend(take(
                    &format!("rnn.weight_ih_l{l}{suffix}"),
                    &[rows, input],
                )?);
                layer
                    .w_hh
                    .extend(take(&format!("rnn.weight_hh_l{l}{suffix}"), &[rows, u])?);
                layer
                    .b_ih
                    .extend(take(&format!("rnn.bias_ih_l{l}{suffix}"), &[rows])?);
                layer
                    .b_hh
                    .extend(take(&format!("rnn.bias_hh_l{l}{suffix}"), &[rows])?);
            }
            rnn.push(layer);
        }

        let linear_w = take("linear.weight", &[cfg.num_labels, 2 * u])?;
        let linear_b = take("linear.bias", &[cfg.num_labels])?;
        ensure!(
            t.is_empty(),
            "checkpoint tensors not read by this config: {:?}",
            t.keys().collect::<Vec<_>>()
        );
        Ok(Self {
            conv,
            rnn,
            linear_w,
            linear_b,
        })
    }
}
