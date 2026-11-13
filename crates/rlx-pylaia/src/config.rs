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

//! `LaiaCRNN` hyperparameters, read from the `model` file PyLaia saves
//! next to `weights.ckpt`.
//!
//! That file is a `torch.save` of `{"module", "name", "args", "kwargs"}`
//! where most numbers are jsonargparse wrappers (`PositiveInt(3)`) and the
//! activation / RNN entries are class references. Nothing is executed:
//! the restricted pickle VM in `rlx-torch-ckpt` keeps both symbolically.

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_torch_ckpt::pickle::Value;
use std::path::Path;

/// Per-block activation (`cnn_activation`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// `torch.nn.LeakyReLU` with its default slope 0.01.
    LeakyRelu(f32),
    Relu,
    Tanh,
    /// No activation.
    Identity,
}

/// Recurrent layer type (`rnn_type`). Always bidirectional in `LaiaCRNN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnKind {
    Lstm,
    Gru,
}

impl RnnKind {
    /// Gate rows per hidden unit in the packed weights.
    pub fn gates(self) -> usize {
        match self {
            RnnKind::Lstm => 4,
            RnnKind::Gru => 3,
        }
    }
}

/// How the CNN feature map becomes a sequence (`image_sequencer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sequencer {
    /// `none-N`: the CNN output must already be `N` rows high.
    Fixed(usize),
    /// `avgpool-N`: adaptive average pooling of the height down to `N`.
    AvgPool(usize),
}

impl Sequencer {
    /// Rows per column after sequencing (`fix_size`).
    pub fn height(self) -> usize {
        match self {
            Sequencer::Fixed(n) | Sequencer::AvgPool(n) => n,
        }
    }
}

/// One `ConvBlock`: conv → (batchnorm) → activation → (max-pool).
#[derive(Debug, Clone, PartialEq)]
pub struct ConvBlockConfig {
    pub out_channels: usize,
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    /// Max-pool window (= stride); `None` when `ph · pw < 2`.
    pub pool: Option<[usize; 2]>,
    pub batchnorm: bool,
    pub activation: Activation,
}

impl ConvBlockConfig {
    /// Same-size padding `(k - 1) / 2 · d`, as `ConvBlock` sets it.
    pub fn padding(&self) -> [usize; 2] {
        [0, 1].map(|d| (self.kernel[d] - 1) / 2 * self.dilation[d])
    }

    /// Output `[h, w]` for an input `[h, w]` (conv, then floor pooling).
    pub fn output_size(&self, size: [usize; 2]) -> Option<[usize; 2]> {
        let pad = self.padding();
        let mut out = [0; 2];
        for d in 0..2 {
            let span = self.dilation[d] * (self.kernel[d] - 1) + 1;
            let padded = size[d] + 2 * pad[d];
            if padded < span {
                return None;
            }
            let conv = (padded - span) / self.stride[d] + 1;
            out[d] = match self.pool {
                Some(p) => conv / p[d],
                None => conv,
            };
            if out[d] == 0 {
                return None;
            }
        }
        Some(out)
    }
}

/// A full `LaiaCRNN`.
#[derive(Debug, Clone, PartialEq)]
pub struct PylaiaConfig {
    /// 1 for grayscale (`L`), 3 for RGB.
    pub in_channels: usize,
    pub blocks: Vec<ConvBlockConfig>,
    pub sequencer: Sequencer,
    pub rnn: RnnKind,
    pub rnn_units: usize,
    pub rnn_layers: usize,
    /// CTC classes including the blank at index 0.
    pub num_labels: usize,
}

impl PylaiaConfig {
    /// Read the `model` file PyLaia writes beside `weights.ckpt`.
    pub fn from_model_file(path: &Path) -> Result<Self> {
        let root = rlx_torch_ckpt::read_torch_pickle_root(path)
            .with_context(|| format!("reading PyLaia model file {}", path.display()))?;
        let name = dict_get(&root, "name").and_then(|v| as_str(&v));
        if let Some(n) = name.as_deref() {
            ensure!(n == "LaiaCRNN", "PyLaia model {n:?} is not a LaiaCRNN");
        }
        let kw = dict_get(&root, "kwargs").ok_or_else(|| anyhow!("model file has no kwargs"))?;
        Self::from_kwargs(&kw)
    }

    /// Build from the `kwargs` mapping of a PyLaia `model` file.
    pub fn from_kwargs(kw: &Value) -> Result<Self> {
        let get = |k: &str| dict_get(kw, k).ok_or_else(|| anyhow!("kwargs.{k} missing"));
        let features = list_of(&get("cnn_num_features")?, as_usize, "cnn_num_features")?;
        let n = features.len();
        let pairs = |k: &str| -> Result<Vec<[usize; 2]>> {
            let v = list_of(&get(k)?, as_pair, k)?;
            ensure!(v.len() == n, "{k}: {} entries for {n} blocks", v.len());
            Ok(v)
        };
        let kernel = pairs("cnn_kernel_size")?;
        let stride = pairs("cnn_stride")?;
        let dilation = pairs("cnn_dilation")?;
        let pool = pairs("cnn_poolsize")?;
        let bn = list_of(&get("cnn_batchnorm")?, as_bool, "cnn_batchnorm")?;
        let acts = list_of(&get("cnn_activation")?, as_activation, "cnn_activation")?;
        ensure!(
            bn.len() == n && acts.len() == n,
            "cnn_* lists disagree on block count"
        );
        if dict_get(kw, "vertical_text").and_then(|v| as_bool(&v)) == Some(true) {
            bail!("vertical_text PyLaia models are not supported");
        }
        let blocks = (0..n)
            .map(|i| ConvBlockConfig {
                out_channels: features[i],
                kernel: kernel[i],
                stride: stride[i],
                dilation: dilation[i],
                pool: (pool[i][0] * pool[i][1] >= 2).then_some(pool[i]),
                batchnorm: bn[i],
                activation: acts[i],
            })
            .collect();
        let seq = as_str(&get("image_sequencer")?).ok_or_else(|| anyhow!("image_sequencer"))?;
        let sequencer = parse_sequencer(&seq)?;
        let rnn = match global_name(&get("rnn_type")?).as_deref() {
            Some("LSTM") => RnnKind::Lstm,
            Some("GRU") => RnnKind::Gru,
            other => bail!("unsupported rnn_type {other:?}"),
        };
        let in_channels = dict_get(kw, "num_input_channels")
            .and_then(|v| as_usize(&v))
            .unwrap_or(1);
        Ok(Self {
            in_channels,
            blocks,
            sequencer,
            rnn,
            rnn_units: as_usize(&get("rnn_units")?).ok_or_else(|| anyhow!("rnn_units"))?,
            rnn_layers: as_usize(&get("rnn_layers")?).ok_or_else(|| anyhow!("rnn_layers"))?,
            num_labels: as_usize(&get("num_output_labels")?)
                .ok_or_else(|| anyhow!("num_output_labels"))?,
        })
    }

    /// CNN output `[h, w]` for an input image `[h, w]`, or `None` when the
    /// image is too small for some block.
    pub fn conv_output_size(&self, size: [usize; 2]) -> Option<[usize; 2]> {
        self.blocks.iter().try_fold(size, |s, b| b.output_size(s))
    }

    /// Feature width of each sequence frame (`C_last · fix_size`).
    pub fn frame_features(&self) -> usize {
        self.blocks
            .last()
            .map_or(self.in_channels, |b| b.out_channels)
            * self.sequencer.height()
    }

    /// Smallest image width that yields at least one frame (PyLaia's
    /// `get_min_valid_image_size`); lines are padded up to it.
    pub fn min_width(&self, height: usize) -> usize {
        (1..4096)
            .find(|&w| self.conv_output_size([height, w]).is_some())
            .unwrap_or(1)
    }

    /// The architecture every Teklia PyLaia release on the Hub uses
    /// (RIMES, IAM, NorHand, …): four 3×3 conv blocks of 12/24/48/48
    /// channels with batchnorm and LeakyReLU, 2×2 pools on blocks 1, 2, 4,
    /// a `none-16` sequencer, and a 3-layer BiLSTM of 256 units.
    pub fn teklia_default(num_labels: usize) -> Self {
        let block = |c, pool: Option<[usize; 2]>| ConvBlockConfig {
            out_channels: c,
            kernel: [3, 3],
            stride: [1, 1],
            dilation: [1, 1],
            pool,
            batchnorm: true,
            activation: Activation::LeakyRelu(0.01),
        };
        Self {
            in_channels: 1,
            blocks: vec![
                block(12, Some([2, 2])),
                block(24, Some([2, 2])),
                block(48, None),
                block(48, Some([2, 2])),
            ],
            sequencer: Sequencer::Fixed(16),
            rnn: RnnKind::Lstm,
            rnn_units: 256,
            rnn_layers: 3,
            num_labels,
        }
    }
}

fn parse_sequencer(s: &str) -> Result<Sequencer> {
    let (kind, n) = s
        .split_once('-')
        .ok_or_else(|| anyhow!("image_sequencer {s:?}"))?;
    let n: usize = n
        .parse()
        .with_context(|| format!("image_sequencer {s:?}"))?;
    ensure!(n > 0, "image_sequencer {s:?}: size 0");
    match kind {
        "none" => Ok(Sequencer::Fixed(n)),
        "avgpool" => Ok(Sequencer::AvgPool(n)),
        other => bail!("image_sequencer kind {other:?} is not supported (none / avgpool)"),
    }
}

/// Strip jsonargparse wrappers: `PositiveInt(3)` → `3`.
fn unwrap(v: &Value) -> Value {
    match v {
        Value::Object { args, .. } if args.len() == 1 => unwrap(&args[0]),
        other => other.clone(),
    }
}

pub(crate) fn dict_get(v: &Value, key: &str) -> Option<Value> {
    let Value::Dict(d) = v else { return None };
    d.borrow()
        .iter()
        .find(|(k, _)| matches!(k, Value::Str(s) if &**s == key))
        .map(|(_, v)| unwrap(v))
}

fn items(v: &Value) -> Option<Vec<Value>> {
    match v {
        Value::List(l) => Some(l.borrow().iter().map(unwrap).collect()),
        Value::Tuple(t) => Some(t.iter().map(unwrap).collect()),
        _ => None,
    }
}

fn list_of<T>(v: &Value, f: impl Fn(&Value) -> Option<T>, what: &str) -> Result<Vec<T>> {
    items(v)
        .ok_or_else(|| anyhow!("{what}: expected a list"))?
        .iter()
        .enumerate()
        .map(|(i, x)| f(x).ok_or_else(|| anyhow!("{what}[{i}]: unexpected value {x:?}")))
        .collect()
}

fn as_usize(v: &Value) -> Option<usize> {
    match unwrap(v) {
        Value::Int(i) => usize::try_from(i).ok(),
        Value::Bool(b) => Some(b as usize),
        _ => None,
    }
}

fn as_bool(v: &Value) -> Option<bool> {
    match unwrap(v) {
        Value::Bool(b) => Some(b),
        Value::Int(i) => Some(i != 0),
        _ => None,
    }
}

fn as_str(v: &Value) -> Option<String> {
    match unwrap(v) {
        Value::Str(s) => Some(s.to_string()),
        _ => None,
    }
}

/// A scalar applies to both axes (`ConvBlock.prepare_dimensional_args`).
fn as_pair(v: &Value) -> Option<[usize; 2]> {
    if let Some(n) = as_usize(v) {
        return Some([n, n]);
    }
    let xs = items(v)?;
    match xs.as_slice() {
        [a, b] => Some([as_usize(a)?, as_usize(b)?]),
        _ => None,
    }
}

fn global_name(v: &Value) -> Option<String> {
    match v {
        Value::Global(_, n) => Some(n.to_string()),
        Value::Object { name, .. } => Some(name.to_string()),
        _ => None,
    }
}

fn as_activation(v: &Value) -> Option<Activation> {
    if matches!(v, Value::None) {
        return Some(Activation::Identity);
    }
    match global_name(v)?.as_str() {
        "LeakyReLU" => Some(Activation::LeakyRelu(0.01)),
        "ReLU" => Some(Activation::Relu),
        "Tanh" => Some(Activation::Tanh),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teklia_geometry() {
        let c = PylaiaConfig::teklia_default(100);
        assert_eq!(c.conv_output_size([128, 2560]), Some([16, 320]));
        assert_eq!(c.frame_features(), 768);
        // Three 2× pools: an 8-pixel image is the smallest with one frame.
        assert_eq!(c.min_width(128), 8);
        assert_eq!(c.conv_output_size([128, 7]), None);
    }

    #[test]
    fn sequencer_parsing() {
        assert_eq!(parse_sequencer("none-16").unwrap(), Sequencer::Fixed(16));
        assert_eq!(parse_sequencer("avgpool-8").unwrap(), Sequencer::AvgPool(8));
        assert!(parse_sequencer("maxpool-8").is_err());
    }
}
