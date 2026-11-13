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

//! The part of Core ML's `Model.proto` / `NeuralNetwork.proto` a Kraken
//! `.mlmodel` uses, read straight off the protobuf wire format.
//!
//! Kraken serializes a VGSL network as a Core ML neural network whose
//! layers carry the weights (`{name}_conv`, `{name}_lin`, LSTMs under the
//! VGSL block name) and whose `userDefined` metadata holds the VGSL spec
//! (`vgsl`), the label codec (`codec`) and training metadata
//! (`kraken_meta`). Only the fields below are decoded; everything else is
//! skipped by wire type, so newer Core ML fields cannot break the reader.
//!
//! Kraken ≥ 6 also writes `.safetensors`; [`MlModel::open_safetensors`]
//! reads those into the same layout.

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::ordered_json::Json;
use std::collections::HashMap;
use std::path::Path;

// Field numbers, from coremltools `mlmodel/format/*.proto`.
const MODEL_DESCRIPTION: u32 = 2;
const MODEL_NEURAL_NETWORK: u32 = 500;
const DESCRIPTION_METADATA: u32 = 100;
const METADATA_USER_DEFINED: u32 = 100;
const NN_LAYERS: u32 = 1;
const LAYER_NAME: u32 = 1;
const LAYER_CONVOLUTION: u32 = 100;
const LAYER_INNER_PRODUCT: u32 = 140;
const LAYER_UNI_LSTM: u32 = 420;
const LAYER_BI_LSTM: u32 = 430;
const CONV_OUTPUT_CHANNELS: u32 = 1;
const CONV_KERNEL_CHANNELS: u32 = 2;
const CONV_GROUPS: u32 = 10;
const CONV_KERNEL_SIZE: u32 = 20;
const CONV_IS_DECONV: u32 = 60;
const CONV_HAS_BIAS: u32 = 70;
const CONV_WEIGHTS: u32 = 90;
const CONV_BIAS: u32 = 91;
const IP_INPUT: u32 = 1;
const IP_OUTPUT: u32 = 2;
const IP_HAS_BIAS: u32 = 10;
const IP_WEIGHTS: u32 = 20;
const IP_BIAS: u32 = 21;
const LSTM_INPUT_SIZE: u32 = 1;
const LSTM_OUTPUT_SIZE: u32 = 2;
const LSTM_PARAMS: u32 = 15;
const LSTM_WEIGHT_PARAMS: u32 = 20;
const LSTM_PARAMS_HAS_BIAS: u32 = 20;
const LSTM_PARAMS_HAS_PEEPHOLE: u32 = 40;
const WEIGHT_FLOAT: u32 = 1;
const WEIGHT_FLOAT16: u32 = 2;
const WEIGHT_RAW: u32 = 30;

/// One protobuf field value, borrowed from the file buffer.
#[derive(Clone, Copy, Debug)]
enum Wire<'a> {
    Varint(u64),
    Fixed64,
    Fixed32(u32),
    Bytes(&'a [u8]),
}

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut out = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *buf.get(*pos).ok_or_else(|| anyhow!("truncated varint"))?;
        *pos += 1;
        out |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(out);
        }
    }
    bail!("varint longer than 10 bytes")
}

/// The fields of one message level, in wire order.
fn fields(buf: &[u8]) -> Result<Vec<(u32, Wire<'_>)>> {
    let mut pos = 0;
    let mut out = Vec::new();
    while pos < buf.len() {
        let key = varint(buf, &mut pos)?;
        let number = u32::try_from(key >> 3).context("field number overflow")?;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8]> {
            let end = pos
                .checked_add(n)
                .filter(|&e| e <= buf.len())
                .ok_or_else(|| anyhow!("field {number} runs past the end"))?;
            let s = &buf[*pos..end];
            *pos = end;
            Ok(s)
        };
        let v = match key & 7 {
            0 => Wire::Varint(varint(buf, &mut pos)?),
            1 => {
                take(&mut pos, 8)?;
                Wire::Fixed64
            }
            2 => {
                let n = usize::try_from(varint(buf, &mut pos)?).context("length overflow")?;
                Wire::Bytes(take(&mut pos, n)?)
            }
            5 => {
                let b = take(&mut pos, 4)?;
                Wire::Fixed32(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            w => bail!("unsupported wire type {w} for field {number}"),
        };
        out.push((number, v));
    }
    Ok(out)
}

fn bytes_of<'a>(fs: &[(u32, Wire<'a>)], n: u32) -> Option<&'a [u8]> {
    fs.iter().find_map(|(k, v)| match (k, v) {
        (k, Wire::Bytes(b)) if *k == n => Some(*b),
        _ => None,
    })
}

fn all_bytes<'a>(fs: &[(u32, Wire<'a>)], n: u32) -> Vec<&'a [u8]> {
    fs.iter()
        .filter_map(|(k, v)| match v {
            Wire::Bytes(b) if *k == n => Some(*b),
            _ => None,
        })
        .collect()
}

fn uint_of(fs: &[(u32, Wire<'_>)], n: u32) -> u64 {
    fs.iter()
        .find_map(|(k, v)| match v {
            Wire::Varint(x) if *k == n => Some(*x),
            _ => None,
        })
        .unwrap_or(0)
}

/// A repeated `uint64`, packed or not.
fn uints_of(fs: &[(u32, Wire<'_>)], n: u32) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for (k, v) in fs {
        if *k != n {
            continue;
        }
        match v {
            Wire::Varint(x) => out.push(*x),
            Wire::Bytes(b) => {
                let mut p = 0;
                while p < b.len() {
                    out.push(varint(b, &mut p)?);
                }
            }
            _ => bail!("field {n}: unexpected wire type"),
        }
    }
    Ok(out)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: m · 2^-24, exactly representable in f32.
        (0, m) => sign | (m as f32 * 2f32.powi(-24)).to_bits(),
        (31, 0) => sign | 0x7f80_0000,
        (31, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// A `WeightParams` message as `f32`s (fp32 packed, fp16, or raw fp32).
fn weights(msg: &[u8]) -> Result<Vec<f32>> {
    let fs = fields(msg)?;
    let mut out = Vec::new();
    for (k, v) in &fs {
        match (*k, v) {
            (WEIGHT_FLOAT, Wire::Bytes(b)) => {
                out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)))
            }
            (WEIGHT_FLOAT, Wire::Fixed32(x)) => out.push(f32::from_bits(*x)),
            (WEIGHT_FLOAT16, Wire::Bytes(b)) => out.extend(
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16_to_f32(u16::from_le_bytes(*c))),
            ),
            (WEIGHT_RAW, Wire::Bytes(b)) if !b.is_empty() => {
                out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)))
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A convolution layer's weights.
#[derive(Debug, Clone)]
pub struct ConvLayer {
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel: [usize; 2],
    /// `[out, in, kh, kw]`.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// An inner-product (linear) layer's weights.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub input: usize,
    pub output: usize,
    /// `[output, input]`.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One LSTM direction in PyTorch gate order (i, f, g, o).
#[derive(Debug, Clone)]
pub struct LstmDirection {
    /// `[4 · hidden, input]`.
    pub w_ih: Vec<f32>,
    /// `[4 · hidden, hidden]`.
    pub w_hh: Vec<f32>,
    /// `[4 · hidden]`, `b_ih + b_hh` (Core ML keeps one bias).
    pub bias: Vec<f32>,
}

/// An LSTM layer: one direction, or forward + backward.
#[derive(Debug, Clone)]
pub struct LstmLayer {
    pub input: usize,
    pub hidden: usize,
    pub directions: Vec<LstmDirection>,
}

/// What the recognizer needs from an `.mlmodel`.
#[derive(Debug, Clone, Default)]
pub struct MlModel {
    /// `userDefined` metadata (`vgsl`, `codec`, `kraken_meta`, …).
    pub metadata: HashMap<String, String>,
    pub convs: HashMap<String, ConvLayer>,
    pub linears: HashMap<String, LinearLayer>,
    pub lstms: HashMap<String, LstmLayer>,
}

impl MlModel {
    pub fn open(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&buf).with_context(|| format!("parsing Core ML model {}", path.display()))
    }

    pub fn parse(buf: &[u8]) -> Result<Self> {
        let top = fields(buf)?;
        let mut out = MlModel::default();
        if let Some(desc) = bytes_of(&top, MODEL_DESCRIPTION) {
            let desc = fields(desc)?;
            if let Some(meta) = bytes_of(&desc, DESCRIPTION_METADATA) {
                let meta = fields(meta)?;
                for entry in all_bytes(&meta, METADATA_USER_DEFINED) {
                    let e = fields(entry)?;
                    let s = |n| {
                        bytes_of(&e, n)
                            .map(|b| String::from_utf8_lossy(b).into_owned())
                            .unwrap_or_default()
                    };
                    out.metadata.insert(s(1), s(2));
                }
            }
        }
        let nn = bytes_of(&top, MODEL_NEURAL_NETWORK)
            .ok_or_else(|| anyhow!("not a Core ML neural network"))?;
        for layer in all_bytes(&fields(nn)?, NN_LAYERS) {
            let lf = fields(layer)?;
            let name = bytes_of(&lf, LAYER_NAME)
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default();
            if let Some(c) = bytes_of(&lf, LAYER_CONVOLUTION) {
                out.convs.insert(name.clone(), conv_layer(c, &name)?);
            } else if let Some(ip) = bytes_of(&lf, LAYER_INNER_PRODUCT) {
                out.linears.insert(name.clone(), linear_layer(ip, &name)?);
            } else if let Some(l) = bytes_of(&lf, LAYER_BI_LSTM) {
                out.lstms.insert(name.clone(), lstm_layer(l, true, &name)?);
            } else if let Some(l) = bytes_of(&lf, LAYER_UNI_LSTM) {
                out.lstms.insert(name.clone(), lstm_layer(l, false, &name)?);
            }
        }
        Ok(out)
    }
}

impl MlModel {
    /// A Kraken ≥ 6 `.safetensors` model: the `TorchVGSLModel` state dict
    /// under a per-model UUID prefix, and a `kraken_meta` header entry
    /// mapping each prefix to its metadata (`vgsl`, `codec`, …). Returns
    /// the recognition model's weights in the same layout the Core ML
    /// reader produces, and its metadata object.
    pub fn open_safetensors(path: &Path) -> Result<(Self, Json)> {
        let meta = safetensors_kraken_meta(path)?;
        let entries = meta
            .as_object()
            .ok_or_else(|| anyhow!("kraken_meta is not an object"))?;
        let (prefix, info) = entries
            .iter()
            .find(|(_, v)| is_recognition(v))
            .ok_or_else(|| anyhow!("{}: no recognition model in kraken_meta", path.display()))?;
        let mut w = rlx_core::load_weight_map(path, &[])?;
        let pfx = format!("{prefix}.nn.");
        let names: Vec<String> = w
            .keys()
            .filter(|k| k.starts_with(&pfx))
            .map(String::from)
            .collect();
        let mut out = MlModel::default();
        let mut take = |n: &str| w.take(n);
        for name in &names {
            let rest = &name[pfx.len()..];
            if let Some(block) = rest.strip_suffix(".co.weight") {
                let (weight, s) = take(name)?;
                ensure!(s.len() == 4, "{name}: shape {s:?}");
                let (bias, _) = take(&format!("{pfx}{block}.co.bias"))?;
                out.convs.insert(
                    format!("{block}_conv"),
                    ConvLayer {
                        out_channels: s[0],
                        in_channels: s[1],
                        kernel: [s[2], s[3]],
                        weight,
                        bias,
                    },
                );
            } else if let Some(block) = rest.strip_suffix(".lin.weight") {
                let (weight, s) = take(name)?;
                let (bias, _) = take(&format!("{pfx}{block}.lin.bias"))?;
                out.linears.insert(
                    format!("{block}_lin"),
                    LinearLayer {
                        input: s[1],
                        output: s[0],
                        weight,
                        bias,
                    },
                );
            } else if let Some(block) = rest.strip_suffix(".layer.weight_ih_l0") {
                let lp = format!("{pfx}{block}.layer");
                let mut directions = Vec::new();
                let mut dims = (0, 0);
                for sfx in ["", "_reverse"] {
                    let Ok((w_ih, s)) = take(&format!("{lp}.weight_ih_l0{sfx}")) else {
                        continue;
                    };
                    let (w_hh, _) = take(&format!("{lp}.weight_hh_l0{sfx}"))?;
                    let (b_ih, _) = take(&format!("{lp}.bias_ih_l0{sfx}"))?;
                    let (b_hh, _) = take(&format!("{lp}.bias_hh_l0{sfx}"))?;
                    dims = (s[1], s[0] / 4);
                    directions.push(LstmDirection {
                        w_ih,
                        w_hh,
                        bias: b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect(),
                    });
                }
                out.lstms.insert(
                    block.to_string(),
                    LstmLayer {
                        input: dims.0,
                        hidden: dims.1,
                        directions,
                    },
                );
            }
        }
        Ok((out, info.clone()))
    }
}

/// `model_type` / `_tasks` holds `recognition` (a string or a list).
fn is_recognition(v: &Json) -> bool {
    ["model_type", "_tasks"].iter().any(|k| match v.get(k) {
        Some(Json::Str(s)) => s.contains("recognition"),
        Some(Json::Array(a)) => a.iter().any(|x| x.as_str() == Some("recognition")),
        _ => false,
    })
}

/// The `kraken_meta` entry of a safetensors header, parsed.
fn safetensors_kraken_meta(path: &Path) -> Result<Json> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = usize::try_from(u64::from_le_bytes(n)).context("header length")?;
    ensure!(
        n < 1 << 28,
        "{}: implausible safetensors header",
        path.display()
    );
    let mut hdr = vec![0u8; n];
    f.read_exact(&mut hdr)?;
    let hdr = Json::parse(std::str::from_utf8(&hdr)?).map_err(|e| anyhow!("header: {e:?}"))?;
    let meta = hdr
        .get("__metadata__")
        .and_then(|m| m.get("kraken_meta"))
        .and_then(Json::as_str)
        .ok_or_else(|| {
            anyhow!(
                "{}: no kraken_meta in the safetensors header",
                path.display()
            )
        })?;
    Json::parse(meta).map_err(|e| anyhow!("kraken_meta: {e:?}"))
}

fn conv_layer(msg: &[u8], name: &str) -> Result<ConvLayer> {
    let f = fields(msg)?;
    ensure!(
        uint_of(&f, CONV_IS_DECONV) == 0,
        "{name}: transposed convolutions are not supported"
    );
    let groups = uint_of(&f, CONV_GROUPS).max(1);
    ensure!(groups == 1, "{name}: grouped convolution ({groups} groups)");
    let out_channels = uint_of(&f, CONV_OUTPUT_CHANNELS) as usize;
    let in_channels = uint_of(&f, CONV_KERNEL_CHANNELS) as usize;
    let ks = uints_of(&f, CONV_KERNEL_SIZE)?;
    ensure!(ks.len() == 2, "{name}: kernel size {ks:?}");
    let kernel = [ks[0] as usize, ks[1] as usize];
    let weight = weights(bytes_of(&f, CONV_WEIGHTS).unwrap_or_default())?;
    ensure!(
        weight.len() == out_channels * in_channels * kernel[0] * kernel[1],
        "{name}: {} weights for [{out_channels}, {in_channels}, {}, {}]",
        weight.len(),
        kernel[0],
        kernel[1]
    );
    let bias = if uint_of(&f, CONV_HAS_BIAS) != 0 {
        weights(bytes_of(&f, CONV_BIAS).unwrap_or_default())?
    } else {
        vec![0.0; out_channels]
    };
    ensure!(
        bias.len() == out_channels,
        "{name}: bias length {}",
        bias.len()
    );
    Ok(ConvLayer {
        out_channels,
        in_channels,
        kernel,
        weight,
        bias,
    })
}

fn linear_layer(msg: &[u8], name: &str) -> Result<LinearLayer> {
    let f = fields(msg)?;
    let input = uint_of(&f, IP_INPUT) as usize;
    let output = uint_of(&f, IP_OUTPUT) as usize;
    let weight = weights(bytes_of(&f, IP_WEIGHTS).unwrap_or_default())?;
    ensure!(
        weight.len() == input * output,
        "{name}: {} weights for [{output}, {input}]",
        weight.len()
    );
    let bias = if uint_of(&f, IP_HAS_BIAS) != 0 {
        weights(bytes_of(&f, IP_BIAS).unwrap_or_default())?
    } else {
        vec![0.0; output]
    };
    ensure!(bias.len() == output, "{name}: bias length {}", bias.len());
    Ok(LinearLayer {
        input,
        output,
        weight,
        bias,
    })
}

fn lstm_layer(msg: &[u8], bidirectional: bool, name: &str) -> Result<LstmLayer> {
    let f = fields(msg)?;
    let input = uint_of(&f, LSTM_INPUT_SIZE) as usize;
    let hidden = uint_of(&f, LSTM_OUTPUT_SIZE) as usize;
    let params = fields(bytes_of(&f, LSTM_PARAMS).unwrap_or_default())?;
    ensure!(
        uint_of(&params, LSTM_PARAMS_HAS_PEEPHOLE) == 0,
        "{name}: peephole LSTMs (ocropy legacy models) are not supported"
    );
    let has_bias = uint_of(&params, LSTM_PARAMS_HAS_BIAS) != 0;
    let dirs = all_bytes(&f, LSTM_WEIGHT_PARAMS);
    ensure!(
        dirs.len() == if bidirectional { 2 } else { 1 },
        "{name}: {} weight sets",
        dirs.len()
    );
    let directions = dirs
        .into_iter()
        .map(|d| lstm_direction(d, input, hidden, has_bias, name))
        .collect::<Result<_>>()?;
    Ok(LstmLayer {
        input,
        hidden,
        directions,
    })
}

/// `LSTMWeightParams` → PyTorch's packed (i, f, g, o) rows. Core ML names
/// the cell candidate `blockInput` (z); Kraken writes each gate to its
/// named field, so reading by name gives PyTorch's order directly.
fn lstm_direction(
    msg: &[u8],
    input: usize,
    hidden: usize,
    has_bias: bool,
    name: &str,
) -> Result<LstmDirection> {
    // (W, R, b) field numbers per gate, in PyTorch order i, f, g(z), o.
    const GATES: [(u32, u32, u32); 4] = [(1, 20, 40), (2, 21, 41), (3, 22, 42), (4, 23, 43)];
    let f = fields(msg)?;
    let get = |n: u32| weights(bytes_of(&f, n).unwrap_or_default());
    let (mut w_ih, mut w_hh, mut bias) = (Vec::new(), Vec::new(), Vec::new());
    for (w, r, b) in GATES {
        let wv = get(w)?;
        ensure!(
            wv.len() == hidden * input,
            "{name}: W gate {w} has {}",
            wv.len()
        );
        let rv = get(r)?;
        ensure!(
            rv.len() == hidden * hidden,
            "{name}: R gate {r} has {}",
            rv.len()
        );
        w_ih.extend(wv);
        w_hh.extend(rv);
        if has_bias {
            let bv = get(b)?;
            ensure!(bv.len() == hidden, "{name}: bias gate {b} has {}", bv.len());
            bias.extend(bv);
        } else {
            bias.extend(std::iter::repeat_n(0.0, hidden));
        }
    }
    Ok(LstmDirection { w_ih, w_hh, bias })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(n: u32, w: u8) -> Vec<u8> {
        let mut out = Vec::new();
        let mut k = (u64::from(n) << 3) | u64::from(w);
        loop {
            let b = (k & 0x7f) as u8;
            k >>= 7;
            if k == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
        out
    }

    fn msg(n: u32, body: &[u8]) -> Vec<u8> {
        let mut out = tag(n, 2);
        out.push(body.len() as u8);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn packed_float_weights() {
        let data: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(weights(&msg(WEIGHT_FLOAT, &data)).unwrap(), vec![1.5, -2.0]);
    }

    #[test]
    fn metadata_and_linear_layer() {
        let entry = [msg(1, b"vgsl"), msg(2, b"[1,8,0,1 O1c3]")].concat();
        let desc = msg(DESCRIPTION_METADATA, &msg(METADATA_USER_DEFINED, &entry));
        let w: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let ip = [
            tag(IP_INPUT, 0),
            vec![2],
            tag(IP_OUTPUT, 0),
            vec![3],
            msg(IP_WEIGHTS, &msg(WEIGHT_FLOAT, &w)),
        ]
        .concat();
        let layer = [msg(LAYER_NAME, b"O_0_lin"), msg(LAYER_INNER_PRODUCT, &ip)].concat();
        let nn = msg(NN_LAYERS, &layer);
        let mut model = msg(MODEL_DESCRIPTION, &desc);
        model.extend(tag(MODEL_NEURAL_NETWORK, 2));
        model.push(nn.len() as u8);
        model.extend(nn);
        let m = MlModel::parse(&model).unwrap();
        assert_eq!(m.metadata["vgsl"], "[1,8,0,1 O1c3]");
        let l = &m.linears["O_0_lin"];
        assert_eq!((l.input, l.output), (2, 3));
        assert_eq!(l.weight, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(l.bias, vec![0.0; 3]);
    }
}
