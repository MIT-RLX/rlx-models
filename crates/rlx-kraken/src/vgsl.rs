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

//! VGSL — the network description language Kraken stores in every model
//! (`[1,120,0,1 Cr3,13,32 Mp2,2 … Lbx200 O1c225]`).
//!
//! Covered: the input block, `C` convolutions (all activations, stride,
//! dilation), `Mp` max-pooling, `Do` dropout (a no-op at inference), `S`
//! reshapes, `L` LSTMs along x or y (uni- or bidirectional, optionally
//! summarizing) and `O1` linear outputs. Kraken's own recognition models
//! use exactly these; parallel/serial sub-blocks, GRUs, group norm and
//! transposed convolutions are rejected with an error naming the block.

use anyhow::{Context, Result, anyhow, bail, ensure};

/// `C` activation letter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvAct {
    /// `s` — Kraken keeps sigmoid outputs as logits (no activation applied).
    Sigmoid,
    Tanh,
    Relu,
    /// `l` — linear.
    Linear,
    /// `lr` — `torch.nn.LeakyReLU()` (slope 0.01).
    LeakyRelu,
    /// `m` — softmax over channels.
    Softmax,
}

/// `O` output nonlinearity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    /// `c` — CTC (softmax over classes).
    Ctc,
    /// `s` — softmax.
    Softmax,
    /// `l` — sigmoid (logits kept).
    Sigmoid,
}

/// One VGSL block.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Conv {
        name: String,
        act: ConvAct,
        kernel: [usize; 2],
        out: usize,
        stride: [usize; 2],
        dilation: [usize; 2],
    },
    MaxPool {
        name: String,
        kernel: [usize; 2],
        stride: [usize; 2],
    },
    Dropout {
        name: String,
    },
    /// `S<d>(<a>x<b>)<e>,<f>` in VGSL axis numbers (0 batch, 1 height,
    /// 2 width, 3 depth); `0` in `a` or `b` means "infer".
    Reshape {
        name: String,
        dim: usize,
        part_a: usize,
        part_b: usize,
        high: usize,
        low: usize,
    },
    Lstm {
        name: String,
        bidirectional: bool,
        /// `y`: run along the height (Kraken's `transpose`).
        along_y: bool,
        summarize: bool,
        hidden: usize,
    },
    Output {
        name: String,
        kind: OutputKind,
        augment: bool,
        out: usize,
    },
}

impl Block {
    pub fn name(&self) -> &str {
        match self {
            Block::Conv { name, .. }
            | Block::MaxPool { name, .. }
            | Block::Dropout { name }
            | Block::Reshape { name, .. }
            | Block::Lstm { name, .. }
            | Block::Output { name, .. } => name,
        }
    }
}

/// `batch, height, width, channels` of the input block; `0` = variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSpec {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

/// A parsed spec.
#[derive(Debug, Clone, PartialEq)]
pub struct Vgsl {
    pub input: InputSpec,
    pub blocks: Vec<Block>,
}

impl Vgsl {
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim();
        let s = s
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| anyhow!("VGSL spec {spec:?} is not a single [ … ] block"))?;
        let mut toks = s.split_whitespace();
        let input = toks.next().ok_or_else(|| anyhow!("empty VGSL spec"))?;
        let dims = nums(input).with_context(|| format!("VGSL input block {input:?}"))?;
        ensure!(dims.len() == 4, "VGSL input block {input:?} needs b,h,w,c");
        let input = InputSpec {
            batch: dims[0],
            height: dims[1],
            width: dims[2],
            channels: dims[3],
        };
        let blocks = toks
            .enumerate()
            .map(|(i, t)| parse_block(t, i).with_context(|| format!("VGSL block {t:?}")))
            .collect::<Result<_>>()?;
        Ok(Self { input, blocks })
    }
}

fn nums(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|x| x.parse().with_context(|| format!("number {x:?}")))
        .collect()
}

/// Split an optional `{name}` off the front; Kraken's default is
/// `{kind}_{index}`.
fn named<'a>(rest: &'a str, kind: &str, idx: usize) -> (String, &'a str) {
    if let Some(r) = rest.strip_prefix('{')
        && let Some(end) = r.find('}')
    {
        return (r[..end].to_string(), &r[end + 1..]);
    }
    (format!("{kind}_{idx}"), rest)
}

fn parse_block(t: &str, idx: usize) -> Result<Block> {
    if t.starts_with(['(', '[']) {
        bail!("parallel / serial sub-blocks are not supported");
    }
    if let Some(r) = t.strip_prefix("Mp") {
        let (name, r) = named(r, "Mp", idx);
        let n = nums(r)?;
        let (kernel, stride) = match n.as_slice() {
            [y, x] => ([*y, *x], [*y, *x]),
            [y, x, sy, sx] => ([*y, *x], [*sy, *sx]),
            _ => bail!("max-pool wants y,x[,sy,sx]"),
        };
        return Ok(Block::MaxPool {
            name,
            kernel,
            stride,
        });
    }
    if let Some(r) = t.strip_prefix("Do") {
        let (name, _) = named(r, "Do", idx);
        return Ok(Block::Dropout { name });
    }
    if t.starts_with("Gn") {
        bail!("group norm is not supported");
    }
    if let Some(r) = t.strip_prefix('C') {
        ensure!(
            !r.starts_with('T'),
            "transposed convolutions are not supported"
        );
        let (act, r) = if let Some(r) = r.strip_prefix("lr") {
            (ConvAct::LeakyRelu, r)
        } else {
            let a = match r.chars().next() {
                Some('s') => ConvAct::Sigmoid,
                Some('t') => ConvAct::Tanh,
                Some('r') => ConvAct::Relu,
                Some('l') => ConvAct::Linear,
                Some('m') => ConvAct::Softmax,
                other => bail!("unknown conv activation {other:?}"),
            };
            (a, &r[1..])
        };
        let (name, r) = named(r, "C", idx);
        let n = nums(r)?;
        let (stride, dilation) = match n.len() {
            3 => ([1, 1], [1, 1]),
            5 => ([n[3], n[4]], [1, 1]),
            7 => ([n[3], n[4]], [n[5], n[6]]),
            _ => bail!("conv wants y,x,d[,sy,sx][,dy,dx]"),
        };
        return Ok(Block::Conv {
            name,
            act,
            kernel: [n[0], n[1]],
            out: n[2],
            stride,
            dilation,
        });
    }
    if let Some(r) = t.strip_prefix('S') {
        let (name, r) = named(r, "S", idx);
        let (dim, r) = r
            .split_once('(')
            .ok_or_else(|| anyhow!("reshape wants d(axb)e,f"))?;
        let (ab, ef) = r
            .split_once(')')
            .ok_or_else(|| anyhow!("reshape wants d(axb)e,f"))?;
        let (a, b) = ab
            .split_once('x')
            .ok_or_else(|| anyhow!("reshape wants (axb)"))?;
        let ef = nums(ef)?;
        ensure!(ef.len() == 2, "reshape wants two target dims");
        let block = Block::Reshape {
            name,
            dim: dim.parse()?,
            part_a: a.parse()?,
            part_b: b.parse()?,
            high: ef[0],
            low: ef[1],
        };
        if let Block::Reshape {
            dim,
            part_a,
            part_b,
            high,
            low,
            ..
        } = &block
        {
            ensure!(
                *dim < 4 && *high < 4 && *low < 4,
                "reshape axis out of range"
            );
            ensure!(
                dim == high || dim == low,
                "reshape: either high or low must be the source axis"
            );
            ensure!(
                *part_a != 0 || *part_b != 0,
                "reshape: only one part may be 0"
            );
        }
        return Ok(block);
    }
    if t.starts_with('G') {
        bail!("GRU layers are not supported");
    }
    if let Some(r) = t.strip_prefix('L') {
        let mut c = r.chars();
        let bidirectional = match c.next() {
            Some('b') => true,
            Some('f' | 'r') => false,
            other => bail!("LSTM direction {other:?}"),
        };
        let along_y = match c.next() {
            Some('y') => true,
            Some('x') => false,
            other => bail!("LSTM axis {other:?}"),
        };
        let mut r = c.as_str();
        let summarize = if let Some(rest) = r.strip_prefix('s') {
            r = rest;
            true
        } else {
            false
        };
        ensure!(
            !r.starts_with(['c', 'o']),
            "legacy clstm / ocropy LSTMs are not supported"
        );
        let (name, r) = named(r, "L", idx);
        return Ok(Block::Lstm {
            name,
            bidirectional,
            along_y,
            summarize,
            hidden: r.parse().with_context(|| format!("LSTM width {r:?}"))?,
        });
    }
    if let Some(r) = t.strip_prefix('O') {
        let (name, r) = named(r, "O", idx);
        let mut c = r.chars();
        ensure!(
            c.next() == Some('1'),
            "only sequence (`O1…`) outputs are supported"
        );
        let kind = match c.next() {
            Some('c') => OutputKind::Ctc,
            Some('s') => OutputKind::Softmax,
            Some('l') => OutputKind::Sigmoid,
            other => bail!("output kind {other:?}"),
        };
        let mut r = c.as_str();
        let augment = if let Some(rest) = r.strip_prefix('a') {
            r = rest;
            true
        } else {
            false
        };
        return Ok(Block::Output {
            name,
            kind,
            augment,
            out: r.parse().with_context(|| format!("output width {r:?}"))?,
        });
    }
    bail!("unsupported block")
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATMUS: &str = "[1,120,0,1 Cr{C_0}3,13,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 \
        Cr{C_3}3,13,32 Do{Do_4}0.1,2 Mp{Mp_5}2,2 Cr{C_6}3,9,64 Do{Do_7}0.1,2 \
        Mp{Mp_8}2,2 Cr{C_9}3,9,64 Do{Do_10}0.1,2 S{S_11}1(1x0)1,3 Lbx{L_12}200 \
        Do{Do_13}0.1,2 Lbx{L_14}200 Do{Do_15}0.1,2 Lbx{L_16}200 Do{Do_17} O{O_18}1c225]";

    #[test]
    fn catmus_spec() {
        let v = Vgsl::parse(CATMUS).unwrap();
        assert_eq!(
            v.input,
            InputSpec {
                batch: 1,
                height: 120,
                width: 0,
                channels: 1
            }
        );
        assert_eq!(v.blocks.len(), 19);
        assert_eq!(
            v.blocks[0],
            Block::Conv {
                name: "C_0".into(),
                act: ConvAct::Relu,
                kernel: [3, 13],
                out: 32,
                stride: [1, 1],
                dilation: [1, 1]
            }
        );
        assert_eq!(
            v.blocks[11],
            Block::Reshape {
                name: "S_11".into(),
                dim: 1,
                part_a: 1,
                part_b: 0,
                high: 1,
                low: 3
            }
        );
        assert!(matches!(
            &v.blocks[12],
            Block::Lstm { name, bidirectional: true, along_y: false, summarize: false, hidden: 200 }
                if name == "L_12"
        ));
        assert!(matches!(
            &v.blocks[18],
            Block::Output {
                kind: OutputKind::Ctc,
                out: 225,
                augment: false,
                ..
            }
        ));
    }

    #[test]
    fn default_names_and_leaky_relu() {
        let v = Vgsl::parse("[1,48,0,1 Clr3,3,16,2,2 Mp2,2 Lfys64 O1s10]").unwrap();
        assert!(matches!(&v.blocks[0],
            Block::Conv { name, act: ConvAct::LeakyRelu, stride: [2, 2], .. } if name == "C_0"));
        assert_eq!(v.blocks[1].name(), "Mp_1");
        assert!(matches!(
            &v.blocks[2],
            Block::Lstm {
                bidirectional: false,
                along_y: true,
                summarize: true,
                hidden: 64,
                ..
            }
        ));
    }

    #[test]
    fn unsupported_blocks_are_named() {
        let e = Vgsl::parse("[1,48,0,1 Gn32 O1c5]").unwrap_err();
        assert!(format!("{e:#}").contains("Gn32"));
    }
}
