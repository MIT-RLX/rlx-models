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

//! The released DAN checkpoints (`dan_rimes_page.pt`, `dan_read_page.pt`,
//! the Faster DAN ones, Zenodo 7244382 / 7568900): a training checkpoint
//! with `encoder_state_dict`, `decoder_state_dict` and the `charset`.
//!
//! The architecture is fixed by the code that trained them
//! (`predict_example.py`): a 6 + 4 block FCN encoder to 256 channels at
//! 1/32 × 1/8 resolution, and an 8-layer, 4-head, 256-wide post-norm
//! decoder with a 256-wide FFN, 2-D sinusoidal positions on the features,
//! 1-D ones on the tokens, and a 100-token self-attention window.

use anyhow::{Context, Result, anyhow, ensure};
use rlx_torch_ckpt::pickle::Value;
use std::collections::HashMap;
use std::path::Path;

/// Feature / model width.
pub const DIM: usize = 256;
pub const LAYERS: usize = 8;
pub const HEADS: usize = 4;
pub const FFN: usize = 256;
/// Tokens a step attends to, itself included (`attention_win`).
pub const WINDOW: usize = 100;
/// `h_max` / `w_max` of the 2-D positional table.
pub const H_MAX: usize = 500;
pub const W_MAX: usize = 1000;
/// `l_max` of the 1-D table — the longest transcription.
pub const L_MAX: usize = 15000;
/// InstanceNorm eps of every encoder block.
pub const NORM_EPS: f32 = 1e-3;

/// Encoder block geometry: (in, out, stride).
pub const CONV_BLOCKS: [(usize, usize, [usize; 2]); 6] = [
    (3, 16, [1, 1]),
    (16, 32, [2, 2]),
    (32, 64, [2, 2]),
    (64, 128, [2, 2]),
    (128, 128, [2, 1]),
    (128, 128, [2, 1]),
];
pub const DSC_BLOCKS: [(usize, usize); 4] = [(128, 128), (128, 128), (128, 128), (128, 256)];

/// Name → `(data, shape)`, prefix stripped.
pub type TensorTable = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// A loaded checkpoint.
pub struct DanCheckpoint {
    pub charset: Vec<String>,
    pub encoder: TensorTable,
    pub decoder: TensorTable,
}

impl DanCheckpoint {
    pub fn open(path: &Path) -> Result<Self> {
        let root = rlx_torch_ckpt::read_torch_pickle_root(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let charset =
            charset_of(&root).ok_or_else(|| anyhow!("{}: no `charset` list", path.display()))?;
        let pt = rlx_torch_ckpt::PtModel::open(path)?;
        let (mut encoder, mut decoder) = (TensorTable::new(), TensorTable::new());
        for name in pt.names() {
            let (table, short) = if let Some(s) = name.strip_prefix("encoder_state_dict.") {
                (&mut encoder, s)
            } else if let Some(s) = name.strip_prefix("decoder_state_dict.") {
                (&mut decoder, s)
            } else {
                continue;
            };
            let t = pt.tensor(&name)?;
            table.insert(short.to_string(), (t.data, t.shape));
        }
        ensure!(
            !encoder.is_empty() && !decoder.is_empty(),
            "{}: not a DAN checkpoint (no encoder/decoder state dicts)",
            path.display()
        );
        let ck = Self {
            charset,
            encoder,
            decoder,
        };
        let v = ck.vocab();
        let emb = ck
            .decoder
            .get("emb.weight")
            .map(|t| t.1.clone())
            .unwrap_or_default();
        ensure!(
            emb == [v + 3, DIM],
            "emb.weight {emb:?} for a {v}-symbol charset (expected [{}, {DIM}])",
            v + 3
        );
        Ok(ck)
    }

    /// Characters in the charset (the decision layer has one more: end).
    pub fn vocab(&self) -> usize {
        self.charset.len()
    }

    /// End-of-transcription token.
    pub fn end_token(&self) -> usize {
        self.vocab()
    }

    /// Start-of-transcription token.
    pub fn start_token(&self) -> usize {
        self.vocab() + 1
    }
}

fn charset_of(root: &Value) -> Option<Vec<String>> {
    let Value::Dict(d) = root else { return None };
    let d = d.borrow();
    let (_, v) = d
        .iter()
        .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "charset"))?;
    let items: Vec<Value> = match v {
        Value::List(l) => l.borrow().clone(),
        Value::Tuple(t) => t.to_vec(),
        _ => return None,
    };
    items
        .iter()
        .map(|x| match x {
            Value::Str(s) => Some(s.to_string()),
            _ => None,
        })
        .collect()
}

/// `div` of DAN's sinusoidal tables: `exp(−(2j / dim) · ln 10000)`, in f32
/// like torch builds it.
fn divs(count: usize) -> Vec<f32> {
    let ln = 10000f32.ln();
    (0..count)
        .map(|j| (-((2 * j) as f32) / DIM as f32 * ln).exp())
        .collect()
}

/// `PositionalEncoding2D` cropped to `[DIM, h, w]`: the first half of the
/// channels encode the row (sin, cos interleaved), the second the column.
pub fn pe_2d(h: usize, w: usize) -> Vec<f32> {
    let half = DIM / 2;
    let div = divs(half / 2);
    let mut pe = vec![0f32; DIM * h * w];
    for (j, &dv) in div.iter().enumerate() {
        for y in 0..h {
            let (s, c) = ((y as f32 * dv).sin(), (y as f32 * dv).cos());
            for x in 0..w {
                pe[((2 * j) * h + y) * w + x] = s;
                pe[((2 * j + 1) * h + y) * w + x] = c;
            }
        }
        for x in 0..w {
            let (s, c) = ((x as f32 * dv).sin(), (x as f32 * dv).cos());
            for y in 0..h {
                pe[((half + 2 * j) * h + y) * w + x] = s;
                pe[((half + 2 * j + 1) * h + y) * w + x] = c;
            }
        }
    }
    pe
}

/// Row `pos` of `PositionalEncoding1D`: `[DIM]`, sin / cos interleaved.
pub fn pe_1d(pos: usize) -> Vec<f32> {
    let div = divs(DIM / 2);
    let mut out = vec![0f32; DIM];
    for (j, &dv) in div.iter().enumerate() {
        let a = pos as f32 * dv;
        out[2 * j] = a.sin();
        out[2 * j + 1] = a.cos();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_tables() {
        // Position 0: sin 0 = 0, cos 0 = 1.
        let p = pe_1d(0);
        assert_eq!((p[0], p[1]), (0.0, 1.0));
        let p = pe_1d(3);
        assert!((p[0] - 3f32.sin()).abs() < 1e-6);
        let t = pe_2d(2, 3);
        // Channel 0 at row 1 is sin(1); channel DIM/2 at column 2 is sin(2).
        assert!((t[3] - 1f32.sin()).abs() < 1e-6);
        assert!((t[(DIM / 2) * 6 + 2] - 2f32.sin()).abs() < 1e-6);
    }
}
