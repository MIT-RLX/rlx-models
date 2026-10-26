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

//! The DiT checkpoint as a [`WeightSource`].
//!
//! FLUX 3 rotates interleaved pairs `(x[2i], x[2i+1])`; the RoPE op rotates
//! half-split pairs `(x[i], x[i + d/2])`. Permuting each head's q/k
//! projection rows (and the QK-norm scales, which act per channel) moves the
//! pairs into place exactly, so the graph uses the ordinary op. Attention is
//! invariant to a shared permutation of q and k, and v is untouched.

use anyhow::Result;
use rlx_flow::{GgufPackedLinear, WeightSource};
use rlx_flux1::weights::{TensorFile, WeightQuant, deinterleave_rows, transpose_half};
use rlx_ir::DType;

use crate::config::Flux3Config;

/// Serves the DiT's tensors with the RoPE permutation applied.
pub struct DitSource<'a> {
    pub file: &'a TensorFile,
    pub cfg: &'a Flux3Config,
    /// Serve projection weights packed (see [`WeightQuant`]).
    pub quant: Option<WeightQuant>,
}

/// q or k projection (rows to permute per head).
fn is_qk(key: &str) -> bool {
    key.ends_with(".q_proj.weight") || key.ends_with(".k_proj.weight")
}

/// QK-norm scale (per-channel, permuted like the rows).
fn is_qk_norm(key: &str) -> bool {
    key.ends_with("_norm.scale")
}

impl DitSource<'_> {
    /// One tensor in f32, permuted as the graph wants it, before any transpose.
    pub fn load(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.file.load(key)?;
        let hd = self.cfg.head_dim;
        if is_qk(key) {
            deinterleave_rows(&mut data, hd, shape[1]);
        } else if is_qk_norm(key) {
            deinterleave_rows(&mut data, hd, 1);
        }
        Ok((data, shape))
    }
}

fn maybe_transpose(data: Vec<f32>, shape: Vec<usize>, transpose: bool) -> (Vec<f32>, Vec<usize>) {
    if transpose && shape.len() == 2 {
        (
            rlx_core::weight_map::transpose_2d(&data, shape[0], shape[1]),
            vec![shape[1], shape[0]],
        )
    } else {
        (data, shape)
    }
}

impl WeightSource for DitSource<'_> {
    /// `[out, in]` rows quantized at load when [`DitSource::quant`] is set.
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let Some(q) = self.quant else {
            return Ok(None);
        };
        if !key.ends_with(".weight") {
            return Ok(None);
        }
        let Some(dtype) = self.file.stored_dtype(key) else {
            return Ok(None);
        };
        let Some((mut bytes, shape)) = self.file.load_half_bytes(key)? else {
            return Ok(None);
        };
        if shape.len() != 2 || !shape[1].is_multiple_of(q.block()) {
            return Ok(None);
        }
        let (rows, cols) = (shape[0], shape[1]);
        if is_qk(key) {
            deinterleave_rows(&mut bytes, self.cfg.head_dim, 2 * cols);
        }
        Ok(Some(GgufPackedLinear {
            w_q: q.encode_half(&bytes, dtype, rows, cols)?,
            scheme: q.scheme(),
            in_dim: cols,
            out_dim: rows,
            bias: Vec::new(),
        }))
    }

    /// Rank-2 weights straight from their half-precision bits, permuted and
    /// transposed in u16, never widened.
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let Some((mut bytes, shape)) = self.file.load_half_bytes(key)? else {
            return Ok(None);
        };
        if shape.len() != 2 {
            return Ok(None);
        }
        let (rows, cols) = (shape[0], shape[1]);
        if is_qk(key) {
            deinterleave_rows(&mut bytes, self.cfg.head_dim, 2 * cols);
        }
        Ok(Some(if transpose {
            (transpose_half(&bytes, rows, cols), vec![cols, rows])
        } else {
            (bytes, shape)
        }))
    }

    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.load(key)?;
        Ok(maybe_transpose(data, shape, transpose))
    }

    fn has(&self, key: &str) -> bool {
        self.file.contains(key)
    }

    fn keys(&self) -> Vec<String> {
        self.file.keys()
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        self.file.stored_dtype(key)
    }
}

/// Text-encoder tensors under a name prefix, half-precision matrices served
/// without the f32 round trip.
pub struct PrefixSource<'a> {
    pub file: &'a TensorFile,
    pub prefix: &'a str,
    pub quant: Option<WeightQuant>,
}

impl PrefixSource<'_> {
    fn real(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }
}

impl WeightSource for PrefixSource<'_> {
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let mut inner = rlx_flux1::weights::PlainSource {
            file: self.file,
            prefix: self.prefix,
            quant: self.quant,
        };
        inner.take_packed(key)
    }

    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let Some((bytes, shape)) = self.file.load_half_bytes(&self.real(key))? else {
            return Ok(None);
        };
        if shape.len() != 2 {
            return Ok(None);
        }
        Ok(Some(if transpose {
            (
                transpose_half(&bytes, shape[0], shape[1]),
                vec![shape[1], shape[0]],
            )
        } else {
            (bytes, shape)
        }))
    }

    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.file.load(&self.real(key))?;
        Ok(maybe_transpose(data, shape, transpose))
    }

    fn has(&self, key: &str) -> bool {
        self.file.contains(&self.real(key))
    }

    fn stored_dtype(&self, key: &str) -> Option<DType> {
        self.file.stored_dtype(&self.real(key))
    }
}
