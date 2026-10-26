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

//! How dense (non-quantized) weight matrices are stored once uploaded.
//!
//! [`WeightLoader::take`](crate::weight_loader::WeightLoader::take) hands
//! builders f32, so a bf16 checkpoint used to land in the arena at 4 B/param —
//! twice its file size, for nothing: widening bf16 to f32 is exact, and every
//! backend's matmul reads an f16/bf16 weight and computes in f32
//! (`tests/half_weight_matmul.rs` pins this on CPU, Metal, MLX and wgpu).
//! Storing the weight at its own width gives the same products at half the
//! memory.
//!
//! Packed quantized weights (GGUF K-quants, MLX affine) are unaffected — they
//! stay packed under every setting. Norm gains, biases, conv taps and other
//! rank < 2 tensors stay f32: they are small and some are read by kernels that
//! expect f32.

use rlx_ir::DType;

/// The storage setting itself lives in `rlx-flow` so every flow-built model
/// shares it ([`rlx_flow::ModelFlow::with_weight_storage`],
/// `RLX_WEIGHT_STORAGE`); re-exported here with the encoder.
pub use rlx_flow::storage::{WEIGHT_STORAGE_ENV, WeightStorage, encode_f32};

/// A host-side table (an embedding / tied LM head) kept at a weight
/// storage width, with row access in f32.
#[derive(Debug, Clone)]
pub enum HostTable {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    /// GGUF quantized rows (e.g. a Q6_K `token_embd`), decoded one row at a
    /// time — lossless and ~4.5x smaller than the f32 table.
    Packed {
        bytes: Vec<u8>,
        dtype: rlx_gguf::GgmlType,
        row_elems: usize,
        row_bytes: usize,
    },
}

impl HostTable {
    /// Store `data` as `dtype` (F16 / BF16; anything else keeps f32).
    pub fn from_f32(data: Vec<f32>, dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self::F16(data.into_iter().map(half::f16::from_f32).collect()),
            DType::BF16 => Self::Bf16(data.into_iter().map(half::bf16::from_f32).collect()),
            _ => Self::F32(data),
        }
    }

    /// Keep GGUF-quantized `[rows, row_elems]` bytes packed. `None` when the
    /// quant's blocks do not tile a row (then widen instead).
    pub fn packed(bytes: Vec<u8>, dtype: rlx_gguf::GgmlType, row_elems: usize) -> Option<Self> {
        let row_bytes = rlx_gguf::bytes_for_public(dtype, row_elems)?;
        (row_bytes > 0 && bytes.len().is_multiple_of(row_bytes)).then_some(Self::Packed {
            bytes,
            dtype,
            row_elems,
            row_bytes,
        })
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::Bf16(v) => v.len(),
            Self::Packed {
                bytes,
                row_elems,
                row_bytes,
                ..
            } => bytes.len() / row_bytes * row_elems,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        match self {
            Self::F32(v) => v.len() * 4,
            Self::F16(v) => v.len() * 2,
            Self::Bf16(v) => v.len() * 2,
            Self::Packed { bytes, .. } => bytes.len(),
        }
    }

    /// Decode the packed rows covering `[start, start + out.len())` into `out`.
    fn read_packed(
        bytes: &[u8],
        dtype: rlx_gguf::GgmlType,
        row_elems: usize,
        row_bytes: usize,
        start: usize,
        out: &mut [f32],
    ) {
        let mut done = 0;
        while done < out.len() {
            let pos = start + done;
            let (row, col) = (pos / row_elems, pos % row_elems);
            let src = &bytes[row * row_bytes..(row + 1) * row_bytes];
            let decoded = rlx_gguf::dequant_typed(dtype, src, row_elems, "host table row")
                .expect("host table row decodes");
            let take = (row_elems - col).min(out.len() - done);
            out[done..done + take].copy_from_slice(&decoded[col..col + take]);
            done += take;
        }
    }

    /// Elements `[start, start + out.len())` widened into `out`.
    pub fn read_into(&self, start: usize, out: &mut [f32]) {
        let n = out.len();
        match self {
            Self::F32(v) => out.copy_from_slice(&v[start..start + n]),
            Self::F16(v) => {
                for (o, x) in out.iter_mut().zip(&v[start..start + n]) {
                    *o = x.to_f32();
                }
            }
            Self::Bf16(v) => {
                for (o, x) in out.iter_mut().zip(&v[start..start + n]) {
                    *o = x.to_f32();
                }
            }
            Self::Packed {
                bytes,
                dtype,
                row_elems,
                row_bytes,
            } => Self::read_packed(bytes, *dtype, *row_elems, *row_bytes, start, out),
        }
    }

    /// `Σ table[start + i] · x[i]` in f64.
    pub fn dot(&self, start: usize, x: &[f32]) -> f64 {
        let n = x.len();
        match self {
            Self::F32(v) => v[start..start + n]
                .iter()
                .zip(x)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum(),
            Self::F16(v) => v[start..start + n]
                .iter()
                .zip(x)
                .map(|(a, &b)| a.to_f64() * b as f64)
                .sum(),
            Self::Bf16(v) => v[start..start + n]
                .iter()
                .zip(x)
                .map(|(a, &b)| a.to_f64() * b as f64)
                .sum(),
            Self::Packed { .. } => {
                let mut row = vec![0f32; n];
                self.read_into(start, &mut row);
                row.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_keeps_the_stored_half_width() {
        assert_eq!(WeightStorage::Auto.resolve(Some(DType::BF16)), DType::BF16);
        assert_eq!(WeightStorage::Auto.resolve(Some(DType::F16)), DType::F16);
        assert_eq!(WeightStorage::Auto.resolve(Some(DType::F32)), DType::F32);
        assert_eq!(WeightStorage::Auto.resolve(None), DType::F32);
        assert_eq!(WeightStorage::F16.resolve(Some(DType::BF16)), DType::F16);
        assert_eq!(WeightStorage::F32.resolve(Some(DType::BF16)), DType::F32);
        assert_eq!(
            "BF16".parse::<WeightStorage>().unwrap(),
            WeightStorage::Bf16
        );
    }

    #[test]
    fn bf16_round_trip_through_f32_is_exact() {
        let vals: Vec<f32> = (0..1000)
            .map(|i| half::bf16::from_bits(i * 61).to_f32())
            .collect();
        let t = HostTable::from_f32(vals.clone(), DType::BF16);
        let mut back = vec![0f32; vals.len()];
        t.read_into(0, &mut back);
        for (a, b) in vals.iter().zip(&back) {
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
        }
        assert_eq!(t.bytes(), 2000);
    }
}
