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

//! Pooling and L2 normalization for sentence/image embeddings.

/// Pooling strategy for reducing token hidden states to one vector per sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// First token ([CLS]).
    Cls,
    /// Attention-mask-weighted mean over tokens.
    Mean,
}

/// Pool `[batch, seq, hidden]` hidden states into `[batch, hidden]` and L2-normalize.
pub fn pool_embeddings(
    hidden: &[f32],
    attention_mask: &[&[u32]],
    batch: usize,
    seq: usize,
    hidden_size: usize,
    pooling: Pooling,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(batch);
    for bi in 0..batch {
        let mut pooled = vec![0f32; hidden_size];
        match pooling {
            Pooling::Cls => {
                pooled.copy_from_slice(
                    &hidden[bi * seq * hidden_size..bi * seq * hidden_size + hidden_size],
                );
            }
            Pooling::Mean => {
                let count: f32 = attention_mask[bi].iter().map(|&v| v as f32).sum();
                let inv = 1.0 / count.max(1.0);
                for si in 0..seq {
                    if attention_mask[bi][si] > 0 {
                        let off = (bi * seq + si) * hidden_size;
                        for j in 0..hidden_size {
                            pooled[j] += hidden[off + j];
                        }
                    }
                }
                for v in &mut pooled {
                    *v *= inv;
                }
            }
        }
        l2_normalize_in_place(&mut pooled);
        out.push(pooled);
    }
    out
}

/// L2-normalize a vector in place (matches fastembed: divide by norm + 1e-12).
pub fn l2_normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12;
    let inv = 1.0 / norm;
    for x in v {
        *x *= inv;
    }
}
