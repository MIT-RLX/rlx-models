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

//! Host sparse-tensor engine for the TRELLIS.2 VAEs
//! (`trellis2/modules/sparse/*`).
//!
//! A [`SparseTensor`] is a set of active voxels: features `[N, C]` (row-major)
//! plus integer coordinates `[N] × (x,y,z)` (the batch column is dropped since
//! inference is a single sample). The ops needed by the shape/texture decoders:
//!
//!   * [`sparse_linear`], [`layer_norm`], [`silu`] — per-voxel channel ops.
//!   * [`submanifold_conv3d`] — stride-1 `SubMConv3d`: output lives on the same
//!     coords; each active voxel gathers its active `3³` neighbours. This
//!     equals a dense `conv3d` on the scattered grid gathered at active sites
//!     (inactive neighbours are zero), which is how it is parity-checked.
//!   * [`channel2spatial`] — the octree upsampler `SparseChannel2Spatial`:
//!     a voxel with `C` channels splits into ≤8 children (per a subdivision
//!     mask); child in octant `o` takes channel block `o` of `C/8` channels and
//!     lands at `2·coord + (o&1, (o>>1)&1, (o>>2)&1)`.
//!   * [`repeat_interleave_channels`] — the C2S residual channel expansion.

use rlx_core::host_kernels::matmul_bt;
use std::collections::HashMap;

/// Active-voxel sparse tensor (single sample). `feats` is `[n, c]` row-major;
/// `coords[i] = (x, y, z)`.
#[derive(Clone)]
pub struct SparseTensor {
    pub feats: Vec<f32>,
    pub coords: Vec<[i32; 3]>,
    pub c: usize,
}

impl SparseTensor {
    pub fn new(feats: Vec<f32>, coords: Vec<[i32; 3]>, c: usize) -> Self {
        debug_assert_eq!(feats.len(), coords.len() * c);
        Self { feats, coords, c }
    }
    pub fn n(&self) -> usize {
        self.coords.len()
    }
    /// Same coords, new features (channel count inferred when `n > 0`).
    ///
    /// When the tensor is empty, the previous channel count is preserved so a
    /// later op does not see `c == 0` after `layer_norm`/`mlp` on zero rows.
    pub fn replace(&self, feats: Vec<f32>) -> Self {
        let c = if self.coords.is_empty() {
            self.c
        } else {
            debug_assert_eq!(feats.len() % self.coords.len(), 0);
            feats.len() / self.coords.len()
        };
        Self {
            feats,
            coords: self.coords.clone(),
            c,
        }
    }
    /// Elementwise add on matching coords (assumes identical coord ordering).
    pub fn add(&self, other: &SparseTensor) -> SparseTensor {
        debug_assert_eq!(self.feats.len(), other.feats.len());
        let feats = self
            .feats
            .iter()
            .zip(&other.feats)
            .map(|(a, b)| a + b)
            .collect();
        self.replace(feats)
    }
}

/// `feats · Wᵀ + b`, `W = [out, in]` (PyTorch layout).
pub fn sparse_linear(
    feats: &[f32],
    n: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    b: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * out_dim];
    matmul_bt(feats, w, &mut out, n, in_dim, out_dim, 1.0);
    if let Some(b) = b {
        for r in 0..n {
            for o in 0..out_dim {
                out[r * out_dim + o] += b[o];
            }
        }
    }
    out
}

/// Per-voxel LayerNorm over channels (eps `1e-6`). `weight`/`bias` optional.
pub fn layer_norm(
    feats: &[f32],
    n: usize,
    c: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let eps = 1e-6f32;
    let mut out = vec![0.0f32; feats.len()];
    for r in 0..n {
        let row = &feats[r * c..(r + 1) * c];
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let o = &mut out[r * c..(r + 1) * c];
        for i in 0..c {
            let mut v = (row[i] - mean) * inv;
            if let Some(w) = weight {
                v *= w[i];
            }
            if let Some(b) = bias {
                v += b[i];
            }
            o[i] = v;
        }
    }
    out
}

/// In-place SiLU on a feature buffer.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// GELU (tanh approximation), in place.
#[allow(dead_code)]
pub fn gelu_tanh_inplace(x: &mut [f32]) {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (K * (t + 0.044715 * t * t * t)).tanh());
    }
}

fn coord_index(coords: &[[i32; 3]]) -> HashMap<[i32; 3], usize> {
    let mut m = HashMap::with_capacity(coords.len());
    for (i, c) in coords.iter().enumerate() {
        m.insert(*c, i);
    }
    m
}

/// Stride-1 submanifold 3-D convolution (`SubMConv3d`, kernel 3). Output coords
/// equal input coords. `weight` is spconv layout `[out, 3, 3, 3, in]`; the
/// weight slot `(kd,kh,kw)` gathers the neighbour at offset `(kd-1,kh-1,kw-1)`.
pub fn submanifold_conv3d(
    x: &SparseTensor,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
) -> SparseTensor {
    let n = x.n();
    let in_c = x.c;
    let idx = coord_index(&x.coords);
    let mut out = vec![0.0f32; n * out_c];
    // weight index: (((co*3 + kd)*3 + kh)*3 + kw)*in_c + ci
    for i in 0..n {
        let [cx, cy, cz] = x.coords[i];
        let orow = &mut out[i * out_c..(i + 1) * out_c];
        orow.copy_from_slice(bias);
        for kd in 0..3i32 {
            for kh in 0..3i32 {
                for kw in 0..3i32 {
                    let nb = [cx + kd - 1, cy + kh - 1, cz + kw - 1];
                    let Some(&j) = idx.get(&nb) else { continue };
                    let jfeat = &x.feats[j * in_c..(j + 1) * in_c];
                    let wbase = ((kd as usize * 3 + kh as usize) * 3 + kw as usize) * in_c;
                    // accumulate: orow[co] += sum_ci W[co, kd,kh,kw, ci] * jfeat[ci]
                    for co in 0..out_c {
                        let wrow = &weight[co * 27 * in_c + wbase..co * 27 * in_c + wbase + in_c];
                        let mut acc = 0.0f32;
                        for ci in 0..in_c {
                            acc += wrow[ci] * jfeat[ci];
                        }
                        orow[co] += acc;
                    }
                }
            }
        }
    }
    SparseTensor::new(out, x.coords.clone(), out_c)
}

/// Octree upsample (`SparseChannel2Spatial`, factor 2). `subdiv_bits[p]` is an
/// 8-bit mask of which octants of parent `p` are active. Child in octant `o`
/// (bit `o`) takes channel block `o` of `c/8` channels and lands at
/// `2·coord + (o&1, (o>>1)&1, (o>>2)&1)`. Children are emitted parent-major,
/// octant-ascending (matching `torch.nonzero`).
pub fn channel2spatial(x: &SparseTensor, subdiv_bits: &[u8]) -> SparseTensor {
    debug_assert_eq!(subdiv_bits.len(), x.n());
    let cin = x.c;
    let cout = cin / 8;
    let mut feats = Vec::new();
    let mut coords = Vec::new();
    for p in 0..x.n() {
        let [px, py, pz] = x.coords[p];
        let bits = subdiv_bits[p];
        for o in 0..8u8 {
            if bits & (1 << o) == 0 {
                continue;
            }
            let ox = (o & 1) as i32;
            let oy = ((o >> 1) & 1) as i32;
            let oz = ((o >> 2) & 1) as i32;
            coords.push([px * 2 + ox, py * 2 + oy, pz * 2 + oz]);
            let src = &x.feats[p * cin + (o as usize) * cout..p * cin + (o as usize + 1) * cout];
            feats.extend_from_slice(src);
        }
    }
    SparseTensor::new(feats, coords, cout)
}

/// Binarize subdivision logits `[n, 8]` (`>0`) into per-voxel 8-bit masks.
pub fn subdiv_to_bits(subdiv_logits: &[f32], n: usize) -> Vec<u8> {
    let mut bits = vec![0u8; n];
    for p in 0..n {
        let mut b = 0u8;
        for o in 0..8 {
            if subdiv_logits[p * 8 + o] > 0.0 {
                b |= 1 << o;
            }
        }
        bits[p] = b;
    }
    bits
}

/// `repeat_interleave(feats, k, dim=1)`: each of `c` channels repeated `k`×.
pub fn repeat_interleave_channels(feats: &[f32], n: usize, c: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * c * k];
    let oc = c * k;
    for r in 0..n {
        for ch in 0..c {
            let v = feats[r * c + ch];
            for j in 0..k {
                out[r * oc + ch * k + j] = v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_preserves_channels_when_empty() {
        let st = SparseTensor::new(vec![], vec![], 7);
        let out = st.replace(vec![]);
        assert_eq!(out.c, 7);
        assert_eq!(out.n(), 0);
    }

    #[test]
    fn c2s_places_children() {
        // one parent at (1,2,3), channels=16, subdiv octants {0,5}
        // octant 0 -> offset (0,0,0), octant 5 -> (1,0,1)
        let feats: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let st = SparseTensor::new(feats, vec![[1, 2, 3]], 16);
        let bits = vec![0b0010_0001u8]; // octants 0 and 5
        let up = channel2spatial(&st, &bits);
        assert_eq!(up.n(), 2);
        assert_eq!(up.c, 2);
        assert_eq!(up.coords[0], [2, 4, 6]); // octant 0
        assert_eq!(up.coords[1], [3, 4, 7]); // octant 5 -> (1,0,1)
        // octant 0 -> channels [0,1]; octant 5 -> channels [10,11]
        assert_eq!(&up.feats[0..2], &[0.0, 1.0]);
        assert_eq!(&up.feats[2..4], &[10.0, 11.0]);
    }

    #[test]
    fn repeat_interleave_basic() {
        let out = repeat_interleave_channels(&[1.0, 2.0], 1, 2, 3);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn submanifold_isolated_voxel_is_center_only() {
        // a single isolated voxel: only the center kernel tap (offset 0) applies.
        let in_c = 2;
        let out_c = 1;
        let x = SparseTensor::new(vec![1.0, 2.0], vec![[5, 5, 5]], in_c);
        let mut w = vec![0.0f32; out_c * 27 * in_c];
        // center slot kd=kh=kw=1 -> index 13; weights [3.0, 4.0]
        let center = ((3 + 1) * 3 + 1) * in_c;
        w[center] = 3.0;
        w[center + 1] = 4.0;
        let out = submanifold_conv3d(&x, &w, &[0.5], out_c);
        assert!((out.feats[0] - (3.0 * 1.0 + 4.0 * 2.0 + 0.5)).abs() < 1e-6);
    }
}
