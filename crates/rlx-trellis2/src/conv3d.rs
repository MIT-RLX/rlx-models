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

//! Dense 3-D convolution + spatial helpers for the sparse-structure decoder
//! (host / CPU reference). All tensors are batch-1, `NCDHW` with the batch
//! dimension dropped: a channel-major `[C, D, H, W]` flat buffer.
//!
//! `conv3d_same` covers the only kernel the decoder needs (`k=3, stride=1,
//! pad=1`) via im2col + a BLAS `matmul_bt`. `pixel_shuffle_3d` matches
//! `trellis2/modules/spatial.py` exactly (channel block index `a·s²+b·s+c`
//! maps to spatial offset `(a,b,c)`).

use rlx_core::host_kernels::matmul_bt;

/// A 3-D volume `[C, D, H, W]` in channel-major order.
#[derive(Clone)]
pub struct Vol {
    pub c: usize,
    pub d: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Vol {
    pub fn new(c: usize, d: usize, h: usize, w: usize) -> Self {
        Self {
            c,
            d,
            h,
            w,
            data: vec![0.0; c * d * h * w],
        }
    }
    #[inline]
    pub fn idx(&self, c: usize, d: usize, h: usize, w: usize) -> usize {
        ((c * self.d + d) * self.h + h) * self.w + w
    }
    pub fn spatial(&self) -> usize {
        self.d * self.h * self.w
    }
}

/// `k=3, stride=1, pad=1` "same" 3-D convolution. `weight` is
/// `[c_out, c_in, 3, 3, 3]` (row-major, PyTorch layout); `bias` is `[c_out]`.
pub fn conv3d_same(x: &Vol, weight: &[f32], bias: &[f32], c_out: usize) -> Vol {
    let (c_in, d, h, w) = (x.c, x.d, x.h, x.w);
    let ksz = 27; // 3^3
    let cols = c_in * ksz;
    let n = d * h * w;
    // im2col: patches[voxel, c_in*27 + kd*9 + kh*3 + kw]
    let mut patches = vec![0.0f32; n * cols];
    for dd in 0..d {
        for hh in 0..h {
            for ww in 0..w {
                let vox = (dd * h + hh) * w + ww;
                let base = vox * cols;
                for ci in 0..c_in {
                    let cbase = base + ci * ksz;
                    for kd in 0..3usize {
                        let sd = dd as isize + kd as isize - 1;
                        if sd < 0 || sd >= d as isize {
                            continue;
                        }
                        for kh in 0..3usize {
                            let sh = hh as isize + kh as isize - 1;
                            if sh < 0 || sh >= h as isize {
                                continue;
                            }
                            for kw in 0..3usize {
                                let sw = ww as isize + kw as isize - 1;
                                if sw < 0 || sw >= w as isize {
                                    continue;
                                }
                                let iv =
                                    ((ci * d + sd as usize) * h + sh as usize) * w + sw as usize;
                                patches[cbase + kd * 9 + kh * 3 + kw] = x.data[iv];
                            }
                        }
                    }
                }
            }
        }
    }
    // out[voxel, c_out] = patches @ weightᵀ ; weight is [c_out, cols]
    let mut out_nc = vec![0.0f32; n * c_out];
    matmul_bt(&patches, weight, &mut out_nc, n, cols, c_out, 1.0);
    // to channel-major [c_out, d, h, w] + bias
    let mut out = Vol::new(c_out, d, h, w);
    for vox in 0..n {
        for co in 0..c_out {
            out.data[co * n + vox] = out_nc[vox * c_out + co] + bias[co];
        }
    }
    out
}

/// 3-D pixel shuffle by `s`: `[C·s³, D, H, W] -> [C, D·s, H·s, W·s]`.
/// Matches `pixel_shuffle_3d`: input channel `c·s³ + (a·s² + b·s + e)` places
/// into spatial offset `(a, b, e)` of output cell `(d,h,w)`.
pub fn pixel_shuffle_3d(x: &Vol, s: usize) -> Vol {
    let c_out = x.c / (s * s * s);
    let (d, h, w) = (x.d, x.h, x.w);
    let mut out = Vol::new(c_out, d * s, h * s, w * s);
    for co in 0..c_out {
        for a in 0..s {
            for b in 0..s {
                for e in 0..s {
                    let cin = co * (s * s * s) + (a * s * s + b * s + e);
                    for dd in 0..d {
                        for hh in 0..h {
                            for ww in 0..w {
                                let src = x.idx(cin, dd, hh, ww);
                                let oi = out.idx(co, dd * s + a, hh * s + b, ww * s + e);
                                out.data[oi] = x.data[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Per-voxel LayerNorm over channels (`ChannelLayerNorm32`), affine, eps `1e-5`.
pub fn channel_layer_norm(x: &Vol, weight: &[f32], bias: &[f32]) -> Vol {
    let n = x.spatial();
    let c = x.c;
    let eps = 1e-5f32;
    let mut out = Vol::new(c, x.d, x.h, x.w);
    for vox in 0..n {
        let mut mean = 0.0f32;
        for ch in 0..c {
            mean += x.data[ch * n + vox];
        }
        mean /= c as f32;
        let mut var = 0.0f32;
        for ch in 0..c {
            let dv = x.data[ch * n + vox] - mean;
            var += dv * dv;
        }
        var /= c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ch in 0..c {
            out.data[ch * n + vox] = (x.data[ch * n + vox] - mean) * inv * weight[ch] + bias[ch];
        }
    }
    out
}

/// In-place SiLU.
pub fn silu(x: &mut Vol) {
    for v in x.data.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv3d_identity_kernel() {
        // center-only kernel (c_out=c_in=1) reproduces the input.
        let mut x = Vol::new(1, 2, 2, 2);
        for (i, v) in x.data.iter_mut().enumerate() {
            *v = i as f32 + 1.0;
        }
        let mut w = vec![0.0f32; 27];
        w[13] = 1.0; // center of 3x3x3
        let out = conv3d_same(&x, &w, &[0.0], 1);
        assert_eq!(out.data, x.data);
    }

    #[test]
    fn pixel_shuffle_roundtrip_shape() {
        let x = Vol::new(8, 2, 2, 2);
        let out = pixel_shuffle_3d(&x, 2);
        assert_eq!((out.c, out.d, out.h, out.w), (1, 4, 4, 4));
    }

    #[test]
    fn pixel_shuffle_places_offsets() {
        // one hot in channel block -> known spatial offset
        let mut x = Vol::new(8, 1, 1, 1);
        // c = a*4+b*2+e = (1,0,1) -> 4*1+2*0+1 = 5
        x.data[5] = 1.0;
        let out = pixel_shuffle_3d(&x, 2);
        assert_eq!(out.data[out.idx(0, 1, 0, 1)], 1.0);
        assert_eq!(out.data.iter().filter(|&&v| v != 0.0).count(), 1);
    }
}
