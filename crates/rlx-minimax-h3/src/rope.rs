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

//! 3-axis rotary embedding over the packed sequence's `(t, h, w)` coordinates.
//!
//! One `inv_freq` buffer of `rope_freq_dim` frequencies is shared by all three
//! axes. Each axis contributes `rope_freq_dim` angles; the three blocks are
//! concatenated to `3 * rope_freq_dim` and that is then concatenated with
//! itself, so the rotate-half convention rotates `2 * 3 * rope_freq_dim` of the
//! `head_dim` channels — 96 of 128 in the released checkpoint. The remaining 32
//! channels pass through unrotated, which is what `rope_n` in `rlx-ir` expresses
//! as partial RoPE.
//!
//! rlx's NeoX rope op takes `cos`/`sin` of width `n_rot / 2` and internally
//! pairs channel `i` with channel `i + n_rot/2`. The reference builds a width
//! `n_rot` table by duplicating the `n_rot / 2` angles, so the half this module
//! emits is exactly the reference's first half.

use anyhow::{Result, ensure};
use rlx_ir::hir::HirMut;
use rlx_ir::{HirGraphExt, HirNodeId};

/// Emit a partial rotary embedding that is correct on every backend.
///
/// `x` is `[1, seq, num_heads * head_dim]`; the leading `n_rot` channels of each
/// head are rotated and the rest pass through.
///
/// # Why this is not just `rope_n`
///
/// In `rlx` **0.2.14**, `rope_n` with `n_rot < head_dim` is wrong on Metal and
/// wgpu — it returns values with a relative error of order 1, not rounding
/// noise, while CPU and MLX are exact. (`examples/rope_probe.rs` isolates it:
/// `hd=128, n_rot=96` gives `metal=2.0e0, wgpu=2.0e0, mlx=0.0`.) The cause is
/// the cos/sin table row stride: those kernels indexed it by `head_dim/2` when
/// the table holds exactly `n_rot/2` angles per token, so every position after
/// the first read into the *next* token's angles. Since MiniMax-H3 rotates 96 of
/// 128 channels in the DiT and 48 of 64 in the video decoder, every accelerated
/// run would be quietly wrong.
///
/// The kernels are **fixed upstream** (`rlx-metal` and `rlx-wgpu` now stride by
/// `n_rot/2`, with `metal_partial_rope_matches_cpu` guarding it). This helper
/// stays because the workspace pins `rlx*` at `^0.2.14` from crates.io for
/// published and fresh-clone builds, which still carry the bug — dropping it
/// would silently mis-rotate for anyone not building against a local `../rlx`.
/// It can go once a release carrying the fix is pinned.
///
/// The way around it costs nothing: slice the rotated channels of every head
/// into their own contiguous tensor, rotate *that* with a **full** rope whose
/// `head_dim` is `n_rot` — the code path every backend gets right — and
/// concatenate the untouched tail back on. The pairing is identical, because
/// NeoX partial rope pairs `(i, i + n_rot/2)` within the rotated block, which is
/// exactly what a full rope over a width-`n_rot` head does.
#[allow(clippy::too_many_arguments)]
pub fn emit_partial_rope(
    gb: &mut HirMut<'_>,
    x: HirNodeId,
    cos: HirNodeId,
    sin: HirNodeId,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    n_rot: usize,
) -> HirNodeId {
    if n_rot == 0 {
        return x;
    }
    if n_rot >= head_dim {
        return gb.rope(x, cos, sin, head_dim);
    }
    let inner = num_heads * head_dim;
    let x4 = gb.reshape_(x, vec![1, seq as i64, num_heads as i64, head_dim as i64]);
    let rot = gb.narrow_(x4, 3, 0, n_rot);
    let pass = gb.narrow_(x4, 3, n_rot, head_dim - n_rot);
    let rot3 = gb.reshape_(rot, vec![1, seq as i64, (num_heads * n_rot) as i64]);
    let rotated = gb.rope(rot3, cos, sin, n_rot);
    let rotated4 = gb.reshape_(rotated, vec![1, seq as i64, num_heads as i64, n_rot as i64]);
    let joined = gb.concat_(vec![rotated4, pass], 3);
    gb.reshape_(joined, vec![1, seq as i64, inner as i64])
}

/// `cos` / `sin` tables for one packed layout.
#[derive(Debug, Clone)]
pub struct RopeTables {
    /// `[seq_len * half]`, row-major.
    pub cos: Vec<f32>,
    /// `[seq_len * half]`, row-major.
    pub sin: Vec<f32>,
    /// `3 * rope_freq_dim` — half the rotated width.
    pub half: usize,
    pub seq_len: usize,
}

impl RopeTables {
    /// Build the tables from the packed `(t, h, w)` grid.
    ///
    /// `position_ids` is `[seq_len * 3]` in `f64`, as [`crate::layout`] produces
    /// it. Angles are accumulated in `f64` and only the final `cos`/`sin` are
    /// narrowed, matching the reference's `position_ids.to(float32)` ordering
    /// closely enough that the rotation is bit-stable in `f32`.
    pub fn build(position_ids: &[f64], rope_freq_dim: usize, rope_theta: f32) -> Result<Self> {
        ensure!(rope_freq_dim > 0, "rope_freq_dim must be positive");
        ensure!(
            position_ids.len().is_multiple_of(3),
            "position_ids len {} is not a multiple of 3",
            position_ids.len()
        );
        let seq_len = position_ids.len() / 3;
        let half = 3 * rope_freq_dim;

        // inv_freq[i] = 1 / theta^(2i / (2 * rope_freq_dim)) for i in 0..freq_dim
        let inv_freq: Vec<f64> = (0..rope_freq_dim)
            .map(|i| {
                let e = (2 * i) as f64 / (2 * rope_freq_dim) as f64;
                1.0 / (rope_theta as f64).powf(e)
            })
            .collect();

        let mut cos = vec![0.0f32; seq_len * half];
        let mut sin = vec![0.0f32; seq_len * half];
        for s in 0..seq_len {
            let pos = &position_ids[s * 3..s * 3 + 3];
            for (axis, &p) in pos.iter().enumerate() {
                for (k, &f) in inv_freq.iter().enumerate() {
                    let angle = p * f;
                    let o = s * half + axis * rope_freq_dim + k;
                    cos[o] = angle.cos() as f32;
                    sin[o] = angle.sin() as f32;
                }
            }
        }
        Ok(Self {
            cos,
            sin,
            half,
            seq_len,
        })
    }

    /// Rotated channels per head: `2 * half`.
    #[must_use]
    pub fn n_rot(&self) -> usize {
        2 * self.half
    }

    /// Apply the rotation on the host, for reference checks.
    ///
    /// `x` is `[seq_len, heads, head_dim]`. Channels beyond `n_rot` pass
    /// through unchanged.
    pub fn apply(&self, x: &[f32], heads: usize, head_dim: usize) -> Result<Vec<f32>> {
        let n_rot = self.n_rot();
        ensure!(
            n_rot <= head_dim,
            "n_rot {n_rot} exceeds head_dim {head_dim}"
        );
        ensure!(
            x.len() == self.seq_len * heads * head_dim,
            "x len {} != seq {} × heads {heads} × head_dim {head_dim}",
            x.len(),
            self.seq_len
        );
        let half = self.half;
        let mut out = x.to_vec();
        for s in 0..self.seq_len {
            for h in 0..heads {
                let base = (s * heads + h) * head_dim;
                for i in 0..half {
                    let c = self.cos[s * half + i];
                    let sn = self.sin[s * half + i];
                    let x1 = x[base + i];
                    let x2 = x[base + i + half];
                    out[base + i] = x1 * c - x2 * sn;
                    out[base + i + half] = x2 * c + x1 * sn;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_shape_matches_released_config() {
        let pos = vec![0.0f64; 4 * 3];
        let t = RopeTables::build(&pos, 16, 10_000.0).unwrap();
        assert_eq!(t.half, 48);
        assert_eq!(t.n_rot(), 96);
        assert_eq!(t.cos.len(), 4 * 48);
        assert_eq!(t.seq_len, 4);
    }

    #[test]
    fn zero_position_gives_identity_rotation() {
        let pos = vec![0.0f64; 2 * 3];
        let t = RopeTables::build(&pos, 16, 10_000.0).unwrap();
        assert!(t.cos.iter().all(|&c| (c - 1.0).abs() < 1e-6));
        assert!(t.sin.iter().all(|&s| s.abs() < 1e-6));
        let x: Vec<f32> = (0..2 * 2 * 128).map(|i| i as f32).collect();
        let y = t.apply(&x, 2, 128).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn channels_beyond_n_rot_pass_through() {
        let pos: Vec<f64> = vec![1.0, 2.0, 3.0];
        let t = RopeTables::build(&pos, 16, 10_000.0).unwrap();
        let head_dim = 128;
        let x: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
        let y = t.apply(&x, 1, head_dim).unwrap();
        // 96 rotated, 32 untouched.
        for i in 96..head_dim {
            assert_eq!(x[i], y[i], "channel {i} must pass through");
        }
        assert!((0..96).any(|i| (x[i] - y[i]).abs() > 1e-6));
    }

    #[test]
    fn each_axis_drives_its_own_frequency_block() {
        // A position that is non-zero on the height axis only must leave the
        // time and width blocks at angle 0.
        let pos = vec![0.0f64, 5.0, 0.0];
        let t = RopeTables::build(&pos, 16, 10_000.0).unwrap();
        for k in 0..16 {
            assert!((t.sin[k]).abs() < 1e-6, "time block must be unrotated");
            assert!(
                (t.sin[32 + k]).abs() < 1e-6,
                "width block must be unrotated"
            );
        }
        assert!(t.sin[16..32].iter().any(|&s| s.abs() > 1e-3));
    }

    #[test]
    fn rotation_preserves_norm() {
        let pos = vec![3.0f64, -1.5, 7.25];
        let t = RopeTables::build(&pos, 16, 10_000.0).unwrap();
        let x: Vec<f32> = (0..128).map(|i| ((i % 7) as f32) - 3.0).collect();
        let y = t.apply(&x, 1, 128).unwrap();
        let nx: f32 = x.iter().map(|v| v * v).sum();
        let ny: f32 = y.iter().map(|v| v * v).sum();
        assert!((nx - ny).abs() < 1e-3 * nx.max(1.0), "{nx} vs {ny}");
    }

    #[test]
    fn inv_freq_is_descending() {
        let pos = vec![1.0f64, 0.0, 0.0];
        let t = RopeTables::build(&pos, 16, 10_000.0).unwrap();
        // With position 1 on the time axis, angle_k = inv_freq[k] and inv_freq
        // decreases, so cos increases toward 1.
        let c: Vec<f32> = t.cos[0..16].to_vec();
        assert!(
            c.windows(2).all(|w| w[1] >= w[0] - 1e-6),
            "cos should rise as the frequency falls: {c:?}"
        );
    }

    #[test]
    fn rejects_malformed_grid() {
        assert!(RopeTables::build(&[0.0, 1.0], 16, 10_000.0).is_err());
        assert!(RopeTables::build(&[0.0, 1.0, 2.0], 0, 10_000.0).is_err());
    }
}
