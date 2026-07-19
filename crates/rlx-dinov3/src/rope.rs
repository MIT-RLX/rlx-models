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

//! DINOv3 2-D axial RoPE tables, host-precomputed as `[seq, head_dim/2]`
//! cos/sin matrices consumable by RLX's NeoX-style [`rope`] op.
//!
//! ## Why this exactly matches HF
//!
//! HF (`DINOv3ViTRopePositionEmbedding`) builds, per patch:
//! ```text
//!   inv_freq[k] = 1 / theta^(4k/head_dim),  k in 0..head_dim/4
//!   coords      = 2·(idx + 0.5)/n_side − 1                    (∈ [−1, 1])
//!   angles      = 2π · coord · inv_freq        → (2, head_dim/4)
//!   angles      = flatten → [y·f₀…y·f_{q−1}, x·f₀…x·f_{q−1}]  (head_dim/2)
//!   angles      = tile(2) → full head_dim
//!   cos,sin     = cos(angles), sin(angles)
//! ```
//! and applies `q·cos + rotate_half(q)·sin` (NeoX rotate-half). Because
//! the angles are *tiled* (`angles[i+half] == angles[i]`), the RLX NeoX
//! rope op — which reads a **half-width** `[seq, head_dim/2]` table and
//! computes `out[i]=x1·cos[i]−x2·sin[i]`, `out[i+half]=x2·cos[i]+x1·sin[i]`
//! — is bit-identical to HF when fed exactly these half tables.
//!
//! RoPE must skip the CLS + register prefix tokens. Rather than slice the
//! sequence in-graph (an MLX/wgpu hazard), we emit **identity rows**
//! (`cos=1, sin=0`) for the prefix, making whole-sequence rope a no-op on
//! those rows.
//!
//! [`rope`]: rlx_ir::HirGraphExt::rope

use std::f64::consts::PI;

/// Build `(cos, sin)` tables of shape `[seq, head_dim/2]` (flat row-major),
/// with identity rows for the first `num_prefix` tokens.
///
/// `num_patches_h` × `num_patches_w` patch grid; `head_dim` must be a
/// multiple of 4 (DINOv3 splits it into y/x halves, each of `head_dim/4`
/// frequencies).
pub fn rope_tables(
    num_patches_h: usize,
    num_patches_w: usize,
    head_dim: usize,
    rope_theta: f64,
    num_prefix: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        head_dim.is_multiple_of(4),
        "DINOv3 RoPE requires head_dim ({head_dim}) divisible by 4"
    );
    let half = head_dim / 2; // table width
    let quarter = head_dim / 4; // frequencies per spatial axis
    let seq = num_prefix + num_patches_h * num_patches_w;

    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];

    // Prefix (CLS + registers): identity rotation → cos=1, sin=0.
    for r in 0..num_prefix {
        let base = r * half;
        for c in 0..half {
            cos[base + c] = 1.0;
            sin[base + c] = 0.0;
        }
    }

    // inv_freq[k] = 1 / theta^(4k/head_dim)
    let inv_freq: Vec<f64> = (0..quarter)
        .map(|k| 1.0 / rope_theta.powf(4.0 * k as f64 / head_dim as f64))
        .collect();

    // meshgrid(indexing="ij"): patch p = py·W + px → (coord_y, coord_x).
    for py in 0..num_patches_h {
        let coord_y = 2.0 * ((py as f64 + 0.5) / num_patches_h as f64) - 1.0;
        for px in 0..num_patches_w {
            let coord_x = 2.0 * ((px as f64 + 0.5) / num_patches_w as f64) - 1.0;
            let row = num_prefix + py * num_patches_w + px;
            let base = row * half;
            for k in 0..quarter {
                // First quarter columns: y-axis; second quarter: x-axis.
                let ang_y = 2.0 * PI * coord_y * inv_freq[k];
                let ang_x = 2.0 * PI * coord_x * inv_freq[k];
                cos[base + k] = ang_y.cos() as f32;
                sin[base + k] = ang_y.sin() as f32;
                cos[base + quarter + k] = ang_x.cos() as f32;
                sin[base + quarter + k] = ang_x.sin() as f32;
            }
        }
    }

    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_rows_are_identity() {
        let (cos, sin) = rope_tables(2, 2, 16, 100.0, 5);
        let half = 8;
        for r in 0..5 {
            for c in 0..half {
                assert_eq!(cos[r * half + c], 1.0);
                assert_eq!(sin[r * half + c], 0.0);
            }
        }
        // A patch row must carry a genuine rotation.
        let patch0 = 5 * half;
        assert!((0..half).any(|c| sin[patch0 + c].abs() > 1e-6));
    }

    #[test]
    fn shape_matches_seq_by_half() {
        let (cos, sin) = rope_tables(14, 14, 64, 100.0, 5);
        let seq = 5 + 14 * 14;
        assert_eq!(cos.len(), seq * 32);
        assert_eq!(sin.len(), seq * 32);
    }
}
