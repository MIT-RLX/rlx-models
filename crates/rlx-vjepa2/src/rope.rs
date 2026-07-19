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

//! 3-D RoPE for V-JEPA2 encoder attention (frame / height / width axes).

/// Decompose flat patch token index into `(frame, height, width)` positions.
pub fn decompose_token_pos(
    token_idx: usize,
    grid_h: usize,
    grid_w: usize,
) -> (usize, usize, usize) {
    let tokens_per_frame = grid_h * grid_w;
    let frame = token_idx / tokens_per_frame;
    let rem = token_idx - tokens_per_frame * frame;
    let height = rem / grid_w;
    let width = rem - grid_w * height;
    (frame, height, width)
}

/// Apply V-JEPA2 3-D RoPE in-place to `q` and `k`.
///
/// When `position_ids` is `Some`, each entry is the patch-grid token index
/// used for frame/height/width decomposition (predictor sorted sequences).
/// Otherwise sequential `0..seq-1` is used (encoder).
pub fn apply_vjepa2_rope(
    q: &mut [f32],
    k: &mut [f32],
    batch_heads: usize,
    seq: usize,
    head_dim: usize,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    d_dim: usize,
    h_dim: usize,
    w_dim: usize,
    position_ids: Option<&[usize]>,
) {
    let rotated = d_dim + h_dim + w_dim;
    debug_assert!(rotated <= head_dim);

    let mut frame_ids = vec![0usize; seq];
    let mut height_ids = vec![0usize; seq];
    let mut width_ids = vec![0usize; seq];
    for t in 0..seq {
        let patch_idx = position_ids.map(|p| p[t]).unwrap_or(t);
        let (f, h, w) = decompose_token_pos(patch_idx, grid_h, grid_w);
        frame_ids[t] = f;
        height_ids[t] = h;
        width_ids[t] = w;
    }
    let _ = grid_t;

    for bh in 0..batch_heads {
        for li in 0..seq {
            let base = (bh * seq + li) * head_dim;
            rotate_segment(
                &mut q[base..base + head_dim],
                0,
                d_dim,
                frame_ids[li] as f32,
            );
            rotate_segment(
                &mut q[base..base + head_dim],
                d_dim,
                h_dim,
                height_ids[li] as f32,
            );
            rotate_segment(
                &mut q[base..base + head_dim],
                d_dim + h_dim,
                w_dim,
                width_ids[li] as f32,
            );
            rotate_segment(
                &mut k[base..base + head_dim],
                0,
                d_dim,
                frame_ids[li] as f32,
            );
            rotate_segment(
                &mut k[base..base + head_dim],
                d_dim,
                h_dim,
                height_ids[li] as f32,
            );
            rotate_segment(
                &mut k[base..base + head_dim],
                d_dim + h_dim,
                w_dim,
                width_ids[li] as f32,
            );
        }
    }
}

/// In-place RoPE on `row[off..off+seg_dim]` with scalar position `pos`.
///
/// Matches Meta / HF `rotate_queries_or_keys` exactly. HF broadcasts the
/// `[seg_dim/2]` cos/sin tables across the segment with `.repeat(..., 2)`
/// (i.e. **tiled**, `[c0..c_{h-1}, c0..c_{h-1}]`, NOT `repeat_interleave`), so
/// element `j` uses frequency index `j % (seg_dim/2)`, while the rotate step
/// pairs *adjacent* elements `(2m, 2m+1)`:
///   `y[2m] = -x[2m+1]`, `y[2m+1] = x[2m]`, `out = x*cos + y*sin`.
/// This deliberately gives the two elements of a pair *different* angles — it is
/// the V-JEPA2 checkpoint's actual (quirky) convention. A textbook "clean"
/// rotation (shared angle per pair) does NOT match the trained weights.
fn rotate_segment(row: &mut [f32], off: usize, seg_dim: usize, pos: f32) {
    if seg_dim == 0 {
        return;
    }
    let half = seg_dim / 2;
    let orig: Vec<f32> = row[off..off + seg_dim].to_vec();
    for j in 0..seg_dim {
        let k = j % half;
        let exp = (2 * k) as f32 / seg_dim as f32;
        let omega = 1.0 / 10_000.0f32.powf(exp);
        let ang = pos * omega;
        let c = ang.cos();
        let s = ang.sin();
        // y_full (HF `stack((-y2, y1)).flatten`): even j → -x[j+1]; odd j → x[j-1].
        let y = if j % 2 == 0 {
            -orig[j + 1]
        } else {
            orig[j - 1]
        };
        row[off + j] = orig[j] * c + y * s;
    }
}

/// Precompute cos/sin tables `[seq, head_dim/2]` for 3-D RoPE (all segments).
///
/// Used by the IR graph builder — one table pair per encoder; identity fill
/// on non-rotated tail pairs.
pub fn build_vjepa2_rope_tables(
    seq: usize,
    head_dim: usize,
    d_dim: usize,
    h_dim: usize,
    w_dim: usize,
    grid_h: usize,
    grid_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    let seg_dims = [d_dim, h_dim, w_dim];

    for t in 0..seq {
        let (frame, height, width) = decompose_token_pos(t, grid_h, grid_w);
        let coords = [frame as f32, height as f32, width as f32];
        let mut pair_base = 0usize;
        for (si, &sd) in seg_dims.iter().enumerate() {
            let pairs = sd / 2;
            for k in 0..pairs {
                let exp = (2 * k) as f32 / sd as f32;
                let omega = 1.0 / 10_000.0f32.powf(exp);
                let ang = coords[si] * omega;
                let gi = t * half + pair_base + k;
                cos[gi] = ang.cos();
                sin[gi] = ang.sin();
            }
            pair_base += pairs;
        }
        for k in pair_base..half {
            cos[t * half + k] = 1.0;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_token_pos_matches_grid() {
        let (f, h, w) = decompose_token_pos(576 + 24 + 5, 24, 24);
        assert_eq!(f, 1);
        assert_eq!(h, 1);
        assert_eq!(w, 5);
    }

    #[test]
    fn rotate_segment_is_involutory_at_zero_angle_offset() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        rotate_segment(&mut v, 0, 4, 0.0);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 2.0).abs() < 1e-6);
    }
}
