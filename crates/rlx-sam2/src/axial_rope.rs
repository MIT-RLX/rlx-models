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

//! SAM2 axial 2-D RoPE (host + cos/sin tables for [`Op::Rope`] IR).

/// Per-token cos/sin for x-axis and y-axis halves (`[seq, head_dim/4]` each).
pub fn axial_rope_cos_sin_for_seq(
    end_x: usize,
    end_y: usize,
    n_tokens: usize,
    head_dim: usize,
    theta: f32,
    repeat_factor: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert!(head_dim.is_multiple_of(4));
    let q4 = head_dim / 4;
    let spatial = end_x * end_y;
    let mut freqs = vec![0f32; q4];
    for i in 0..q4 {
        freqs[i] = 1.0 / theta.powf((4 * i) as f32 / head_dim as f32);
    }
    let mut cs_x = vec![0f32; spatial * q4];
    let mut sn_x = vec![0f32; spatial * q4];
    let mut cs_y = vec![0f32; spatial * q4];
    let mut sn_y = vec![0f32; spatial * q4];
    for pos in 0..spatial {
        let tx = (pos % end_x) as f32;
        let ty = (pos / end_x) as f32;
        for c in 0..q4 {
            let ax = tx * freqs[c];
            let ay = ty * freqs[c];
            cs_x[pos * q4 + c] = ax.cos();
            sn_x[pos * q4 + c] = ax.sin();
            cs_y[pos * q4 + c] = ay.cos();
            sn_y[pos * q4 + c] = ay.sin();
        }
    }
    let mut cos_x = vec![0f32; n_tokens * q4];
    let mut sin_x = vec![0f32; n_tokens * q4];
    let mut cos_y = vec![0f32; n_tokens * q4];
    let mut sin_y = vec![0f32; n_tokens * q4];
    for tok in 0..n_tokens {
        let pos = tok / repeat_factor;
        for c in 0..q4 {
            cos_x[tok * q4 + c] = cs_x[pos * q4 + c];
            sin_x[tok * q4 + c] = sn_x[pos * q4 + c];
            cos_y[tok * q4 + c] = cs_y[pos * q4 + c];
            sin_y[tok * q4 + c] = sn_y[pos * q4 + c];
        }
    }
    (cos_x, sin_x, cos_y, sin_y)
}

/// Apply axial 2-D RoPE on `x` shaped `[nh, n_tokens, head_dim]`.
pub fn apply_axial_rope_2d(
    x: &[f32],
    nh: usize,
    n_tokens: usize,
    dh: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    repeat_factor: usize,
) -> Vec<f32> {
    debug_assert!(
        dh.is_multiple_of(4),
        "RoPE expects head_dim multiple of 4 (got {dh})"
    );
    let half = dh / 2;
    let q4 = dh / 4;
    let spatial = end_x * end_y;
    debug_assert_eq!(
        n_tokens,
        spatial * repeat_factor,
        "RoPE token count mismatch"
    );

    let (cs_x, sn_x, cs_y, sn_y) =
        axial_rope_cos_sin_for_seq(end_x, end_y, n_tokens, dh, theta, repeat_factor);

    let mut out = vec![0f32; nh * n_tokens * dh];
    for h in 0..nh {
        for tok in 0..n_tokens {
            let pos = tok / repeat_factor;
            let src_base = (h * n_tokens + tok) * dh;
            let dst_base = src_base;
            for c in 0..q4 {
                let ix0 = src_base + 2 * c;
                let ix1 = src_base + 2 * c + 1;
                let x0 = x[ix0];
                let x1 = x[ix1];
                let co = cs_x[pos * q4 + c];
                let si = sn_x[pos * q4 + c];
                out[dst_base + 2 * c] = x0 * co - x1 * si;
                out[dst_base + 2 * c + 1] = x0 * si + x1 * co;
            }
            for c in 0..q4 {
                let ix0 = src_base + half + 2 * c;
                let ix1 = src_base + half + 2 * c + 1;
                let x0 = x[ix0];
                let x1 = x[ix1];
                let co = cs_y[pos * q4 + c];
                let si = sn_y[pos * q4 + c];
                out[dst_base + half + 2 * c] = x0 * co - x1 * si;
                out[dst_base + half + 2 * c + 1] = x0 * si + x1 * co;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interleaved pair rotation on `[nh, n, half]` (matches [`apply_axial_rope_2d`] x/y halves).
    fn rope_interleaved_half(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        nh: usize,
        n: usize,
        half: usize,
    ) -> Vec<f32> {
        let q4 = half / 2;
        let mut out = x.to_vec();
        for h in 0..nh {
            for tok in 0..n {
                let base = (h * n + tok) * half;
                for c in 0..q4 {
                    let ix0 = base + 2 * c;
                    let ix1 = base + 2 * c + 1;
                    let co = cos[tok * q4 + c];
                    let si = sin[tok * q4 + c];
                    let x0 = out[ix0];
                    let x1 = out[ix1];
                    out[ix0] = x0 * co - x1 * si;
                    out[ix1] = x0 * si + x1 * co;
                }
            }
        }
        out
    }

    #[test]
    fn dual_rope_n_matches_apply_axial() {
        let nh = 1usize;
        let n = 64usize;
        let dh = 256usize;
        let end_x = 8usize;
        let end_y = 8usize;
        let x: Vec<f32> = (0..nh * n * dh).map(|i| i as f32 * 0.001).collect();
        let host = apply_axial_rope_2d(&x, nh, n, dh, end_x, end_y, 10000.0, 1);
        let half = dh / 2;
        let (cx, sx, cy, sy) = axial_rope_cos_sin_for_seq(end_x, end_y, n, dh, 10000.0, 1);
        let mut lo = vec![0f32; nh * n * half];
        let mut hi = vec![0f32; nh * n * half];
        for h in 0..nh {
            for tok in 0..n {
                let base = (h * n + tok) * dh;
                let off = (h * n + tok) * half;
                lo[off..off + half].copy_from_slice(&x[base..base + half]);
                hi[off..off + half].copy_from_slice(&x[base + half..base + dh]);
            }
        }
        let lo_r = rope_interleaved_half(&lo, &cx, &sx, nh, n, half);
        let hi_r = rope_interleaved_half(&hi, &cy, &sy, nh, n, half);
        let mut manual = vec![0f32; nh * n * dh];
        for h in 0..nh {
            for tok in 0..n {
                let base = (h * n + tok) * dh;
                let off = (h * n + tok) * half;
                manual[base..base + half].copy_from_slice(&lo_r[off..off + half]);
                manual[base + half..base + dh].copy_from_slice(&hi_r[off..off + half]);
            }
        }
        let fd = host
            .iter()
            .zip(&manual)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 2e-2, "dual rope_n vs axial max |Δ| = {fd:.3e}");
    }
}
