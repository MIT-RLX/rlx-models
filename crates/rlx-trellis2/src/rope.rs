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

//! TRELLIS 3-D axial rotary position embedding.
//!
//! Upstream (`trellis2/modules/attention/rope.py`, `RotaryPositionEmbedder`)
//! builds, per voxel coordinate `(x, y, z)`, an **interleaved** complex phase
//! table of length `head_dim/2` and applies it as
//! `pair_i = (q[2i], q[2i+1]) · e^{iθ_i}`.
//!
//! For each head with `head_dim = 2H` channels the angle vector is
//! ```text
//!   θ_i = coord[axis(i)] · freq[i mod F]   for i < 3F,   else 0   (identity pad)
//! ```
//! where `F = head_dim / 2 / 3` and `freq[k] = base0 / base1^{k/F}`
//! (defaults `base0 = 1`, `base1 = 10000`).
//!
//! Two consumers are supported:
//!   * [`apply_interleaved_rope`] — the exact host reference (pairs adjacent
//!     channels), used by the CPU DiT reference and parity tests.
//!   * [`deinterleave_perm`] + NeoX-style tables from [`RopeTables::neox`] —
//!     lets the graph reuse the stock split-half `rope` op. Permuting the
//!     `head_dim` channels of **both** q and k by [`deinterleave_perm`] turns
//!     interleaved rotation into split-half rotation with identical attention
//!     scores (a permutation preserves `q·k`, and v/output are untouched).

/// Rotary frequencies for a `dim`-axis embedding over `head_dim` channels.
///
/// `F = head_dim / 2 / dim` frequencies, `freq[k] = base.0 / base.1^{k/F}`.
pub fn axial_freqs(head_dim: usize, dim: usize, base: (f32, f32)) -> Vec<f32> {
    let f = head_dim / 2 / dim;
    (0..f)
        .map(|k| base.0 / base.1.powf(k as f32 / f as f32))
        .collect()
}

/// Per-position rotation angles `θ` of length `head_dim/2`.
///
/// `coords`: `[n_pos * dim]` row-major integer voxel coordinates (as f32).
/// Returns `[n_pos * (head_dim/2)]` angles; the final `head_dim/2 - dim*F`
/// entries are the identity pad (`θ = 0`).
pub fn axial_angles(
    coords: &[f32],
    n_pos: usize,
    head_dim: usize,
    dim: usize,
    base: (f32, f32),
) -> Vec<f32> {
    let freqs = axial_freqs(head_dim, dim, base);
    let f = freqs.len();
    let half = head_dim / 2;
    debug_assert_eq!(coords.len(), n_pos * dim);
    let mut ang = vec![0.0f32; n_pos * half];
    for p in 0..n_pos {
        let out = &mut ang[p * half..(p + 1) * half];
        for axis in 0..dim {
            let c = coords[p * dim + axis];
            for k in 0..f {
                out[axis * f + k] = c * freqs[k];
            }
        }
        // remaining [dim*F .. half) stay 0 (identity pad)
    }
    ang
}

/// Cos/sin tables in **NeoX / split-half** layout for the stock `rope` graph
/// op: both `[n_pos * (head_dim/2)]`, indexed the same as [`axial_angles`].
pub struct RopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

impl RopeTables {
    /// Build from voxel coordinates. Pair with [`deinterleave_perm`]-permuted
    /// q/k weights to reproduce the interleaved reference exactly.
    pub fn neox(
        coords: &[f32],
        n_pos: usize,
        head_dim: usize,
        dim: usize,
        base: (f32, f32),
    ) -> Self {
        let ang = axial_angles(coords, n_pos, head_dim, dim, base);
        let cos = ang.iter().map(|a| a.cos()).collect();
        let sin = ang.iter().map(|a| a.sin()).collect();
        Self {
            cos,
            sin,
            half: head_dim / 2,
        }
    }
}

/// De-interleave permutation on a single head's `head_dim` channels: even
/// indices map to the first half, odd indices to the second half. Returns a
/// length-`head_dim` array `perm` such that `y[i] = x[perm[i]]` de-interleaves.
///
/// After `y[i] = x[perm[i]]`, `neox_rope(y)[i] == interleaved_rope(x)[perm[i]]`,
/// so applying it to q **and** k (and their per-head RMS-norm gammas) leaves
/// `q·k` unchanged while letting the split-half rope op stand in for the
/// interleaved one.
pub fn deinterleave_perm(head_dim: usize) -> Vec<usize> {
    let half = head_dim / 2;
    let mut perm = vec![0usize; head_dim];
    for i in 0..half {
        perm[i] = 2 * i; // first half ← even channels
        perm[i + half] = 2 * i + 1; // second half ← odd channels
    }
    perm
}

/// Exact interleaved-RoPE reference for one `[n_pos, n_heads, head_dim]` tensor
/// (row-major). Rotates adjacent channel pairs `(2i, 2i+1)` by `θ_i` from
/// [`axial_angles`]. Mutates in place.
pub fn apply_interleaved_rope(
    x: &mut [f32],
    coords: &[f32],
    n_pos: usize,
    n_heads: usize,
    head_dim: usize,
    dim: usize,
    base: (f32, f32),
) {
    let ang = axial_angles(coords, n_pos, head_dim, dim, base);
    let half = head_dim / 2;
    for p in 0..n_pos {
        let a = &ang[p * half..(p + 1) * half];
        for h in 0..n_heads {
            let off = (p * n_heads + h) * head_dim;
            for i in 0..half {
                let (c, s) = (a[i].cos(), a[i].sin());
                let e = x[off + 2 * i];
                let o = x[off + 2 * i + 1];
                x[off + 2 * i] = e * c - o * s;
                x[off + 2 * i + 1] = e * s + o * c;
            }
        }
    }
}

/// Split-half (NeoX) RoPE reference used to validate the permutation trick:
/// rotates `(x[i], x[i+H])` by `θ_i`. Mutates in place.
pub fn apply_neox_rope(
    x: &mut [f32],
    tables: &RopeTables,
    n_pos: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let half = tables.half;
    debug_assert_eq!(half, head_dim / 2);
    for p in 0..n_pos {
        let cos = &tables.cos[p * half..(p + 1) * half];
        let sin = &tables.sin[p * half..(p + 1) * half];
        for h in 0..n_heads {
            let off = (p * n_heads + h) * head_dim;
            for i in 0..half {
                let x1 = x[off + i];
                let x2 = x[off + i + half];
                x[off + i] = x1 * cos[i] - x2 * sin[i];
                x[off + i + half] = x1 * sin[i] + x2 * cos[i];
            }
        }
    }
}

/// Dense `[res, res, res]` grid coordinates flattened row-major (`ij` order),
/// as f32, shape `[res³ * 3]` — the sparse-structure DiT's token positions.
pub fn grid_coords(res: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(res * res * res * 3);
    for x in 0..res {
        for y in 0..res {
            for z in 0..res {
                out.push(x as f32);
                out.push(y as f32);
                out.push(z as f32);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: (f32, f32) = (1.0, 10000.0);

    #[test]
    fn deinterleave_matches_interleaved_rope() {
        // The permutation trick: neox_rope(perm(x)) == perm(interleaved_rope(x)).
        let (head_dim, dim, n_heads) = (128usize, 3usize, 2usize);
        let n_pos = 5usize;
        let coords: Vec<f32> = (0..n_pos * dim).map(|i| (i % 7) as f32).collect();

        // random-ish deterministic input
        let mut x: Vec<f32> = (0..n_pos * n_heads * head_dim)
            .map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x0 = x.clone();

        // reference: interleaved rope
        apply_interleaved_rope(&mut x, &coords, n_pos, n_heads, head_dim, dim, BASE);

        // trick path: permute channels, apply neox rope
        let perm = deinterleave_perm(head_dim);
        let mut y = vec![0.0f32; x0.len()];
        for p in 0..n_pos {
            for h in 0..n_heads {
                let off = (p * n_heads + h) * head_dim;
                for i in 0..head_dim {
                    y[off + i] = x0[off + perm[i]];
                }
            }
        }
        let tables = RopeTables::neox(&coords, n_pos, head_dim, dim, BASE);
        apply_neox_rope(&mut y, &tables, n_pos, n_heads, head_dim);

        // neox(perm(x))[i] must equal interleaved(x)[perm[i]]
        for p in 0..n_pos {
            for h in 0..n_heads {
                let off = (p * n_heads + h) * head_dim;
                for i in 0..head_dim {
                    let a = y[off + i];
                    let b = x[off + perm[i]];
                    assert!(
                        (a - b).abs() < 1e-5,
                        "mismatch at p{p} h{h} i{i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn qk_dot_is_preserved() {
        // The whole point: attention scores are identical under the trick.
        let (head_dim, dim) = (128usize, 3usize);
        let coords_q: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
        let coords_k: Vec<f32> = (0..dim).map(|i| (2 * i + 3) as f32).collect();
        let q0: Vec<f32> = (0..head_dim).map(|i| (i as f32).sin()).collect();
        let k0: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.7).cos()).collect();

        // interleaved reference dot
        let mut q = q0.clone();
        let mut k = k0.clone();
        apply_interleaved_rope(&mut q, &coords_q, 1, 1, head_dim, dim, BASE);
        apply_interleaved_rope(&mut k, &coords_k, 1, 1, head_dim, dim, BASE);
        let dot_ref: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();

        // trick dot
        let perm = deinterleave_perm(head_dim);
        let mut qp: Vec<f32> = perm.iter().map(|&j| q0[j]).collect();
        let mut kp: Vec<f32> = perm.iter().map(|&j| k0[j]).collect();
        apply_neox_rope(
            &mut qp,
            &RopeTables::neox(&coords_q, 1, head_dim, dim, BASE),
            1,
            1,
            head_dim,
        );
        apply_neox_rope(
            &mut kp,
            &RopeTables::neox(&coords_k, 1, head_dim, dim, BASE),
            1,
            1,
            head_dim,
        );
        let dot_trick: f32 = qp.iter().zip(&kp).map(|(a, b)| a * b).sum();

        assert!(
            (dot_ref - dot_trick).abs() < 1e-4,
            "{dot_ref} vs {dot_trick}"
        );
    }
}
