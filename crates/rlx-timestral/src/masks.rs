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

//! Attention masks and time positions over the `[Enc ; Dec]` sequence
//! (upstream `timestral.masks`).
//!
//! Masks are built as booleans (true = may attend) in row-major `Vec<bool>`s and turned
//! into additive biases by [`to_bias`]. Every mask keeps its diagonal open, so no query
//! row is ever fully masked and the additive form equals the boolean SDPA mask.

/// Additive value for a masked key. Large enough that `exp` underflows to exactly 0.
pub const MASKED: f32 = -1e9;

pub fn to_bias(mask: &[bool]) -> Vec<f32> {
    mask.iter().map(|&m| if m { 0.0 } else { MASKED }).collect()
}

/// `(R, N)` encoder key validity from `(R, N)` per-patch observation flags.
///
/// A patch is a valid key if it is observed or, with `attend_missing`, if the row has
/// started at or before it. Leading unobserved patches are left padding.
pub fn key_validity(obs_per_patch: &[bool], n: usize, attend_missing: bool) -> Vec<bool> {
    if !attend_missing {
        return obs_per_patch.to_vec();
    }
    let mut out = vec![false; obs_per_patch.len()];
    for (row_in, row_out) in obs_per_patch.chunks(n).zip(out.chunks_mut(n)) {
        let mut started = false;
        for (o, v) in row_in.iter().zip(row_out.iter_mut()) {
            started |= *o;
            *v = started;
        }
    }
    out
}

/// `(R, L+h, L+h)` time mask for one block without covariates.
///
/// The encoder is causal over valid keys; each decoder token reads the whole valid
/// context and its causal decoder chain; the context never reads the horizon.
pub fn forecast_time_mask(key_valid: &[bool], l: usize, h: usize) -> Vec<bool> {
    let r = key_valid.len() / l;
    let s = l + h;
    let mut m = vec![false; r * s * s];
    for row in 0..r {
        let kv = &key_valid[row * l..(row + 1) * l];
        let base = row * s * s;
        for i in 0..s {
            for j in 0..s {
                let allow = if i < l {
                    j < l && ((j <= i && kv[j]) || i == j)
                } else if j < l {
                    kv[j]
                } else {
                    j <= i
                };
                m[base + i * s + j] = allow;
            }
        }
    }
    m
}

/// `(R, 2N, 2N)` time mask over a full `[Enc ; Dec]` window (the covariate path).
///
/// - Enc to Enc: causal over valid keys; full for `bidirectional` rows.
/// - Enc to Dec: never.
/// - Dec to Enc: a decoder token in the block anchored at `b` reads valid `Enc[j]`, `j < b`.
/// - Dec to Dec: causal within the same block; every decoder token reads itself.
pub fn two_stream_time_mask(
    min_context: &[usize],
    n: usize,
    block_size: usize,
    key_valid: &[bool],
    bidirectional: &[bool],
) -> Vec<bool> {
    let r = min_context.len();
    let s = 2 * n;
    let mut m = vec![false; r * s * s];
    for row in 0..r {
        let kv = &key_valid[row * n..(row + 1) * n];
        let start = min_context[row];
        let layout = |p: usize| -> (usize, bool) {
            if p >= start {
                (start + block_size * ((p - start) / block_size), true)
            } else {
                (0, false)
            }
        };
        let base = row * s * s;
        for i in 0..n {
            for j in 0..n {
                let enc = if bidirectional[row] {
                    kv[j] || i == j
                } else {
                    (j <= i && kv[j]) || i == j
                };
                m[base + i * s + j] = enc;

                let (anchor_i, active_i) = layout(i);
                let (anchor_j, active_j) = layout(j);
                m[base + (n + i) * s + j] = active_i && j < anchor_i && kv[j];
                let dec = anchor_i == anchor_j && j <= i && active_i && active_j;
                m[base + (n + i) * s + n + j] = dec || i == j;
            }
        }
    }
    m
}

/// `(S, R, R)` variate mask over `S = n_enc + n_dec` slots.
///
/// A row reads another row of the same series; on encoder slots the key row must also
/// be a valid key there. Two firewalls hold on every slot: a covariate row never reads
/// a target row, and a known-future (bidirectional) row never reads a causal row.
pub fn variate_mask(
    group_ids: &[usize],
    enc_key_valid: &[bool],
    n_enc: usize,
    n_dec: usize,
    target_rows: Option<&[bool]>,
    bidirectional: Option<&[bool]>,
) -> Vec<bool> {
    let r = group_ids.len();
    let s = n_enc + n_dec;
    let mut m = vec![false; s * r * r];
    for slot in 0..s {
        for q in 0..r {
            for k in 0..r {
                let key_ok = slot >= n_enc || enc_key_valid[k * n_enc + slot];
                let mut allow = group_ids[q] == group_ids[k] && key_ok;
                if let Some(t) = target_rows {
                    allow &= !(!t[q] && t[k]);
                }
                if let Some(b) = bidirectional {
                    allow &= !(b[q] && !b[k]);
                }
                m[(slot * r + q) * r + k] = allow || q == k;
            }
        }
    }
    m
}

/// `(G,)` first patch any row of each group observes (0 if none).
pub fn group_first_observed_patch(
    obs_per_patch: &[bool],
    n: usize,
    group_ids: &[usize],
) -> Vec<usize> {
    let groups = group_ids.iter().copied().max().map_or(0, |g| g + 1);
    let mut first = vec![usize::MAX; groups];
    for (row, &g) in group_ids.iter().enumerate() {
        if let Some(p) = obs_per_patch[row * n..(row + 1) * n]
            .iter()
            .position(|&o| o)
        {
            first[g] = first[g].min(p);
        }
    }
    first
        .into_iter()
        .map(|f| if f == usize::MAX { 0 } else { f })
        .collect()
}

/// Per-row time positions of an `[Enc ; Dec]` sequence, anchored at the row's first
/// observed patch. `two_stream` gives the encoder and decoder token of a patch the same
/// time (`(R, 2N)`); otherwise positions run `0..L+h` (`(R, L+h)`).
pub fn positions(first_obs: &[usize], l: usize, h: usize, two_stream: bool) -> Vec<i64> {
    let s = if two_stream { 2 * l } else { l + h };
    let mut out = Vec::with_capacity(first_obs.len() * s);
    for &f in first_obs {
        for p in 0..s {
            let t = if two_stream && p >= l { p - l } else { p };
            out.push(t as i64 - f as i64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_mask_blocks_context_from_horizon() {
        let kv = [false, true, true];
        let m = forecast_time_mask(&kv, 3, 2);
        let s = 5;
        assert!(m[0], "diagonal stays open on a padding patch");
        assert!(!m[s], "padding is not a valid key");
        assert!(
            m[2 * s + 1] && !m[2 * s + 3],
            "context never reads the horizon"
        );
        assert!(m[3 * s + 1] && m[3 * s + 2] && !m[3 * s]);
        assert!(m[4 * s + 3] && !m[3 * s + 4], "decoder chain is causal");
    }

    #[test]
    fn variate_firewall_keeps_covariates_off_targets() {
        let groups = [0, 0];
        let kv = [true, true];
        let m = variate_mask(&groups, &kv, 1, 1, Some(&[true, false]), None);
        // Slot 0: target(0) reads covariate(1); covariate never reads the target.
        assert!(m[1] && !m[2]);
    }
}
