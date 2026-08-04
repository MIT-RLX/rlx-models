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

//! Host-side math shared by the flow builders: RoPE cos/sin tables for
//! arbitrary `position_ids`, the block-causal additive attention mask, and the
//! flow-matching sinusoidal time embedding.

use std::f64::consts::PI;

/// Large negative value used for masked attention entries (matches the
/// reference `torch.finfo(dtype).min` clamp closely enough for f32 softmax).
pub const MASK_NEG: f32 = -1e9;

/// `position_ids = cumsum(pad) - 1` (reference
/// `build_attention_mask_and_position_ids`). Masked tokens keep the previous
/// position (harmless — they are masked out in attention).
pub fn position_ids_from_pad(pad: &[bool]) -> Vec<i64> {
    let mut acc: i64 = 0;
    pad.iter()
        .map(|&p| {
            if p {
                acc += 1;
            }
            acc - 1
        })
        .collect()
}

/// Per-pair inverse frequencies `1 / theta^(2i/head_dim)`, length `head_dim/2`.
pub fn inv_freq(theta: f64, head_dim: usize) -> Vec<f64> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64))
        .collect()
}

/// RoPE cos/sin tables for the given `position_ids`, each flat
/// `[seq · head_dim/2]` (row `i` holds the angles for `position_ids[i]`).
///
/// The RLX `Op::Rope` kernel indexes the table by token index within the
/// sequence (row `i` → token `i`), so pre-baking per-token positions here lets
/// us realize the reference's `cumsum(pad)-1` positions exactly.
pub fn rope_tables(theta: f64, head_dim: usize, position_ids: &[i64]) -> (Vec<f32>, Vec<f32>) {
    let freqs = inv_freq(theta, head_dim);
    let half = freqs.len();
    let seq = position_ids.len();
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (t, &pos) in position_ids.iter().enumerate() {
        let p = pos.max(0) as f64;
        for (i, &f) in freqs.iter().enumerate() {
            let angle = p * f;
            cos[t * half + i] = angle.cos() as f32;
            sin[t * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

/// Block-causal additive attention bias `[heads · seq · seq]` (batch 1),
/// matching `build_attention_mask_and_position_ids`:
/// ```text
///   c[j]        = cumsum(att)[j]
///   allow[i,j]  = (c[j] <= c[i]) && pad[i] && pad[j]
///   bias[i,j]   = allow ? 0 : MASK_NEG
/// ```
/// `att` marks block boundaries (0 within a bidirectional block, 1 opens a new
/// block). The same `[seq,seq]` pattern is broadcast across all heads.
pub fn block_causal_bias(pad: &[bool], att: &[i32], heads: usize) -> Vec<f32> {
    let seq = pad.len();
    assert_eq!(att.len(), seq, "att/pad length mismatch");
    let mut cum = vec![0i64; seq];
    let mut acc = 0i64;
    for (j, &a) in att.iter().enumerate() {
        acc += a as i64;
        cum[j] = acc;
    }
    // [seq, seq] pattern first.
    let mut pat = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let allow = cum[j] <= cum[i] && pad[i] && pad[j];
            pat[i * seq + j] = if allow { 0.0 } else { MASK_NEG };
        }
    }
    // Broadcast to [heads, seq, seq].
    let mut out = vec![0f32; heads * seq * seq];
    for h in 0..heads {
        out[h * seq * seq..(h + 1) * seq * seq].copy_from_slice(&pat);
    }
    out
}

/// Flow-matching sinusoidal time embedding (reference
/// `create_sinusoidal_pos_embedding`): `dim`-vector for a scalar `time`.
/// ```text
///   fraction = linspace(0, 1, dim/2)
///   period   = min_period * (max_period/min_period)^fraction
///   scale    = 1/period * 2π
///   emb      = concat(sin(scale·time), cos(scale·time))
/// ```
pub fn sinusoidal_time_embedding(
    time: f32,
    dim: usize,
    min_period: f32,
    max_period: f32,
) -> Vec<f32> {
    assert!(dim.is_multiple_of(2), "time embedding dim must be even");
    let half = dim / 2;
    let mut emb = vec![0f32; dim];
    let t = time as f64;
    let min_p = min_period as f64;
    let max_p = max_period as f64;
    for i in 0..half {
        let fraction = if half > 1 {
            i as f64 / (half - 1) as f64
        } else {
            0.0
        };
        let period = min_p * (max_p / min_p).powf(fraction);
        let scale = 1.0 / period * 2.0 * PI;
        let arg = scale * t;
        emb[i] = arg.sin() as f32;
        emb[half + i] = arg.cos() as f32;
    }
    emb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_skip_padding() {
        let pad = [true, true, false, true];
        assert_eq!(position_ids_from_pad(&pad), vec![0, 1, 1, 2]);
    }

    #[test]
    fn prefix_block_is_bidirectional_suffix_is_blocked() {
        // 2 prefix tokens (att=0,0) then 2 suffix tokens (att=1,0), all real.
        let pad = [true, true, true, true];
        let att = [0, 0, 1, 0];
        let bias = block_causal_bias(&pad, &att, 1);
        let seq = 4;
        let at = |i: usize, j: usize| bias[i * seq + j];
        // prefix→prefix bidirectional (all allowed).
        assert_eq!(at(0, 1), 0.0);
        assert_eq!(at(1, 0), 0.0);
        // prefix cannot attend to suffix (c[j]>c[i]).
        assert_eq!(at(0, 2), MASK_NEG);
        // suffix attends to prefix.
        assert_eq!(at(2, 0), 0.0);
        assert_eq!(at(3, 1), 0.0);
        // suffix token 2 (c=1) and 3 (c=1) attend to each other (same block).
        assert_eq!(at(2, 3), 0.0);
        assert_eq!(at(3, 2), 0.0);
    }

    #[test]
    fn time_embedding_shape_and_bounds() {
        let e = sinusoidal_time_embedding(1.0, 8, 4e-3, 4.0);
        assert_eq!(e.len(), 8);
        assert!(e.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}
