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

//! Block attention masks for MTP (ported from HF `mask_sdpa_utils`).

use crate::config::LocateAnythingConfig;

const MASK_BLOCKED: f32 = f32::NEG_INFINITY;

/// Build a causal float mask `[seq, seq]`: `0` = attend, `MASK_BLOCKED` = masked.
pub fn causal_mask_f32(seq: usize) -> Vec<f32> {
    let mut m = vec![0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            if k > q {
                m[q * seq + k] = MASK_BLOCKED;
            }
        }
    }
    m
}

/// MTP inference window mask (last `block_size` tokens bidirectional within the window).
pub fn mtp_window_mask_f32(seq: usize, block_size: usize, use_cache: bool) -> Vec<f32> {
    let mut m = causal_mask_f32(seq);
    if seq < block_size {
        return m;
    }
    let start = seq - block_size;
    for q in start..seq {
        for k in start..seq {
            m[q * seq + k] = 0.0;
        }
    }
    if use_cache && start > 0 {
        let mask_col = start - 1;
        for q in start..seq {
            m[q * seq + mask_col] = MASK_BLOCKED;
        }
    }
    m
}

/// HF `update_causal_mask_for_one_gen_window_2d`.
pub fn update_causal_mask_for_one_gen_window_2d(
    attn_mask_2d: &mut [f32],
    seq: usize,
    block_size: usize,
    use_cache: bool,
    causal_attn: bool,
) {
    if causal_attn || seq < block_size {
        return;
    }
    let start = seq - block_size;
    for q in start..seq {
        for k in start..seq {
            attn_mask_2d[q * seq + k] = 0.0;
        }
    }
    if use_cache && start > 0 {
        let mask_col = start - 1;
        for q in start..seq {
            attn_mask_2d[q * seq + mask_col] = MASK_BLOCKED;
        }
    }
}

/// HF `update_causal_mask_with_pad_non_visible_2d` for `text_mask` placeholder tokens.
pub fn update_causal_mask_with_pad_non_visible_2d(
    input_ids: &[u32],
    attn_mask_2d: &mut [f32],
    text_mask_token_id: u32,
    causal_attn: bool,
) {
    let seq = input_ids.len();
    if seq == 0 {
        return;
    }
    let mut input_mask = vec![false; seq];
    let mut input_before_mask = vec![false; seq];
    for i in 0..seq {
        input_mask[i] = input_ids[i] == text_mask_token_id;
    }
    let tail = seq.saturating_sub(1);
    input_before_mask[..tail].copy_from_slice(&input_mask[1..tail + 1]);
    let mask_cols: Vec<bool> = input_mask
        .iter()
        .zip(input_before_mask.iter())
        .map(|(&m, &b)| m || b)
        .collect();
    let non_mask: Vec<bool> = mask_cols.iter().map(|&m| !m).collect();

    let mut prev_non_mask = vec![0usize; seq];
    let mut running = 0usize;
    for i in 0..seq {
        if non_mask[i] {
            running = i;
        }
        prev_non_mask[i] = running;
    }

    let mut next_non_mask = vec![seq; seq];
    let mut running = seq;
    for i in (0..seq).rev() {
        if non_mask[i] {
            running = i;
        }
        next_non_mask[i] = running;
    }

    for q in 0..seq {
        for k in 0..seq {
            // Match HF `mask_sdpa_utils`: `prev_non_mask` / `next_non_mask` index the key column.
            let infra = k > prev_non_mask[k] && q >= next_non_mask[k] && mask_cols[k];
            if infra {
                attn_mask_2d[q * seq + k] = MASK_BLOCKED;
            }
        }
    }

    if !causal_attn {
        for q in 0..seq {
            for k in 0..seq {
                let visible = q > prev_non_mask[k] && q < k && mask_cols[k];
                if visible {
                    attn_mask_2d[q * seq + k] = 0.0;
                }
            }
        }
    }
}

/// HF `create_block_diff_mask_by_pe` — block-causal prefix + mutual MTP blocks + prefix visibility.
pub fn block_diffusion_mask_2d(
    seq: usize,
    block_size: usize,
    x0_len: usize,
    position_ids: &[u32],
    causal_attn: bool,
) -> Vec<f32> {
    let mut m = vec![MASK_BLOCKED; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            let q_idx = q as i64;
            let kv_idx = k as i64;
            let x0 = x0_len as i64;

            let in_x0_q = q_idx < x0;
            let in_x0_kv = kv_idx < x0;
            if in_x0_q && in_x0_kv && q_idx >= kv_idx {
                m[q * seq + k] = 0.0;
                continue;
            }

            if q_idx >= x0 && kv_idx >= x0 {
                let q_blk = (q_idx - x0) / block_size as i64;
                let kv_blk = (kv_idx - x0) / block_size as i64;
                let mutual_ok = if causal_attn { q_idx >= kv_idx } else { true };
                if q_blk == kv_blk && mutual_ok {
                    m[q * seq + k] = 0.0;
                    continue;
                }
            }

            if q_idx >= x0 && kv_idx < x0 {
                let q_blk = (q_idx - x0) / block_size as i64;
                let blk_start = (x0 as usize).saturating_add(q_blk as usize * block_size);
                if blk_start < seq {
                    let prefix_len = position_ids[blk_start] as usize;
                    if kv_idx < prefix_len as i64 {
                        m[q * seq + k] = 0.0;
                    }
                }
            }
        }
    }
    m
}

/// Build MTP prefill mask for a generation window (causal prefix + bidirectional MTP block + text_mask rules).
pub fn mtp_prefill_mask_2d(
    input_ids: &[u32],
    text_mask_token_id: u32,
    block_size: usize,
    use_cache: bool,
    causal_attn: bool,
) -> Vec<f32> {
    let seq = input_ids.len();
    let mut m = causal_mask_f32(seq);
    update_causal_mask_for_one_gen_window_2d(&mut m, seq, block_size, use_cache, causal_attn);
    update_causal_mask_with_pad_non_visible_2d(input_ids, &mut m, text_mask_token_id, causal_attn);
    m
}

/// Expand `[seq, seq]` additive mask to RLX `MaskKind::Bias` layout `[batch, num_heads, seq, seq]`.
pub fn attn_bias_from_2d(batch: usize, num_heads: usize, seq: usize, mask_2d: &[f32]) -> Vec<f32> {
    attn_bias_from_query_keys(batch, num_heads, seq, seq, mask_2d)
}

/// Expand `[q_len, k_len]` mask rows (MTP query block × cached+new keys) to RLX bias layout.
pub fn attn_bias_for_incremental(
    batch: usize,
    num_heads: usize,
    past_len: usize,
    q_len: usize,
    full_mask_2d: &[f32],
    full_seq: usize,
) -> Vec<f32> {
    let k_len = past_len + q_len;
    let mut qk = vec![0f32; q_len * k_len];
    for qi in 0..q_len {
        let full_q = past_len + qi;
        for ki in 0..k_len {
            qk[qi * k_len + ki] = full_mask_2d[full_q * full_seq + ki];
        }
    }
    attn_bias_from_query_keys(batch, num_heads, q_len, k_len, &qk)
}

/// Like [`attn_bias_for_incremental`], but pads `k_len` up to the compile-time bucket `upper + q_len`.
///
/// `full_mask_2d` is `[full_seq, full_seq]` for the *true* (unpadded) window length `past_len + q_len`.
/// When `upper > past_len`, we insert masked-out padding keys in `[past_len, upper)`, and shift the
/// query block's key columns to start at `upper`.
pub fn attn_bias_for_incremental_padded(
    batch: usize,
    num_heads: usize,
    past_len: usize,
    upper: usize,
    q_len: usize,
    full_mask_2d: &[f32],
    full_seq: usize,
) -> Vec<f32> {
    let k_len = upper + q_len;
    let mut qk = vec![MASK_BLOCKED; q_len * k_len];
    for qi in 0..q_len {
        let full_q = past_len + qi;
        // Past keys: [0, past_len)
        let src_past = &full_mask_2d[full_q * full_seq..full_q * full_seq + past_len];
        qk[qi * k_len..qi * k_len + past_len].copy_from_slice(src_past);
        // Query block keys: [past_len, past_len + q_len) → [upper, upper + q_len)
        let src_q =
            &full_mask_2d[full_q * full_seq + past_len..full_q * full_seq + past_len + q_len];
        let dst = qi * k_len + upper;
        qk[dst..dst + q_len].copy_from_slice(src_q);
    }
    attn_bias_from_query_keys(batch, num_heads, q_len, k_len, &qk)
}

fn attn_bias_from_query_keys(
    batch: usize,
    num_heads: usize,
    q_len: usize,
    k_len: usize,
    mask_qk: &[f32],
) -> Vec<f32> {
    let per_head = q_len * k_len;
    let mut out = vec![0f32; batch * num_heads * per_head];
    for b in 0..batch {
        for h in 0..num_heads {
            let off = (b * num_heads + h) * per_head;
            out[off..off + per_head].copy_from_slice(mask_qk);
        }
    }
    out
}

/// Decode-step custom mask row: `1.0` = attend, `0.0` = block (RLX `MaskKind::Custom` convention).
pub fn decode_custom_mask_from_row(row_additive: &[f32]) -> Vec<f32> {
    row_additive
        .iter()
        .map(|&v| if v.is_finite() && v >= 0.0 { 1.0 } else { 0.0 })
        .collect()
}

/// Padded RLX custom mask for MTP decode (`cap_len` = compile-time `past_seq + 1`).
pub fn mtp_decode_mask_padded(block_size: usize, past_len: usize, cap_len: usize) -> Vec<f32> {
    let row = last_row_decode_mask(block_size, past_len);
    let mut m = decode_custom_mask_from_row(&row);
    if m.len() < cap_len {
        m.resize(cap_len, 0.0);
    } else if m.len() > cap_len {
        m.truncate(cap_len);
    }
    m
}

/// Last query row for decode with MTP window (additive scores; convert via [`decode_custom_mask_from_row`]).
pub fn last_row_decode_mask(block_size: usize, past: usize) -> Vec<f32> {
    let total = past + 1;
    let mut row = vec![0f32; total];
    let q = past;
    for k in 0..total {
        if k > q {
            row[k] = MASK_BLOCKED;
        }
    }
    let win_start = past.saturating_sub(block_size.saturating_sub(1));
    for k in win_start..=past {
        row[k] = 0.0;
    }
    row
}

/// Position ids for MTP window tokens: `0..prefix_len-1` on content, then repeat last prefix index on masks.
pub fn position_ids_for_window(input_ids: &[u32], text_mask_token_id: u32) -> Vec<u32> {
    let seq = input_ids.len();
    let mut pos = Vec::with_capacity(seq);
    let mut last = 0u32;
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok == text_mask_token_id {
            pos.push(last);
        } else {
            last = i as u32;
            pos.push(last);
        }
    }
    pos
}

/// Prefix length (tokens before the trailing MTP mask run) for block-diffusion masks.
pub fn x0_prefix_len(input_ids: &[u32], text_mask_token_id: u32, block_size: usize) -> usize {
    let seq = input_ids.len();
    if seq < block_size {
        return seq;
    }
    let tail_start = seq - block_size;
    if input_ids[tail_start..]
        .iter()
        .all(|&t| t == text_mask_token_id)
    {
        tail_start
    } else {
        seq
    }
}

pub fn block_size(cfg: &LocateAnythingConfig) -> usize {
    cfg.text_config.block_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_window_unblocks_tail_block() {
        let m = mtp_window_mask_f32(8, 4, false);
        assert_eq!(m[7 * 8 + 7], 0.0);
        assert!(m[7].is_infinite());
    }

    #[test]
    fn mtp_decode_mask_pads_to_cap() {
        let m = mtp_decode_mask_padded(6, 10, 16);
        assert_eq!(m.len(), 16);
        assert_eq!(m[10], 1.0);
        assert_eq!(m[15], 0.0);
    }

    #[test]
    fn text_mask_allows_query_before_mask_col() {
        let ids = vec![1u32, 2, 15, 15, 15];
        let mut m = causal_mask_f32(5);
        update_causal_mask_with_pad_non_visible_2d(&ids, &mut m, 15, false);
        assert_eq!(m[5 + 2], 0.0);
    }
}
