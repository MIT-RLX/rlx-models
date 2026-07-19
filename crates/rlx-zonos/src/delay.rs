// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! DAC delay pattern (Zyphra/Zonos `codebook_pattern.py`).

use crate::config::N_CODEBOOKS;

/// Apply delay pattern: pad last dim by `n_q`, then roll codebook `k` by `k+1`.
///
/// Layout: `[n_q][T]` → `[n_q][T + n_q]` (matches PyTorch `F.pad(..., (0, n_q))`).
pub fn apply_delay_pattern(codes: &[Vec<i64>], mask_token: i64) -> Vec<Vec<i64>> {
    assert_eq!(codes.len(), N_CODEBOOKS);
    let t = codes[0].len();
    let n_q = N_CODEBOOKS;
    let out_t = t + n_q;
    let mut out = vec![vec![mask_token; out_t]; n_q];
    for k in 0..n_q {
        // Pad: original codes in [0, T), mask in [T, T+n_q).
        for i in 0..t {
            out[k][i] = codes[k][i];
        }
        // torch.roll(..., shifts = k+1) along time — right shift with wrap.
        let shift = k + 1;
        let src = out[k].clone();
        for i in 0..out_t {
            out[k][(i + shift) % out_t] = src[i];
        }
    }
    out
}

/// Inverse: `[n_q][T_delayed]` → `[n_q][T_aligned]`.
pub fn revert_delay_pattern(codes: &[Vec<i64>]) -> Vec<Vec<i64>> {
    assert_eq!(codes.len(), N_CODEBOOKS);
    let seq_len = codes[0].len();
    let n_q = N_CODEBOOKS;
    let out_t = seq_len.saturating_sub(n_q);
    let mut out = vec![vec![0i64; out_t]; n_q];
    for k in 0..n_q {
        let start = k + 1;
        let end = seq_len - n_q + k + 1;
        out[k].copy_from_slice(&codes[k][start..end]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MASKED_TOKEN_ID;

    #[test]
    fn delay_roundtrip() {
        let t = 5;
        let codes: Vec<Vec<i64>> = (0..N_CODEBOOKS)
            .map(|q| (0..t).map(|i| (q * 100 + i) as i64).collect())
            .collect();
        let delayed = apply_delay_pattern(&codes, MASKED_TOKEN_ID);
        assert_eq!(delayed[0].len(), t + N_CODEBOOKS);
        let back = revert_delay_pattern(&delayed);
        assert_eq!(back, codes);
    }
}
