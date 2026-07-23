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

//! Rolling sliding-window attention (RSWA) mask helpers.
//!
//! `config.sliding_window` (128) bounds how far back each query position may
//! attend once decode has moved past the window; the causal mask stays a
//! plain lower-triangular mask until then. Graph wiring (compiled attention
//! kernels) lands with the full LM flow ([`crate::lm_flow`]).

/// Whether `key_pos` is inside the rolling window ending at `query_pos`
/// (inclusive), given `window` (0 disables the window — plain causal mask).
pub fn within_window(query_pos: usize, key_pos: usize, window: usize) -> bool {
    if key_pos > query_pos {
        return false;
    }
    if window == 0 {
        return true;
    }
    query_pos - key_pos < window
}

/// Dense `[n_queries, n_keys]` additive mask (`0.0` visible, `-inf` masked)
/// for a rolling sliding-window causal block starting at `key_offset`.
pub fn build_rswa_mask(
    n_queries: usize,
    n_keys: usize,
    key_offset: usize,
    window: usize,
) -> Vec<f32> {
    let mut mask = vec![0f32; n_queries * n_keys];
    for q in 0..n_queries {
        let query_pos = key_offset + q;
        for (k, slot) in mask[q * n_keys..(q + 1) * n_keys].iter_mut().enumerate() {
            if !within_window(query_pos, k, window) {
                *slot = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_zero_is_plain_causal() {
        assert!(within_window(10, 0, 0));
        assert!(within_window(10, 10, 0));
        assert!(!within_window(5, 6, 0));
    }

    #[test]
    fn window_limits_lookback() {
        assert!(within_window(200, 199, 128));
        assert!(within_window(200, 73, 128));
        assert!(!within_window(200, 71, 128));
    }

    #[test]
    fn mask_shape_and_diagonal_visible() {
        let mask = build_rswa_mask(2, 4, 2, 2);
        assert_eq!(mask.len(), 8);
        // query_pos = 2 (row 0): keys 0..4, window=2 -> visible keys {1,2}.
        assert_eq!(mask[0], f32::NEG_INFINITY);
        assert_eq!(mask[1], 0.0);
        assert_eq!(mask[2], 0.0);
        assert_eq!(mask[3], f32::NEG_INFINITY);
    }
}
