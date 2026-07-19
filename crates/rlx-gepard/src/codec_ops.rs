// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! FSQ mixed-radix codec operations for NanoCodec / Gepard.
//!
//! # Codec geometry
//!
//! The NanoCodec operates at 21.5 fps with 8 FSQ groups and levels
//! `[8, 7, 6, 6]` per group (4 dimensions × 8 groups = 32 channels/frame).
//!
//! Each group packs its 4-dimensional FSQ code as a single integer
//! `c ∈ [0, 8×7×6×6) = [0, 2016)` using **little-endian mixed-radix**:
//!
//! ```text
//! c = d0 + 8*(d1 + 7*(d2 + 6*d3))
//! ```
//!
//! ## Unfold (packed codes → per-dimension channels)
//!
//! Given 8 packed codes `[c_0, …, c_7]` (one per group), this produces
//! 32 channel values `[d0_0, d1_0, d2_0, d3_0, d0_1, …, d3_7]`.
//!
//! ## Fold (per-dimension channels → packed codes)
//!
//! The inverse: given 32 channels, produce 8 packed codes.
//!
//! ## Dequantise
//!
//! Maps an integer channel value `v ∈ [0, L)` with the NeMo/Gepard formula
//! `(v - L//2) / (L//2)` (for even `L`, the max code lands at `(L-1-L/2)/(L/2)`,
//! not exactly `+1`).

use anyhow::{Result, anyhow};

/// FSQ levels for the NanoCodec: `[8, 7, 6, 6]` (repeated 8 times for 32 ch).
pub const FSQ_LEVELS: [u32; 4] = [8, 7, 6, 6];
/// Number of FSQ groups (codec layers).
pub const NUM_GROUPS: usize = 8;
/// Number of channels per group = len(FSQ_LEVELS).
pub const CHANNELS_PER_GROUP: usize = 4;
/// Total channels per frame = NUM_GROUPS × CHANNELS_PER_GROUP.
pub const NUM_CHANNELS: usize = NUM_GROUPS * CHANNELS_PER_GROUP;
/// Packed code capacity per group = 8×7×6×6 = 2016.
pub const PACKED_VOCAB: u32 = 2016;

/// Unfold `num_groups` packed codes into `num_groups × 4` per-dimension channels.
///
/// Input layout: `codes[group]`  
/// Output layout: `channels[group * 4 + dim]`
pub fn unfold_codes(codes: &[u32]) -> Result<Vec<u32>> {
    if codes.len() != NUM_GROUPS {
        return Err(anyhow!(
            "unfold_codes: expected {} packed codes, got {}",
            NUM_GROUPS,
            codes.len()
        ));
    }
    let mut out = vec![0u32; NUM_CHANNELS];
    for (g, &c) in codes.iter().enumerate() {
        if c >= PACKED_VOCAB {
            return Err(anyhow!(
                "unfold_codes: code {} at group {} exceeds max {}",
                c,
                g,
                PACKED_VOCAB - 1
            ));
        }
        let base = g * CHANNELS_PER_GROUP;
        // Little-endian mixed-radix decomposition
        let mut rem = c;
        for (d, &level) in FSQ_LEVELS.iter().enumerate() {
            out[base + d] = rem % level;
            rem /= level;
        }
    }
    Ok(out)
}

/// Fold 32 per-dimension channels back into 8 packed codes.
///
/// Input layout: `channels[group * 4 + dim]`  
/// Output layout: `codes[group]`
pub fn fold_channels(channels: &[u32]) -> Result<Vec<u32>> {
    if channels.len() != NUM_CHANNELS {
        return Err(anyhow!(
            "fold_channels: expected {} channels, got {}",
            NUM_CHANNELS,
            channels.len()
        ));
    }
    let mut out = vec![0u32; NUM_GROUPS];
    for g in 0..NUM_GROUPS {
        let base = g * CHANNELS_PER_GROUP;
        // Little-endian packing: c = d0 + L0*(d1 + L1*(d2 + L2*d3))
        let mut stride = 1u32;
        let mut c = 0u32;
        for (d, &level) in FSQ_LEVELS.iter().enumerate() {
            let val = channels[base + d];
            if val >= level {
                return Err(anyhow!(
                    "fold_channels: channel {} value {} exceeds level {}",
                    base + d,
                    val,
                    level
                ));
            }
            c += val * stride;
            stride *= level;
        }
        out[g] = c;
    }
    Ok(out)
}

/// Dequantise a per-dimension channel value to roughly `[-1.0, 1.0]`.
///
/// Official Gepard / NeMo formula: `(v - L//2) / (L//2)`.
#[inline]
pub fn dequantize_channel(value: u32, level: u32) -> f32 {
    let half = (level / 2).max(1) as f32;
    (value as f32 - half) / half
}

/// Dequantise 32 channels into 32 float values using the per-channel levels.
pub fn dequantize_frame(channels: &[u32]) -> Vec<f32> {
    assert_eq!(channels.len(), NUM_CHANNELS);
    channels
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let level = FSQ_LEVELS[i % CHANNELS_PER_GROUP];
            dequantize_channel(v, level)
        })
        .collect()
}

/// Quantise a float in `[-1.0, 1.0]` back to a channel integer.
#[inline]
pub fn quantize_channel(value: f32, level: u32) -> u32 {
    let max = (level - 1) as f32;
    let clamped = value.clamp(-1.0, 1.0);
    ((clamped + 1.0) / 2.0 * max).round() as u32
}

/// Number of waveform samples per codec frame at 22050 Hz and 21.5 fps.
pub fn samples_per_frame() -> usize {
    // 22050 / 21.5 ≈ 1025.6 → round to nearest integer
    (22050.0_f32 / 21.5_f32).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unfold_fold_roundtrip() {
        // Generate all codes [0..8), [8..16), … one per group
        let codes: Vec<u32> = (0..NUM_GROUPS as u32).collect();
        let channels = unfold_codes(&codes).unwrap();
        let recovered = fold_channels(&channels).unwrap();
        assert_eq!(codes, recovered, "fold(unfold(codes)) must equal codes");
    }

    #[test]
    fn unfold_fold_max_code() {
        // Use PACKED_VOCAB-1 for every group
        let codes = vec![PACKED_VOCAB - 1; NUM_GROUPS];
        let channels = unfold_codes(&codes).unwrap();
        // All dims should be at their max level-1
        for (i, &ch) in channels.iter().enumerate() {
            assert_eq!(ch, FSQ_LEVELS[i % CHANNELS_PER_GROUP] - 1);
        }
        let recovered = fold_channels(&channels).unwrap();
        assert_eq!(codes, recovered);
    }

    #[test]
    fn dequantize_extremes() {
        let level = 8u32;
        let half = (level / 2) as f32;
        let lo = dequantize_channel(0, level);
        let hi = dequantize_channel(level - 1, level);
        assert!((lo + 1.0).abs() < 1e-6, "min should be -1.0, got {lo}");
        let expected_hi = ((level - 1) as f32 - half) / half;
        assert!(
            (hi - expected_hi).abs() < 1e-6,
            "max should be {expected_hi}, got {hi}"
        );
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let level = 7u32;
        for v in 0..level {
            let f = dequantize_channel(v, level);
            let back = quantize_channel(f, level);
            assert_eq!(v, back, "roundtrip failed for level={level} v={v}");
        }
    }
}
