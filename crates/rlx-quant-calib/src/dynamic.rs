// RLX models — calibration quantization.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Dynamic quantization — assign a per-layer bit-width by sensitivity to hit a
//! target average bits-per-weight (mlx-lm's `dynamic_quant`). No training: a
//! sensitivity score per layer (e.g. its round-to-nearest reconstruction error
//! or activation energy) drives a greedy bit budget.

/// Estimate a layer's quant sensitivity as the RTN reconstruction error at
/// `bits` — larger ⇒ keep more precision.
pub fn rtn_sensitivity(w: &[f32], out: usize, inn: usize, bits: u32, group_size: usize) -> f32 {
    let q = crate::quant::quantize_rtn(w, out, inn, bits, group_size);
    crate::quant::mse(w, &crate::quant::dequantize(&q))
}

/// Allocate a bit-width from `bit_choices` to each layer so the
/// parameter-weighted average bits ≈ `target_avg_bits`, upgrading the most
/// sensitive layers first. Every layer starts at the lowest choice.
pub fn dynamic_bit_allocation(
    sensitivity: &[f32],
    sizes: &[usize],
    bit_choices: &[u32],
    target_avg_bits: f32,
) -> Vec<u32> {
    let n = sensitivity.len();
    assert_eq!(n, sizes.len(), "sensitivity / sizes length mismatch");
    let mut choices = bit_choices.to_vec();
    choices.sort_unstable();
    choices.dedup();
    let lo = *choices.first().expect("at least one bit choice");

    let total: f32 = sizes.iter().map(|&s| s as f32).sum::<f32>().max(1.0);
    let avg = |b: &[u32]| -> f32 {
        b.iter()
            .zip(sizes)
            .map(|(&bi, &sz)| bi as f32 * sz as f32)
            .sum::<f32>()
            / total
    };

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sensitivity[b]
            .partial_cmp(&sensitivity[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut bits = vec![lo; n];
    loop {
        if avg(&bits) >= target_avg_bits {
            break;
        }
        let mut changed = false;
        for &l in &order {
            if let Some(&next) = choices.iter().find(|&&c| c > bits[l]) {
                bits[l] = next;
                changed = true;
                if avg(&bits) >= target_avg_bits {
                    return bits;
                }
            }
        }
        if !changed {
            break; // every layer at the max choice
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_gives_sensitive_layers_more_bits() {
        let sens = vec![10.0, 0.1, 8.0, 0.2];
        let sizes = vec![100usize; 4];
        let bits = dynamic_bit_allocation(&sens, &sizes, &[2, 4, 8], 5.0);
        // The most sensitive layer ends up with strictly more bits than the
        // least sensitive one.
        assert!(bits[0] > bits[3], "{bits:?}");
        assert!(bits[0] >= bits[1] && bits[2] >= bits[3], "{bits:?}");
        let avg: f32 = bits.iter().map(|&b| b as f32).sum::<f32>() / 4.0;
        assert!(avg >= 5.0 - 1e-6, "avg {avg} should reach the target");
    }

    #[test]
    fn rtn_sensitivity_tracks_magnitude() {
        // A wider-range weight has larger absolute RTN error.
        let small: Vec<f32> = (0..32).map(|i| (i as f32) * 0.001).collect();
        let large: Vec<f32> = (0..32).map(|i| (i as f32) * 1.0).collect();
        assert!(rtn_sensitivity(&large, 4, 8, 3, 8) > rtn_sensitivity(&small, 4, 8, 3, 8));
    }
}
