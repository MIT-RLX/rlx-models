//! MLX grouped affine dequant (Kyutai `model.q4/q8.safetensors`).

use anyhow::{Result, ensure};

/// Dequantize MLX affine grouped weights: `w = scale * code - bias`.
pub fn dequantize_affine(
    packed: &[u32],
    packed_cols: usize,
    scales_bf16: &[u16],
    biases_bf16: &[u16],
    out_rows: usize,
    out_cols: usize,
    group_size: usize,
    bits: u32,
) -> Result<Vec<f32>> {
    ensure!(
        bits == 4 || bits == 8,
        "mlx dequant supports 4 or 8 bits, got {bits}"
    );
    let bits_us = bits as usize;
    ensure!(
        group_size > 0 && out_cols.is_multiple_of(group_size),
        "invalid group_size {group_size} for cols {out_cols}"
    );
    let codes_per_u32 = 32 / bits_us;
    let expected_packed_cols = out_cols / codes_per_u32;
    ensure!(
        packed_cols == expected_packed_cols,
        "packed cols {packed_cols} != expected {expected_packed_cols}"
    );
    let n_groups = out_cols / group_size;
    ensure!(
        scales_bf16.len() >= out_rows * n_groups,
        "scales too small: {} vs {}",
        scales_bf16.len(),
        out_rows * n_groups
    );
    ensure!(
        biases_bf16.len() >= out_rows * n_groups,
        "biases too small: {} vs {}",
        biases_bf16.len(),
        out_rows * n_groups
    );

    let mut out = vec![0f32; out_rows * out_cols];
    for row in 0..out_rows {
        for pcol in 0..packed_cols {
            let word = packed[row * packed_cols + pcol];
            for slot in 0..codes_per_u32 {
                let col = pcol * codes_per_u32 + slot;
                if col >= out_cols {
                    break;
                }
                let shift = 32 - (slot + 1) * bits_us;
                let mask = (1u32 << bits_us) - 1;
                let code = ((word >> shift) & mask) as f32;
                let g = col / group_size;
                let scale = bf16_to_f32(scales_bf16[row * n_groups + g]);
                let bias = bf16_to_f32(biases_bf16[row * n_groups + g]);
                out[row * out_cols + col] = scale * code - bias;
            }
        }
    }
    Ok(out)
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_identity_shape() {
        // 1 row, 32 cols, group 32, 4-bit → 4 u32 cols
        let packed = vec![0x7654_3210u32; 4];
        let scales = vec![f32_to_bf16(1.0); 1];
        let biases = vec![f32_to_bf16(0.0); 1];
        let out = dequantize_affine(&packed, 4, &scales, &biases, 1, 32, 32, 4).unwrap();
        assert_eq!(out.len(), 32);
    }
}
