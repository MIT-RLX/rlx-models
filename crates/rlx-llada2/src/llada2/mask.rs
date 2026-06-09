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

// RLX — block-diffusion attention mask (TIDE `generate` in modeling_llada2_moe.py).

/// Block-diagonal causal mask over blocks, full attention inside each block.
///
/// Returns additive mask `[batch, 1, seq, seq]` flattened (row-major `seq×seq`
/// per batch). Allowed positions are `0.0`; blocked positions are `-inf` (TIDE
/// uses `block_mask.log()` on `{0,1}` values).
pub fn block_diffusion_attention_mask(batch: usize, seq: usize, block_length: usize) -> Vec<f32> {
    assert!(block_length > 0);
    let num_blocks = seq.div_ceil(block_length);
    let mut one_block = vec![f32::NEG_INFINITY; seq * seq];
    for bi in 0..num_blocks {
        for bj in 0..=bi {
            let r0 = bi * block_length;
            let c0 = bj * block_length;
            let r1 = (r0 + block_length).min(seq);
            let c1 = (c0 + block_length).min(seq);
            for r in r0..r1 {
                for c in c0..c1 {
                    one_block[r * seq + c] = 0.0;
                }
            }
        }
    }
    let mut out = Vec::with_capacity(batch * seq * seq);
    for _ in 0..batch {
        out.extend_from_slice(&one_block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_mask_is_lower_block_triangular() {
        let m = block_diffusion_attention_mask(1, 8, 4);
        assert_eq!(m[0], 0.0);
        assert_eq!(m[3 * 8 + 3], 0.0);
        assert!(m[4].is_infinite() && m[4] < 0.0);
        assert_eq!(m[4 * 8 + 4], 0.0);
        assert_eq!(m[7 * 8], 0.0);
    }
}
