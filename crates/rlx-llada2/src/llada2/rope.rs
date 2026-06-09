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

// RLX — LLaDA2 RoPE (transformers `LLaDA2MoeRotaryEmbedding` + NeoX `rope_n` tables).

use crate::config::LLaDA2MoeConfig;

/// `inv_freq` length `rope_dim / 2` (transformers default RoPE init).
pub fn inv_freq(cfg: &LLaDA2MoeConfig) -> Vec<f32> {
    let dim = cfg.rope_dim();
    let theta = cfg.rope_theta as f32;
    (0..dim)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f32 / dim as f32))
        .collect()
}

/// Cos/sin tables for [`rlx_ir::Op::Rope`]: `[max_seq, head_dim / 2]`.
///
/// Matches `emb = cat(freqs, freqs)` then `.cos()` / `.sin()` in PyTorch
/// (`freqs` from `inv_freq @ position_ids`).
pub fn build_rope_tables(
    cfg: &LLaDA2MoeConfig,
    inv_freq: &[f32],
    max_seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = cfg.head_dim();
    let rope_dim = cfg.rope_dim();
    let tab_half = head_dim / 2;
    let rot_half = rope_dim / 2;
    let mut cos = vec![0f32; max_seq * tab_half];
    let mut sin = vec![0f32; max_seq * tab_half];
    for pos in 0..max_seq {
        let base = pos * tab_half;
        for j in 0..rot_half {
            let angle = pos as f32 * inv_freq[j];
            let c = angle.cos();
            let s = angle.sin();
            cos[base + j] = c;
            sin[base + j] = s;
            if rope_dim > 2 && base + rot_half + j < cos.len() {
                cos[base + rot_half + j] = c;
                sin[base + rot_half + j] = s;
            }
        }
    }
    (cos, sin)
}
