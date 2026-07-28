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

//! Llama-4 RoPE cos/sin tables `[seq, head_dim/2]` for the interleaved
//! (GptJ/complex) rotary. `inv_freq[j] = theta^(-2j/head_dim)`,
//! `cos[pos,j] = cos(pos·inv_freq[j])`.

/// Build `(cos, sin)`, each `[seq * head_dim/2]` row-major over `(pos, j)`.
pub fn build_rope_tables(head_dim: usize, theta: f32, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0 / theta.powf(2.0 * j as f32 / head_dim as f32))
        .collect();
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for pos in 0..seq {
        for j in 0..half {
            let a = pos as f32 * inv_freq[j];
            cos[pos * half + j] = a.cos();
            sin[pos * half + j] = a.sin();
        }
    }
    (cos, sin)
}

/// 2D-axial vision RoPE cos/sin `[num_patches, head_dim/2]`
/// (`Llama4VisionRotaryEmbedding`). The first `half/2` rotary pairs use the
/// patch x-coordinate, the next `half/2` use the y-coordinate; the trailing
/// class token gets zero rotation (`cos=1, sin=0`).
pub fn build_vision_rope_tables(
    image_size: usize,
    patch_size: usize,
    hidden: usize,
    heads: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let idx = image_size / patch_size;
    let np = idx * idx + 1;
    let head_dim = hidden / heads;
    let half = head_dim / 2;
    let freq_dim = head_dim / 2; // hidden/heads/2
    let nfreq = freq_dim / 2;
    let rope_freq: Vec<f32> = (0..nfreq)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / freq_dim as f32))
        .collect();

    let mut cos = vec![0.0f32; np * half];
    let mut sin = vec![0.0f32; np * half];
    for p in 0..np {
        let is_cls = p == idx * idx;
        let x = if is_cls { 0 } else { p % idx };
        let y = if is_cls { 0 } else { p / idx };
        for k in 0..half {
            let a = if is_cls {
                0.0
            } else if k < nfreq {
                (x as f32 + 1.0) * rope_freq[k]
            } else {
                (y as f32 + 1.0) * rope_freq[k - nfreq]
            };
            cos[p * half + k] = a.cos();
            sin[p * half + k] = a.sin();
        }
    }
    (cos, sin)
}
