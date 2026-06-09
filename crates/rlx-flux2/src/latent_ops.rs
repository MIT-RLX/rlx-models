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

//! FLUX.2 latent pack/unpack, patchify, and BN denorm (matches diffusers / mflux).

use anyhow::{Result, ensure};

/// Pixel `(height, width)` → `(latent_h, latent_w, eff_h, eff_w)` (mflux FLUX.2 geometry).
pub fn flux2_latent_geometry(height: usize, width: usize) -> (usize, usize, usize, usize) {
    let eff_h = 2 * (height / 16);
    let eff_w = 2 * (width / 16);
    (eff_h / 2, eff_w / 2, eff_h, eff_w)
}

/// FLUX.2 latent position ids `[batch, h*w, 4]` with `(t=0, h, w, l=0)`.
pub fn prepare_latent_ids(batch: usize, latent_h: usize, latent_w: usize) -> Vec<f32> {
    prepare_latent_ids_with_t(batch, latent_h, latent_w, 0)
}

/// Position ids with a custom time coordinate (reference images in edit mode use `10 + 10*i`).
pub fn prepare_latent_ids_with_t(
    batch: usize,
    latent_h: usize,
    latent_w: usize,
    t_coord: i32,
) -> Vec<f32> {
    let seq = latent_h * latent_w;
    let t_val = t_coord as f32;
    let mut ids = vec![0.0f32; batch * seq * 4];
    for b in 0..batch {
        for h in 0..latent_h {
            for w in 0..latent_w {
                let t = b * seq + h * latent_w + w;
                ids[t * 4] = t_val;
                ids[t * 4 + 1] = h as f32;
                ids[t * 4 + 2] = w as f32;
                ids[t * 4 + 3] = 0.0;
            }
        }
    }
    ids
}

/// `[batch, seq, channels]` → `[batch, channels, h, w]` using position ids (axis 1=h, 2=w).
pub fn unpack_latents_with_ids(
    packed: &[f32],
    img_ids: &[f32],
    batch: usize,
    seq: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>> {
    ensure!(packed.len() == batch * seq * channels);
    ensure!(img_ids.len() >= batch * seq * 4);
    let spatial = height * width;
    let mut out = vec![0.0f32; batch * channels * spatial];
    for b in 0..batch {
        for t in 0..seq {
            let h = img_ids[(b * seq + t) * 4 + 1] as usize;
            let w = img_ids[(b * seq + t) * 4 + 2] as usize;
            if h >= height || w >= width {
                continue;
            }
            let src_base = (b * seq + t) * channels;
            for c in 0..channels {
                let dst_idx = b * channels * spatial + c * spatial + h * width + w;
                out[dst_idx] = packed[src_base + c];
            }
        }
    }
    Ok(out)
}

/// `[batch, C, 2h, 2w]` → `[batch, C*4, h, w]` (inverse of [`unpatchify_latents`]).
pub fn patchify_latents(
    latents: &[f32],
    batch: usize,
    base_c: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let packed_c = base_c * 4;
    let h2 = h * 2;
    let w2 = w * 2;
    let mut out = vec![0.0f32; batch * packed_c * h * w];
    for b in 0..batch {
        for c in 0..base_c {
            for y in 0..h {
                for x in 0..w {
                    for py in 0..2 {
                        for px in 0..2 {
                            let pc = c * 4 + py * 2 + px;
                            let src_y = y * 2 + py;
                            let src_x = x * 2 + px;
                            let src = b * base_c * h2 * w2 + c * h2 * w2 + src_y * w2 + src_x;
                            let dst = b * packed_c * h * w + pc * h * w + y * w + x;
                            out[dst] = latents[src];
                        }
                    }
                }
            }
        }
    }
    out
}

/// BN normalize patchified latents: `(x - mean) / std` (encode path).
pub fn bn_normalize_patchified_latents(
    latents: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> Vec<f32> {
    let n = latents.len();
    let ch = running_mean.len();
    assert!(n.is_multiple_of(ch), "latent len must divide BN channels");
    let spatial = n / ch;
    let mut out = vec![0.0f32; n];
    for c in 0..ch {
        let std = (running_var[c] + eps).sqrt();
        let mean = running_mean[c];
        for i in 0..spatial {
            let idx = c * spatial + i;
            out[idx] = (latents[idx] - mean) / std;
        }
    }
    out
}

/// `[batch, C*4, h, w]` → `[batch, C, 2h, 2w]`.
pub fn unpatchify_latents(
    latents: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let packed_c = channels;
    let base_c = packed_c / 4;
    let mut out = vec![0.0f32; batch * base_c * (h * 2) * (w * 2)];
    for b in 0..batch {
        for c in 0..base_c {
            for y in 0..h {
                for x in 0..w {
                    for py in 0..2 {
                        for px in 0..2 {
                            let pc = c * 4 + py * 2 + px;
                            let src = b * packed_c * h * w + pc * h * w + y * w + x;
                            let dst_y = y * 2 + py;
                            let dst_x = x * 2 + px;
                            let dst = b * base_c * (h * 2) * (w * 2)
                                + c * (h * 2) * (w * 2)
                                + dst_y * (w * 2)
                                + dst_x;
                            out[dst] = latents[src];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of training-time BN on patchified latents: `x * std + mean`.
pub fn denorm_patchified_latents(
    latents: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> Vec<f32> {
    let n = latents.len();
    let ch = running_mean.len();
    assert!(n.is_multiple_of(ch), "latent len must divide BN channels");
    let spatial = n / ch;
    let mut out = vec![0.0f32; n];
    for c in 0..ch {
        let std = (running_var[c] + eps).sqrt();
        let mean = running_mean[c];
        for i in 0..spatial {
            let idx = c * spatial + i;
            out[idx] = latents[idx] * std + mean;
        }
    }
    out
}

/// Pack `[batch, C, H, W]` → `[batch, H*W, C]`.
pub fn pack_latents(
    latents: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let seq = h * w;
    let mut out = vec![0.0f32; batch * seq * channels];
    for b in 0..batch {
        for t in 0..seq {
            let y = t / w;
            let x = t % w;
            let dst = (b * seq + t) * channels;
            for c in 0..channels {
                out[dst + c] = latents[b * channels * seq + c * seq + y * w + x];
            }
        }
    }
    out
}

/// `(1 - sigma) * clean + sigma * noise` for img2img init.
pub fn blend_latents_with_noise(clean: &[f32], noise: &[f32], sigma: f32) -> Vec<f32> {
    clean
        .iter()
        .zip(noise)
        .map(|(&c, &n)| (1.0 - sigma) * c + sigma * n)
        .collect()
}

/// Concatenate packed latent tokens along the sequence axis.
pub fn concat_packed_latents(a: &[f32], b: &[f32], batch: usize, channels: usize) -> Vec<f32> {
    let seq_a = a.len() / (batch * channels);
    let seq_b = b.len() / (batch * channels);
    let mut out = vec![0.0f32; batch * (seq_a + seq_b) * channels];
    for bi in 0..batch {
        let dst = bi * (seq_a + seq_b) * channels;
        out[dst..dst + seq_a * channels]
            .copy_from_slice(&a[bi * seq_a * channels..(bi + 1) * seq_a * channels]);
        out[dst + seq_a * channels..dst + (seq_a + seq_b) * channels]
            .copy_from_slice(&b[bi * seq_b * channels..(bi + 1) * seq_b * channels]);
    }
    out
}

/// Concatenate `[batch, seq, 4]` id tensors.
pub fn concat_latent_ids(a: &[f32], b: &[f32], batch: usize) -> Vec<f32> {
    let seq_a = a.len() / (batch * 4);
    let seq_b = b.len() / (batch * 4);
    let mut out = vec![0.0f32; batch * (seq_a + seq_b) * 4];
    for bi in 0..batch {
        let dst = bi * (seq_a + seq_b) * 4;
        out[dst..dst + seq_a * 4].copy_from_slice(&a[bi * seq_a * 4..(bi + 1) * seq_a * 4]);
        out[dst + seq_a * 4..dst + (seq_a + seq_b) * 4]
            .copy_from_slice(&b[bi * seq_b * 4..(bi + 1) * seq_b * 4]);
    }
    out
}

/// Keep only the first `gen_seq` tokens from a noise prediction.
pub fn slice_gen_noise(noise: &[f32], batch: usize, channels: usize, gen_seq: usize) -> Vec<f32> {
    let total_seq = noise.len() / (batch * channels);
    assert!(gen_seq <= total_seq);
    let mut out = vec![0.0f32; batch * gen_seq * channels];
    for bi in 0..batch {
        let src = bi * total_seq * channels;
        let dst = bi * gen_seq * channels;
        out[dst..dst + gen_seq * channels].copy_from_slice(&noise[src..src + gen_seq * channels]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_unpatchify_roundtrip() {
        let b = 1usize;
        let base_c = 8usize;
        let h = 2usize;
        let w = 2usize;
        let h2 = h * 2;
        let w2 = w * 2;
        let orig: Vec<f32> = (0..b * base_c * h2 * w2).map(|i| i as f32 * 0.01).collect();
        let packed = patchify_latents(&orig, b, base_c, h, w);
        let back = unpatchify_latents(&packed, b, base_c * 4, h, w);
        assert_eq!(back.len(), orig.len());
        for (a, o) in back.iter().zip(&orig) {
            assert!((a - o).abs() < 1e-5, "mismatch {a} vs {o}");
        }
    }

    #[test]
    fn bn_norm_denorm_roundtrip() {
        let mean = vec![0.1f32, -0.2, 0.3, 0.0];
        let var = vec![1.0f32, 4.0, 0.25, 1.0];
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let norm = bn_normalize_patchified_latents(&x, &mean, &var, 1e-6);
        let back = denorm_patchified_latents(&norm, &mean, &var, 1e-6);
        for (a, o) in back.iter().zip(&x) {
            assert!((a - o).abs() < 1e-4);
        }
    }

    #[test]
    fn unpatchify_roundtrip_dims() {
        let b = 1usize;
        let h = 2usize;
        let w = 2usize;
        let base_c = 8usize;
        let packed_c = base_c * 4;
        let latents: Vec<f32> = (0..b * packed_c * h * w).map(|i| i as f32).collect();
        let out = unpatchify_latents(&latents, b, packed_c, h, w);
        assert_eq!(out.len(), b * base_c * (h * 2) * (w * 2));
    }
}
