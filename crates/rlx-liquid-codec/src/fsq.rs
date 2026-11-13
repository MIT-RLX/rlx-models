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

//! Grouped finite scalar quantization (`LiquidCodecQuantizer`), on the host.
//!
//! The latent `[frames, n_q * d]` is `n_q` groups of `d = len(levels)`
//! dimensions; each group is one token. A dimension with `L` levels is
//! bounded by `tanh` to `L` integer steps around zero and rounded; the
//! token is the mixed-radix number of the steps (`basis = cumprod(levels)`,
//! first dimension least significant). Decoding maps each digit back to
//! `(digit - L/2) / (L/2)`.

use anyhow::{Result, bail};

use crate::config::LiquidCodecConfig;

/// Tokens `[frames][n_q]` of latents `[frames, n_q * d]` (`project_in`'s
/// output). Arithmetic in f32 and round-half-to-even, as torch does.
pub fn encode(latents: &[f32], cfg: &LiquidCodecConfig) -> Vec<Vec<u32>> {
    let levels = &cfg.fsq_levels;
    let d = levels.len();
    // `_bound`: tanh(z + shift) * half_levels - offset.
    let consts: Vec<(f32, f32, f32, f32, u32)> = levels
        .iter()
        .scan(1u32, |basis, &l| {
            let half_levels = (l as f32 - 1.0) * (1.0 + 1e-3) / 2.0;
            let offset = if l.is_multiple_of(2) { 0.5f32 } else { 0.0 };
            let shift = (offset / half_levels).atanh();
            let half_width = (l / 2) as f32;
            let b = *basis;
            *basis *= l as u32;
            Some((half_levels, offset, shift, half_width, b))
        })
        .collect();
    latents
        .chunks_exact(cfg.n_q * d)
        .map(|frame| {
            frame
                .chunks_exact(d)
                .map(|group| {
                    group
                        .iter()
                        .zip(&consts)
                        .map(|(&z, &(hl, off, shift, hw, basis))| {
                            let step = ((z + shift).tanh() * hl - off).round_ties_even();
                            (step + hw) as u32 * basis
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Latents `[frames, n_q * d]` of tokens `[frames][n_q]` (`project_out`'s
/// input).
pub fn decode(codes: &[Vec<u32>], cfg: &LiquidCodecConfig) -> Result<Vec<f32>> {
    let levels = &cfg.fsq_levels;
    let size = cfg.codebook_size() as u32;
    let mut out = Vec::with_capacity(codes.len() * cfg.n_q * levels.len());
    for (t, frame) in codes.iter().enumerate() {
        if frame.len() != cfg.n_q {
            bail!(
                "frame {t}: {} codes, the codec has {}",
                frame.len(),
                cfg.n_q
            );
        }
        for &c in frame {
            if c >= size {
                bail!("frame {t}: code {c} is not below the codebook size {size}");
            }
            let mut rest = c;
            for &l in levels {
                let l = l as u32;
                let half = (l / 2) as f32;
                out.push(((rest % l) as f32 - half) / half);
                rest /= l;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_then_encode_is_identity() {
        let cfg = LiquidCodecConfig::default();
        let codes: Vec<Vec<u32>> = (0..cfg.codebook_size() as u32)
            .collect::<Vec<_>>()
            .chunks(cfg.n_q)
            .map(<[u32]>::to_vec)
            .collect();
        // Latents at the level centres, pushed through `atanh` so the bound
        // lands on them: encode must give the codes back.
        let lat = decode(&codes, &cfg).unwrap();
        let d = cfg.codebook_dim();
        let pre: Vec<f32> = lat
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let l = cfg.fsq_levels[i % d] as f32;
                let hl = (l - 1.0) * (1.0 + 1e-3) / 2.0;
                let off = if (l as u32).is_multiple_of(2) {
                    0.5
                } else {
                    0.0
                };
                let shift = (off / hl).atanh();
                ((v * (l / 2.0).floor() + off) / hl).atanh() - shift
            })
            .collect();
        assert_eq!(encode(&pre, &cfg), codes);
        assert!(decode(&[vec![2016; 12]], &cfg).is_err());
    }
}
