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

//! Rectified-flow sampling for FLUX.1 (BFL `flux/sampling.py`).

/// BFL `get_schedule`: `steps + 1` times from 1 to 0, optionally shifted by
/// `mu = lerp(base_shift, max_shift)` over image sequence length 256 → 4096
/// (`[dev]` shifts, `[schnell]` does not).
pub fn schedule(steps: usize, image_seq_len: usize, shift: bool) -> Vec<f32> {
    schedule_with(steps, image_seq_len, shift, 0.5, 1.15)
}

pub fn schedule_with(
    steps: usize,
    image_seq_len: usize,
    shift: bool,
    base_shift: f64,
    max_shift: f64,
) -> Vec<f32> {
    let steps = steps.max(1);
    let ts: Vec<f64> = (0..=steps).map(|i| 1.0 - i as f64 / steps as f64).collect();
    if !shift {
        return ts.into_iter().map(|t| t as f32).collect();
    }
    let m = (max_shift - base_shift) / (4096.0 - 256.0);
    let b = base_shift - m * 256.0;
    let mu = m * image_seq_len as f64 + b;
    ts.into_iter()
        .map(|t| {
            if t <= 0.0 {
                0.0
            } else {
                (mu.exp() / (mu.exp() + (1.0 / t - 1.0))) as f32
            }
        })
        .collect()
}

/// Latent grid for a pixel size: `2 · ceil(px / 16)` (one latent per 8 px,
/// rounded so the 2×2 packing tiles it).
pub fn latent_hw(height_px: usize, width_px: usize) -> (usize, usize) {
    (2 * height_px.div_ceil(16), 2 * width_px.div_ceil(16))
}

/// `[c, h, w]` → `[(h/2)(w/2), c·4]`, channel order `(c, ph, pw)`
/// (`rearrange "c (h ph) (w pw) -> (h w) (c ph pw)"`).
pub fn pack(latents: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (gh, gw) = (h / 2, w / 2);
    let mut out = vec![0.0f32; latents.len()];
    for y in 0..gh {
        for x in 0..gw {
            let row = (y * gw + x) * c * 4;
            for ch in 0..c {
                for ph in 0..2 {
                    for pw in 0..2 {
                        out[row + ch * 4 + ph * 2 + pw] =
                            latents[(ch * h + 2 * y + ph) * w + 2 * x + pw];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`pack`].
pub fn unpack(packed: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (gh, gw) = (h / 2, w / 2);
    let mut out = vec![0.0f32; packed.len()];
    for y in 0..gh {
        for x in 0..gw {
            let row = (y * gw + x) * c * 4;
            for ch in 0..c {
                for ph in 0..2 {
                    for pw in 0..2 {
                        out[(ch * h + 2 * y + ph) * w + 2 * x + pw] =
                            packed[row + ch * 4 + ph * 2 + pw];
                    }
                }
            }
        }
    }
    out
}

/// Seeded standard-normal noise (SplitMix64 + Box–Muller). Deterministic
/// across platforms; it does not reproduce torch's or MLX's generators — pass
/// reference noise explicitly for bit-level comparisons.
pub fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        // (0, 1] so ln never sees 0.
        let u1 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (next() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        out.push((r * th.cos()) as f32);
        out.push((r * th.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Euler integration over `timesteps`: `x += (t_next - t) · v(x, t)`.
pub fn euler<F>(mut x: Vec<f32>, timesteps: &[f32], mut velocity: F) -> anyhow::Result<Vec<f32>>
where
    F: FnMut(&[f32], f32, usize) -> anyhow::Result<Vec<f32>>,
{
    for (i, w) in timesteps.windows(2).enumerate() {
        let (t, t_next) = (w[0], w[1]);
        let v = velocity(&x, t, i)?;
        anyhow::ensure!(
            v.len() == x.len(),
            "velocity len {} != latent len {}",
            v.len(),
            x.len()
        );
        let dt = t_next - t;
        x.iter_mut().zip(&v).for_each(|(a, b)| *a += dt * b);
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrips() {
        let (c, h, w) = (3, 4, 6);
        let x: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let p = pack(&x, c, h, w);
        // Token (0,0), channel 1: pixels (0,0),(0,1),(1,0),(1,1) of channel 1.
        assert_eq!(&p[4..8], &[24.0, 25.0, 30.0, 31.0]);
        assert_eq!(unpack(&p, c, h, w), x);
    }

    #[test]
    fn schedule_endpoints_and_shift() {
        let s = schedule(4, 4096, true);
        assert_eq!(s.len(), 5);
        assert_eq!((s[0], s[4]), (1.0, 0.0));
        // mu = 1.15 at 4096 tokens: t=0.5 → e^mu / (e^mu + 1).
        let want = (1.15f64.exp() / (1.15f64.exp() + 1.0)) as f32;
        assert!((s[2] - want).abs() < 1e-6);
        assert_eq!(schedule(4, 4096, false)[2], 0.5);
    }

    #[test]
    fn noise_is_standard_normal_ish() {
        let n = gaussian_noise(100_000, 7);
        let mean = n.iter().map(|&v| v as f64).sum::<f64>() / n.len() as f64;
        let var = n.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n.len() as f64;
        assert!(
            mean.abs() < 0.02 && (var - 1.0).abs() < 0.02,
            "mean {mean} var {var}"
        );
        assert_eq!(gaussian_noise(5, 7), n[..5]);
    }
}
