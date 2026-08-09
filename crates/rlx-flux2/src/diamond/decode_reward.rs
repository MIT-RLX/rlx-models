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

//! Reward on VAE-decoded RGB (requires loaded VAE).

use crate::runner::Flux2Runner;
use anyhow::Result;
use rlx_diamond::{LatentReward, grad_xt_via_z};

/// Latent reward with optional decode path for evaluation (gradients stay latent).
pub struct HybridLatentDecodeReward<'a, R: LatentReward + ?Sized> {
    pub inner: &'a R,
    pub runner: &'a Flux2Runner,
    pub img_ids: &'a [f32],
    pub latent_h: usize,
    pub latent_w: usize,
    pub use_decode_for_reward: bool,
}

impl<R: LatentReward + ?Sized> LatentReward for HybridLatentDecodeReward<'_, R> {
    fn reward(&self, z: &[f32]) -> f32 {
        if self.use_decode_for_reward && self.runner.has_vae() {
            decoded_blueness_reward(self.runner, z, self.img_ids, self.latent_h, self.latent_w)
                .unwrap_or_else(|_| self.inner.reward(z))
        } else {
            self.inner.reward(z)
        }
    }

    fn grad_wrt_z(&self, z: &[f32]) -> Vec<f32> {
        self.inner.grad_wrt_z(z)
    }
}

/// Mean blue channel in `[0,1]` on decoded RGB (planar u8 HWC interleaved in decode output).
pub fn decoded_blueness_reward(
    runner: &Flux2Runner,
    packed_latents: &[f32],
    img_ids: &[f32],
    latent_h: usize,
    latent_w: usize,
) -> Result<f32> {
    let (rgb, _w, _h) = runner.decode_to_rgb(packed_latents, img_ids, latent_h, latent_w)?;
    let n = rgb.len() / 3;
    if n == 0 {
        return Ok(0.0);
    }
    let blue_sum: u32 = (0..n).map(|i| rgb[i * 3 + 2] as u32).sum();
    Ok(blue_sum as f32 / (n as f32 * 255.0))
}

/// Wrap latent reward; gradients unchanged (decode not differentiated).
pub fn hybrid_reward<'a, R: LatentReward + ?Sized>(
    inner: &'a R,
    runner: &'a Flux2Runner,
    img_ids: &'a [f32],
    latent_h: usize,
    latent_w: usize,
    use_decode: bool,
) -> HybridLatentDecodeReward<'a, R> {
    HybridLatentDecodeReward {
        inner,
        runner,
        img_ids,
        latent_h,
        latent_w,
        use_decode_for_reward: use_decode,
    }
}

/// Proxy grad for hybrid rewards.
pub fn hybrid_grad_xt<R: LatentReward + ?Sized>(
    reward: &HybridLatentDecodeReward<'_, R>,
    z: &[f32],
) -> Vec<f32> {
    grad_xt_via_z(&reward.grad_wrt_z(z))
}
