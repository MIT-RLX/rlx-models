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

//! FLUX.2 img2img / edit latent conditioning (matches mflux).

use super::latent_ops::{
    bn_normalize_patchified_latents, concat_latent_ids, concat_packed_latents, pack_latents,
    patchify_latents, prepare_latent_ids, prepare_latent_ids_with_t,
};
use super::scheduler::{flow_match_init_timestep, flow_match_sigmas};
use super::vae::{Flux2VaeConfig, Flux2VaeWeights, flux2_vae_encode};
use anyhow::{Result, ensure};
use std::path::Path;

/// Reference image conditioning for edit mode.
#[derive(Debug, Clone)]
pub struct Flux2ReferenceConditioning {
    pub packed: Vec<f32>,
    pub img_ids: Vec<f32>,
    pub seq: usize,
}

/// Crop `[batch, C, H, W]` to even H/W if needed.
pub fn crop_latents_to_even(
    latents: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
) -> (Vec<f32>, usize, usize) {
    let mut out_h = h;
    let mut out_w = w;
    if !out_h.is_multiple_of(2) {
        out_h -= 1;
    }
    if !out_w.is_multiple_of(2) {
        out_w -= 1;
    }
    if out_h == h && out_w == w {
        return (latents.to_vec(), h, w);
    }
    let mut out = vec![0.0f32; batch * channels * out_h * out_w];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..out_h {
                for x in 0..out_w {
                    let src = b * channels * h * w + c * h * w + y * w + x;
                    let dst = b * channels * out_h * out_w + c * out_h * out_w + y * out_w + x;
                    out[dst] = latents[src];
                }
            }
        }
    }
    (out, out_h, out_w)
}

/// Center-crop or zero-pad spatial dims to `(target_h, target_w)`.
pub fn match_latent_spatial_size(
    latents: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    target_h: usize,
    target_w: usize,
) -> Vec<f32> {
    if h == target_h && w == target_w {
        return latents.to_vec();
    }
    let mut out = vec![0.0f32; batch * channels * target_h * target_w];
    if h >= target_h {
        let off_y = (h - target_h) / 2;
        let off_x = if w >= target_w { (w - target_w) / 2 } else { 0 };
        let use_w = target_w.min(w);
        for b in 0..batch {
            for c in 0..channels {
                for y in 0..target_h {
                    for x in 0..use_w {
                        let src = b * channels * h * w + c * h * w + (y + off_y) * w + (x + off_x);
                        let dst = b * channels * target_h * target_w
                            + c * target_h * target_w
                            + y * target_w
                            + x;
                        out[dst] = latents[src];
                    }
                }
            }
        }
    } else {
        let pad_y = (target_h - h) / 2;
        let pad_x = if w < target_w { (target_w - w) / 2 } else { 0 };
        let use_w = w.min(target_w);
        for b in 0..batch {
            for c in 0..channels {
                for y in 0..h {
                    for x in 0..use_w {
                        let src = b * channels * h * w + c * h * w + y * w + x;
                        let dst = b * channels * target_h * target_w
                            + c * target_h * target_w
                            + (y + pad_y) * target_w
                            + (x + pad_x);
                        out[dst] = latents[src];
                    }
                }
            }
        }
    }
    out
}

/// Post-process VAE-encoded latents → packed transformer input.
pub fn pack_encoded_latents(
    vae_weights: &Flux2VaeWeights,
    vae_cfg: &Flux2VaeConfig,
    encoded: Vec<f32>,
    batch: usize,
    enc_h: usize,
    enc_w: usize,
    eff_h: usize,
    eff_w: usize,
    latent_h: usize,
    latent_w: usize,
) -> Result<Vec<f32>> {
    let (cropped, ch, cw) =
        crop_latents_to_even(&encoded, batch, vae_cfg.latent_channels, enc_h, enc_w);
    let encoded = match_latent_spatial_size(
        &cropped,
        batch,
        vae_cfg.latent_channels,
        ch,
        cw,
        eff_h,
        eff_w,
    );
    let patch = patchify_latents(&encoded, batch, vae_cfg.latent_channels, latent_h, latent_w);
    let norm = bn_normalize_patchified_latents(
        &patch,
        &vae_weights.bn_running_mean,
        &vae_weights.bn_running_var,
        vae_cfg.batch_norm_eps,
    );
    Ok(pack_latents(
        &norm,
        batch,
        vae_cfg.bn_channels(),
        latent_h,
        latent_w,
    ))
}

/// VAE encode → patchify → BN → pack for transformer input.
pub fn encode_rgb_to_packed(
    vae_weights: &Flux2VaeWeights,
    vae_cfg: &Flux2VaeConfig,
    rgb: &[f32],
    batch: usize,
    pixel_h: usize,
    pixel_w: usize,
    eff_h: usize,
    eff_w: usize,
    latent_h: usize,
    latent_w: usize,
) -> Result<Vec<f32>> {
    let stride = vae_cfg.encode_spatial_stride();
    let enc_h = pixel_h / stride;
    let enc_w = pixel_w / stride;
    ensure!(
        enc_h > 0 && enc_w > 0,
        "encoded spatial dims too small for {pixel_h}x{pixel_w}"
    );
    let encoded = flux2_vae_encode(vae_weights, vae_cfg, rgb, batch, pixel_h, pixel_w)?;
    ensure!(
        encoded.len() == batch * vae_cfg.latent_channels * enc_h * enc_w,
        "encoded len {} != expected {}",
        encoded.len(),
        batch * vae_cfg.latent_channels * enc_h * enc_w
    );
    pack_encoded_latents(
        vae_weights,
        vae_cfg,
        encoded,
        batch,
        enc_h,
        enc_w,
        eff_h,
        eff_w,
        latent_h,
        latent_w,
    )
}

/// img2img init: noisy blend of encoded source + fresh noise.
pub fn prepare_img2img_latents(
    vae_weights: &Flux2VaeWeights,
    vae_cfg: &Flux2VaeConfig,
    rgb: &[f32],
    batch: usize,
    pixel_h: usize,
    pixel_w: usize,
    latent_h: usize,
    latent_w: usize,
    eff_h: usize,
    eff_w: usize,
    noise: &[f32],
    image_strength: f32,
    num_inference_steps: usize,
) -> Result<Vec<f32>> {
    let clean = encode_rgb_to_packed(
        vae_weights,
        vae_cfg,
        rgb,
        batch,
        pixel_h,
        pixel_w,
        eff_h,
        eff_w,
        latent_h,
        latent_w,
    )?;
    ensure!(clean.len() == noise.len());
    let sigmas = flow_match_sigmas(num_inference_steps);
    let init_step = flow_match_init_timestep(image_strength, num_inference_steps);
    let sigma = sigmas[init_step.min(sigmas.len() - 1)];
    Ok(super::latent_ops::blend_latents_with_noise(
        &clean, noise, sigma,
    ))
}

/// Encode one or more reference images for edit-mode concat conditioning.
pub fn prepare_reference_conditioning(
    vae_weights: &Flux2VaeWeights,
    vae_cfg: &Flux2VaeConfig,
    images: &[(&[f32], usize, usize)],
    batch: usize,
    eff_h: usize,
    eff_w: usize,
    latent_h: usize,
    latent_w: usize,
) -> Result<Flux2ReferenceConditioning> {
    ensure!(
        !images.is_empty(),
        "edit requires at least one reference image"
    );
    let channels = vae_cfg.bn_channels();
    let mut packed_acc: Option<Vec<f32>> = None;
    let mut ids_acc: Option<Vec<f32>> = None;
    let mut total_seq = 0usize;

    for (i, (rgb, ph, pw)) in images.iter().enumerate() {
        let packed = encode_rgb_to_packed(
            vae_weights,
            vae_cfg,
            rgb,
            batch,
            *ph,
            *pw,
            eff_h,
            eff_w,
            latent_h,
            latent_w,
        )?;
        let seq = packed.len() / (batch * channels);
        total_seq += seq;
        let ids = prepare_latent_ids_with_t(batch, latent_h, latent_w, 10 + 10 * i as i32);
        packed_acc = Some(match packed_acc {
            Some(prev) => concat_packed_latents(&prev, &packed, batch, channels),
            None => packed,
        });
        ids_acc = Some(match ids_acc {
            Some(prev) => concat_latent_ids(&prev, &ids, batch),
            None => ids,
        });
    }

    Ok(Flux2ReferenceConditioning {
        packed: packed_acc.unwrap(),
        img_ids: ids_acc.unwrap(),
        seq: total_seq,
    })
}

/// Gen-only ids for txt2img / img2img / edit output tokens.
pub fn prepare_generation_ids(batch: usize, latent_h: usize, latent_w: usize) -> Vec<f32> {
    prepare_latent_ids(batch, latent_h, latent_w)
}

/// Placeholder for future path-based loading in rlx-models tests.
pub fn encode_image_path_to_packed(
    _vae_weights: &Flux2VaeWeights,
    _vae_cfg: &Flux2VaeConfig,
    _path: &Path,
    _batch: usize,
    _pixel_h: usize,
    _pixel_w: usize,
    _eff_h: usize,
    _eff_w: usize,
    _latent_h: usize,
    _latent_w: usize,
) -> Result<Vec<f32>> {
    anyhow::bail!("use encode_rgb_to_packed with planar RGB from the caller")
}
