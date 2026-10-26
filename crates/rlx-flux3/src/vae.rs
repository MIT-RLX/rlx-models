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

//! The FLUX 3 autoencoder: the FLUX.2 VAE (32 channels, ×8) with its 2×2
//! patchify and latent batch-norm, ×16 overall. Latent tokens are
//! `[h·w, 128]` in row-major grid order, channel `c·4 + dy·2 + dx`.
//!
//! The graphs are `rlx-flux2`'s; this module maps BFL's `ae.safetensors`
//! names onto them and does the token ⇄ image reshapes.

use anyhow::{Context, Result, ensure};
use rlx_core::weight_map::WeightMap;
use rlx_flux2::latent_ops::{
    bn_normalize_patchified_latents, denorm_patchified_latents, patchify_latents,
    unpatchify_latents,
};
use rlx_flux2::vae::{
    Flux2VaeConfig, Flux2VaeWeights, build_flux2_vae_decoder_built, compile_flux2_vae_encoder_hir,
    extract_flux2_vae_weights,
};
use rlx_runtime::{CompiledGraph, Device};
use std::path::Path;

/// Pixels per latent token side.
pub const DOWNSAMPLING_RATIO: usize = 16;

/// BFL `ae.safetensors` (LDM names) or a diffusers `vae/` directory.
pub fn load_ae(path: &Path) -> Result<(Flux2VaeConfig, Flux2VaeWeights)> {
    let cfg = Flux2VaeConfig::flux2_klein();
    let (file, ldm) = if path.is_dir() {
        match ["ae.safetensors", "diffusion_pytorch_model.safetensors"]
            .iter()
            .map(|f| path.join(f))
            .find(|p| p.is_file())
        {
            Some(p) => {
                let ldm = p.file_name().is_some_and(|n| n == "ae.safetensors");
                (p, ldm)
            }
            None => anyhow::bail!("no autoencoder safetensors in {}", path.display()),
        }
    } else {
        (path.to_path_buf(), true)
    };
    let mut wm = WeightMap::from_file(file.to_str().context("non-utf8 autoencoder path")?)?;
    if ldm {
        let levels = cfg.block_out_channels.len();
        wm.remap_keys(|k| match k.as_str() {
            "encoder.quant_conv.weight" => "quant_conv.weight".into(),
            "encoder.quant_conv.bias" => "quant_conv.bias".into(),
            "decoder.post_quant_conv.weight" => "post_quant_conv.weight".into(),
            "decoder.post_quant_conv.bias" => "post_quant_conv.bias".into(),
            _ => rlx_flux1::vae::ldm_to_diffusers(&k, levels).unwrap_or(k),
        });
    }
    let weights = extract_flux2_vae_weights(wm, &cfg)?;
    Ok((cfg, weights))
}

/// Decoder for one latent grid.
pub struct AeDecoder {
    graph: CompiledGraph,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl AeDecoder {
    pub fn compile(
        cfg: &Flux2VaeConfig,
        weights: &Flux2VaeWeights,
        grid_h: usize,
        grid_w: usize,
        device: Device,
    ) -> Result<Self> {
        let built = build_flux2_vae_decoder_built(cfg, weights, 1, 2 * grid_h, 2 * grid_w)?;
        Ok(Self {
            graph: rlx_core::flow_util::compile_built(built, device)?,
            grid_h,
            grid_w,
        })
    }

    /// Tokens `[h·w, 128]` → planar RGB `[3, 16h, 16w]` in `[-1, 1]`.
    pub fn decode(
        &mut self,
        cfg: &Flux2VaeConfig,
        weights: &Flux2VaeWeights,
        tokens: &[f32],
    ) -> Result<Vec<f32>> {
        let (h, w) = (self.grid_h, self.grid_w);
        let c = cfg.bn_channels();
        ensure!(tokens.len() == h * w * c, "latent tokens {}", tokens.len());
        let planar = tokens_to_planar(tokens, h, w, c);
        let z = denorm_patchified_latents(
            &planar,
            &weights.bn_running_mean,
            &weights.bn_running_var,
            cfg.batch_norm_eps,
        );
        let z = unpatchify_latents(&z, 1, c, h, w);
        let out = self.graph.run(&[("latents", &z)]);
        out.into_iter().next().context("decoder produced no output")
    }
}

/// Encoder for one image size (multiples of 16).
pub struct AeEncoder {
    graph: CompiledGraph,
    pub height: usize,
    pub width: usize,
}

impl AeEncoder {
    pub fn compile(
        cfg: &Flux2VaeConfig,
        weights: &Flux2VaeWeights,
        height: usize,
        width: usize,
        device: Device,
    ) -> Result<Self> {
        ensure!(
            height.is_multiple_of(DOWNSAMPLING_RATIO) && width.is_multiple_of(DOWNSAMPLING_RATIO),
            "{width}x{height}: sides must be multiples of {DOWNSAMPLING_RATIO}"
        );
        let (graph, _) =
            compile_flux2_vae_encoder_hir(cfg, weights, 1, height, width, device, None)?;
        Ok(Self {
            graph,
            height,
            width,
        })
    }

    /// Interleaved RGB8 `[H, W, 3]` → tokens `[h·w, 128]` (bn-normalized).
    pub fn encode_rgb8(
        &mut self,
        cfg: &Flux2VaeConfig,
        weights: &Flux2VaeWeights,
        rgb: &[u8],
    ) -> Result<Vec<f32>> {
        let (hh, ww) = (self.height, self.width);
        ensure!(rgb.len() == 3 * hh * ww, "expected {ww}x{hh} RGB");
        let mut planar = vec![0.0f32; 3 * hh * ww];
        for i in 0..hh * ww {
            for c in 0..3 {
                planar[c * hh * ww + i] = rgb[i * 3 + c] as f32 / 127.5 - 1.0;
            }
        }
        let out = self.graph.run(&[("rgb", &planar)]);
        let mean = out
            .into_iter()
            .next()
            .context("encoder produced no output")?;
        let (h, w) = (hh / DOWNSAMPLING_RATIO, ww / DOWNSAMPLING_RATIO);
        let z = patchify_latents(&mean, 1, cfg.latent_channels, h, w);
        let z = bn_normalize_patchified_latents(
            &z,
            &weights.bn_running_mean,
            &weights.bn_running_var,
            cfg.batch_norm_eps,
        );
        Ok(planar_to_tokens(&z, h, w, cfg.bn_channels()))
    }
}

/// `[h·w, c]` → `[c, h, w]`.
pub fn tokens_to_planar(tokens: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens.len()];
    for i in 0..h * w {
        for ch in 0..c {
            out[ch * h * w + i] = tokens[i * c + ch];
        }
    }
    out
}

/// `[c, h, w]` → `[h·w, c]`.
pub fn planar_to_tokens(planar: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; planar.len()];
    for i in 0..h * w {
        for ch in 0..c {
            out[i * c + ch] = planar[ch * h * w + i];
        }
    }
    out
}

/// Planar `[3, H, W]` in `[-1, 1]` → interleaved RGB8, as the reference's
/// `to_pil` (truncation, not rounding).
pub fn rgb_to_u8(planar: &[f32], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; 3 * h * w];
    for c in 0..3 {
        for i in 0..h * w {
            let v = (planar[c * h * w + i].clamp(-1.0, 1.0) + 1.0) * 127.5;
            out[i * 3 + c] = v as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_layout_round_trips() {
        let (h, w, c) = (2, 3, 4);
        let t: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let p = tokens_to_planar(&t, h, w, c);
        // Token 1 (row 0, col 1), channel 2 sits at plane 2, pixel 1.
        assert_eq!(p[2 * h * w + 1], t[c + 2]);
        assert_eq!(planar_to_tokens(&p, h, w, c), t);
    }

    #[test]
    fn rgb_conversion_truncates_like_to_pil() {
        let px = rgb_to_u8(&[0.999, 0.999, 0.999], 1, 1);
        // (0.999 + 1) · 127.5 = 254.87 → 254 (torch `.to(uint8)` truncates).
        assert_eq!(px, [254, 254, 254]);
    }
}
