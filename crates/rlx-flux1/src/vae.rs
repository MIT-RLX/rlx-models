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

//! FLUX.1 autoencoder (16 latent channels), decoded with the rlx-flux2 VAE
//! graph.
//!
//! The FLUX.1 AE is the standard LDM `AutoencoderKL` (no quant convs, no
//! batch-norm statistics); diffusers' `AutoencoderKL` is the same network
//! under different names. BFL's `ae.safetensors` uses the LDM names
//! (`decoder.up.3.block.0`, `mid.attn_1.q`), diffusers' `vae/` folder the
//! diffusers ones — both are accepted; LDM keys are renamed (up blocks run in
//! reverse order in diffusers) before the shared loader reads them.

use anyhow::{Context, Result};
use rlx_core::weight_map::WeightMap;
use rlx_flux2::vae::{
    Flux2VaeConfig, Flux2VaeWeights, build_flux2_vae_decoder_built, extract_flux2_vae_weights,
};
use rlx_runtime::{CompiledGraph, Device};
use std::path::Path;

/// FLUX.1 latent normalization: `z_vae = z / scale + shift`.
pub const SCALING_FACTOR: f32 = 0.3611;
pub const SHIFT_FACTOR: f32 = 0.1159;

/// The FLUX.1 VAE in the rlx-flux2 config vocabulary.
pub fn flux1_vae_config() -> Flux2VaeConfig {
    Flux2VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 16,
        layers_per_block: 2,
        norm_num_groups: 32,
        block_out_channels: vec![128, 256, 512, 512],
        act_fn: "silu".into(),
        batch_norm_eps: 1e-4,
        mid_block_add_attention: true,
        use_post_quant_conv: false,
        scaling_factor: SCALING_FACTOR,
        shift_factor: SHIFT_FACTOR,
    }
}

/// LDM (BFL `ae.safetensors`) name → diffusers name; `None` when already diffusers.
pub fn ldm_to_diffusers(key: &str, num_levels: usize) -> Option<String> {
    let (side, rest) = key.split_once('.')?;
    if side != "encoder" && side != "decoder" {
        return None;
    }
    let mid = |r: &str| -> Option<String> {
        Some(match r.split_once('.')? {
            ("block_1", t) => format!("mid_block.resnets.0.{t}"),
            ("block_2", t) => format!("mid_block.resnets.1.{t}"),
            ("attn_1", t) => {
                let (n, p) = t.split_once('.')?;
                let n = match n {
                    "norm" => "group_norm",
                    "q" => "to_q",
                    "k" => "to_k",
                    "v" => "to_v",
                    "proj_out" => "to_out.0",
                    _ => return None,
                };
                format!("mid_block.attentions.0.{n}.{p}")
            }
            _ => return None,
        })
    };
    let resnet_tail = |t: &str| t.replacen("nin_shortcut", "conv_shortcut", 1);
    let out = if let Some(r) = rest.strip_prefix("mid.") {
        mid(r)?
    } else if let Some(r) = rest.strip_prefix("norm_out.") {
        format!("conv_norm_out.{r}")
    } else if let Some(r) = rest.strip_prefix("down.").filter(|_| side == "encoder") {
        let (i, t) = r.split_once('.')?;
        match t.split_once('.')? {
            ("block", t) => {
                let (j, t) = t.split_once('.')?;
                format!("down_blocks.{i}.resnets.{j}.{}", resnet_tail(t))
            }
            ("downsample", t) => format!("down_blocks.{i}.downsamplers.0.{t}"),
            _ => return None,
        }
    } else {
        let r = rest.strip_prefix("up.").filter(|_| side == "decoder")?;
        let (i, t) = r.split_once('.')?;
        let i: usize = i.parse().ok()?;
        let j = num_levels - 1 - i;
        match t.split_once('.')? {
            ("block", t) => {
                let (k, t) = t.split_once('.')?;
                format!("up_blocks.{j}.resnets.{k}.{}", resnet_tail(t))
            }
            ("upsample", t) => format!("up_blocks.{j}.upsamplers.0.{t}"),
            _ => return None,
        }
    };
    Some(format!("{side}.{out}"))
}

/// Load FLUX.1 VAE weights (LDM or diffusers names) into the rlx-flux2 layout.
pub fn load_flux1_vae(path: &Path) -> Result<(Flux2VaeConfig, Flux2VaeWeights)> {
    let (cfg, wm) = flux1_vae_weight_map(path)?;
    let weights = extract_flux2_vae_weights(wm, &cfg)?;
    Ok((cfg, weights))
}

/// The VAE's tensors under the FLUX.2 names [`load_flux1_vae`] builds from,
/// with the identity layers FLUX.1 lacks filled in (the web export records it).
pub fn flux1_vae_weight_map(path: &Path) -> Result<(Flux2VaeConfig, WeightMap)> {
    let cfg = flux1_vae_config();
    let file = if path.is_dir() {
        ["ae.safetensors", "diffusion_pytorch_model.safetensors"]
            .iter()
            .map(|f| path.join(f))
            .find(|p| p.is_file())
            .with_context(|| format!("no VAE safetensors in {}", path.display()))?
    } else {
        path.to_path_buf()
    };
    let mut wm = WeightMap::from_file(file.to_str().context("non-utf8 VAE path")?)?;
    let levels = cfg.block_out_channels.len();
    wm.remap_keys(|k| ldm_to_diffusers(&k, levels).unwrap_or(k));
    let lc2 = 2 * cfg.latent_channels;
    // FLUX.1 has no quant_conv: the encoder's moments pass straight through.
    if !wm.has("quant_conv.weight") {
        let mut eye = vec![0.0f32; lc2 * lc2];
        (0..lc2).for_each(|i| eye[i * lc2 + i] = 1.0);
        wm.insert("quant_conv.weight", eye, vec![lc2, lc2, 1, 1]);
        wm.insert("quant_conv.bias", vec![0.0; lc2], vec![lc2]);
    }
    // Nor FLUX.2's latent batch-norm: identity statistics (unused by decode).
    let bn = cfg.bn_channels();
    wm.insert("bn.running_mean", vec![0.0; bn], vec![bn]);
    wm.insert("bn.running_var", vec![1.0; bn], vec![bn]);
    Ok((cfg, wm))
}

/// A compiled VAE decoder for one latent size.
pub struct Flux1VaeDecoder {
    graph: CompiledGraph,
    pub latent_h: usize,
    pub latent_w: usize,
}

impl Flux1VaeDecoder {
    pub fn compile(
        cfg: &Flux2VaeConfig,
        weights: &Flux2VaeWeights,
        latent_h: usize,
        latent_w: usize,
        device: Device,
    ) -> Result<Self> {
        let built = build_flux2_vae_decoder_built(cfg, weights, 1, latent_h, latent_w)?;
        Ok(Self {
            graph: rlx_core::flow_util::compile_built(built, device)?,
            latent_h,
            latent_w,
        })
    }

    /// Denoiser-space latents `[16, h, w]` → RGB `[3, 8h, 8w]` in `[-1, 1]`.
    pub fn decode(&mut self, latents: &[f32]) -> Result<Vec<f32>> {
        let z: Vec<f32> = latents
            .iter()
            .map(|v| v / SCALING_FACTOR + SHIFT_FACTOR)
            .collect();
        let out = self.graph.run(&[("latents", &z)]);
        out.into_iter().next().context("VAE produced no output")
    }

    /// [`Self::decode`] where a GPU readback cannot block (the browser).
    #[cfg(target_arch = "wasm32")]
    pub async fn decode_async(&mut self, latents: &[f32]) -> Result<Vec<f32>> {
        let z: Vec<f32> = latents
            .iter()
            .map(|v| v / SCALING_FACTOR + SHIFT_FACTOR)
            .collect();
        let out = self
            .graph
            .run_async(&[("latents", z.as_slice())])
            .await
            .map_err(|e| anyhow::anyhow!("VAE: {e}"))?;
        out.into_iter().next().context("VAE produced no output")
    }
}

/// A compiled VAE encoder for one image size (mean of the posterior, no
/// sampling — what conditioning a sampler on an image needs).
pub struct Flux1VaeEncoder {
    graph: CompiledGraph,
    pub h: usize,
    pub w: usize,
}

impl Flux1VaeEncoder {
    pub fn compile(
        cfg: &Flux2VaeConfig,
        weights: &Flux2VaeWeights,
        h: usize,
        w: usize,
        device: Device,
    ) -> Result<Self> {
        let (graph, _) =
            rlx_flux2::vae::compile_flux2_vae_encoder_hir(cfg, weights, 1, h, w, device, None)?;
        Ok(Self { graph, h, w })
    }

    /// Interleaved RGB8 `[h, w, 3]` → denoiser-space latents `[16, h/8, w/8]`
    /// (`(z - shift) · scale`, the inverse of [`Flux1VaeDecoder::decode`]).
    pub fn encode_rgb8(&mut self, rgb: &[u8]) -> Result<Vec<f32>> {
        let (h, w) = (self.h, self.w);
        anyhow::ensure!(rgb.len() == 3 * h * w, "expected {}x{} RGB", w, h);
        let mut planar = vec![0.0f32; 3 * h * w];
        for i in 0..h * w {
            for c in 0..3 {
                planar[c * h * w + i] = rgb[i * 3 + c] as f32 / 127.5 - 1.0;
            }
        }
        let out = self.graph.run(&[("rgb", &planar)]);
        let z = out
            .into_iter()
            .next()
            .context("VAE encoder produced no output")?;
        Ok(z.into_iter()
            .map(|v| (v - SHIFT_FACTOR) * SCALING_FACTOR)
            .collect())
    }
}

/// Planar `[3, h, w]` in `[-1, 1]` → interleaved RGB8 (diffusers rounding).
pub fn rgb_to_u8(planar: &[f32], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; 3 * h * w];
    for c in 0..3 {
        for i in 0..h * w {
            let v = ((planar[c * h * w + i] / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round();
            out[i * 3 + c] = v as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldm_names_map_to_diffusers() {
        let m = |k: &str| ldm_to_diffusers(k, 4).unwrap();
        assert_eq!(
            m("decoder.up.3.block.0.conv1.weight"),
            "decoder.up_blocks.0.resnets.0.conv1.weight"
        );
        assert_eq!(
            m("decoder.up.1.upsample.conv.bias"),
            "decoder.up_blocks.2.upsamplers.0.conv.bias"
        );
        assert_eq!(
            m("decoder.up.0.block.0.nin_shortcut.weight"),
            "decoder.up_blocks.3.resnets.0.conv_shortcut.weight"
        );
        assert_eq!(
            m("encoder.down.1.block.0.nin_shortcut.bias"),
            "encoder.down_blocks.1.resnets.0.conv_shortcut.bias"
        );
        assert_eq!(
            m("encoder.down.0.downsample.conv.weight"),
            "encoder.down_blocks.0.downsamplers.0.conv.weight"
        );
        assert_eq!(
            m("decoder.mid.attn_1.proj_out.weight"),
            "decoder.mid_block.attentions.0.to_out.0.weight"
        );
        assert_eq!(
            m("decoder.mid.block_2.norm1.bias"),
            "decoder.mid_block.resnets.1.norm1.bias"
        );
        assert_eq!(m("encoder.norm_out.weight"), "encoder.conv_norm_out.weight");
        assert_eq!(ldm_to_diffusers("decoder.conv_in.weight", 4), None);
        assert_eq!(
            ldm_to_diffusers("decoder.up_blocks.0.resnets.0.conv1.weight", 4),
            None
        );
    }
}
