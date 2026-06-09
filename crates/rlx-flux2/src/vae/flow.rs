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

//! Native FLUX.2 VAE flows — decoder (`latents` → RGB) and encoder (`rgb` → latents).

use anyhow::Result;
use rlx_flow::{BuiltModel, MapWeights, ModelFlow};
use rlx_ir::{DType, Shape};

use super::config::Flux2VaeConfig;
use super::hir_builder::VaeHirBuilder;
use super::weights::Flux2VaeWeights;

fn decoder_output_hw(weights: &Flux2VaeWeights, h: usize, w: usize) -> (usize, usize) {
    let mut hh = h;
    let mut ww = w;
    for block in &weights.up_blocks {
        if block.upsample.is_some() {
            hh *= 2;
            ww *= 2;
        }
    }
    (hh, ww)
}

/// Tier-0 FLUX.2 VAE decoder flow.
#[derive(Clone)]
pub struct Flux2VaeDecoderFlow<'a> {
    cfg: &'a Flux2VaeConfig,
    weights: &'a Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
}

impl<'a> Flux2VaeDecoderFlow<'a> {
    pub fn new(
        cfg: &'a Flux2VaeConfig,
        weights: &'a Flux2VaeWeights,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Self {
        Self {
            cfg,
            weights,
            batch,
            h,
            w,
        }
    }

    pub fn build(self) -> Result<BuiltModel> {
        build_flux2_vae_decoder_built(self.cfg, self.weights, self.batch, self.h, self.w)
    }
}

/// Tier-0 FLUX.2 VAE encoder flow.
#[derive(Clone)]
pub struct Flux2VaeEncoderFlow<'a> {
    cfg: &'a Flux2VaeConfig,
    weights: &'a Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
}

impl<'a> Flux2VaeEncoderFlow<'a> {
    pub fn new(
        cfg: &'a Flux2VaeConfig,
        weights: &'a Flux2VaeWeights,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Self {
        Self {
            cfg,
            weights,
            batch,
            h,
            w,
        }
    }

    pub fn build(self) -> Result<BuiltModel> {
        build_flux2_vae_encoder_built(self.cfg, self.weights, self.batch, self.h, self.w)
    }
}

pub fn build_flux2_vae_decoder_built(
    cfg: &Flux2VaeConfig,
    weights: &Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let lc = cfg.latent_channels;
    let in_shape = Shape::new(&[batch, lc, h, w], f);
    let (out_h, out_w) = decoder_output_hw(weights, h, w);
    let out_shape = Shape::new(&[batch, cfg.out_channels, out_h, out_w], f);

    let cfg = cfg.clone();
    let weights = weights.clone();
    ModelFlow::new("flux2_vae_decoder")
        .input("latents", in_shape)
        .plugin_named("flux2_vae.decoder", move |emit, input| {
            let latents = input
                .ok_or_else(|| anyhow::anyhow!("VAE decoder requires latents input"))?
                .hir_id();
            let (hir, params) = emit.hir_and_params();
            let mut b = VaeHirBuilder::from_emit_parts(hir, params, &cfg, &weights, batch, h, w);
            let (out, _, _, _) = b.emit_decoder(latents)?;
            Ok(Some(emit.wrap(out, out_shape.clone())))
        })
        .output("rgb")
        .build(&mut MapWeights::default())
}

pub fn build_flux2_vae_encoder_built(
    cfg: &Flux2VaeConfig,
    weights: &Flux2VaeWeights,
    batch: usize,
    h: usize,
    w: usize,
) -> Result<BuiltModel> {
    let f = DType::F32;
    let in_c = cfg.in_channels;
    let in_shape = Shape::new(&[batch, in_c, h, w], f);
    let mean_c = weights.quant_conv.out_c / 2;
    let out_shape = Shape::new(&[batch, mean_c, h, w], f);

    let cfg = cfg.clone();
    let weights = weights.clone();
    ModelFlow::new("flux2_vae_encoder")
        .input("rgb", in_shape)
        .plugin_named("flux2_vae.encoder", move |emit, input| {
            let rgb = input
                .ok_or_else(|| anyhow::anyhow!("VAE encoder requires rgb input"))?
                .hir_id();
            let (hir, params) = emit.hir_and_params();
            let mut b = VaeHirBuilder::from_emit_parts(hir, params, &cfg, &weights, batch, h, w);
            let out = b.emit_encoder(rgb)?;
            Ok(Some(emit.wrap(out, out_shape.clone())))
        })
        .output("latents")
        .build(&mut MapWeights::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vae::{
        Flux2VaeConfig, build_flux2_vae_encoder_hir, build_flux2_vae_hir, synthetic_vae_weights,
    };

    #[test]
    fn vae_decoder_flow_matches_hir_node_count() {
        let cfg = Flux2VaeConfig::tiny();
        let w = synthetic_vae_weights(&cfg);
        let batch = 1;
        let h = 4;
        let w_px = 4;

        let ref_hir = build_flux2_vae_hir(&cfg, &w, batch, h, w_px).unwrap().hir;
        let built = Flux2VaeDecoderFlow::new(&cfg, &w, batch, h, w_px)
            .build()
            .unwrap();
        let flow_hir = built.into_hir().unwrap();

        assert_eq!(
            flow_hir.len(),
            ref_hir.len(),
            "VAE decoder flow should match hir_builder node count"
        );
    }

    #[test]
    fn vae_encoder_flow_matches_hir_node_count() {
        let cfg = Flux2VaeConfig::tiny();
        let w = synthetic_vae_weights(&cfg);
        let batch = 1;
        let h = 32;
        let w_px = 32;

        let ref_hir = build_flux2_vae_encoder_hir(&cfg, &w, batch, h, w_px)
            .unwrap()
            .hir;
        let built = Flux2VaeEncoderFlow::new(&cfg, &w, batch, h, w_px)
            .build()
            .unwrap();
        let flow_hir = built.into_hir().unwrap();

        assert_eq!(
            flow_hir.len(),
            ref_hir.len(),
            "VAE encoder flow should match hir_builder node count"
        );
    }
}
