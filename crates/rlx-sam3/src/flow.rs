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

//! Tier-0 SAM3 detector encoder/decoder flow.

use anyhow::Result;
use rlx_flow::{BuiltModel, ModelFlow, plugin_named};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;
use std::collections::HashMap;

use super::detector_decoder::Sam3DecoderWeights;
use super::detector_decoder_ir::Sam3CompiledDecoder;
use super::detector_encoder::{N_LAYERS, Sam3EncoderWeights};
use super::detector_encoder_ir::emit_sam3_detector_encoder_layer;
use rlx_core::flow_util::built_from_hir_with_profile;
use rlx_flow::CompileProfile;

const SAM3_SRC: &str = "sam3.encoder.src";
const SAM3_SRC_POS: &str = "sam3.encoder.src_pos";
const SAM3_PROMPT: &str = "sam3.encoder.prompt";
const SAM3_PROMPT_KPM: &str = "sam3.encoder.prompt_kpm_inv";

const D_MODEL: usize = 256;

#[derive(Clone)]
pub struct Sam3DetectorEncoderFlow<'a> {
    weights: &'a Sam3EncoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
    profile: CompileProfile,
}

impl<'a> Sam3DetectorEncoderFlow<'a> {
    pub fn new(weights: &'a Sam3EncoderWeights, batch: usize, hw: usize, seq: usize) -> Self {
        Self::new_with_profile(weights, batch, hw, seq, CompileProfile::sam3())
    }

    pub fn new_with_profile(
        weights: &'a Sam3EncoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        profile: CompileProfile,
    ) -> Self {
        Self {
            weights,
            batch,
            hw,
            seq,
            profile,
        }
    }

    pub fn build(self) -> Result<BuiltModel> {
        let (hir, params) =
            build_sam3_detector_encoder_model_flow(self.weights, self.batch, self.hw, self.seq)?;
        built_from_hir_with_profile(hir, params, self.profile)
    }
}

/// Native SAM3 detector encoder via [`ModelFlow`] + [`emit_sam3_detector_encoder_layer`].
pub fn build_sam3_detector_encoder_model_flow(
    weights: &Sam3EncoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
) -> Result<(HirModule, std::collections::HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let tgt_shape = Shape::new(&[batch, hw, D_MODEL], f);
    let weights_c = weights.clone();
    let bind_out = tgt_shape.clone();

    let mut flow = ModelFlow::new("sam3_detector_encoder")
        .input("src", tgt_shape.clone())
        .input("src_pos", tgt_shape.clone())
        .input("prompt", Shape::new(&[batch, seq, D_MODEL], f))
        .input("prompt_kpm_inv", Shape::new(&[batch, seq], f));

    flow = flow.plugin_named("sam3.encoder.bind", move |emit, _| {
        let src = emit.flow_input("src")?.hir_id();
        emit.set_named(SAM3_SRC, src);
        emit.set_named(SAM3_SRC_POS, emit.flow_input("src_pos")?.hir_id());
        emit.set_named(SAM3_PROMPT, emit.flow_input("prompt")?.hir_id());
        emit.set_named(SAM3_PROMPT_KPM, emit.flow_input("prompt_kpm_inv")?.hir_id());
        Ok(Some(emit.wrap(src, bind_out.clone())))
    });

    let weights_layers = weights_c.clone();
    let layer_count = weights_layers.layers.len().min(N_LAYERS);
    flow = flow.repeat_layers(layer_count, move |li| {
        let weights = weights_layers.clone();
        let out_shape = tgt_shape.clone();
        plugin_named(format!("sam3.encoder.l{li}"), move |emit, input| {
            let tgt_in = input.ok_or_else(|| anyhow::anyhow!("sam3 encoder layer requires tgt"))?;
            let src_pos = emit.named(SAM3_SRC_POS)?;
            let prompt = emit.named(SAM3_PROMPT)?;
            let prompt_kpm = emit.named(SAM3_PROMPT_KPM)?;
            let hir = emit
                .module
                .as_hir_mut()
                .expect("sam3 encoder flow requires HIR stage");
            let mut gb = HirMut::new(hir);
            let layer = weights
                .layers
                .get(li)
                .ok_or_else(|| anyhow::anyhow!("sam3 encoder layer {li} missing"))?;
            let mut typed_params = Vec::new();
            let mut gguf_w_cache = HashMap::new();
            let h = emit_sam3_detector_encoder_layer(
                &mut gb,
                emit.params,
                &mut typed_params,
                &mut gguf_w_cache,
                None,
                &weights.prefix,
                li,
                layer,
                batch,
                hw,
                seq,
                tgt_in.hir_id(),
                src_pos,
                prompt,
                prompt_kpm,
            )?;
            Ok(Some(emit.wrap(h, out_shape.clone())))
        })
    });

    flow = flow.output("tgt");

    struct Sam3EncoderParams;
    impl rlx_flow::WeightSource for Sam3EncoderParams {
        fn take(&mut self, _key: &str, _transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
            anyhow::bail!("sam3 encoder flow does not load via WeightSource")
        }
    }

    let built = flow.build(&mut Sam3EncoderParams)?;
    built.into_parts()
}

pub fn build_sam3_detector_encoder_built(
    weights: &Sam3EncoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
) -> Result<BuiltModel> {
    Sam3DetectorEncoderFlow::new(weights, batch, hw, seq).build()
}

pub fn build_sam3_detector_encoder_built_with_profile(
    weights: &Sam3EncoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
    profile: &CompileProfile,
) -> Result<BuiltModel> {
    Sam3DetectorEncoderFlow::new_with_profile(weights, batch, hw, seq, profile.clone()).build()
}

/// Compile-once SAM3 detector decoder (six per-layer graphs + host glue).
pub struct Sam3DetectorDecoderBuilt {
    pub inner: Sam3CompiledDecoder,
}

#[derive(Clone)]
pub struct Sam3DetectorDecoderFlow<'a> {
    weights: &'a Sam3DecoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
    device: Device,
    profile: CompileProfile,
}

impl<'a> Sam3DetectorDecoderFlow<'a> {
    pub fn new(
        weights: &'a Sam3DecoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        device: Device,
    ) -> Self {
        Self::new_with_profile(weights, batch, hw, seq, device, CompileProfile::sam3())
    }

    pub fn new_with_profile(
        weights: &'a Sam3DecoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        device: Device,
        profile: CompileProfile,
    ) -> Self {
        Self {
            weights,
            batch,
            hw,
            seq,
            device,
            profile,
        }
    }

    pub fn build(self) -> Result<Sam3DetectorDecoderBuilt> {
        Ok(Sam3DetectorDecoderBuilt {
            inner: Sam3CompiledDecoder::new_with_profile(
                self.weights,
                self.batch,
                self.hw,
                self.seq,
                self.device,
                &self.profile,
            )?,
        })
    }
}

pub fn build_sam3_detector_decoder_built(
    weights: &Sam3DecoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
    device: Device,
) -> Result<Sam3DetectorDecoderBuilt> {
    Sam3DetectorDecoderFlow::new(weights, batch, hw, seq, device).build()
}

pub fn build_sam3_detector_decoder_built_with_profile(
    weights: &Sam3DecoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
    device: Device,
    profile: &CompileProfile,
) -> Result<Sam3DetectorDecoderBuilt> {
    Sam3DetectorDecoderFlow::new_with_profile(weights, batch, hw, seq, device, profile.clone())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector_encoder::{N_LAYERS, Sam3EncoderLayerWeights, Sam3EncoderWeights};

    fn tiny_layer() -> Sam3EncoderLayerWeights {
        let d = D_MODEL;
        let ff = 2048usize;
        Sam3EncoderLayerWeights {
            self_attn_in_w_t: vec![0.0; d * 3 * d],
            self_attn_in_b: vec![0.0; 3 * d],
            self_attn_in_gguf_key: None,
            self_attn_out_w_t: vec![0.0; d * d],
            self_attn_out_b: vec![0.0; d],
            self_attn_out_gguf_key: None,
            cross_attn_in_w_t: vec![0.0; d * 3 * d],
            cross_attn_in_b: vec![0.0; 3 * d],
            cross_attn_in_gguf_key: None,
            cross_attn_out_w_t: vec![0.0; d * d],
            cross_attn_out_b: vec![0.0; d],
            cross_attn_out_gguf_key: None,
            linear1_w_t: vec![0.0; d * ff],
            linear1_b: vec![0.0; ff],
            linear1_gguf_key: None,
            linear2_w_t: vec![0.0; ff * d],
            linear2_b: vec![0.0; d],
            linear2_gguf_key: None,
            norm1_w: vec![1.0; d],
            norm1_b: vec![0.0; d],
            norm2_w: vec![1.0; d],
            norm2_b: vec![0.0; d],
            norm3_w: vec![1.0; d],
            norm3_b: vec![0.0; d],
        }
    }

    #[test]
    fn sam3_encoder_model_flow_matches_hir_node_count() {
        let weights = Sam3EncoderWeights {
            loaded: true,
            prefix: "transformer.encoder".to_string(),
            layers: vec![tiny_layer(); N_LAYERS],
        };
        let (hir_flow, _) = build_sam3_detector_encoder_model_flow(&weights, 1, 8, 4).unwrap();
        let parts = crate::detector_encoder_ir::build_encoder_hir(&weights, 1, 8, 4, None).unwrap();
        assert_eq!(hir_flow.len(), parts.hir.len());
        assert_eq!(hir_flow.outputs.len(), 1);
    }
}
