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

//! Native FLUX.2 text encoder flow — Qwen3-shaped causal trunk → joint prompt embeds.

use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use rlx_flow::{BuiltModel, MapWeights, ModelFlow, plugin_named};
use rlx_ir::{DType, HirNodeId, Shape};

use super::hir_builder::TextEncoderHirBuilder;
use super::weights::Flux2TextEncoderWeights;
use rlx_qwen3::Qwen3Config;

const ROPE_COS: &str = "flux2_te.rope_cos";
const ROPE_SIN: &str = "flux2_te.rope_sin";

/// Tier-0 FLUX.2 text encoder flow (Qwen3-shaped causal LM trunk).
#[derive(Clone)]
pub struct Flux2TextEncoderFlow<'a> {
    cfg: &'a Qwen3Config,
    weights: &'a Flux2TextEncoderWeights,
    batch: usize,
    seq: usize,
    hidden_state_layers: Vec<usize>,
}

impl<'a> Flux2TextEncoderFlow<'a> {
    pub fn new(
        cfg: &'a Qwen3Config,
        weights: &'a Flux2TextEncoderWeights,
        batch: usize,
        seq: usize,
        hidden_state_layers: &[usize],
    ) -> Self {
        Self {
            cfg,
            weights,
            batch,
            seq,
            hidden_state_layers: hidden_state_layers.to_vec(),
        }
    }

    pub fn build(self) -> Result<Flux2TextEncoderBuilt> {
        build_flux2_text_encoder_built(
            self.cfg,
            self.weights,
            self.batch,
            self.seq,
            &self.hidden_state_layers,
        )
    }
}

pub struct Flux2TextEncoderBuilt {
    pub model: BuiltModel,
    pub joint_dim: usize,
}

pub fn build_flux2_text_encoder_built(
    cfg: &Qwen3Config,
    weights: &Flux2TextEncoderWeights,
    batch: usize,
    seq: usize,
    hidden_state_layers: &[usize],
) -> Result<Flux2TextEncoderBuilt> {
    ensure!(
        cfg.num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads),
        "num_attention_heads must divide num_key_value_heads"
    );
    let joint_dim = cfg.hidden_size * hidden_state_layers.len();
    let h = cfg.hidden_size;
    let f = DType::F32;
    let hidden_shape = Shape::new(&[batch, seq, h], f);
    let out_shape = Shape::new(&[batch, seq, joint_dim], f);

    let cfg = cfg.clone();
    let weights = weights.clone();
    let hidden_state_layers = hidden_state_layers.to_vec();
    let checkpoints: Arc<Mutex<Vec<HirNodeId>>> = Arc::new(Mutex::new(Vec::new()));
    let embed_hidden_shape = hidden_shape.clone();
    let layer_hidden_shape = hidden_shape.clone();

    let mut flow = ModelFlow::new("flux2_text_encoder")
        .input("input_ids", Shape::new(&[batch, seq], f))
        .plugin_named("flux2_te.embed", {
            let cfg = cfg.clone();
            let weights = weights.clone();
            let checkpoints = checkpoints.clone();
            move |emit, _| {
                let ids = emit.flow_input("input_ids")?.hir_id();
                let (hir, params) = emit.hir_and_params();
                let mut b =
                    TextEncoderHirBuilder::from_emit_parts(hir, params, &cfg, &weights, batch, seq);
                let hidden = b.emit_embed(ids)?;
                checkpoints.lock().unwrap().push(hidden);
                Ok(Some(emit.wrap(hidden, embed_hidden_shape.clone())))
            }
        })
        .plugin_named("flux2_te.rope", {
            let cfg = cfg.clone();
            let weights = weights.clone();
            move |emit, primary| {
                let (hir, params) = emit.hir_and_params();
                let mut b =
                    TextEncoderHirBuilder::from_emit_parts(hir, params, &cfg, &weights, batch, seq);
                let (cos, sin) = b.rope_tables()?;
                emit.set_named(ROPE_COS, cos);
                emit.set_named(ROPE_SIN, sin);
                Ok(primary)
            }
        });

    for (li, layer) in weights.layers.iter().enumerate() {
        let layer = layer.clone();
        let cfg = cfg.clone();
        let weights = weights.clone();
        let checkpoints = checkpoints.clone();
        let layer_shape = layer_hidden_shape.clone();
        flow = flow.raw_stage(plugin_named(
            format!("flux2_te.layer{li}"),
            move |emit, input| {
                let hidden =
                    input.ok_or_else(|| anyhow::anyhow!("text encoder layer requires hidden"))?;
                let cos = emit.named(ROPE_COS)?;
                let sin = emit.named(ROPE_SIN)?;
                let (hir, params) = emit.hir_and_params();
                let mut b =
                    TextEncoderHirBuilder::from_emit_parts(hir, params, &cfg, &weights, batch, seq);
                let out = b.layer_forward(&layer, li, hidden.hir_id(), cos, sin)?;
                checkpoints.lock().unwrap().push(out);
                Ok(Some(emit.wrap(out, layer_shape.clone())))
            },
        ));
    }

    let built = flow
        .plugin_named("flux2_te.joint", {
            let cfg = cfg.clone();
            let weights = weights.clone();
            let checkpoints = checkpoints.clone();
            let hidden_state_layers = hidden_state_layers.clone();
            move |emit, primary| {
                let hidden = primary
                    .ok_or_else(|| anyhow::anyhow!("joint output requires hidden"))?
                    .hir_id();
                let ckpts = checkpoints.lock().unwrap().clone();
                let (hir, params) = emit.hir_and_params();
                let mut b =
                    TextEncoderHirBuilder::from_emit_parts(hir, params, &cfg, &weights, batch, seq);
                let out = b.emit_joint_output(&ckpts, &hidden_state_layers, joint_dim)?;
                let _ = hidden;
                Ok(Some(emit.wrap(out, out_shape.clone())))
            }
        })
        .output("prompt_embeds")
        .build(&mut MapWeights::default())?;

    Ok(Flux2TextEncoderBuilt {
        model: built,
        joint_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_encoder::{
        TINY_TEXT_ENCODER_LAYERS, build_flux2_text_encoder_hir, synthetic_text_encoder_weights,
        tiny_text_encoder_config,
    };

    #[test]
    fn text_encoder_flow_matches_hir_node_count() {
        let cfg = tiny_text_encoder_config();
        let w = synthetic_text_encoder_weights(&cfg);
        let batch = 1;
        let seq = 4;
        let layers = TINY_TEXT_ENCODER_LAYERS;

        let ref_hir = build_flux2_text_encoder_hir(&cfg, &w, batch, seq, layers)
            .unwrap()
            .hir;
        let built = Flux2TextEncoderFlow::new(&cfg, &w, batch, seq, layers)
            .build()
            .unwrap();
        let flow_hir = built.model.into_hir().unwrap();

        assert_eq!(
            flow_hir.len(),
            ref_hir.len(),
            "text encoder flow should match hir_builder node count (flow={}, builder={})",
            flow_hir.len(),
            ref_hir.len()
        );
    }
}
