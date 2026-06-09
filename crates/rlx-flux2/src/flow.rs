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

//! Fluent FLUX.2 assembly — dual-stream blocks via `rlx-flow` streams + plugins.

use std::fmt;
use std::sync::Arc;

use anyhow::Result;
use rlx_flow::stream::id as stream_id;
use rlx_flow::{BuiltModel, CompileProfile, MapWeights, ModelFlow};
use rlx_ir::{DType, Shape};

use super::config::Flux2Config;
use super::hir_builder::{Flux2DoubleMod, Flux2HirBuilder, Flux2TypedParams};
use super::packed::Flux2PackedParams;
use super::typed_linear::TypedLinearStore;
use super::weights::Flux2Weights;

/// Named handles for FLUX dual-stream conditioning (stored in [`rlx_flow::FlowState::named`]).
const MOD_IMG_KEY: &str = "flux2.mod_img";
const MOD_TXT_KEY: &str = "flux2.mod_txt";
const ROPE_COS_KEY: &str = "flux2.rope_cos";
const ROPE_SIN_KEY: &str = "flux2.rope_sin";

/// Tier-0 FLUX.2 dual-stream flow builder.
#[derive(Clone)]
pub struct Flux2Flow<'a> {
    cfg: &'a Flux2Config,
    weights: &'a Flux2Weights,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: Arc<Vec<f32>>,
    txt_ids: Arc<Vec<f32>>,
    profile: Option<CompileProfile>,
}

impl fmt::Debug for Flux2Flow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Flux2Flow")
            .field("batch", &self.batch)
            .field("img_seq", &self.img_seq)
            .field("txt_seq", &self.txt_seq)
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

impl<'a> Flux2Flow<'a> {
    pub fn new(cfg: &'a Flux2Config, weights: &'a Flux2Weights) -> Self {
        Self {
            cfg,
            weights,
            batch: 1,
            img_seq: 64,
            txt_seq: 128,
            img_ids: Arc::new(Vec::new()),
            txt_ids: Arc::new(Vec::new()),
            profile: None,
        }
    }

    pub fn batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    pub fn img_seq(mut self, seq: usize) -> Self {
        self.img_seq = seq;
        self
    }

    pub fn txt_seq(mut self, seq: usize) -> Self {
        self.txt_seq = seq;
        self
    }

    pub fn position_ids(mut self, img_ids: Vec<f32>, txt_ids: Vec<f32>) -> Self {
        self.img_ids = Arc::new(img_ids);
        self.txt_ids = Arc::new(txt_ids);
        self
    }

    pub fn profile(mut self, profile: CompileProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Build img/txt embed inputs + dual-stream transformer blocks (no single-stream tail).
    pub fn build_dual_blocks(self) -> Result<BuiltModel> {
        flux2_dual_flow(
            "flux2_dual",
            self.cfg,
            self.weights,
            self.batch,
            self.img_seq,
            self.txt_seq,
            self.img_ids,
            self.txt_ids,
            self.profile.unwrap_or_default(),
        )
        .load_stream(stream_id::IMG)
        .output("hidden")
        .build(&mut MapWeights::default())
    }

    /// Full denoiser forward: native dual-stream blocks + single-stream tail.
    pub fn build_forward(self, img_ids: &[f32], txt_ids: &[f32]) -> Result<Flux2ForwardBuilt> {
        let cfg = self.cfg.clone();
        let batch = self.batch;
        let img_seq = self.img_seq;
        let txt_seq = self.txt_seq;
        let out_shape = Shape::new(&[batch, img_seq, cfg.proj_out_dim()], DType::F32);
        let built = flux2_dual_flow(
            "flux2_forward",
            self.cfg,
            self.weights,
            batch,
            img_seq,
            txt_seq,
            Arc::new(img_ids.to_vec()),
            Arc::new(txt_ids.to_vec()),
            self.profile.unwrap_or_default(),
        )
        .plugin_named("flux2.single_tail", {
            let cfg = cfg.clone();
            let weights = self.weights.clone();
            move |emit, _| {
                let img = emit
                    .state
                    .streams
                    .get(stream_id::IMG)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing img stream after dual blocks"))?;
                let txt = emit
                    .state
                    .streams
                    .get(stream_id::TXT)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing txt stream after dual blocks"))?;
                let cos = emit.named(ROPE_COS_KEY)?;
                let sin = emit.named(ROPE_SIN_KEY)?;
                let temb = emit.flow_input("temb")?.hir_id();
                let mut typed = Flux2TypedParams::new();
                let out = {
                    let (hir, params) = emit.hir_and_params();
                    let mut b = Flux2HirBuilder::from_emit_parts(
                        hir, params, &mut typed, &cfg, &weights, batch, img_seq, txt_seq,
                    );
                    b.emit_single_stream_tail(img.hir_id(), txt.hir_id(), cos, sin, temb)?
                };
                Ok(Some(emit.wrap(out, out_shape.clone())))
            }
        })
        .output("hidden")
        .build(&mut MapWeights::default())?;

        Ok(Flux2ForwardBuilt {
            graph_params: built.params.clone(),
            typed_params: Flux2TypedParams::new(),
            model: built,
        })
    }

    /// Compile-minimal path: x_embedder → proj_out.
    pub fn build_minimal(self) -> Result<BuiltModel> {
        build_flux2_minimal_built(self.cfg, self.weights, self.batch, self.img_seq)
    }
}

/// Compile-minimal FLUX.2 flow: `hidden` → x_embedder → proj_out.
pub fn build_flux2_minimal_built(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
) -> Result<BuiltModel> {
    let cfg = cfg.clone();
    let x_embedder = weights.x_embedder.clone();
    let proj_out = weights.proj_out.clone();
    let in_ch = cfg.in_channels;
    let out_dim = cfg.proj_out_dim();
    let f = DType::F32;
    let hidden_shape = Shape::new(&[batch, img_seq, in_ch], f);
    let embed_shape = Shape::new(&[batch, img_seq, x_embedder.out_dim], f);
    let out_shape = Shape::new(&[batch, img_seq, out_dim], f);

    ModelFlow::new("flux2_minimal")
        .input("hidden", hidden_shape.clone())
        .plugin_named("flux2_minimal.embed", {
            let x_embedder = x_embedder.clone();
            let embed_shape = embed_shape.clone();
            move |emit, _| {
                let hidden = emit.flow_input("hidden")?.hir_id();
                let hir = emit
                    .module
                    .as_hir_mut()
                    .expect("flux2 minimal flow requires HIR stage");
                let embedded = super::builder::linear_hir(
                    hir,
                    emit.params,
                    hidden,
                    &x_embedder,
                    "x_embedder",
                    embed_shape.clone(),
                )?;
                Ok(Some(emit.wrap(embedded, embed_shape.clone())))
            }
        })
        .plugin_named("flux2_minimal.proj", {
            let proj_out = proj_out.clone();
            let out_shape = out_shape.clone();
            move |emit, primary| {
                let embedded = primary
                    .ok_or_else(|| anyhow::anyhow!("flux2 minimal proj requires embed output"))?
                    .hir_id();
                let hir = emit
                    .module
                    .as_hir_mut()
                    .expect("flux2 minimal flow requires HIR stage");
                let out = super::builder::linear_hir(
                    hir,
                    emit.params,
                    embedded,
                    &proj_out,
                    "proj_out",
                    out_shape.clone(),
                )?;
                Ok(Some(emit.wrap(out, out_shape.clone())))
            }
        })
        .output("output")
        .build(&mut MapWeights::default())
}

/// Full forward build product (includes non-f32 typed param blobs).
pub struct Flux2ForwardBuilt {
    pub model: BuiltModel,
    pub typed_params: Flux2TypedParams,
    pub graph_params: crate::builder::Flux2GraphParams,
}

/// Compile denoiser via tier-0 [`Flux2Flow`] wrapper (same numerics as [`super::hir_builder::compile_flux2_forward`]).
pub fn compile_flux2_forward_via_flow(
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: &[f32],
    txt_ids: &[f32],
    device: rlx_runtime::Device,
    packed: Option<&Flux2PackedParams>,
    typed_linears: Option<&TypedLinearStore>,
    aot: Option<&rlx_runtime::AotCache>,
) -> Result<(rlx_runtime::CompiledGraph, crate::builder::Flux2GraphParams)> {
    use crate::compile_util::{compile_hir_cached, flux2_denoiser_aot_key};

    super::device::assert_flux2_device_available(device)?;
    let Flux2ForwardBuilt {
        model,
        typed_params,
        graph_params,
    } = Flux2Flow::new(cfg, weights)
        .batch(batch)
        .img_seq(img_seq)
        .txt_seq(txt_seq)
        .position_ids(img_ids.to_vec(), txt_ids.to_vec())
        .build_forward(img_ids, txt_ids)?;

    let key = format!(
        "{}_flow",
        flux2_denoiser_aot_key(
            device,
            batch,
            img_seq,
            txt_seq,
            img_ids,
            txt_ids,
            packed.is_some()
        )
    );
    let hir = model
        .into_hir()
        .ok_or_else(|| anyhow::anyhow!("Flux2Flow build did not produce HIR"))?;
    let profile = CompileProfile::flux2();
    let mut compiled = compile_hir_cached(device, aot, &key, hir, &profile)?;
    for (name, data) in &graph_params {
        compiled.set_param(name, data);
    }
    for (name, data, dtype) in &typed_params {
        compiled.set_param_typed(name, data, *dtype);
    }
    let _ = (packed, typed_linears);
    Ok((compiled, graph_params))
}

/// Tier-0 CFG combine: `neg + scale * (pos - neg)`.
#[derive(Debug, Clone, Copy)]
pub struct Flux2CfgCombineFlow {
    pub batch: usize,
    pub seq: usize,
    pub channels: usize,
}

impl Flux2CfgCombineFlow {
    pub fn new(batch: usize, seq: usize, channels: usize) -> Self {
        Self {
            batch,
            seq,
            channels,
        }
    }

    pub fn build(self) -> Result<BuiltModel> {
        super::cfg::build_flux2_cfg_combine_built(self.batch, self.seq, self.channels)
    }
}

fn flux2_dual_flow(
    name: &str,
    cfg: &Flux2Config,
    weights: &Flux2Weights,
    batch: usize,
    img_seq: usize,
    txt_seq: usize,
    img_ids: Arc<Vec<f32>>,
    txt_ids: Arc<Vec<f32>>,
    profile: CompileProfile,
) -> ModelFlow {
    let cfg = cfg.clone();
    let weights = weights.clone();
    let dim = cfg.inner_dim();
    let f = DType::F32;
    let img_shape = Shape::new(&[batch, img_seq, cfg.in_channels], f);
    let txt_shape = Shape::new(&[batch, txt_seq, cfg.joint_attention_dim], f);
    let temb_shape = Shape::new(&[batch, dim], f);

    let mut flow = ModelFlow::new(name)
        .with_profile(profile)
        .input("hidden", img_shape.clone())
        .input("encoder", txt_shape.clone())
        .input("temb", temb_shape)
        .bind_inputs_to_streams([("hidden", stream_id::IMG), ("encoder", stream_id::TXT)])
        .plugin_named("flux2.embed", {
            let cfg = cfg.clone();
            let weights = weights.clone();
            move |emit, _| {
                let img = emit
                    .state
                    .streams
                    .get(stream_id::IMG)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing img stream"))?;
                let txt = emit
                    .state
                    .streams
                    .get(stream_id::TXT)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing txt stream"))?;
                let mut typed = Flux2TypedParams::new();
                let (hir, params) = emit.hir_and_params();
                let mut b = Flux2HirBuilder::from_emit_parts(
                    hir, params, &mut typed, &cfg, &weights, batch, img_seq, txt_seq,
                );
                let img_e = b.linear(
                    img.hir_id(),
                    &weights.x_embedder,
                    "x_embedder",
                    Shape::new(&[batch, img_seq, dim], f),
                )?;
                let txt_e = b.linear(
                    txt.hir_id(),
                    &weights.context_embedder,
                    "context_embedder",
                    Shape::new(&[batch, txt_seq, dim], f),
                )?;
                let img_out = emit.wrap(img_e, Shape::new(&[batch, img_seq, dim], f));
                let txt_out = emit.wrap(txt_e, Shape::new(&[batch, txt_seq, dim], f));
                emit.state
                    .streams
                    .insert(stream_id::IMG.into(), img_out.clone());
                emit.state.streams.insert(stream_id::TXT.into(), txt_out);
                Ok(Some(img_out))
            }
        })
        .plugin_named("flux2.cond", {
            let cfg = cfg.clone();
            let weights = weights.clone();
            let img_ids = img_ids.clone();
            let txt_ids = txt_ids.clone();
            move |emit, primary| {
                let temb = emit.flow_input("temb")?.hir_id();
                let mut typed = Flux2TypedParams::new();
                let (mod_img, mod_txt, cos, sin) = {
                    let (hir, params) = emit.hir_and_params();
                    let mut b = Flux2HirBuilder::from_emit_parts(
                        hir, params, &mut typed, &cfg, &weights, batch, img_seq, txt_seq,
                    );
                    let mod_img = b.modulation_params(&weights.double_mod_img, "mod_img", temb)?;
                    let mod_txt = b.modulation_params(&weights.double_mod_txt, "mod_txt", temb)?;
                    let (cos, sin) = b.rope_params(&img_ids, &txt_ids)?;
                    (mod_img, mod_txt, cos, sin)
                };
                store_double_mod(emit, MOD_IMG_KEY, &mod_img);
                store_double_mod(emit, MOD_TXT_KEY, &mod_txt);
                emit.set_named(ROPE_COS_KEY, cos);
                emit.set_named(ROPE_SIN_KEY, sin);
                Ok(primary)
            }
        });

    let block_count = weights.transformer_blocks.len();
    for li in 0..block_count {
        let block = weights.transformer_blocks[li].clone();
        let cfg = cfg.clone();
        let weights = weights.clone();
        flow = flow.dual_stream(
            format!("blk{li}"),
            stream_id::IMG,
            stream_id::TXT,
            move |emit, img, txt| {
                let mod_img = load_double_mod(emit, MOD_IMG_KEY)?;
                let mod_txt = load_double_mod(emit, MOD_TXT_KEY)?;
                let cos = emit.named(ROPE_COS_KEY)?;
                let sin = emit.named(ROPE_SIN_KEY)?;
                let mut typed = Flux2TypedParams::new();
                let (h, e) = {
                    let (hir, params) = emit.hir_and_params();
                    let mut b = Flux2HirBuilder::from_emit_parts(
                        hir, params, &mut typed, &cfg, &weights, batch, img_seq, txt_seq,
                    );
                    b.emit_dual_stream_block(
                        li,
                        &block,
                        img.hir_id(),
                        txt.hir_id(),
                        &mod_img,
                        &mod_txt,
                        cos,
                        sin,
                    )?
                };
                Ok((
                    emit.wrap(h, img.shape.clone()),
                    emit.wrap(e, txt.shape.clone()),
                ))
            },
        );
    }
    flow
}

fn store_double_mod(emit: &mut rlx_flow::Emit<'_>, prefix: &str, m: &Flux2DoubleMod) {
    emit.set_named(format!("{prefix}.msa.s"), m.0.0);
    emit.set_named(format!("{prefix}.msa.c"), m.0.1);
    emit.set_named(format!("{prefix}.msa.g"), m.0.2);
    emit.set_named(format!("{prefix}.mlp.s"), m.1.0);
    emit.set_named(format!("{prefix}.mlp.c"), m.1.1);
    emit.set_named(format!("{prefix}.mlp.g"), m.1.2);
}

fn load_double_mod(emit: &rlx_flow::Emit<'_>, prefix: &str) -> Result<Flux2DoubleMod> {
    Ok((
        (
            emit.named(&format!("{prefix}.msa.s"))?,
            emit.named(&format!("{prefix}.msa.c"))?,
            emit.named(&format!("{prefix}.msa.g"))?,
        ),
        (
            emit.named(&format!("{prefix}.mlp.s"))?,
            emit.named(&format!("{prefix}.mlp.c"))?,
            emit.named(&format!("{prefix}.mlp.g"))?,
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{extract_flux2_weights, prepare_weight_map, synthetic_weights};

    #[test]
    fn cfg_flow_matches_hir_node_count() {
        let batch = 1;
        let seq = 2;
        let channels = 2;
        let ref_hir = crate::cfg::build_flux2_cfg_combine_hir(batch, seq, channels).hir;
        let built = crate::cfg::build_flux2_cfg_combine_built(batch, seq, channels).unwrap();
        let flow_hir = built.into_hir().unwrap();
        assert_eq!(flow_hir.len(), ref_hir.len());
    }

    #[test]
    fn dual_block_flow_matches_builder_node_count() {
        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let weights = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let batch = 1;
        let img_seq = 4;
        let txt_seq = 3;
        let img_ids = vec![0.0f32; img_seq * 4];
        let txt_ids = vec![0.0f32; txt_seq * 4];

        let ref_hir = super::super::hir_builder::build_flux2_dual_section_hir(
            &cfg, &weights, batch, img_seq, txt_seq, &img_ids, &txt_ids,
        )
        .unwrap()
        .hir;

        let built = Flux2Flow::new(&cfg, &weights)
            .batch(batch)
            .img_seq(img_seq)
            .txt_seq(txt_seq)
            .position_ids(img_ids, txt_ids)
            .build_dual_blocks()
            .unwrap();
        let flow_hir = built.into_hir().unwrap();

        assert_eq!(
            flow_hir.len(),
            ref_hir.len(),
            "dual-stream flow should match hir_builder node count (flow={}, builder={})",
            flow_hir.len(),
            ref_hir.len()
        );
    }

    #[test]
    fn forward_flow_compile_matches_hir_cpu() {
        use super::super::hir_builder::compile_flux2_forward;

        let cfg = Flux2Config::tiny();
        let wm = synthetic_weights(&cfg);
        let weights = extract_flux2_weights(prepare_weight_map(wm), &cfg).unwrap();
        let batch = 1usize;
        let img_seq = 4usize;
        let txt_seq = 3usize;
        let img_ids = vec![0.0f32; img_seq * 4];
        let txt_ids = vec![0.0f32; txt_seq * 4];

        let (mut flow_c, _) = super::compile_flux2_forward_via_flow(
            &cfg,
            &weights,
            batch,
            img_seq,
            txt_seq,
            &img_ids,
            &txt_ids,
            rlx_runtime::Device::Cpu,
            None,
            None,
            None,
        )
        .unwrap();
        let (mut hir_c, _) = compile_flux2_forward(
            &cfg,
            &weights,
            batch,
            img_seq,
            txt_seq,
            &img_ids,
            &txt_ids,
            rlx_runtime::Device::Cpu,
            None,
            None,
            None,
        )
        .unwrap();

        let hidden = vec![0.1f32; batch * img_seq * cfg.in_channels];
        let encoder = vec![0.2f32; batch * txt_seq * cfg.joint_attention_dim];
        let temb =
            super::super::hir_builder::host_temb(&weights, &cfg, &[0.5], Some(&[3.5])).unwrap();
        let out_flow = flow_c
            .run(&[
                ("hidden", hidden.as_slice()),
                ("encoder", encoder.as_slice()),
                ("temb", temb.as_slice()),
            ])
            .remove(0);
        let out_hir = hir_c
            .run(&[
                ("hidden", hidden.as_slice()),
                ("encoder", encoder.as_slice()),
                ("temb", temb.as_slice()),
            ])
            .remove(0);
        assert_eq!(out_flow.len(), out_hir.len());
        let mae: f32 = out_flow
            .iter()
            .zip(out_hir.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / out_flow.len() as f32;
        assert!(mae < 1e-4, "flow vs hir mae={mae}");
    }
}
