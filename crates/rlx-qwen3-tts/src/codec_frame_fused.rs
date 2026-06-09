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

//! Codec-frame fusion: host CP greedy + compiled talker decode (production), optional full megagraph.
//!
//! **Production (`run_codec_frame`):** one CPU eager CP backbone + talker
//! [`TalkerEngine::decode_hidden_into`] (bucketed cache / GPU KV) — no duplicate CP or compile cache.
//! **Parity/bench (`run_full_megagraph`):** CP prefill + unrolled decode + talker in one tier-1 graph
//! (`RLX_QWEN3_TTS_CODEC_FRAME_MEGAGRAPH=1`).

use crate::codec_frame::{
    Qwen3TtsGraphProfiles, Qwen3TtsGraphRole, build_qwen3_tts_codec_frame_decode_built,
};
use crate::compile_opts::{
    metal_compile_guard, metal_mpsgraph_run_guard, talker_compile_options,
    talker_decode_compile_device,
};
use crate::config::{CodePredictorConfig, TalkerConfig};
use crate::cp_frame::{
    CP_DECODE_BACKBONE_STEPS, CP_PREFILL_TWO, build_qwen3_tts_cp_decode_step_built,
    build_qwen3_tts_cp_prefill_two_built,
};
use crate::cp_megakernel::CpMegakernelGreedy;
use crate::hir_stitch::{InputBindMap, SegmentOutputs, append_graph_segment};
use crate::load::{Qwen3TtsWeightStore, remap_code_predictor_weights, remap_talker_weights};
use crate::mrope::talker_decode_rope_into;
use crate::talker::engine::TalkerEngine;
use crate::talker::rope::{build_inv_freq, rope_slice};
use anyhow::{Result, ensure};
use ndarray::ArrayView1;
use rlx_core::autoregressive::split_bucketed_decode_kv;
use rlx_core::flow_util::compile_cache_ensure_built_with_options;
use rlx_flow::BuiltModel;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_qwen3::Qwen3Config;
use rlx_runtime::compile_cache::CompileCache;
use rlx_runtime::{CompileOptions, Device};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

fn kv_past_bind(segment_outs: &[NodeId], n_layers: usize) -> InputBindMap {
    let mut bind = InputBindMap::new();
    for layer in 0..n_layers {
        bind.insert(format!("past_k_{layer}"), segment_outs[1 + 2 * layer]);
        bind.insert(format!("past_v_{layer}"), segment_outs[2 + 2 * layer]);
    }
    bind
}

fn decode_step_bind(
    embed_id: NodeId,
    rope_cos_id: NodeId,
    rope_sin_id: NodeId,
    mask_id: NodeId,
    kv_from: &[NodeId],
    n_layers: usize,
) -> InputBindMap {
    let mut bind = kv_past_bind(kv_from, n_layers);
    bind.insert("inputs_embeds".into(), embed_id);
    bind.insert("rope_cos".into(), rope_cos_id);
    bind.insert("rope_sin".into(), rope_sin_id);
    bind.insert("mask".into(), mask_id);
    bind
}

fn snapshot_loader(
    cache: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> crate::weights::SnapshotLoader {
    crate::weights::SnapshotLoader::new(cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

/// Stitch CP prefill, [`CP_DECODE_BACKBONE_STEPS`] decode steps, and talker decode into one graph.
pub fn build_qwen3_tts_codec_frame_built(
    talker_qwen3: &Qwen3Config,
    cp_qwen3: &Qwen3Config,
    talker_weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    cp_weights: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    talker_profiles: &Qwen3TtsGraphProfiles,
    cp_profiles: &Qwen3TtsGraphProfiles,
    talker_past_upper: usize,
) -> Result<BuiltModel> {
    ensure!(
        CP_DECODE_BACKBONE_STEPS > 0,
        "codec frame fused graph needs at least one CP decode step"
    );

    let f = DType::F32;
    let cp_h = cp_qwen3.hidden_size;
    let _cp_kv_dim = cp_qwen3.kv_proj_dim();
    let cp_half = cp_qwen3.head_dim / 2;
    let cp_layers = cp_qwen3.num_hidden_layers;

    let talk_h = talker_qwen3.hidden_size;
    let talk_kv_dim = talker_qwen3.kv_proj_dim();
    let talk_half = talker_qwen3.head_dim / 2;
    let talk_layers = talker_qwen3.num_hidden_layers;

    let head_half = cp_half;
    let rope_placeholder = vec![0f32; CP_PREFILL_TWO * head_half];

    let mut cp_loader = snapshot_loader(cp_weights);
    let cp_prefill_built = build_qwen3_tts_cp_prefill_two_built(
        cp_qwen3,
        &mut cp_loader,
        &cp_profiles.prefill,
        Some(rope_placeholder.clone()),
        Some(rope_placeholder),
    )?;
    let (cp_prefill_graph, cp_prefill_params) = cp_prefill_built.into_graph_parts()?;

    let mut decode_parts = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
    for step in 0..CP_DECODE_BACKBONE_STEPS {
        let past_seq = CP_PREFILL_TWO + step;
        let mut cp_loader = snapshot_loader(cp_weights);
        let built = build_qwen3_tts_cp_decode_step_built(
            cp_qwen3,
            &mut cp_loader,
            past_seq,
            &cp_profiles.decode,
        )?;
        decode_parts.push(built.into_graph_parts()?);
    }

    let mut talk_loader = snapshot_loader(talker_weights);
    let talker_built = build_qwen3_tts_codec_frame_decode_built(
        talker_qwen3,
        &mut talk_loader,
        talker_past_upper,
        &talker_profiles.decode,
    )?;
    let (talker_graph, talker_params) = talker_built.into_graph_parts()?;

    let mut graph = Graph::new("qwen3_tts_codec_frame");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut shared_params: HashMap<String, NodeId> = HashMap::new();

    let cp_prefill_embeds = graph.input(
        "cp_prefill_embeds",
        Shape::new(&[1, CP_PREFILL_TWO, cp_h], f),
    );

    let mut cp_step_embed = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
    let mut cp_rope_cos = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
    let mut cp_rope_sin = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
    let mut cp_mask = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
    for step in 0..CP_DECODE_BACKBONE_STEPS {
        let past_seq = CP_PREFILL_TWO + step;
        cp_step_embed.push(graph.input(
            format!("cp_step_embed_{step}"),
            Shape::new(&[1, 1, cp_h], f),
        ));
        cp_rope_cos.push(graph.input(format!("cp_rope_cos_{step}"), Shape::new(&[1, cp_half], f)));
        cp_rope_sin.push(graph.input(format!("cp_rope_sin_{step}"), Shape::new(&[1, cp_half], f)));
        cp_mask.push(graph.input(format!("cp_mask_{step}"), Shape::new(&[1, past_seq + 1], f)));
    }

    let talker_codec_emb = graph.input("talker_codec_emb", Shape::new(&[1, 1, talk_h], f));
    let talker_rope_cos = graph.input("talker_rope_cos", Shape::new(&[1, talk_half], f));
    let talker_rope_sin = graph.input("talker_rope_sin", Shape::new(&[1, talk_half], f));
    let talker_mask = graph.input("talker_mask", Shape::new(&[1, talker_past_upper + 1], f));
    let mut talker_past_k = Vec::with_capacity(talk_layers);
    let mut talker_past_v = Vec::with_capacity(talk_layers);
    for layer in 0..talk_layers {
        let past_kv = Shape::new(&[1, talker_past_upper, talk_kv_dim], f);
        talker_past_k.push(graph.input(format!("talker_past_k_{layer}"), past_kv.clone()));
        talker_past_v.push(graph.input(format!("talker_past_v_{layer}"), past_kv));
    }

    let mut prefill_bind = InputBindMap::new();
    prefill_bind.insert("inputs_embeds".into(), cp_prefill_embeds);
    let cp_prefill_outs = append_graph_segment(
        &mut graph,
        &mut params,
        &cp_prefill_graph,
        &cp_prefill_params,
        "cp_pf.",
        &prefill_bind,
        &mut shared_params,
    )?;

    let cp_prefill_hidden = cp_prefill_outs[0];
    let mut decode_outs: Vec<SegmentOutputs> = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
    let mut kv_source = cp_prefill_outs;
    for (step, (dec_graph, dec_params)) in decode_parts.into_iter().enumerate() {
        let bind = decode_step_bind(
            cp_step_embed[step],
            cp_rope_cos[step],
            cp_rope_sin[step],
            cp_mask[step],
            &kv_source,
            cp_layers,
        );
        let outs = append_graph_segment(
            &mut graph,
            &mut params,
            &dec_graph,
            &dec_params,
            "cp_dec.",
            &bind,
            &mut shared_params,
        )?;
        kv_source = outs.clone();
        decode_outs.push(outs);
    }

    let mut talk_bind = InputBindMap::new();
    talk_bind.insert("inputs_embeds".into(), talker_codec_emb);
    talk_bind.insert("rope_cos".into(), talker_rope_cos);
    talk_bind.insert("rope_sin".into(), talker_rope_sin);
    talk_bind.insert("mask".into(), talker_mask);
    for layer in 0..talk_layers {
        talk_bind.insert(format!("past_k_{layer}"), talker_past_k[layer]);
        talk_bind.insert(format!("past_v_{layer}"), talker_past_v[layer]);
    }
    let talker_outs = append_graph_segment(
        &mut graph,
        &mut params,
        &talker_graph,
        &talker_params,
        "talk.",
        &talk_bind,
        &mut shared_params,
    )?;

    let mut outputs = Vec::new();
    outputs.push(cp_prefill_hidden);
    for outs in &decode_outs {
        outputs.push(outs[0]);
    }
    outputs.push(talker_outs[0]);
    for layer in 0..talk_layers {
        outputs.push(talker_outs[1 + 2 * layer]);
        outputs.push(talker_outs[2 + 2 * layer]);
    }
    graph.set_outputs(outputs);

    let mut built = BuiltModel::from_graph(graph, params)?;
    built.profile = talker_profiles.decode.clone();
    Ok(built)
}

fn cp_step_mask(past_seq: usize) -> Vec<f32> {
    vec![1.0; past_seq + 1]
}

/// Warmed codec-frame engine: host CP greedy + optional megagraph cache (parity/bench).
pub struct CodecFrameFusedEngine {
    session_device: Device,
    megagraph_cache: CompileCache,
    cp_greedy: CpMegakernelGreedy,
    talker_qwen3: Qwen3Config,
    cp_qwen3: Qwen3Config,
    talk_weights: Arc<crate::load::TensorSnapshot>,
    cp_weights: Arc<crate::load::TensorSnapshot>,
    talk_profiles: Qwen3TtsGraphProfiles,
    cp_profiles: Qwen3TtsGraphProfiles,
    talk_hidden: usize,
    cp_hidden: usize,
    talk_layers: usize,
    talk_kv_dim: usize,
    talk_cfg: TalkerConfig,
    talk_inv_freq: Vec<f64>,
    decode_opts: CompileOptions,
    prefill_flat: Vec<f32>,
    cp_masks: Vec<Vec<f32>>,
    cp_rope_cos: Vec<Vec<f32>>,
    cp_rope_sin: Vec<Vec<f32>>,
    talk_rope_cos: Vec<f32>,
    talk_rope_sin: Vec<f32>,
    talk_mask_buf: Vec<f32>,
}

impl CodecFrameFusedEngine {
    pub fn open(
        store: &Qwen3TtsWeightStore,
        talker_cfg: &TalkerConfig,
        cp_cfg: &CodePredictorConfig,
        session_device: Device,
    ) -> Result<Self> {
        Self::open_at(store.model_dir(), store, talker_cfg, cp_cfg, session_device)
    }

    pub fn open_at(
        model_dir: &Path,
        store: &Qwen3TtsWeightStore,
        talker_cfg: &TalkerConfig,
        cp_cfg: &CodePredictorConfig,
        session_device: Device,
    ) -> Result<Self> {
        let megagraph = crate::synth_opts::codec_frame_megagraph_enabled();
        let compile_device = talker_decode_compile_device(session_device);
        let (talk_weights, cp_weights, talk_profiles, cp_profiles, decode_opts) = if megagraph {
            let mut talk_wm = store.load_talker_backbone()?;
            let talk_weights = Arc::new(remap_talker_weights(&mut talk_wm)?);
            let mut cp_wm = store.load_code_predictor_backbone()?;
            let cp_weights = Arc::new(remap_code_predictor_weights(&mut cp_wm)?);
            let talk_profiles = Qwen3TtsGraphProfiles::for_role(
                model_dir,
                Qwen3TtsGraphRole::Talker,
                compile_device,
            );
            let decode_opts = talker_compile_options(&talk_profiles.decode, compile_device);
            let cp_profiles = Qwen3TtsGraphProfiles::for_role(
                model_dir,
                Qwen3TtsGraphRole::CodePredictor,
                compile_device,
            );
            (
                talk_weights,
                cp_weights,
                talk_profiles,
                cp_profiles,
                decode_opts,
            )
        } else {
            let talk_profiles = Qwen3TtsGraphProfiles::for_role(
                model_dir,
                Qwen3TtsGraphRole::Talker,
                compile_device,
            );
            let decode_opts = talker_compile_options(&talk_profiles.decode, compile_device);
            let cp_profiles = Qwen3TtsGraphProfiles::for_role(
                model_dir,
                Qwen3TtsGraphRole::CodePredictor,
                compile_device,
            );
            (
                Arc::new(HashMap::new()),
                Arc::new(HashMap::new()),
                talk_profiles,
                cp_profiles,
                decode_opts,
            )
        };

        let mut cp_masks = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
        let mut cp_rope_cos = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
        let mut cp_rope_sin = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
        for step in 0..CP_DECODE_BACKBONE_STEPS {
            let past_seq = CP_PREFILL_TWO + step;
            cp_masks.push(cp_step_mask(past_seq));
            let (cos, sin) = rope_slice(
                &build_inv_freq(cp_cfg.head_dim, cp_cfg.rope_theta),
                past_seq,
                cp_cfg.head_dim,
            );
            cp_rope_cos.push(cos);
            cp_rope_sin.push(sin);
        }

        let cp_greedy = CpMegakernelGreedy::open(store, cp_cfg)?;

        Ok(Self {
            session_device,
            megagraph_cache: CompileCache::new(compile_device, 4),
            cp_greedy,
            talker_qwen3: talker_cfg.to_qwen3_config(),
            cp_qwen3: cp_cfg.to_qwen3_config(),
            talk_weights,
            cp_weights,
            talk_profiles,
            cp_profiles,
            talk_hidden: talker_cfg.hidden_size,
            cp_hidden: cp_cfg.hidden_size,
            talk_layers: talker_cfg.num_hidden_layers,
            talk_kv_dim: talker_cfg.num_key_value_heads * talker_cfg.head_dim,
            talk_cfg: talker_cfg.clone(),
            talk_inv_freq: build_inv_freq(talker_cfg.head_dim, talker_cfg.rope_theta),
            decode_opts,
            prefill_flat: vec![0.0; CP_PREFILL_TWO * cp_cfg.hidden_size],
            cp_masks,
            cp_rope_cos,
            cp_rope_sin,
            talk_rope_cos: vec![0.0; talker_cfg.head_dim / 2],
            talk_rope_sin: vec![0.0; talker_cfg.head_dim / 2],
            talk_mask_buf: Vec::new(),
        })
    }

    pub fn device(&self) -> Device {
        self.session_device
    }

    pub fn warmup(&mut self, talker: &TalkerEngine, horizon: usize) -> Result<()> {
        self.cp_greedy.warmup()?;
        if !crate::synth_opts::codec_frame_megagraph_enabled() {
            return Ok(());
        }
        if crate::synth_opts::lazy_talk_buckets() {
            let upper = talker.decode_bucket_upper(talker.past_len());
            self.ensure_megagraph_compiled(upper)?;
            return Ok(());
        }
        for past in 0..=horizon {
            let upper = talker.decode_bucket_upper(past);
            self.ensure_megagraph_compiled(upper)?;
        }
        Ok(())
    }

    fn ensure_megagraph_compiled(
        &mut self,
        talker_upper: usize,
    ) -> Result<&mut rlx_runtime::CompiledGraph> {
        let key = talker_upper as u64;
        let talker_qwen3 = self.talker_qwen3.clone();
        let cp_qwen3 = self.cp_qwen3.clone();
        let talk_weights = Arc::clone(&self.talk_weights);
        let cp_weights = Arc::clone(&self.cp_weights);
        let talk_profiles = self.talk_profiles.clone();
        let cp_profiles = self.cp_profiles.clone();
        let opts = self.decode_opts.clone();
        metal_compile_guard(self.session_device, || {
            let built = build_qwen3_tts_codec_frame_built(
                &talker_qwen3,
                &cp_qwen3,
                talk_weights.as_ref(),
                cp_weights.as_ref(),
                &talk_profiles,
                &cp_profiles,
                talker_upper,
            )?;
            compile_cache_ensure_built_with_options(&mut self.megagraph_cache, key, built, &opts)
        })
    }

    /// Host CP greedy (groups + summed codec embed); no talker decode.
    pub fn predict_codec_groups(
        &mut self,
        talker_hidden: &[f32],
        group0: u32,
        pad: &[f32],
        codec_emb: &mut [f32],
    ) -> Result<Vec<u32>> {
        self.cp_greedy
            .predict_groups_fill_emb(talker_hidden, group0, pad, codec_emb)
    }

    /// Production path: host CP greedy once, then compiled talker decode.
    pub fn run_codec_frame(
        &mut self,
        talker: &mut TalkerEngine,
        talker_hidden: &[f32],
        group0: u32,
        pad: &[f32],
        codec_emb: &mut [f32],
        hidden_out: &mut [f32],
    ) -> Result<Vec<u32>> {
        let groups = self.predict_codec_groups(talker_hidden, group0, pad, codec_emb)?;
        self.run_talk_decode(talker, codec_emb, hidden_out)?;
        Ok(groups)
    }

    /// Talker decode via [`TalkerEngine`] bucketed cache (no duplicate graph build per frame).
    pub fn run_talk_decode(
        &mut self,
        talker: &mut TalkerEngine,
        codec_emb: &[f32],
        hidden_out: &mut [f32],
    ) -> Result<()> {
        talker.decode_hidden_into(ArrayView1::from(codec_emb), hidden_out)
    }

    /// Full megagraph (CP + talker); parity/bench only — duplicates CP vs [`Self::run_codec_frame`].
    pub fn run_full_megagraph(
        &mut self,
        talker: &mut TalkerEngine,
        talker_hidden: &[f32],
        codec_emb: &[f32],
        step_embeds: &[Vec<f32>],
        hidden_out: &mut [f32],
    ) -> Result<()> {
        ensure!(talker_hidden.len() == self.cp_hidden);
        ensure!(codec_emb.len() == self.talk_hidden);
        ensure!(hidden_out.len() == self.talk_hidden);
        ensure!(
            step_embeds.len() == CP_DECODE_BACKBONE_STEPS,
            "step_embeds len {}",
            step_embeds.len()
        );

        let past_seq = talker.past_len();
        let upper = talker.decode_bucket_upper(past_seq);

        self.prefill_flat[..self.cp_hidden].copy_from_slice(talker_hidden);

        talker_decode_rope_into(
            &self.talk_cfg,
            &self.talk_inv_freq,
            past_seq,
            talker.rope_delta(),
            &mut self.talk_rope_cos,
            &mut self.talk_rope_sin,
        );
        self.talk_mask_buf.resize(upper + 1, 0.0);
        for (i, slot) in self.talk_mask_buf.iter_mut().enumerate().take(upper + 1) {
            *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
        }

        let kv = talker.kv_state();
        let (padded_k, padded_v) = kv.pad_layers_to_upper(upper as u64, self.talk_kv_dim);

        self.ensure_megagraph_compiled(upper)?;

        let prefill_flat = self.prefill_flat.clone();
        let cp_rope_cos = self.cp_rope_cos.clone();
        let cp_rope_sin = self.cp_rope_sin.clone();
        let cp_masks = self.cp_masks.clone();
        let talk_cos = self.talk_rope_cos.clone();
        let talk_sin = self.talk_rope_sin.clone();
        let talk_mask = self.talk_mask_buf.clone();

        let mut named: Vec<(String, &[f32])> = Vec::new();
        named.push(("cp_prefill_embeds".into(), prefill_flat.as_slice()));
        for step in 0..CP_DECODE_BACKBONE_STEPS {
            named.push((
                format!("cp_step_embed_{step}"),
                step_embeds[step].as_slice(),
            ));
            named.push((format!("cp_rope_cos_{step}"), cp_rope_cos[step].as_slice()));
            named.push((format!("cp_rope_sin_{step}"), cp_rope_sin[step].as_slice()));
            named.push((format!("cp_mask_{step}"), cp_masks[step].as_slice()));
        }
        named.push(("talker_codec_emb".into(), codec_emb));
        named.push(("talker_rope_cos".into(), talk_cos.as_slice()));
        named.push(("talker_rope_sin".into(), talk_sin.as_slice()));
        named.push(("talker_mask".into(), talk_mask.as_slice()));
        for layer in 0..self.talk_layers {
            named.push((format!("talker_past_k_{layer}"), padded_k[layer].as_slice()));
            named.push((format!("talker_past_v_{layer}"), padded_v[layer].as_slice()));
        }
        let inputs: Vec<(&str, &[f32])> = named.iter().map(|(n, d)| (n.as_str(), *d)).collect();
        let key = upper as u64;
        let opts = self.decode_opts.clone();

        let outputs = metal_mpsgraph_run_guard(self.session_device, || {
            self.megagraph_cache
                .get_or_compile_with_options(
                    key,
                    || panic!("megagraph cache miss for key {key}"),
                    &opts,
                )
                .run(&inputs)
        });
        let talk_idx = 1 + CP_DECODE_BACKBONE_STEPS;
        let talk_outputs = outputs[talk_idx..talk_idx + 1 + 2 * self.talk_layers].to_vec();
        let (hidden_vec, new_k, new_v) = split_bucketed_decode_kv(
            talk_outputs,
            past_seq,
            self.talk_kv_dim,
            self.talk_layers,
            1,
        )?;
        talker.import_fused_decode_outputs(&hidden_vec, &new_k, &new_v, past_seq)?;
        hidden_out.copy_from_slice(talker.last_hidden_view().as_slice().unwrap());
        Ok(())
    }

    /// Fill CP prefill row-1 with group-0 codec embed (megagraph feeds).
    pub fn set_cp_prefill_g0_embed(&mut self, g0_embed: &[f32]) {
        let h = self.cp_hidden;
        self.prefill_flat[h..h * 2].copy_from_slice(g0_embed);
    }

    pub fn codec_embed_row(&self, group_idx: usize, token: u32) -> Result<Vec<f32>> {
        self.cp_greedy.codec_embed_row(group_idx, token)
    }

    pub fn cp_step_embeds_from_groups(&self, groups: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.cp_greedy.cp_step_embeds_from_groups(groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Qwen3TtsConfig;
    use crate::codec_frame::Qwen3TtsGraphRole;
    use crate::load::{Qwen3TtsWeightStore, remap_code_predictor_weights, remap_talker_weights};
    use rlx_runtime::Device;
    use std::path::PathBuf;

    #[test]
    fn codec_frame_fused_graph_builds() {
        let Some(model_dir) = std::env::var("RLX_QWEN3_TTS_DIR").ok().map(PathBuf::from) else {
            eprintln!("skip: RLX_QWEN3_TTS_DIR");
            return;
        };
        let cfg = Qwen3TtsConfig::from_model_dir(&model_dir).expect("config");
        let store = Qwen3TtsWeightStore::open(&model_dir).expect("store");
        let talker_cfg = cfg.talker().to_qwen3_config();
        let cp_cfg = cfg.code_predictor().to_qwen3_config();

        let mut talk_wm = store.load_talker_backbone().expect("talker wm");
        let talk_weights = remap_talker_weights(&mut talk_wm).expect("talker remap");
        let mut cp_wm = store.load_code_predictor_backbone().expect("cp wm");
        let cp_weights = remap_code_predictor_weights(&mut cp_wm).expect("cp remap");

        let talk_profiles = Qwen3TtsGraphProfiles::for_role(
            store.model_dir(),
            Qwen3TtsGraphRole::Talker,
            Device::Cpu,
        );
        let cp_profiles = Qwen3TtsGraphProfiles::for_role(
            store.model_dir(),
            Qwen3TtsGraphRole::CodePredictor,
            Device::Cpu,
        );

        let built = build_qwen3_tts_codec_frame_built(
            &talker_cfg,
            &cp_cfg,
            &talk_weights,
            &cp_weights,
            &talk_profiles,
            &cp_profiles,
            16,
        )
        .expect("fused codec frame built");

        let (graph, params) = built.into_graph_parts().expect("graph parts");
        assert!(!graph.nodes().is_empty());
        assert!(!params.is_empty());
        let expected_outs = 1 + CP_DECODE_BACKBONE_STEPS + 1 + 2 * talker_cfg.num_hidden_layers;
        assert_eq!(
            graph.outputs.len(),
            expected_outs,
            "hidden taps + talker hidden + talker kv"
        );
    }
}
