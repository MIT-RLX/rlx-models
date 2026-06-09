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

//! Nemotron-H hybrid runner — single-step decode with Mamba2 state.

use anyhow::{Context, Result, anyhow};
use rlx_core::flow_util::{WeightMapSource, attach_built_params, graph_from_built};
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, FlowValue, ModelFlow, SideOutputs};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{Device, Session};
use std::path::PathBuf;

use super::config::{NemotronHybridConfig, NemotronLayerKind};
use super::flow::{mamba2_decode_layer_plugin_with_sink, stateless_attention_layer_plugin};

fn build_decode_flow(cfg: &NemotronHybridConfig, sink: SideOutputs) -> ModelFlow {
    let cfg_for_lm = cfg.clone();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let mh = cfg.mamba2_num_heads;
    let mn = cfg.mamba2_state_size;

    let mut flow = ModelFlow::new("nemotron_hybrid_decode")
        .with_profile(CompileProfile::encoder())
        .input("token_id", Shape::new(&[1, 1], DType::F32));

    // Per-Mamba2-layer state input.
    for (l, kind) in cfg.layer_kinds.iter().enumerate() {
        if matches!(kind, NemotronLayerKind::Mamba2) {
            flow = flow.input(
                format!("state_in_{l}"),
                Shape::new(&[1, mh, mn], DType::F32),
            );
        }
    }

    flow = flow.plugin_named("embed", move |emit, _input| {
        let token_id_val = emit.flow_input("token_id")?;
        let w = emit.load_param("token_embd.weight", false)?;
        let mut gb = HirMut::new(emit.hir());
        let h_out = gb.gather_(w, token_id_val.hir_id(), 0);
        let reshaped = gb.reshape_(h_out, vec![1, 1, hidden as i64]);
        Ok(Some(FlowValue::new(
            reshaped,
            Shape::new(&[1, 1, hidden], DType::F32),
        )))
    });

    let sink_inner = sink.inner();
    for (l, kind) in cfg.layer_kinds.iter().copied().enumerate() {
        match kind {
            NemotronLayerKind::Mamba2 => {
                let cfg_l = cfg.clone();
                let sink_l = sink_inner.clone();
                flow = flow.plugin_named(format!("bind_state_{l}"), move |emit, input| {
                    let state_val = emit.flow_input(&format!("state_in_{l}"))?;
                    emit.state
                        .named
                        .insert(format!("nemotron.state_in_{l}"), state_val.hir_id());
                    Ok(input)
                });
                flow = flow.plugin_named(
                    format!("mamba2_layer_{l}"),
                    mamba2_decode_layer_plugin_with_sink(cfg_l, l, Some(sink_l)),
                );
            }
            NemotronLayerKind::Attention => {
                let cfg_l = cfg.clone();
                flow = flow.plugin_named(
                    format!("attn_layer_{l}"),
                    stateless_attention_layer_plugin(cfg_l, l),
                );
            }
        }
    }

    flow = flow.plugin_named("lm_head", move |emit, input| {
        let h_in = input.ok_or_else(|| anyhow!("lm_head requires input"))?;
        let norm_w = emit.load_param("output_norm.weight", false)?;
        let eps = cfg_for_lm.rms_norm_eps as f32;
        let beta = emit.synth_zeros("output_norm.zero_beta", cfg_for_lm.hidden_size);
        let normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(h_in.hir_id(), norm_w, beta, eps)
        };
        let lm_w_key = if cfg_for_lm.tie_word_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        let lm_w = emit.load_param(lm_w_key, true)?;
        let mut gb = HirMut::new(emit.hir());
        let logits = gb.mm(normed, lm_w);
        Ok(Some(FlowValue::new(
            logits,
            Shape::new(&[1, 1, vocab], DType::F32),
        )))
    });

    flow.output("logits")
}

pub struct NemotronHybridRunner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: NemotronHybridConfig,
    /// Per-Mamba2-layer state buffer (other layers contribute no state).
    /// Indexed by logical layer index; attention layers store `Vec::new()`.
    state_buffers: Vec<Vec<f32>>,
    _device: Device,
}

impl NemotronHybridRunner {
    pub fn builder() -> NemotronHybridRunnerBuilder {
        NemotronHybridRunnerBuilder::default()
    }
    pub fn config(&self) -> &NemotronHybridConfig {
        &self.cfg
    }

    pub fn reset_state(&mut self) {
        for buf in self.state_buffers.iter_mut() {
            for x in buf.iter_mut() {
                *x = 0.0;
            }
        }
    }

    pub fn step(&mut self, token: u32) -> Vec<f32> {
        let token_f32 = vec![token as f32];
        let mut named: Vec<(String, &[f32])> = Vec::new();
        named.push(("token_id".to_string(), token_f32.as_slice()));
        for (l, kind) in self.cfg.layer_kinds.iter().enumerate() {
            if matches!(kind, NemotronLayerKind::Mamba2) {
                named.push((format!("state_in_{l}"), self.state_buffers[l].as_slice()));
            }
        }
        let inputs: Vec<(&str, &[f32])> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let mut outs = self.compiled.run(&inputs);
        let mut state_idx = 1usize;
        for (l, kind) in self.cfg.layer_kinds.iter().enumerate() {
            if matches!(kind, NemotronLayerKind::Mamba2) {
                let state_out = &outs[state_idx];
                self.state_buffers[l].copy_from_slice(state_out);
                state_idx += 1;
            }
        }
        outs.swap_remove(0)
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        self.reset_state();
        let mut last = Vec::new();
        for &t in prompt {
            last = self.step(t);
        }
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax_u32(&last);
            out.push(next);
            on_token(next);
            last = self.step(next);
        }
        out
    }
}

fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[derive(Debug, Clone, Default)]
pub struct NemotronHybridRunnerBuilder {
    weights: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    config: Option<NemotronHybridConfig>,
    device: Option<Device>,
}

impl NemotronHybridRunnerBuilder {
    pub fn weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn hf_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(p.into());
        self
    }
    pub fn config(mut self, c: NemotronHybridConfig) -> Self {
        self.config = Some(c);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<NemotronHybridRunner> {
        rlx_ssm::register_ir_ops();
        rlx_ssm::register_ssm_kernels();

        let device = self.device.unwrap_or(Device::Cpu);
        let weights_path = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow!("weights path required"))?
            .clone();
        let cfg = match (self.config, self.hf_config) {
            (Some(c), _) => c,
            (None, Some(p)) => NemotronHybridConfig::from_hf_config_json(&p)
                .with_context(|| format!("rlx-nemotron: parse HF config {p:?}"))?,
            (None, None) => {
                let raw_gguf = rlx_gguf::GgufFile::from_path(&weights_path)
                    .with_context(|| format!("rlx-nemotron: parse GGUF {weights_path:?}"))?;
                NemotronHybridConfig::from_gguf(&raw_gguf)?
            }
        };

        let path_str = weights_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
        let mut wm = WeightMap::from_file(path_str)
            .with_context(|| format!("rlx-nemotron: load {weights_path:?}"))?;

        let sink = SideOutputs::new();
        let flow = build_decode_flow(&cfg, sink.clone());
        let built = flow.build(&mut WeightMapSource(&mut wm))?;
        let typed = built.typed_params.clone();
        let built = built.with_extra_hir_outputs(sink.drain());
        let (graph, params) = graph_from_built(built)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        attach_built_params(&mut compiled, params, &typed);

        let state_len = cfg.mamba2_num_heads * cfg.mamba2_state_size;
        let state_buffers: Vec<Vec<f32>> = cfg
            .layer_kinds
            .iter()
            .map(|k| match k {
                NemotronLayerKind::Mamba2 => vec![0f32; state_len],
                NemotronLayerKind::Attention => Vec::new(),
            })
            .collect();

        rlx_core::validate_standard_device("nemotron", device)?;

        Ok(NemotronHybridRunner {
            compiled,
            cfg,
            state_buffers,
            _device: device,
        })
    }
}
