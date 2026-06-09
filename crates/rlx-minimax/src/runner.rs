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

//! MiniMax M2 runner — full LM decode loop.
//!
//! Builds a single-step decode graph (token_id + per-layer state-in
//! inputs → logits + per-layer state-out side outputs), compiles it on
//! the chosen device, and drives a token-by-token generate loop while
//! carrying Lightning Attention state across calls.

use anyhow::{Context, Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, FlowValue, ModelFlow, SideOutputs};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{Device, Session};
use std::path::PathBuf;

use super::config::MiniMaxConfig;
use super::flow::minimax_decode_layer_plugin_with_sink;

/// Build the complete MiniMax single-step decode graph.
fn build_decode_flow(cfg: &MiniMaxConfig, sink: SideOutputs) -> ModelFlow {
    let cfg_owned = cfg.clone();
    let cfg_for_layers = cfg.clone();
    let cfg_for_lmhead = cfg.clone();
    let _hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let h = cfg.num_attention_heads;
    let n = cfg.head_dim;
    let layers = cfg.num_hidden_layers;

    let mut flow = ModelFlow::new("minimax_decode")
        .with_profile(CompileProfile::encoder())
        .input("token_id", Shape::new(&[1, 1], DType::F32));
    for l in 0..layers {
        flow = flow.input(
            format!("state_in_{l}"),
            Shape::new(&[1, h, n, n], DType::F32),
        );
    }

    // Token-embed: gather one row from token_embd.weight using token_id.
    flow = flow.plugin_named("embed", move |emit, _input| {
        let token_id_val = emit.flow_input("token_id")?;
        let w = emit.load_param("token_embd.weight", false)?;
        let mut gb = HirMut::new(emit.hir());
        // Gather: select row[token_id[0,0]] from w[vocab, hidden].
        // Use `gather_rows` style: convert token_id to i64 index.
        // Since rlx-ir gather expects indices as a node, we use Gather.
        let h_out = gb.gather_(w, token_id_val.hir_id(), 0);
        let shape = Shape::new(&[1, 1, cfg_owned.hidden_size], DType::F32);
        let reshaped = gb.reshape_(h_out, vec![1, 1, cfg_owned.hidden_size as i64]);
        Ok(Some(FlowValue::new(reshaped, shape)))
    });

    // Per-layer plugins.
    let sink_inner = sink.inner();
    for l in 0..layers {
        let cfg_layer = cfg_for_layers.clone();
        let sink_layer = sink_inner.clone();
        flow = flow.plugin_named(format!("layer_{l}_bind"), move |emit, input| {
            // Bind state_in_{l} into named state.
            let state_val = emit.flow_input(&format!("state_in_{l}"))?;
            emit.state
                .named
                .insert(format!("minimax.state_in_{l}"), state_val.hir_id());
            Ok(input)
        });
        flow = flow.plugin_named(
            format!("layer_{l}"),
            minimax_decode_layer_plugin_with_sink(cfg_layer, l, Some(sink_layer)),
        );
    }

    // Final RMSNorm + LM head.
    flow = flow.plugin_named("lm_head", move |emit, input| {
        let h_in = input.ok_or_else(|| anyhow!("lm_head requires input"))?;
        let norm_w = emit.load_param("output_norm.weight", false)?;
        let eps = cfg_for_lmhead.rms_norm_eps as f32;
        let beta = emit.synth_zeros("output_norm.zero_beta", cfg_for_lmhead.hidden_size);
        let normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(h_in.hir_id(), norm_w, beta, eps)
        };
        // tie_word_embeddings ? token_embd : output
        let lm_w_key = if cfg_for_lmhead.tie_word_embeddings {
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

pub struct MiniMaxRunner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: MiniMaxConfig,
    state_buffers: Vec<Vec<f32>>,
    _device: Device,
}

impl MiniMaxRunner {
    pub fn builder() -> MiniMaxRunnerBuilder {
        MiniMaxRunnerBuilder::default()
    }
    pub fn config(&self) -> &MiniMaxConfig {
        &self.cfg
    }

    /// Reset all per-layer Lightning Attention state buffers to zero.
    pub fn reset_state(&mut self) {
        for buf in self.state_buffers.iter_mut() {
            for x in buf.iter_mut() {
                *x = 0.0;
            }
        }
    }

    /// Run one decode step. Returns logits over `vocab_size`.
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        let token_f32 = vec![token as f32];
        let mut named: Vec<(String, &[f32])> = Vec::with_capacity(self.state_buffers.len() + 1);
        named.push(("token_id".to_string(), token_f32.as_slice()));
        for (i, buf) in self.state_buffers.iter().enumerate() {
            named.push((format!("state_in_{i}"), buf.as_slice()));
        }
        let inputs: Vec<(&str, &[f32])> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let mut outs = self.compiled.run(&inputs);
        // Outputs: [logits, state_out_0, state_out_1, ..., state_out_{L-1}].
        for (i, buf) in self.state_buffers.iter_mut().enumerate() {
            let state_out = outs
                .get(i + 1)
                .ok_or(())
                .expect("missing state_out side output");
            buf.copy_from_slice(state_out);
        }
        outs.swap_remove(0)
    }

    /// Greedy generate `n_new` tokens after the prompt, calling
    /// `on_token` for each. Returns the generated tokens.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        self.reset_state();
        // Consume prompt (last token's logits become the first sample).
        let mut last_logits = Vec::new();
        for &t in prompt {
            last_logits = self.step(t);
        }
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax_u32(&last_logits);
            out.push(next);
            on_token(next);
            last_logits = self.step(next);
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
pub struct MiniMaxRunnerBuilder {
    weights: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    config: Option<MiniMaxConfig>,
    device: Option<Device>,
}

impl MiniMaxRunnerBuilder {
    pub fn weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn hf_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(p.into());
        self
    }
    pub fn config(mut self, c: MiniMaxConfig) -> Self {
        self.config = Some(c);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<MiniMaxRunner> {
        // Required at runtime — register SSM IR ops + CPU kernels.
        rlx_ssm::register_ir_ops();
        rlx_ssm::register_ssm_kernels();

        let device = self.device.unwrap_or(Device::Cpu);
        let weights_path = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?
            .clone();
        let cfg = match (self.config, self.hf_config) {
            (Some(c), _) => c,
            (None, Some(p)) => MiniMaxConfig::from_hf_config_json(&p)
                .with_context(|| format!("rlx-minimax: parse HF config {p:?}"))?,
            (None, None) => {
                let raw_gguf = rlx_gguf::GgufFile::from_path(&weights_path)
                    .with_context(|| format!("rlx-minimax: parse GGUF {weights_path:?}"))?;
                MiniMaxConfig::from_gguf(&raw_gguf)?
            }
        };

        let path_str = weights_path
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
        let mut wm = WeightMap::from_file(path_str)
            .with_context(|| format!("rlx-minimax: load {weights_path:?}"))?;

        let sink = SideOutputs::new();
        let flow = build_decode_flow(&cfg, sink.clone());
        let built = flow.build(&mut WeightMapSource(&mut wm))?;
        let typed = built.typed_params.clone();
        // Append per-layer state outputs.
        let mut built = built;
        built = built.with_extra_hir_outputs(sink.drain());
        let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);

        // Allocate per-layer state buffers.
        let state_len = cfg.num_attention_heads * cfg.head_dim * cfg.head_dim;
        let state_buffers = (0..cfg.num_hidden_layers)
            .map(|_| vec![0f32; state_len])
            .collect();

        rlx_core::validate_standard_device("minimax", device)?;

        Ok(MiniMaxRunner {
            compiled,
            cfg,
            state_buffers,
            _device: device,
        })
    }
}
