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

//! LFM2.5 runner — full LM decode loop.

use anyhow::{Context, Result, anyhow};
use rlx_core::flow_util::WeightMapSource;
use rlx_core::weight_map::WeightMap;
use rlx_flow::{CompileProfile, FlowValue, ModelFlow, SideOutputs};
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_runtime::{Device, Session};
use std::path::PathBuf;

use super::config::LfmConfig;
use super::flow::lfm_decode_layer_plugin_with_sink;

fn build_decode_flow(cfg: &LfmConfig, sink: SideOutputs) -> ModelFlow {
    let cfg_for_layers = cfg.clone();
    let cfg_for_lmhead = cfg.clone();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let c = cfg.ssm_channels;
    let n = cfg.ssm_state_size;
    let layers = cfg.num_hidden_layers;

    let mut flow = ModelFlow::new("lfm_decode")
        .with_profile(CompileProfile::encoder())
        .input("token_id", Shape::new(&[1, 1], DType::F32));
    for l in 0..layers {
        flow = flow.input(format!("state_in_{l}"), Shape::new(&[1, c, n], DType::F32));
    }
    let h2 = hidden;
    flow = flow.plugin_named("embed", move |emit, _input| {
        let token_id_val = emit.flow_input("token_id")?;
        let w = emit.load_param("token_embd.weight", false)?;
        let mut gb = HirMut::new(emit.hir());
        let h_out = gb.gather_(w, token_id_val.hir_id(), 0);
        let reshaped = gb.reshape_(h_out, vec![1, 1, h2 as i64]);
        Ok(Some(FlowValue::new(
            reshaped,
            Shape::new(&[1, 1, h2], DType::F32),
        )))
    });

    let sink_inner = sink.inner();
    for l in 0..layers {
        let cfg_l = cfg_for_layers.clone();
        let sink_l = sink_inner.clone();
        flow = flow.plugin_named(format!("layer_{l}_bind"), move |emit, input| {
            let state_val = emit.flow_input(&format!("state_in_{l}"))?;
            emit.state
                .named
                .insert(format!("lfm.state_in_{l}"), state_val.hir_id());
            Ok(input)
        });
        flow = flow.plugin_named(
            format!("layer_{l}"),
            lfm_decode_layer_plugin_with_sink(cfg_l, l, Some(sink_l)),
        );
    }

    flow = flow.plugin_named("lm_head", move |emit, input| {
        let h_in = input.ok_or_else(|| anyhow!("lm_head requires input"))?;
        let norm_w = emit.load_param("output_norm.weight", false)?;
        let eps = cfg_for_lmhead.rms_norm_eps as f32;
        let beta = emit.synth_zeros("output_norm.zero_beta", cfg_for_lmhead.hidden_size);
        let normed = {
            let mut gb = HirMut::new(emit.hir());
            gb.rms_norm(h_in.hir_id(), norm_w, beta, eps)
        };
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

pub struct LfmRunner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: LfmConfig,
    state_buffers: Vec<Vec<f32>>,
    _device: Device,
}

impl LfmRunner {
    pub fn builder() -> LfmRunnerBuilder {
        LfmRunnerBuilder::default()
    }
    pub fn config(&self) -> &LfmConfig {
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
        let mut named: Vec<(String, &[f32])> = Vec::with_capacity(self.state_buffers.len() + 1);
        named.push(("token_id".to_string(), token_f32.as_slice()));
        for (i, buf) in self.state_buffers.iter().enumerate() {
            named.push((format!("state_in_{i}"), buf.as_slice()));
        }
        let inputs: Vec<(&str, &[f32])> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let mut outs = self.compiled.run(&inputs);
        for (i, buf) in self.state_buffers.iter_mut().enumerate() {
            let state_out = outs.get(i + 1).expect("missing state_out side output");
            buf.copy_from_slice(state_out);
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

impl rlx_cli::LmRunner for LfmRunner {
    fn family(&self) -> &'static str {
        "lfm"
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        if prompt_ids.is_empty() {
            return Err(anyhow::anyhow!("LfmRunner::predict_logits: empty prompt"));
        }
        self.reset_state();
        let mut last = Vec::new();
        for &t in prompt_ids {
            last = self.step(t);
        }
        Ok(last)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        // LfmRunner's inherent generate(prompt, n_new, FnMut(u32)) ignores
        // the return value of `on_token`. Inline the same logic so we can
        // honour the `false → stop` signal expected by the LmRunner trait.
        self.reset_state();
        let mut last = Vec::new();
        for &t in prompt_ids {
            last = self.step(t);
        }
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax_u32(&last);
            out.push(next);
            if !on_token(next) {
                break;
            }
            last = self.step(next);
        }
        Ok(out)
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
pub struct LfmRunnerBuilder {
    weights: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    config: Option<LfmConfig>,
    device: Option<Device>,
}

impl LfmRunnerBuilder {
    pub fn weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn hf_config(mut self, p: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(p.into());
        self
    }
    pub fn config(mut self, c: LfmConfig) -> Self {
        self.config = Some(c);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn build(self) -> Result<LfmRunner> {
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
            (None, Some(p)) => LfmConfig::from_hf_config_json(&p)
                .with_context(|| format!("rlx-lfm: parse HF config {p:?}"))?,
            (None, None) => {
                let raw_gguf = rlx_gguf::GgufFile::from_path(&weights_path)
                    .with_context(|| format!("rlx-lfm: parse GGUF {weights_path:?}"))?;
                LfmConfig::from_gguf(&raw_gguf)?
            }
        };

        // Load both safetensors and GGUF (incl. K-quant). LfmRunner
        // doesn't have a packed-matmul lowering yet, so every K-quant
        // tensor is force-dequantized to F32 via `WeightLoader::take`.
        // Memory cost: ~4× the packed file size; correctness is
        // preserved. A future LFM milestone can wire packed matmul
        // (mirroring `Qwen35Weights::from_loader_packed`) for the
        // memory win.
        let mut loader = rlx_core::weight_registry::open_weight_loader(&weights_path)
            .with_context(|| format!("rlx-lfm: open {weights_path:?}"))?;
        let mut wm = WeightMap::from_weight_loader_dequant_all(loader.as_mut())
            .with_context(|| format!("rlx-lfm: dequant load {weights_path:?}"))?;

        let sink = SideOutputs::new();
        let flow = build_decode_flow(&cfg, sink.clone());
        let built = flow.build(&mut WeightMapSource(&mut wm))?;
        let typed = built.typed_params.clone();
        let built = built.with_extra_hir_outputs(sink.drain());
        let (graph, params) = rlx_core::flow_util::graph_from_built(built)?;
        let opts =
            rlx_core::flow_bridge::compile_options_for_profile(&CompileProfile::encoder(), device);
        let mut compiled = Session::new(device).compile_with(graph, &opts);
        rlx_core::flow_util::attach_built_params(&mut compiled, params, &typed);

        let state_len = cfg.ssm_channels * cfg.ssm_state_size;
        let state_buffers = (0..cfg.num_hidden_layers)
            .map(|_| vec![0f32; state_len])
            .collect();

        rlx_core::validate_standard_device("lfm", device)?;

        Ok(LfmRunner {
            compiled,
            cfg,
            state_buffers,
            _device: device,
        })
    }
}
