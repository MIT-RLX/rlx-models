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

//! GPU/MLX/Metal path for speech decoder `pre_transformer` (8-layer Qwen3 + sliding window).

use super::decode::DecoderConfig;
use super::ops::{linear2, rms_norm};
use crate::compile_opts::{metal_compile_guard, talker_compile_options};
use crate::weights::weight_map_from_cache;
use anyhow::{Context, Result, ensure};
use ndarray::{Array2, ArrayView2};
use rlx_core::flow_util::compile_cache_ensure_built_with_options;
use rlx_flow::CompileProfile;
use rlx_qwen3::{
    Qwen3Config, Qwen3PrefillOpts, build_qwen3_prefill_embeds_built, qwen3_profile_near_weights,
};
use rlx_runtime::Device;
use rlx_runtime::compile_cache::CompileCache;
use std::collections::HashMap;
use std::path::Path;

const PT_PREFIX: &str = "decoder.pre_transformer";
const PT_MAX_SEQ: usize = 512;

pub fn speech_pt_use_compiled(device: Device) -> bool {
    if std::env::var("RLX_QWEN3_TTS_SPEECH_EAGER").ok().as_deref() == Some("1") {
        return false;
    }
    if std::env::var("RLX_QWEN3_TTS_SPEECH_COMPILED")
        .ok()
        .as_deref()
        == Some("1")
    {
        return device != Device::Cpu;
    }
    if device == Device::Cpu {
        return false;
    }
    if crate::gpu_pipeline::gpu_session_enabled(device)
        && crate::gpu_pipeline::speech_compiled_default()
    {
        return matches!(
            device,
            Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm
        );
    }
    matches!(device, Device::Mlx | Device::Cuda | Device::Rocm)
}

pub fn speech_pt_backend_label(device: Device) -> &'static str {
    if speech_pt_use_compiled(device) {
        "compiled"
    } else {
        "CPU eager"
    }
}

fn map_pt_key(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix(PT_PREFIX)?;
    let rest = rest.strip_prefix('.').unwrap_or(rest);
    match rest {
        s if s.starts_with("layers.") => Some(format!("model.{rest}")),
        "norm.weight" => Some("model.norm.weight".into()),
        _ => None,
    }
}

fn remap_pre_transformer_weights(
    map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut out = HashMap::new();
    for (k, v) in map {
        if let Some(m) = map_pt_key(k) {
            out.insert(m, v.clone());
        }
    }
    out
}

fn decoder_to_qwen3(cfg: &DecoderConfig) -> Qwen3Config {
    Qwen3Config {
        vocab_size: 1,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        max_position_embeddings: cfg.sliding_window.max(PT_MAX_SEQ),
        rms_norm_eps: cfg.rms_norm_eps as f64,
        rope_theta: cfg.rope_theta,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        qk_norm: false,
        sliding_window: Some(cfg.sliding_window),
        max_window_layers: 0,
        use_sliding_window: true,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

pub struct PreTransformerGpu {
    device: Device,
    qwen3: Qwen3Config,
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    prefill_cache: CompileCache,
    profile: CompileProfile,
    inv_freq: Vec<f64>,
}

impl PreTransformerGpu {
    pub fn open(
        model_dir: &Path,
        cfg: &DecoderConfig,
        map: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        device: Device,
    ) -> Result<Self> {
        let mut weights = remap_pre_transformer_weights(map);
        ensure!(
            !weights.is_empty(),
            "no pre_transformer weights for GPU path"
        );
        super::layer_scale::bake_layer_scales_into_qwen3_weights(
            &mut weights,
            map,
            cfg.num_hidden_layers,
        )?;
        let head_half = cfg.head_dim / 2;
        let inv_freq: Vec<f64> = (0..head_half)
            .map(|j| 1.0 / cfg.rope_theta.powf(2.0 * j as f64 / cfg.head_dim as f64))
            .collect();
        Ok(Self {
            device,
            qwen3: decoder_to_qwen3(cfg),
            weights,
            prefill_cache: CompileCache::new(device, 32),
            profile: qwen3_profile_near_weights(model_dir, false),
            inv_freq,
        })
    }

    pub fn available(device: Device) -> bool {
        speech_pt_use_compiled(device)
    }

    pub fn warmup(&mut self, seq: usize) -> Result<()> {
        let seq = seq.clamp(1, 32);
        let mut h = Array2::<f32>::zeros((seq, self.qwen3.hidden_size));
        h[[0, 0]] = 1e-5;
        let _ = self.forward(h.view())?;
        Ok(())
    }

    /// `[seq, hidden]` in → `[seq, hidden]` out (input_proj / output_proj outside).
    pub fn forward(&mut self, seq: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (t, h) = seq.dim();
        ensure!(h == self.qwen3.hidden_size, "pt hidden mismatch");
        ensure!(t <= PT_MAX_SEQ, "pt seq {t} > {PT_MAX_SEQ}");
        let flat: Vec<f32> = seq.iter().copied().collect();
        let rope_table_len = self.qwen3.max_position_embeddings;
        let (rope_cos, rope_sin) = crate::talker::rope::rope_tables_full(
            &self.inv_freq,
            rope_table_len,
            self.qwen3.head_dim,
        );
        let opts = talker_compile_options(&self.profile, self.device);
        let key = ((1u64) << 32) | (t as u64);
        let qwen3 = self.qwen3.clone();
        let weights = self.weights.clone();
        let profile = self.profile.clone();
        let built = {
            let mut wm = weight_map_from_cache(&weights)?;
            build_qwen3_prefill_embeds_built(
                &qwen3,
                &mut wm,
                &Qwen3PrefillOpts {
                    batch: 1,
                    seq: t,
                    with_kv_outputs: false,
                    with_qk_outputs: false,
                    with_lm_head: false,
                    last_logits_only: false,
                    profile: Some(profile),
                    rope_cos: Some(rope_cos),
                    rope_sin: Some(rope_sin),
                },
            )?
        };
        let compiled = metal_compile_guard(self.device, || {
            compile_cache_ensure_built_with_options(&mut self.prefill_cache, key, built, &opts)
        })?;
        let mut outputs = compiled.run(&[("inputs_embeds", flat.as_slice())]);
        let hidden = outputs.pop().context("pt gpu forward: no hidden output")?;
        Ok(Array2::from_shape_vec((t, h), hidden)?)
    }
}

/// Run input_proj → GPU transformer → norm → output_proj (matches eager `run_pre_transformer`).
pub fn run_pre_transformer_hybrid(
    cfg: &DecoderConfig,
    pt: &super::decode::PreTransformer,
    mut h: Array2<f32>,
    gpu: &mut PreTransformerGpu,
) -> Result<Array2<f32>> {
    h = linear2(
        h.view(),
        pt.input_proj_w.view(),
        Some(pt.input_proj_b.view()),
    );
    h = gpu.forward(h.view())?;
    h = rms_norm(h.view(), pt.norm_w.view(), cfg.rms_norm_eps);
    h = linear2(
        h.view(),
        pt.output_proj_w.view(),
        Some(pt.output_proj_b.view()),
    );
    Ok(h)
}
