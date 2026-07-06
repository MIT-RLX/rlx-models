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

//! End-to-end Qwen2.5-VL runner — vision encoder + mRoPE LM.

use crate::aif::{
    AifConfig, AifDynamicsMode, AifProbe, NativePrefillProbeInputs, VisionKeySpan,
    decode_mask_row_causal, dynamics_from_graph_qk_decode_step, dynamics_from_graph_qk_layers,
    native_prefill_probe,
};
use crate::config::Qwen25VlLmConfig;
use crate::lm::load_lm_config_from_gguf;
use crate::lm_flow::{
    Qwen25VlPrefillOpts, build_qwen25_vl_prefill_mrope_built, mrope_decode_feeds,
};
use crate::mrope::mrope_prefill_feeds;
use crate::multimodal::{MultimodalPrefill, MultimodalPrompt};
use crate::vision::{Qwen25VlVisionEncoder, VisionEncodeOutput};
use anyhow::{Context, Result, bail};
use rlx_core::WeightMap;
use rlx_core::autoregressive::{KvCacheState, split_decode_logits_kv};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::{ArcCacheLoader, ArcF32Tensor, GgufLoader};
use rlx_qwen3::flow::{Qwen3DecodeOpts, build_qwen3_decode_built};
use rlx_qwen3::{Qwen3ConfigSource, Qwen3Runner, Qwen3RunnerBuilder, SampleOpts};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct Qwen25VlRunnerBuilder {
    weights: Option<PathBuf>,
    mmproj: Option<PathBuf>,
    hf_config: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
    sample: Option<SampleOpts>,
    lm_config: Option<Qwen25VlLmConfig>,
    inline_lm_weights: Option<HashMap<String, (Vec<f32>, Vec<usize>)>>,
    inline_mmproj: Option<(crate::vision::MmProjConfig, crate::vision::MmProjWeights)>,
    aif_dynamics_mode: AifDynamicsMode,
}

impl Qwen25VlRunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    pub fn mmproj(mut self, path: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(path.into());
        self
    }

    pub fn hf_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.hf_config = Some(path.into());
        self
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    pub fn sample(mut self, opts: SampleOpts) -> Self {
        self.sample = Some(opts);
        self
    }

    pub fn lm_config(mut self, cfg: Qwen25VlLmConfig) -> Self {
        self.lm_config = Some(cfg);
        self
    }

    pub fn inline_lm_weights(mut self, weights: HashMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        self.inline_lm_weights = Some(weights);
        self
    }

    pub fn inline_mmproj(
        mut self,
        cfg: crate::vision::MmProjConfig,
        weights: crate::vision::MmProjWeights,
    ) -> Self {
        self.inline_mmproj = Some((cfg, weights));
        self
    }

    pub fn aif_dynamics_mode(mut self, mode: AifDynamicsMode) -> Self {
        self.aif_dynamics_mode = mode;
        self
    }

    pub fn build(self) -> Result<Qwen25VlRunner> {
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(4096);

        let lm_cfg = match self.lm_config {
            Some(c) => c,
            None if self
                .weights
                .as_ref()
                .and_then(|p| p.extension().and_then(|s| s.to_str()))
                == Some("gguf") =>
            {
                load_lm_config_from_gguf(self.weights.as_ref().unwrap())?.0
            }
            None => {
                let hf = self.hf_config.ok_or_else(|| {
                    anyhow::anyhow!("non-GGUF weights require .hf_config(...) or .lm_config(...)")
                })?;
                crate::config::config_from_hf_json(&hf)?.lm
            }
        };

        let mut lm_builder = Qwen3RunnerBuilder::default()
            .device(device)
            .config(Qwen3ConfigSource::Explicit(lm_cfg.lm.clone()))
            .max_seq(max_seq);
        if let Some(s) = self.sample {
            lm_builder = lm_builder.sample(s);
        }
        // ModelFlow owns prefill/decode; skip Qwen3Runner to avoid a second packed-GGUF session.
        let _ = lm_builder;
        let lm = None;
        if self.weights.is_none() && self.inline_lm_weights.is_none() {
            bail!("Qwen25VlRunnerBuilder: .weights(...) or .inline_lm_weights(...) required");
        }

        let vision = match self.inline_mmproj {
            Some((cfg, weights)) => Some(Qwen25VlVisionEncoder::from_parts(cfg, weights, 4, 4)?),
            None => self
                .mmproj
                .map(|p| {
                    let path_str = p.to_str().context("mmproj path utf8")?;
                    let loader = GgufLoader::from_file(path_str)?;
                    let cfg = crate::vision::MmProjConfig::from_gguf(loader.file())?;
                    let side = cfg.image_size.max(cfg.patch_size * cfg.n_merge * 2);
                    Qwen25VlVisionEncoder::from_mmproj_device(p, side, side, device)
                })
                .transpose()
                .with_context(|| "load mmproj")?,
        };

        let lm_weights = self.inline_lm_weights.unwrap_or_default();

        Ok(Qwen25VlRunner {
            lm,
            lm_cfg,
            vision,
            device,
            max_seq,
            lm_weights,
            lm_weights_cache: HashMap::new(),
            weights_path: self.weights,
            decode_cache: None,
            mrope_positions: None,
            lm_head_cached: None,
            embed_cached: None,
            last_prefill_hidden: None,
            last_assembled_prefill: None,
            last_prefill_qk_layers: None,
            vision_key_span: None,
            aif_blocked_keys: None,
            aif_dynamics_mode: self.aif_dynamics_mode,
            last_prefill_logits: None,
            last_decode_qk_layers: None,
        })
    }
}

pub struct Qwen25VlRunner {
    lm: Option<Qwen3Runner>,
    lm_cfg: Qwen25VlLmConfig,
    vision: Option<Qwen25VlVisionEncoder>,
    device: Device,
    max_seq: usize,
    lm_weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// Dequantized LM weights for graph builds (Q4 GGUF and safetensors).
    lm_weights_cache: HashMap<String, ArcF32Tensor>,
    weights_path: Option<PathBuf>,
    decode_cache: Option<KvCacheState>,
    mrope_positions: Option<Vec<[usize; 4]>>,
    lm_head_cached: Option<(Vec<f32>, Vec<usize>)>,
    embed_cached: Option<Vec<f32>>,
    last_prefill_hidden: Option<Vec<f32>>,
    /// Cached multimodal prefill for native AIF probe.
    last_assembled_prefill: Option<MultimodalPrefill>,
    /// Per-layer (Q, K) side outputs from the last probe prefill, when enabled.
    last_prefill_qk_layers: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    vision_key_span: Option<VisionKeySpan>,
    /// When set, decode uses `MaskKind::Custom` and blocks these KV indices.
    aif_blocked_keys: Option<Vec<usize>>,
    aif_dynamics_mode: AifDynamicsMode,
    last_prefill_logits: Option<Vec<f32>>,
    /// Per-layer (Q, K) from the last decode-step AIF probe (when enabled).
    last_decode_qk_layers: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

impl Qwen25VlRunner {
    pub fn builder() -> Qwen25VlRunnerBuilder {
        Qwen25VlRunnerBuilder::default()
    }

    pub fn lm_config(&self) -> &Qwen25VlLmConfig {
        &self.lm_cfg
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn device(&self) -> Device {
        self.lm.as_ref().map(|l| l.device()).unwrap_or(self.device)
    }

    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.lm
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("predict_logits requires .weights(...) LM GGUF"))?
            .predict_logits(prompt_ids)
    }

    pub fn generate_text(&mut self, prompt_ids: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        self.lm
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("generate_text requires .weights(...) LM GGUF"))?
            .generate_stoppable(prompt_ids, max_tokens, |_| true)
    }

    pub fn encode_image(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionEncodeOutput> {
        self.encode_image_resized(rgb, w, h, None, None)
    }

    /// Vision encode at HF-resized dimensions when `target_w/h` are set.
    pub fn encode_image_resized(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        target_w: Option<usize>,
        target_h: Option<usize>,
    ) -> Result<VisionEncodeOutput> {
        let enc = self
            .vision
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("encode_image requires .mmproj(...) at build time"))?;
        enc.encode_rgb_resized(rgb, w, h, target_w, target_h)
    }

    /// Last-position hidden state from the most recent [`Self::prefill_from_assembled`].
    pub fn last_prefill_hidden(&self) -> Option<&[f32]> {
        self.last_prefill_hidden.as_deref()
    }

    /// Token embedding table (`vocab * hidden`) — cached from inline weights or GGUF.
    pub fn embed_table(&mut self) -> Result<Vec<f32>> {
        self.ensure_embed_cached()
    }

    /// Multimodal prefill: encode image, assemble prompt, run mRoPE LM prefill.
    pub fn prefill_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Vec<f32>> {
        let vision = self.encode_image(rgb, w, h)?;
        let n_embd = self.lm_cfg.lm.hidden_size;
        let embed = self.ensure_embed_cached()?;
        let mm = MultimodalPrompt {
            prompt,
            vision: &vision,
        };
        let prefill = mm.assemble(tokenizer, &embed, n_embd, 0)?;
        self.prefill_from_assembled(prefill)
    }

    /// Greedy decode after [`Self::prefill_multimodal`] or [`Self::prefill_from_assembled`].
    pub fn generate_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        max_new_tokens: usize,
        mut tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        self.clear_aif_decode();
        self.generate_multimodal_inner(
            prompt,
            rgb,
            w,
            h,
            max_new_tokens,
            &mut tokenizer,
            stop_token,
        )
    }

    /// Fig. 6 — modulated greedy decode after a paper [`AifProbe`].
    pub fn generate_multimodal_aif(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        max_new_tokens: usize,
        mut tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
        stop_token: Option<u32>,
        config: &AifConfig,
    ) -> Result<Vec<u32>> {
        let logits = self.prefill_multimodal(prompt, rgb, w, h, &mut tokenizer)?;
        self.apply_aif_config(config)?;
        self.generate_from_prefill_logits(logits, max_new_tokens, stop_token)
    }

    /// Fig. 6 with native RLX probe (Eq. 2 prefill or decode-step dynamics, no Python).
    pub fn generate_multimodal_aif_native(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        max_new_tokens: usize,
        mut tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        let logits = if self.aif_dynamics_mode == AifDynamicsMode::DecodeStep {
            self.prefill_multimodal(prompt, rgb, w, h, &mut tokenizer)?
        } else {
            self.prefill_multimodal_aif_probe(prompt, rgb, w, h, &mut tokenizer)?
        };
        let probe = self.probe_aif_native()?;
        self.apply_aif_probe(&probe)?;
        self.generate_from_prefill_logits(logits, max_new_tokens, stop_token)
    }

    pub fn set_aif_dynamics_mode(&mut self, mode: AifDynamicsMode) {
        self.aif_dynamics_mode = mode;
    }

    pub fn aif_dynamics_mode(&self) -> AifDynamicsMode {
        self.aif_dynamics_mode
    }

    /// Multimodal prefill that also exports per-layer Q/K side outputs for AIF.
    pub fn prefill_multimodal_aif_probe(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        tokenizer: &mut impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Vec<f32>> {
        let vision = self.encode_image(rgb, w, h)?;
        let n_embd = self.lm_cfg.lm.hidden_size;
        let embed = self.ensure_embed_cached()?;
        let mm = MultimodalPrompt {
            prompt,
            vision: &vision,
        };
        let prefill = mm.assemble(tokenizer, &embed, n_embd, 0)?;
        self.prefill_from_assembled_probe(prefill)
    }

    /// Build [`AifProbe`] from the last multimodal prefill (native CPU replay or graph Q/K).
    pub fn probe_aif_native(&mut self) -> Result<AifProbe> {
        match self.aif_dynamics_mode {
            AifDynamicsMode::PrefillV2t => self.probe_aif_native_prefill(),
            AifDynamicsMode::DecodeStep => self.probe_aif_native_decode_step(),
        }
    }

    fn probe_aif_native_prefill(&mut self) -> Result<AifProbe> {
        let prefill = self
            .last_assembled_prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("probe_aif_native: run prefill first"))?
            .clone();
        let vision = self
            .vision_key_span
            .ok_or_else(|| anyhow::anyhow!("probe_aif_native: no vision tokens in prefill"))?;
        let seq = prefill.hidden.len() / self.lm_cfg.lm.hidden_size;

        if let Some((ref q_layers, ref k_layers)) = self.last_prefill_qk_layers {
            let q_refs: Vec<&[f32]> = q_layers.iter().map(|v| v.as_slice()).collect();
            let k_refs: Vec<&[f32]> = k_layers.iter().map(|v| v.as_slice()).collect();
            let dynamics =
                dynamics_from_graph_qk_layers(&q_refs, &k_refs, vision, seq, &self.lm_cfg.lm)?;
            return Ok(AifProbe::build(dynamics));
        }

        if !self.lm_weights.is_empty() {
            return native_prefill_probe(&NativePrefillProbeInputs {
                cfg: &self.lm_cfg,
                weights: &self.lm_weights,
                hidden: &prefill.hidden,
                mrope_sections: &prefill.mrope_sections,
                vision,
                seq,
            });
        }

        bail!(
            "probe_aif_native: run prefill_from_assembled_probe first, or provide inline LM weights"
        )
    }

    fn probe_aif_native_decode_step(&mut self) -> Result<AifProbe> {
        let vision = self
            .vision_key_span
            .ok_or_else(|| anyhow::anyhow!("probe_aif_native: no vision tokens in prefill"))?;
        let logits = self
            .last_prefill_logits
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("probe_aif_native decode_step: run prefill first"))?;
        let probe_token = argmax_token(logits);
        self.run_decode_probe_qk(probe_token)?;
        let (q_layers, k_layers) = self
            .last_decode_qk_layers
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode-step probe missing Q/K side outputs"))?;
        let q_refs: Vec<&[f32]> = q_layers.iter().map(|v| v.as_slice()).collect();
        let k_refs: Vec<&[f32]> = k_layers.iter().map(|v| v.as_slice()).collect();
        let dynamics =
            dynamics_from_graph_qk_decode_step(&q_refs, &k_refs, vision, &self.lm_cfg.lm)?;
        Ok(AifProbe::build(dynamics))
    }

    /// One decode forward with Q/K side outputs; does not mutate the live KV cache.
    pub fn run_decode_probe_qk(&mut self, token_id: u32) -> Result<()> {
        let cache = self
            .decode_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("run_decode_probe_qk: prefill first"))?;
        let past_seq = cache.past_len;
        let layers_k = cache.layers_k.clone();
        let layers_v = cache.layers_v.clone();

        let abs_pos = self
            .mrope_positions
            .as_ref()
            .and_then(|s| s.get(past_seq.saturating_sub(1)))
            .map(|sec| sec[0])
            .unwrap_or(past_seq);
        let (cos, sin) = mrope_decode_feeds(&self.lm_cfg, abs_pos + 1);

        self.ensure_lm_weights_cache()?;
        let mut weight_loader = ArcCacheLoader::new(&self.lm_weights_cache);
        let opts = Qwen3DecodeOpts {
            batch: 1,
            past_seq,
            export_qk: true,
            use_custom_mask: false,
            ..Default::default()
        };
        let built = build_qwen3_decode_built(&self.lm_cfg.lm, &mut weight_loader, &opts)?;
        let mut compiled = compile_built(built, self.device)?;

        let input_ids = [token_id as f32];
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", input_ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ];
        let key_strs: Vec<String> = (0..self.lm_cfg.lm.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        for i in 0..self.lm_cfg.lm.num_hidden_layers {
            inputs.push((&key_strs[2 * i], layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], layers_v[i].as_slice()));
        }

        let outputs = compiled.run(&inputs);
        let n_layers = self.lm_cfg.lm.num_hidden_layers;
        let layout = crate::decode_side::DecodeSideLayout {
            n_layers,
            export_qk: true,
        };
        anyhow::ensure!(
            outputs.len() == layout.expected_outputs(),
            "decode probe produced {} outputs, expected {}",
            outputs.len(),
            layout.expected_outputs()
        );
        let (_logits, _new_k, _new_v, qk) = layout.parse_kv_qk(outputs.into_iter())?;
        self.last_decode_qk_layers = qk;
        Ok(())
    }

    /// Same as [`Self::generate_multimodal_aif`] with a raw probe.
    pub fn generate_multimodal_aif_paper(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        max_new_tokens: usize,
        mut tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
        stop_token: Option<u32>,
        probe: &AifProbe,
    ) -> Result<Vec<u32>> {
        self.generate_multimodal_aif(
            prompt,
            rgb,
            w,
            h,
            max_new_tokens,
            &mut tokenizer,
            stop_token,
            &AifConfig::from(probe),
        )
    }

    pub fn apply_aif_probe(&mut self, probe: &AifProbe) -> Result<()> {
        self.apply_aif_config(&AifConfig::from(probe))
    }

    pub fn apply_aif_config(&mut self, config: &AifConfig) -> Result<()> {
        let span = self
            .vision_key_span
            .ok_or_else(|| anyhow::anyhow!("apply_aif_config: run prefill first"))?;
        self.aif_blocked_keys = Some(config.blocked_keys(span));
        Ok(())
    }

    #[deprecated(note = "use apply_aif_config")]
    pub fn apply_aif_lite(&mut self, config: &AifConfig) -> Result<()> {
        self.apply_aif_config(config)
    }

    /// Greedy decode with a cached or on-disk AIF probe for this sample id.
    pub fn generate_multimodal_aif_sample(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        max_new_tokens: usize,
        mut tokenizer: impl FnMut(&str) -> Result<Vec<u32>>,
        stop_token: Option<u32>,
        probe_dir: &Path,
        sample_id: &str,
    ) -> Result<Vec<u32>> {
        let probe = crate::probe::load_probe_sample(probe_dir, sample_id)?;
        self.generate_multimodal_aif_paper(
            prompt,
            rgb,
            w,
            h,
            max_new_tokens,
            &mut tokenizer,
            stop_token,
            &probe,
        )
    }

    /// Visual KV span from the last multimodal prefill.
    pub fn vision_key_span(&self) -> Option<VisionKeySpan> {
        self.vision_key_span
    }

    /// Enable AIF decode masking for subsequent [`Self::decode_step`] calls.
    pub fn set_aif_blocked_keys(&mut self, blocked: Vec<usize>) {
        self.aif_blocked_keys = Some(blocked);
    }

    pub fn clear_aif_decode(&mut self) {
        self.aif_blocked_keys = None;
    }

    fn generate_multimodal_inner(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        w: usize,
        h: usize,
        max_new_tokens: usize,
        tokenizer: &mut impl FnMut(&str) -> Result<Vec<u32>>,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        let logits = self.prefill_multimodal(prompt, rgb, w, h, tokenizer)?;
        self.generate_from_prefill_logits(logits, max_new_tokens, stop_token)
    }

    fn generate_from_prefill_logits(
        &mut self,
        mut logits: Vec<f32>,
        max_new_tokens: usize,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for _ in 0..max_new_tokens {
            let next = argmax_token(&logits);
            if stop_token == Some(next) {
                break;
            }
            out.push(next);
            logits = self.decode_step(next)?;
        }
        Ok(out)
    }

    pub fn prefill_from_assembled(&mut self, prefill: MultimodalPrefill) -> Result<Vec<f32>> {
        let seq = prefill.seq.len();
        self.prefill_from_assembled_opts(prefill, Qwen25VlPrefillOpts::vlm_prefill(1, seq))
    }

    /// Prefill with optional AIF Q/K side outputs (`opts.export_aif_qk`).
    pub fn prefill_from_assembled_probe(&mut self, prefill: MultimodalPrefill) -> Result<Vec<f32>> {
        let seq = prefill.seq.len();
        let opts = Qwen25VlPrefillOpts::vlm_prefill_aif_probe(1, seq);
        self.prefill_from_assembled_opts(prefill, opts)
    }

    fn prefill_from_assembled_opts(
        &mut self,
        prefill: MultimodalPrefill,
        mut opts: Qwen25VlPrefillOpts,
    ) -> Result<Vec<f32>> {
        let seq = prefill.seq.len();
        if seq == 0 {
            bail!("multimodal prefill: empty sequence");
        }
        if seq > self.max_seq {
            bail!("multimodal seq {seq} exceeds max_seq {}", self.max_seq);
        }
        opts.batch = 1;
        opts.seq = seq;
        self.mrope_positions = Some(prefill.mrope_sections.clone());
        if prefill.n_vision_tokens > 0 {
            self.vision_key_span = Some(VisionKeySpan {
                start: prefill.vision_start_idx,
                end: prefill.vision_start_idx + prefill.n_vision_tokens,
            });
        } else {
            self.vision_key_span = None;
        }

        self.ensure_lm_weights_cache()?;
        let mut weight_loader = ArcCacheLoader::new(&self.lm_weights_cache);
        let built = build_qwen25_vl_prefill_mrope_built(
            &self.lm_cfg,
            &mut weight_loader,
            &opts,
            Some(&prefill.mrope_sections),
        )?;
        let mut compiled = compile_built(built, self.device)?;

        let head_half = self.lm_cfg.head_half();
        let (rope_cos, rope_sin) =
            mrope_prefill_feeds(&self.lm_cfg, seq, Some(&prefill.mrope_sections), head_half);

        let outputs = compiled.run(&[
            ("prefill_hidden", prefill.hidden.as_slice()),
            ("rope_cos", rope_cos.as_slice()),
            ("rope_sin", rope_sin.as_slice()),
        ]);

        let n_layers = self.lm_cfg.lm.num_hidden_layers;
        let layout = crate::prefill_side::PrefillSideLayout {
            n_layers,
            export_qk: opts.export_aif_qk,
        };
        anyhow::ensure!(
            outputs.len() == layout.expected_outputs(),
            "prefill produced {} outputs, expected {}",
            outputs.len(),
            layout.expected_outputs()
        );
        let (hidden, layers_k, layers_v, qk) = layout.parse_kv_qk(outputs.into_iter())?;
        if let Some((q_layers, k_layers)) = qk {
            self.last_prefill_qk_layers = Some((q_layers, k_layers));
        } else {
            self.last_prefill_qk_layers = None;
        }
        let h = self.lm_cfg.lm.hidden_size;
        let hidden_last = hidden[hidden.len().saturating_sub(h)..].to_vec();
        self.last_prefill_hidden = Some(hidden_last);
        self.last_assembled_prefill = Some(prefill);
        let logits = self.lm_head_logits(&hidden)?;
        let kv = KvCacheState {
            past_len: seq,
            layers_kv_base: vec![0; layers_k.len()],
            layers_k,
            layers_v,
        };
        self.decode_cache = Some(kv);
        self.last_decode_qk_layers = None;

        let vocab = self.lm_cfg.lm.vocab_size;
        let logits_vec = logits[..vocab.min(logits.len())].to_vec();
        self.last_prefill_logits = Some(logits_vec.clone());
        Ok(logits_vec)
    }

    pub fn prefill_from_token_ids(
        &mut self,
        input_ids: &[u32],
        vision_start_idx: usize,
        n_vision: usize,
        vision: &VisionEncodeOutput,
        text_start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_embd = self.lm_cfg.lm.hidden_size;
        let embed = self.ensure_embed_cached()?;
        let prefill = crate::multimodal::assemble_from_token_ids(
            input_ids,
            vision_start_idx,
            n_vision,
            vision,
            &embed,
            n_embd,
            text_start_pos,
        )?;
        self.prefill_from_assembled(prefill)
    }

    /// Like [`Self::prefill_from_token_ids`] but exports Q/K side outputs for native AIF probe.
    pub fn prefill_from_token_ids_probe(
        &mut self,
        input_ids: &[u32],
        vision_start_idx: usize,
        n_vision: usize,
        vision: &VisionEncodeOutput,
        text_start_pos: usize,
    ) -> Result<Vec<f32>> {
        let n_embd = self.lm_cfg.lm.hidden_size;
        let embed = self.ensure_embed_cached()?;
        let prefill = crate::multimodal::assemble_from_token_ids(
            input_ids,
            vision_start_idx,
            n_vision,
            vision,
            &embed,
            n_embd,
            text_start_pos,
        )?;
        self.prefill_from_assembled_probe(prefill)
    }

    pub fn decode_step(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let blocked = self.aif_blocked_keys.clone();
        self.decode_step_masked(token_id, blocked.as_deref())
    }

    fn decode_step_masked(
        &mut self,
        token_id: u32,
        blocked_keys: Option<&[usize]>,
    ) -> Result<Vec<f32>> {
        let past_seq = self
            .decode_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_step: call prefill_from_assembled first"))?
            .past_len;
        let layers_k = self.decode_cache.as_ref().unwrap().layers_k.clone();
        let layers_v = self.decode_cache.as_ref().unwrap().layers_v.clone();

        let abs_pos = self
            .mrope_positions
            .as_ref()
            .and_then(|s| s.get(past_seq.saturating_sub(1)))
            .map(|sec| sec[0])
            .unwrap_or(past_seq);
        let (cos, sin) = mrope_decode_feeds(&self.lm_cfg, abs_pos + 1);

        self.ensure_lm_weights_cache()?;
        let mut weight_loader = ArcCacheLoader::new(&self.lm_weights_cache);
        let use_custom_mask = blocked_keys.is_some_and(|b| !b.is_empty());
        let opts = Qwen3DecodeOpts {
            batch: 1,
            past_seq,
            use_custom_mask,
            ..Default::default()
        };
        let built = build_qwen3_decode_built(&self.lm_cfg.lm, &mut weight_loader, &opts)?;
        let mut compiled = compile_built(built, self.device)?;

        let input_ids = [token_id as f32];
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", input_ids.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ];
        let mask_row;
        if let Some(blocked) = blocked_keys {
            mask_row = decode_mask_row_causal(past_seq, blocked);
            inputs.push(("mask", mask_row.as_slice()));
        }
        let key_strs: Vec<String> = (0..self.lm_cfg.lm.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        for i in 0..self.lm_cfg.lm.num_hidden_layers {
            inputs.push((&key_strs[2 * i], layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], layers_v[i].as_slice()));
        }

        let outputs = compiled.run(&inputs);
        let (logits, new_k, new_v) =
            split_decode_logits_kv(outputs, self.lm_cfg.lm.num_hidden_layers)?;
        let cache_mut = self.decode_cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;

        let vocab = self.lm_cfg.lm.vocab_size;
        Ok(logits[..vocab.min(logits.len())].to_vec())
    }

    pub fn assemble_multimodal_prefill(&self, prefill: &MultimodalPrefill) -> Result<()> {
        let _ = prefill;
        bail!("use MultimodalPrompt::assemble + prefill_from_assembled")
    }

    fn ensure_lm_weights_cache(&mut self) -> Result<()> {
        if !self.lm_weights_cache.is_empty() {
            return Ok(());
        }
        if !self.lm_weights.is_empty() {
            for (key, (data, shape)) in &self.lm_weights {
                self.lm_weights_cache
                    .insert(key.clone(), (Arc::new(data.clone()), shape.clone()));
            }
            return Ok(());
        }
        let path = self
            .weights_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("LM weights path required for multimodal prefill"))?;
        let file = rlx_core::weights::pick_default(path)?;
        let mut loader = rlx_core::open_weight_loader(&file)
            .with_context(|| format!("open LM weights {file:?}"))?;
        // ModelFlow prefill/decode uses F32 matmuls; force-dequant K-quants like rlx-lfm.
        let mut map = WeightMap::from_weight_loader_dequant_all(loader.as_mut())
            .with_context(|| format!("dequant LM weights {file:?}"))?;
        let keys: Vec<String> = map.keys().map(str::to_string).collect();
        for key in keys {
            let (data, shape) = map.take(&key)?;
            let canonical =
                rlx_core::weight_loader::gguf_to_hf_name(&key).unwrap_or_else(|| key.clone());
            self.lm_weights_cache
                .insert(canonical, (Arc::new(data), shape));
        }
        Ok(())
    }

    fn lm_head_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        let h = self.lm_cfg.lm.hidden_size;
        let vocab = self.lm_cfg.lm.vocab_size;
        anyhow::ensure!(
            hidden.len() >= h,
            "hidden len {} < hidden_size {h}",
            hidden.len()
        );
        let h_row = &hidden[hidden.len() - h..];
        self.ensure_lm_head_cached()?;
        let (w, shape) = self.lm_head_cached.as_ref().context("lm_head cache")?;
        let (out, inp) = if shape.len() == 2 && shape[1] == h {
            (shape[0], shape[1])
        } else if shape.len() == 2 && shape[0] == h {
            let out = shape[1];
            let mut logits = vec![0f32; out];
            for v in 0..out {
                logits[v] = h_row
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * w[i * out + v])
                    .sum();
            }
            return Ok(logits);
        } else if shape == &[vocab, h] {
            (vocab, h)
        } else if shape == &[h, vocab] {
            let mut logits = vec![0f32; vocab];
            for v in 0..vocab {
                logits[v] = h_row
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * w[i * vocab + v])
                    .sum();
            }
            return Ok(logits);
        } else {
            bail!("unexpected lm head shape {:?}", shape);
        };
        let mut logits = vec![0f32; out];
        for v in 0..out {
            logits[v] = h_row
                .iter()
                .enumerate()
                .map(|(i, &x)| x * w[v * inp + i])
                .sum();
        }
        Ok(logits)
    }

    fn ensure_embed_cached(&mut self) -> Result<Vec<f32>> {
        if let Some(ref e) = self.embed_cached {
            return Ok(e.clone());
        }
        if let Some((data, shape)) = self.lm_weights.get("model.embed_tokens.weight") {
            let n_embd = self.lm_cfg.lm.hidden_size;
            anyhow::ensure!(
                shape.len() == 2 && (shape[1] == n_embd || shape[0] == n_embd),
                "embed shape {:?}",
                shape
            );
            self.embed_cached = Some(data.clone());
            return Ok(data.clone());
        }
        self.ensure_lm_weights_cache()?;
        let (data, shape) = self
            .lm_weights_cache
            .get("model.embed_tokens.weight")
            .ok_or_else(|| anyhow::anyhow!("model.embed_tokens.weight missing from LM cache"))?;
        let n_embd = self.lm_cfg.lm.hidden_size;
        anyhow::ensure!(
            shape.len() == 2 && shape[1] == n_embd,
            "embed shape {:?}",
            shape
        );
        let embed = data.as_ref().clone();
        self.embed_cached = Some(embed.clone());
        Ok(embed)
    }

    fn ensure_lm_head_cached(&mut self) -> Result<()> {
        if self.lm_head_cached.is_some() {
            return Ok(());
        }
        for key in ["lm_head.weight", "model.embed_tokens.weight"] {
            if let Some((data, shape)) = self.lm_weights.get(key) {
                self.lm_head_cached = Some((data.clone(), shape.clone()));
                return Ok(());
            }
        }
        self.ensure_lm_weights_cache()?;
        for key in ["lm_head.weight", "model.embed_tokens.weight"] {
            if let Some((data, shape)) = self.lm_weights_cache.get(key) {
                self.lm_head_cached = Some((data.as_ref().clone(), shape.clone()));
                return Ok(());
            }
        }
        bail!("lm_head_logits: missing lm_head or embed weights")
    }

    pub fn assert_paper_checkpoint(path: &Path) -> Result<Qwen25VlLmConfig> {
        let (cfg, _) = load_lm_config_from_gguf(path)?;
        let h = cfg.lm.hidden_size;
        let layers = cfg.lm.num_hidden_layers;
        if h < 3000 || layers < 20 {
            eprintln!(
                "[rlx-qwen25-vl] warning: {path:?} looks smaller than Qwen2.5-VL-7B \
                 (hidden={h}, layers={layers})"
            );
        }
        Ok(cfg)
    }
}

fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
