// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Combined Ministral / Mistral Medium VL runner (Pixtral mmproj + mistral3/4 LM).

use crate::encoder::PixtralVisionEncoder;
use anyhow::{Context, Result, anyhow, ensure};
use rlx_cli::LmRunner;
use rlx_core::WeightMap;
use rlx_core::autoregressive::{KvCacheState, split_decode_logits_kv};
use rlx_core::flow_util::compile_built;
use rlx_core::weight_loader::{ArcCacheLoader, ArcF32Tensor};
use rlx_llama32::Llama32Config;
use rlx_llama32::flow::{
    Llama32DecodeOpts, Llama32PrefillOpts, build_llama32_decode_built, build_llama32_prefill_built,
};
use rlx_mistral::MistralRunner;
use rlx_qwen35::encode_prompt_auto;
use rlx_runtime::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const MEDIA_MARKER: &str = "<__media__>";
pub const IMAGE_MARKER: &str = "<image>";

#[derive(Debug, Clone, Default)]
pub struct MistralVlRunnerBuilder {
    weights: Option<PathBuf>,
    mmproj: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
}

impl MistralVlRunnerBuilder {
    pub fn weights(mut self, p: impl Into<PathBuf>) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn mmproj(mut self, p: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(p.into());
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

    pub fn build(self) -> Result<MistralVlRunner> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow!("MistralVlRunner: .weights(...) required"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(4096);
        // Accelerators run the LM **packed** (K-quant `input_embeddings`), so the
        // 24B weights never dequant to F32 (~96 GB). CPU has no packed matmul
        // path, so it falls back to the F32 embed-splice (small models only).
        let packed_lm = matches!(
            device,
            Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm
        );
        let mut lm_b = MistralRunner::builder()
            .weights(&weights)
            .device(device)
            .max_seq(max_seq)
            // A paired Pixtral mmproj confirms this is a Mistral-3 VL model even
            // when the LM GGUF is tagged with the legacy `llama` arch.
            .accept_llama_arch(self.mmproj.is_some());
        if !packed_lm {
            lm_b = lm_b.packed_weights(false);
        }
        let lm = lm_b
            .build()
            .with_context(|| format!("MistralVlRunner: load LM {weights:?}"))?;

        let vision = match self.mmproj {
            Some(mp) => Some(
                PixtralVisionEncoder::from_mmproj_on_device(&mp, device)
                    .with_context(|| format!("MistralVlRunner: load mmproj {mp:?}"))?,
            ),
            None => None,
        };

        Ok(MistralVlRunner {
            lm,
            vision,
            weights_path: weights,
            device,
            max_seq,
            packed_lm,
            lm_weights_cache: HashMap::new(),
            embed_cached: None,
            decode_cache: None,
            llama_cfg: None,
        })
    }
}

pub struct MistralVlRunner {
    lm: MistralRunner,
    vision: Option<PixtralVisionEncoder>,
    weights_path: PathBuf,
    device: Device,
    max_seq: usize,
    /// LM runs packed on-device (Metal/MLX/CUDA/ROCm) via the embed-splice
    /// prefill; `false` → CPU F32 embed path.
    packed_lm: bool,
    lm_weights_cache: HashMap<String, ArcF32Tensor>,
    embed_cached: Option<Vec<f32>>,
    decode_cache: Option<KvCacheState>,
    llama_cfg: Option<Llama32Config>,
}

impl MistralVlRunner {
    pub fn builder() -> MistralVlRunnerBuilder {
        MistralVlRunnerBuilder::default()
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    fn llama_cfg(&mut self) -> &Llama32Config {
        if self.llama_cfg.is_none() {
            self.llama_cfg = Some(self.lm.inner().config().clone());
        }
        self.llama_cfg.as_ref().unwrap()
    }

    fn ensure_lm_weights_cache(&mut self) -> Result<()> {
        if !self.lm_weights_cache.is_empty() {
            return Ok(());
        }
        let file = rlx_core::weights::pick_default(&self.weights_path)?;
        let mut loader = rlx_core::open_weight_loader(&file)
            .with_context(|| format!("open LM weights {file:?}"))?;
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

    fn ensure_embed_cached(&mut self) -> Result<Vec<f32>> {
        if let Some(e) = &self.embed_cached {
            return Ok(e.clone());
        }
        self.ensure_lm_weights_cache()?;
        let key_candidates = [
            "token_embd.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ];
        let mut found = None;
        for k in key_candidates {
            if let Some((data, _shape)) = self.lm_weights_cache.get(k) {
                found = Some(data.as_ref().clone());
                break;
            }
        }
        let data = found.ok_or_else(|| {
            anyhow!("MistralVlRunner: token embedding table not found in LM weights")
        })?;
        self.embed_cached = Some(data.clone());
        Ok(data)
    }

    fn assemble_hidden(
        &mut self,
        prompt: &str,
        vision_embd: &[f32],
        n_vision: usize,
        tokenizer: Option<&Path>,
    ) -> Result<Vec<f32>> {
        let n_embd = self.llama_cfg().hidden_size;
        ensure!(
            vision_embd.len() == n_vision * n_embd,
            "vision dim mismatch: {} vs {}×{}",
            vision_embd.len(),
            n_vision,
            n_embd
        );
        let prompt = normalize_prompt(prompt);
        let parts: Vec<&str> = prompt.split(MEDIA_MARKER).collect();
        ensure!(
            parts.len() == 2,
            "prompt must contain exactly one `{MEDIA_MARKER}`"
        );
        let weights = self.weights_path.clone();
        let tok = |s: &str| -> Result<Vec<u32>> { encode_prompt_auto(&weights, tokenizer, s) };
        let before = tok(parts[0])?;
        let after = tok(parts[1])?;
        let emb = self.ensure_embed_cached()?;
        let vocab = emb.len() / n_embd;

        let mut seq = Vec::new();
        seq.extend_from_slice(&before);
        let vision_start = seq.len();
        seq.extend(std::iter::repeat_n(0u32, n_vision));
        seq.extend_from_slice(&after);

        let mut hidden = Vec::with_capacity(seq.len() * n_embd);
        for &tok_id in &seq {
            let row = tok_id as usize;
            ensure!(row < vocab, "token {tok_id} out of vocab {vocab}");
            let off = row * n_embd;
            hidden.extend_from_slice(&emb[off..off + n_embd]);
        }
        for t in 0..n_vision {
            let dst = (vision_start + t) * n_embd;
            let src = t * n_embd;
            hidden[dst..dst + n_embd].copy_from_slice(&vision_embd[src..src + n_embd]);
        }
        Ok(hidden)
    }

    fn prefill_from_embeds(&mut self, hidden: &[f32], seq: usize) -> Result<Vec<f32>> {
        ensure!(seq > 0, "empty multimodal sequence");
        ensure!(
            seq <= self.max_seq,
            "multimodal seq {seq} exceeds max_seq {}",
            self.max_seq
        );
        let n_embd = {
            let c = self.llama_cfg();
            c.hidden_size
        };
        ensure!(hidden.len() == seq * n_embd);
        self.ensure_lm_weights_cache()?;
        let cfg = self.llama_cfg().clone();
        let n_layers = cfg.num_hidden_layers;
        let vocab = cfg.vocab_size;
        let mut weight_loader = ArcCacheLoader::new(&self.lm_weights_cache);
        let mut opts = Llama32PrefillOpts::static_prefill(1, seq);
        opts.inputs_embeds = true;
        opts.with_kv_outputs = true;
        opts.with_lm_head = true;
        opts.last_logits_only = true;
        let built = build_llama32_prefill_built(&cfg, &mut weight_loader, &opts)?;
        let mut compiled = compile_built(built, self.device)?;
        let outputs = compiled.run(&[("inputs_embeds", hidden)]);
        ensure!(
            outputs.len() > 2 * n_layers,
            "prefill produced {} outputs, need >= {} (1 logits + 2/layer)",
            outputs.len(),
            1 + 2 * n_layers
        );
        let logits = outputs[0].clone();
        let kv_start = if outputs.len() == 1 + 2 * n_layers {
            1
        } else {
            2
        };
        let mut layers_k = Vec::with_capacity(n_layers);
        let mut layers_v = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers_k.push(outputs[kv_start + 2 * i].clone());
            layers_v.push(outputs[kv_start + 2 * i + 1].clone());
        }
        self.decode_cache = Some(KvCacheState {
            past_len: seq,
            layers_kv_base: vec![0; n_layers],
            layers_k,
            layers_v,
        });
        Ok(logits[..vocab.min(logits.len())].to_vec())
    }

    fn decode_step(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let past_seq = self
            .decode_cache
            .as_ref()
            .ok_or_else(|| anyhow!("decode_step: call multimodal prefill first"))?
            .past_len;
        let layers_k = self.decode_cache.as_ref().unwrap().layers_k.clone();
        let layers_v = self.decode_cache.as_ref().unwrap().layers_v.clone();
        self.ensure_lm_weights_cache()?;
        let cfg = self.llama_cfg().clone();
        let n_layers = cfg.num_hidden_layers;
        let vocab = cfg.vocab_size;
        let mut weight_loader = ArcCacheLoader::new(&self.lm_weights_cache);
        let opts = Llama32DecodeOpts {
            batch: 1,
            past_seq,
            dynamic_past: false,
            use_custom_mask: false,
            profile: None,
        };
        let built = build_llama32_decode_built(&cfg, &mut weight_loader, &opts)?;
        let mut compiled = compile_built(built, self.device)?;
        let input_ids = [token_id as f32];
        let mut inputs: Vec<(&str, &[f32])> = vec![("input_ids", input_ids.as_slice())];
        let key_strs: Vec<String> = (0..n_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        for i in 0..n_layers {
            inputs.push((&key_strs[2 * i], layers_k[i].as_slice()));
            inputs.push((&key_strs[2 * i + 1], layers_v[i].as_slice()));
        }
        let outputs = compiled.run(&inputs);
        let (logits, new_k, new_v) = split_decode_logits_kv(outputs, n_layers)?;
        let cache_mut = self.decode_cache.as_mut().unwrap();
        cache_mut.past_len = past_seq + 1;
        cache_mut.layers_k = new_k;
        cache_mut.layers_v = new_v;
        Ok(logits[..vocab.min(logits.len())].to_vec())
    }

    pub fn generate_multimodal_rgb(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let proj_dim = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow!("MistralVlRunner: generate_multimodal requires .mmproj(...)"))?
            .config()
            .projector_output_dim;
        let n_embd = {
            let c = self.llama_cfg();
            c.hidden_size
        };
        ensure!(
            proj_dim == n_embd,
            "MistralVlRunner: projector_output_dim {proj_dim} != LM hidden {n_embd}"
        );
        let vision_embd = self
            .vision
            .as_mut()
            .unwrap()
            .encode_rgb(rgb, img_w, img_h)?;
        ensure!(vision_embd.len() % n_embd == 0);
        let n_vision = vision_embd.len() / n_embd;

        // Accelerator: keep the LM packed and splice the vision soft tokens into
        // the packed `input_embeddings` prefill (no 96 GB F32 dequant).
        if self.packed_lm {
            return self.generate_multimodal_packed(
                prompt,
                &vision_embd,
                n_vision,
                tokenizer,
                n_new,
                on_token,
            );
        }

        // CPU fallback: F32 embed assembly + custom prefill/decode.
        let hidden = self.assemble_hidden(prompt, &vision_embd, n_vision, tokenizer)?;
        let seq = hidden.len() / n_embd;
        let mut logits = self.prefill_from_embeds(&hidden, seq)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let next = argmax_u32(&logits);
            out.push(next);
            if !on_token(next) {
                break;
            }
            logits = self.decode_step(next)?;
        }
        Ok(out)
    }

    /// Packed on-device multimodal generation: tokenize the prompt around the
    /// media marker, reserve `n_vision` placeholder ids, register the vision
    /// embeds as a one-shot splice, then run the packed LM `generate` (packed
    /// prefill with `input_embeddings` + packed decode). Weights stay K-quant.
    fn generate_multimodal_packed(
        &mut self,
        prompt: &str,
        vision_embd: &[f32],
        n_vision: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let weights = self.weights_path.clone();
        let prompt = normalize_prompt(prompt);
        let parts: Vec<&str> = prompt.split(MEDIA_MARKER).collect();
        ensure!(
            parts.len() == 2,
            "prompt must contain exactly one `{MEDIA_MARKER}`"
        );
        let before = encode_prompt_auto(&weights, tokenizer, parts[0])?;
        let after = encode_prompt_auto(&weights, tokenizer, parts[1])?;
        let vision_start = before.len();
        let seq_len = before.len() + n_vision + after.len();
        ensure!(
            seq_len <= self.max_seq,
            "multimodal seq {seq_len} exceeds max_seq {}",
            self.max_seq
        );
        // Placeholder ids at the vision positions — their gathered embeds are
        // overwritten by the splice, so any in-vocab id works.
        let mut seq_ids = Vec::with_capacity(seq_len);
        seq_ids.extend_from_slice(&before);
        seq_ids.extend(std::iter::repeat_n(0u32, n_vision));
        seq_ids.extend_from_slice(&after);

        self.lm
            .set_multimodal_embed_override(vision_start, vision_embd.to_vec());
        // `generate_until` (unlike `MistralRunner::generate`) honors the stop
        // callback, so decode halts at EOS instead of always emitting `n_new`.
        let result = self
            .lm
            .inner_mut()
            .generate_until(&seq_ids, n_new, on_token);
        // The packed prefill consumes the splice; a lingering override means the
        // packed path wasn't taken (vision dropped). Read it BEFORE clearing, and
        // always clear so an unconsumed splice (e.g. an error before prefill) can
        // never leak into a later generation on this runner.
        let vision_dropped = self.lm.multimodal_override_pending();
        self.lm.clear_multimodal_embed_override();
        let out = result?;
        ensure!(
            !vision_dropped,
            "MistralVlRunner: packed prefill path not taken on {:?} — vision tokens \
             were dropped (needs a block-quantized GGUF on a packed-capable device)",
            self.device
        );
        Ok(out)
    }
}

impl LmRunner for MistralVlRunner {
    fn family(&self) -> &'static str {
        "mistral-vl"
    }
    fn vocab_size(&self) -> usize {
        self.lm.inner().config().vocab_size
    }
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.lm.predict_logits(prompt_ids)
    }
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        LmRunner::generate(&mut self.lm, prompt_ids, n_new, on_token)
    }
    fn supports_multimodal(&self) -> bool {
        self.has_vision()
    }
    fn generate_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_multimodal_rgb(prompt, rgb, img_w, img_h, tokenizer, n_new, on_token)
    }
}

fn normalize_prompt(prompt: &str) -> String {
    if prompt.contains(MEDIA_MARKER) {
        return prompt.to_string();
    }
    if prompt.contains(IMAGE_MARKER) {
        return prompt.replacen(IMAGE_MARKER, MEDIA_MARKER, 1);
    }
    let mut p = prompt.to_string();
    if !p.is_empty() && !p.ends_with(|c: char| c.is_whitespace()) {
        p.push(' ');
    }
    p.push_str(MEDIA_MARKER);
    p
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

#[cfg(test)]
mod tests {
    use super::{IMAGE_MARKER, MEDIA_MARKER, normalize_prompt};

    #[test]
    fn normalize_prompt_variants() {
        // An explicit media marker is left exactly as-is.
        let p = format!("look {MEDIA_MARKER} here");
        assert_eq!(normalize_prompt(&p), p);
        // `<image>` is rewritten to the media marker (first occurrence only).
        assert_eq!(
            normalize_prompt(&format!("a {IMAGE_MARKER} b")),
            format!("a {MEDIA_MARKER} b")
        );
        // No marker at all → appended with a separating space.
        assert_eq!(
            normalize_prompt("describe"),
            format!("describe {MEDIA_MARKER}")
        );
    }

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(super::argmax_u32(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(super::argmax_u32(&[5.0, -1.0, 2.0]), 0);
    }
}
