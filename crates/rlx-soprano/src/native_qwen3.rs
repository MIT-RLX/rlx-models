// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native (ort-free) soprano backbone via **rlx-qwen3 reuse**.
//!
//! soprano's backbone is a stock `Qwen3ForCausalLM` (`ekwek/Soprano-1.1-80M`,
//! Apache-2.0). The shipped ONNX (`soprano_backbone_kv_fp32.onnx`) is kept ONLY
//! for parity; it mis-broadcasts `past>1` on import (forcing full-prefix
//! recompute bucketed to seq≤128) and diverges numerically on CUDA. This path
//! rebuilds the backbone with rlx-qwen3's TTS-backbone graphs
//! ([`build_qwen3_prefill_embeds_built`], `with_lm_head=false` → post-norm
//! `hidden_states` = soprano's 512-d latent), which are cuda-bit-exact and have
//! a real KV cache (no seq≤128 limit).
//!
//! Milestone ladder (validate each before the next):
//!   1. prefill-logit parity vs the ONNX backbone  ← this module
//!   2. AR decode loop with KV cache (mirror rlx-qwen3-tts talker)
//!   3. swap the ONNX Vocos decoder for a native one (rlx-fft ISTFT)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_ir::DType;
use rlx_qwen3::{Qwen3Config, Qwen3PrefillOpts, build_qwen3_prefill_embeds_built};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, Device};

use crate::native::{EOS, HIDDEN, VOCAB};

/// Static config for `ekwek/Soprano-1.1-80M` (stock Qwen3ForCausalLM). serde
/// fills the MoE / sliding-window defaults (all off for soprano).
pub fn soprano_qwen3_config() -> Qwen3Config {
    serde_json::from_str(
        r#"{
            "vocab_size": 8192,
            "hidden_size": 512,
            "intermediate_size": 2304,
            "num_hidden_layers": 17,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "attention_bias": false,
            "qk_norm": true
        }"#,
    )
    .expect("static soprano qwen3 config parses")
}

pub struct SopranoQwen3 {
    cfg: Qwen3Config,
    device: Device,
    st_path: PathBuf,
    cache: AotCache,
    graphs: Mutex<HashMap<String, CompiledGraph>>,
    /// `model.embed_tokens.weight` [vocab, hidden], row-major (host embed gather).
    embed_tokens: Vec<f32>,
    /// `lm_head.weight` [vocab, hidden], row-major (host logits matmul).
    lm_head: Vec<f32>,
}

impl SopranoQwen3 {
    /// `backbone_st` = the `model.safetensors` from `ekwek/Soprano-1.1-80M`.
    pub fn open(backbone_st: &Path, device: Device) -> Result<Self> {
        let cfg = soprano_qwen3_config();
        // embed_tokens + lm_head are NOT consumed by the embeds builder (it takes
        // `inputs_embeds` and emits post-norm hidden, no LM head) — read them for
        // host-side gather / logits.
        let mut wm = WeightMap::from_file(
            backbone_st
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 path"))?,
        )
        .with_context(|| format!("load soprano backbone {}", backbone_st.display()))?;
        let embed_tokens = take_rows(&mut wm, "model.embed_tokens.weight", VOCAB * HIDDEN)?;
        let lm_head = take_rows(&mut wm, "lm_head.weight", VOCAB * HIDDEN)?;
        let cache = AotCache::new(std::env::temp_dir().join(format!("rlx_soprano_q3_{device:?}")));
        Ok(Self {
            cfg,
            device,
            st_path: backbone_st.to_path_buf(),
            cache,
            graphs: Mutex::new(HashMap::new()),
            embed_tokens,
            lm_head,
        })
    }

    fn embed(&self, ids: &[i64]) -> Vec<f32> {
        let mut out = vec![0f32; ids.len() * HIDDEN];
        for (t, &id) in ids.iter().enumerate() {
            let src = (id as usize) * HIDDEN;
            out[t * HIDDEN..(t + 1) * HIDDEN]
                .copy_from_slice(&self.embed_tokens[src..src + HIDDEN]);
        }
        out
    }

    fn compile_prefill(&self, seq: usize) -> Result<CompiledGraph> {
        // Fresh loader per compile: the builder `take`s the projection weights.
        let mut wm = WeightMap::from_file(self.st_path.to_str().unwrap())
            .context("reload soprano backbone for build")?;
        let opts = Qwen3PrefillOpts {
            batch: 1,
            seq,
            with_lm_head: false,
            with_kv_outputs: false,
            with_qk_outputs: false,
            last_logits_only: false,
            packed: false,
            profile: None,
            rope_cos: None,
            rope_sin: None,
        };
        let built = build_qwen3_prefill_embeds_built(&self.cfg, &mut wm, &opts)
            .context("build qwen3 prefill embeds")?;
        let (hir, mut params) = built.into_parts()?;
        let key = format!("sopq3_prefill_{:?}_s{seq}", self.device);
        let mut g = self
            .cache
            .compile_hir_cached(&key, self.device, hir, &CompileOptions::default())
            .map_err(|e| anyhow!("compile qwen3 prefill: {e}"))?;
        for (n, d) in params.drain() {
            g.set_param(&n, &d);
        }
        g.finalize_params();
        Ok(g)
    }

    fn graph_mut<'a>(
        cache: &'a mut HashMap<String, CompiledGraph>,
        key: &str,
        build: impl FnOnce() -> Result<CompiledGraph>,
    ) -> Result<&'a mut CompiledGraph> {
        if !cache.contains_key(key) {
            cache.insert(key.to_string(), build()?);
        }
        Ok(cache.get_mut(key).unwrap())
    }

    /// Prefill over `ids`, end-padded to a bucket (few graph compiles instead of
    /// one per length). Returns `(real_len, hidden_states[bucket, HIDDEN])`. The
    /// causal mask means real positions never attend to the zero padding, so the
    /// last-real row is exact.
    fn run_prefill(&self, ids: &[i64]) -> Result<(usize, Vec<f32>)> {
        anyhow::ensure!(!ids.is_empty(), "empty prompt");
        let real = ids.len();
        let seq = bucket(real);
        let mut embeds = self.embed(ids);
        embeds.resize(seq * HIDDEN, 0.0);
        let key = format!("prefill_s{seq}");
        let mut cache = self.graphs.lock().unwrap();
        let g = Self::graph_mut(&mut cache, &key, || self.compile_prefill(seq))?;
        let eb: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&embeds).to_vec();
        let outs = g.run_typed(&[("inputs_embeds", &eb, DType::F32)]);
        let hidden = as_f32(&outs[0].0);
        anyhow::ensure!(
            hidden.len() >= seq * HIDDEN,
            "hidden {} short",
            hidden.len()
        );
        Ok((real, hidden))
    }

    /// logits = hidden[pos] @ lm_head^T.
    fn logits_at(&self, hidden: &[f32], pos: usize) -> Vec<f32> {
        let h = &hidden[pos * HIDDEN..(pos + 1) * HIDDEN];
        let mut logits = vec![0f32; VOCAB];
        for (v, lg) in logits.iter_mut().enumerate() {
            let w = &self.lm_head[v * HIDDEN..(v + 1) * HIDDEN];
            *lg = h.iter().zip(w).map(|(a, b)| a * b).sum();
        }
        logits
    }

    /// Native prefill logits over the LAST prompt position.
    /// Parity target: `NativeSoprano::prefill_logits` (the ONNX backbone).
    pub fn prefill_logits(&self, ids: &[i64]) -> Result<Vec<f32>> {
        let (real, hidden) = self.run_prefill(ids)?;
        Ok(self.logits_at(&hidden, real - 1))
    }

    /// Greedy AR (full-recompute, no seq≤128 cap): prompt ids →
    /// `(latents [T][HIDDEN], audio-token stream)`. Mirrors
    /// `NativeSoprano::generate_latents`: prefill samples the first token and
    /// discards its hidden; each subsequent non-EOS token contributes its
    /// last-position hidden as a Vocos latent.
    pub fn generate_latents_greedy(
        &self,
        prompt: &[i64],
        max_new: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<i64>)> {
        let mut ids = prompt.to_vec();
        let mut latents: Vec<Vec<f32>> = Vec::new();
        let mut toks: Vec<i64> = Vec::new();

        let (real, hidden) = self.run_prefill(&ids)?;
        let mut next = argmax(&self.logits_at(&hidden, real - 1)) as i64;
        toks.push(next);
        if next == EOS {
            return Ok((latents, toks));
        }
        ids.push(next);

        for _ in 0..max_new {
            let (real, hidden) = self.run_prefill(&ids)?;
            let tok = argmax(&self.logits_at(&hidden, real - 1)) as i64;
            toks.push(tok);
            if tok != EOS {
                latents.push(hidden[(real - 1) * HIDDEN..real * HIDDEN].to_vec());
            }
            next = tok;
            ids.push(next);
            if next == EOS {
                break;
            }
        }
        Ok((latents, toks))
    }
}

fn bucket(real: usize) -> usize {
    for b in [32usize, 64, 128, 256, 512, 1024, 2048] {
        if real <= b {
            return b;
        }
    }
    real.next_power_of_two()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

fn take_rows(wm: &mut WeightMap, name: &str, expect: usize) -> Result<Vec<f32>> {
    let (v, _shape) = wm
        .take(name)
        .with_context(|| format!("missing weight {name}"))?;
    anyhow::ensure!(v.len() == expect, "{name} len {} != {expect}", v.len());
    Ok(v)
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(bytes).to_vec()
}
