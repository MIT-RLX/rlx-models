// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Native-RLX Moshi LM driver: runs the temporal-transformer decode graph with a
//! persistent KV cache and the DepFormer graphs each 12.5 Hz frame — the candle-
//! free counterpart to [`crate::lm::LmModel`] + [`crate::generate::GenerateState`].

use crate::checkpoint::MoshiCheckpoint;
use crate::config::{GenerateConfig, LmConfig, MoshiVariant};
use crate::generate::{ForcedAudioTokens, UNGENERATED};
use crate::rlx_lm::{
    DepDims, HeliumDims, build_temporal_decode_graph_bucketed, compile_depformer_slice,
    decode_bucketed_run, depformer_slice_run, set_temporal_params,
};
use crate::sampling::LogitsProcessor;
use anyhow::{Result, ensure};
use ndarray::ArrayView1;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::Path;

/// Moshi LM held as a dequantized weight map driven by native RLX graphs.
pub struct RlxLm {
    cfg: LmConfig,
    dims: HeliumDims,
    dep: Option<DepDims>,
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: Device,
    /// Set once the temporal-layer weights have been baked into a compiled graph
    /// and dropped from `weights` (steady-state RAM saving). A pruned `RlxLm` can
    /// no longer compile the temporal graph — it's tied to its `RlxGenerateState`.
    temporal_pruned: bool,
}

impl RlxLm {
    /// Build from an already-loaded dequantized weight map (GGUF / MLX / bf16).
    pub fn from_weights(
        cfg: LmConfig,
        weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
        device: Device,
    ) -> Result<Self> {
        let dims = HeliumDims::from_cfg(&cfg.transformer, cfg.text_out_vocab_size);
        let dep = cfg.depformer.as_ref().map(|d| {
            // The DepFormer head (`linear_out`) outputs the audio codes only —
            // `audio_vocab_size - 1` (the pad token is input-embedding-only). Read
            // the actual row count from the checkpoint rather than assuming.
            let head = weights
                .get("depformer.0.linear_out.weight")
                .map(|(_, s)| s[0])
                .unwrap_or(cfg.audio_vocab_size);
            DepDims::from_cfg(d, cfg.transformer.d_model, head)
        });
        Ok(Self {
            cfg,
            dims,
            dep,
            weights,
            device,
            temporal_pruned: false,
        })
    }

    /// Drop the temporal-layer weights once they've been copied into the compiled
    /// graph — they account for the bulk of the model, so this roughly halves the
    /// steady-state footprint. Embeddings (`text_emb`/`emb.*`) and DepFormer
    /// weights are kept (still needed for `sum_embeds` / the depth decoder).
    fn prune_temporal_weights(&mut self) {
        self.weights.retain(|k, _| {
            !(k.starts_with("transformer.layers.")
                || k == "out_norm.alpha"
                || k == "text_linear.weight")
        });
        self.temporal_pruned = true;
    }

    /// Load a checkpoint into a native-RLX LM. GGUF and bf16 are candle-free;
    /// MLX dequant currently still routes through the candle loader (to be
    /// replaced by the native `mlx_dequant` path).
    pub fn open(
        model_dir: &Path,
        variant: MoshiVariant,
        checkpoint: MoshiCheckpoint,
        device: Device,
    ) -> Result<Self> {
        let cfg = variant.lm_config();
        let weights = if checkpoint.is_gguf() {
            crate::gguf::load_gguf_weight_map(&checkpoint.lm_weights_path(model_dir), &cfg)?
        } else if checkpoint.is_mlx() {
            crate::mlx_weights::load_eager_weight_map(
                &checkpoint.lm_weights_path(model_dir),
                checkpoint,
                &cfg,
            )?
        } else {
            crate::weights::load_weight_map(model_dir)?
        };
        Self::from_weights(cfg, weights, device)
    }

    pub fn config(&self) -> &LmConfig {
        &self.cfg
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Compile the bucketed temporal decode graph once for a fixed `upper`
    /// (max past). Params (weights) are set into the compiled graph here, so a
    /// single instance is reused across all frames — no per-frame recompile.
    fn compile_temporal_bucketed(&self, upper: usize) -> Result<CompiledGraph> {
        ensure!(
            !self.temporal_pruned,
            "RlxLm temporal weights were pruned after compile; build a fresh RlxLm to compile again"
        );
        let mut c = Session::new(self.device)
            .compile(build_temporal_decode_graph_bucketed(&self.dims, upper));
        // Incremental param-setting to keep peak RAM ≈ weights + graph at 7B scale.
        set_temporal_params(&mut c, &self.dims, &self.weights)?;
        Ok(c)
    }

    /// Sum the input embeddings (text + per-codebook audio) for one frame.
    fn sum_embeds(&self, text_token: Option<u32>, audio_tokens: &[Option<u32>]) -> Vec<f32> {
        let d = self.dims.d_model;
        let mut emb = vec![0.0f32; d];
        if let Some(tt) = text_token {
            if let Some((data, shape)) = self.weights.get("text_emb.weight") {
                let row = shape[1];
                let base = tt as usize * row;
                for i in 0..d {
                    emb[i] += data[base + i];
                }
            }
        }
        for (cb, tok) in audio_tokens.iter().enumerate() {
            if let Some(t) = tok {
                if let Some((data, shape)) = self.weights.get(&format!("emb.{cb}.weight")) {
                    let row = shape[1];
                    let base = *t as usize * row;
                    for i in 0..d {
                        emb[i] += data[base + i];
                    }
                }
            }
        }
        emb
    }
}

/// Per-generation autoregressive state for [`RlxLm`] — mirrors
/// [`crate::generate::GenerateState`] but holds the temporal KV cache and runs
/// the RLX decode + DepFormer graphs instead of the eager `LmModel`.
pub struct RlxGenerateState {
    audio_tokens: Vec<Vec<u32>>,
    text_tokens: Vec<u32>,
    text_lp: LogitsProcessor,
    audio_lp: LogitsProcessor,
    step_idx: usize,
    forced_audio_tokens: ForcedAudioTokens,
    cfg: GenerateConfig,
    /// Per-layer temporal KV cache, each `(k, v)` flattened `[step, nh, hd]` (real,
    /// unpadded — padded to `max_upper` per frame inside the bucketed graph).
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Fixed decode bucket (= max past). The temporal graph is compiled once for it.
    max_upper: usize,
    /// Compiled bucketed temporal decode graph (lazy; reused every frame).
    temporal_compiled: Option<CompiledGraph>,
    /// Compiled DepFormer slice graphs, one per slice (lazy; reused every frame).
    dep_compiled: Vec<Option<CompiledGraph>>,
}

impl RlxGenerateState {
    pub fn new(
        max_steps: usize,
        text_lp: LogitsProcessor,
        audio_lp: LogitsProcessor,
        cfg: GenerateConfig,
    ) -> Self {
        let buf = max_steps + cfg.acoustic_delay;
        let audio_tokens = vec![vec![UNGENERATED; cfg.total_audio_codebooks()]; buf];
        let text_tokens = vec![UNGENERATED; buf];
        let forced = ForcedAudioTokens::new(cfg.acoustic_delay, cfg.audio_pad_token(), &[8, 8]);
        Self {
            audio_tokens,
            text_tokens,
            text_lp,
            audio_lp,
            step_idx: 0,
            forced_audio_tokens: forced,
            cfg,
            kv: Vec::new(),
            // `buf` > every reachable `step_idx`, so past_seq < max_upper always.
            max_upper: buf,
            temporal_compiled: None,
            dep_compiled: Vec::new(),
        }
    }

    /// Sample all DepFormer audio codebooks using the per-slice compiled-graph
    /// cache (each slice graph compiled once, reused every frame).
    fn depformer_sample_cached(
        &mut self,
        lm: &RlxLm,
        dd: &DepDims,
        hidden: &[f32],
        text_token: u32,
        forced: &[Option<u32>],
    ) -> Result<Vec<u32>> {
        if self.dep_compiled.len() != dd.num_slices {
            self.dep_compiled = (0..dd.num_slices).map(|_| None).collect();
        }
        let mut tokens = Vec::with_capacity(dd.num_slices);
        let mut last_token = Some(text_token);
        let mut past_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for si in 0..dd.num_slices {
            // Slice `si` always inherits `si` cached positions → graph past = si.
            let mut compiled = match self.dep_compiled[si].take() {
                Some(c) => c,
                None => compile_depformer_slice(dd, &lm.weights, si, si, lm.device)?,
            };
            let (logits, new_kv) = depformer_slice_run(
                &mut compiled,
                dd,
                &lm.weights,
                hidden,
                si,
                last_token,
                &past_kv,
            )?;
            self.dep_compiled[si] = Some(compiled);
            past_kv = new_kv;
            let token = self.audio_lp.sample(ArrayView1::from(&logits))?;
            tokens.push(token);
            last_token = Some(forced.get(si).copied().flatten().unwrap_or(token));
        }
        Ok(tokens)
    }

    pub fn config(&self) -> &GenerateConfig {
        &self.cfg
    }

    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    pub fn text_tokens(&self) -> &[u32] {
        let n = self.step_idx.min(self.text_tokens.len());
        &self.text_tokens[..n]
    }

    /// Advance one 12.5 Hz frame. `input_audio` is user codebooks (empty for one-way).
    pub fn step(&mut self, lm: &mut RlxLm, text_token: u32, input_audio: &[u32]) -> Result<u32> {
        ensure!(
            input_audio.len() == self.cfg.input_audio_codebooks,
            "expected {} user codebooks, got {}",
            self.cfg.input_audio_codebooks,
            input_audio.len()
        );
        for (ci, &t) in input_audio.iter().enumerate() {
            let idx = ci + self.cfg.generated_audio_codebooks;
            self.audio_tokens[self.step_idx][idx] = t;
        }
        let pad = self.cfg.audio_pad_token();
        let mut delayed = Vec::with_capacity(self.cfg.total_audio_codebooks());
        for codebook in 0..self.cfg.total_audio_codebooks() {
            let t = if codebook == 0 || codebook == self.cfg.generated_audio_codebooks {
                if self.step_idx == 0 {
                    pad
                } else {
                    self.audio_tokens[self.step_idx - 1][codebook]
                }
            } else if self.step_idx <= self.cfg.acoustic_delay {
                pad
            } else {
                self.audio_tokens[self.step_idx - self.cfg.acoustic_delay - 1][codebook]
            };
            ensure!(
                t != UNGENERATED,
                "internal: ungenerated audio at step {}",
                self.step_idx
            );
            delayed.push(Some(t));
        }

        // Temporal transformer decode via the cached bucketed graph (compiled
        // once, reused). KV is taken out of `self` to satisfy the borrow checker,
        // then restored.
        let emb = lm.sum_embeds(Some(text_token), &delayed);
        // Compile the temporal graph on the first frame, then prune the now-baked
        // temporal weights from `lm` to free their (large) source copy.
        if self.temporal_compiled.is_none() {
            self.temporal_compiled = Some(lm.compile_temporal_bucketed(self.max_upper)?);
            lm.prune_temporal_weights();
        }
        let mut compiled = self.temporal_compiled.take().unwrap();
        let (text_logits, hidden, new_kv) = decode_bucketed_run(
            &mut compiled,
            &lm.dims,
            &emb,
            &self.kv,
            self.step_idx,
            self.max_upper,
        )?;
        self.temporal_compiled = Some(compiled);
        // Append this token's single-step K/V to the real (unpadded) cache.
        if self.kv.len() != lm.dims.n_layers {
            self.kv = (0..lm.dims.n_layers)
                .map(|_| (Vec::new(), Vec::new()))
                .collect();
        }
        for (li, (k, v)) in new_kv.iter().enumerate() {
            self.kv[li].0.extend_from_slice(k);
            self.kv[li].1.extend_from_slice(v);
        }
        let sampled_text = self.text_lp.sample(ArrayView1::from(&text_logits))?;
        self.text_tokens[self.step_idx] = sampled_text;

        // DepFormer audio codebooks via the per-slice compiled-graph cache.
        if let Some(dd) = lm.dep {
            let forced = self.forced_audio_tokens.forced_tokens(self.step_idx);
            let tokens = self.depformer_sample_cached(lm, &dd, &hidden, sampled_text, &forced)?;
            for (ci, &tok) in tokens.iter().enumerate() {
                let delay = if ci == 0 { 0 } else { self.cfg.acoustic_delay };
                let pos = self.step_idx.saturating_sub(delay);
                self.audio_tokens[pos][ci] = tok;
            }
        }
        self.step_idx += 1;
        Ok(sampled_text)
    }

    /// Moshi output codebooks ready for Mimi decode (past acoustic delay).
    pub fn last_audio_frame(&self) -> Option<Vec<u32>> {
        if self.step_idx <= self.cfg.acoustic_delay {
            return None;
        }
        let pos = self.step_idx - self.cfg.acoustic_delay - 1;
        let frame = &self.audio_tokens[pos];
        let pad = self.cfg.audio_pad_token();
        if frame[..self.cfg.generated_audio_codebooks]
            .iter()
            .any(|&t| t >= pad)
        {
            return None;
        }
        Some(frame[..self.cfg.generated_audio_codebooks].to_vec())
    }

    pub fn reset(&mut self) {
        self.step_idx = 0;
        self.kv.clear();
        // Keep `temporal_compiled` — weights are unchanged, so the compiled graph
        // is still valid; only the KV cache + token buffers are cleared.
        let buf = self.audio_tokens.len();
        let tc = self.cfg.total_audio_codebooks();
        self.audio_tokens = vec![vec![UNGENERATED; tc]; buf];
        self.text_tokens = vec![UNGENERATED; self.text_tokens.len()];
    }
}
