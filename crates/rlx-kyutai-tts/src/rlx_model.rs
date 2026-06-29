//! Native RLX GPU backbone for Kyutai TTS (DepFormer + conditioners stay eager).

use crate::config::KyutaiTtsConfig;
use crate::depformer_stream::DepformerStream;
use crate::model::{ConditionerBundle, TextEmbedding};
use crate::nn::Embedding;
use crate::rlx_lm::{
    TtsDims, build_temporal_decode_graph_bucketed, decode_bucketed_run, prepare_cross_ctx,
    set_temporal_params,
};
use crate::sampling::StreamSampler;
use crate::util::take_mat2;
use crate::weights::{WeightMap, load_weight_map};
use anyhow::Result;
use ndarray::{Array1, Array2};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::Path;

/// Re-export for RLX graph sizing (see `model::MAX_SPEAKER_CROSS_FRAMES`).
pub use crate::model::MAX_SPEAKER_CROSS_FRAMES;

/// Kyutai TTS LM with the temporal backbone on native RLX graphs.
pub struct RlxKyutaiTtsModel {
    pub cfg: KyutaiTtsConfig,
    dims: TtsDims,
    text_emb: TextEmbedding,
    audio_embs: Vec<Embedding>,
    depformer: DepformerStream,
    conditioners: ConditionerBundle,
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: Device,
    temporal_pruned: bool,
    sum_offset: Option<Array1<f32>>,
    cross_ctx: Vec<f32>,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    seq_len: usize,
    temporal_compiled: Option<CompiledGraph>,
    max_upper: usize,
}

impl RlxKyutaiTtsModel {
    pub fn open(
        model_dir: &Path,
        cfg: KyutaiTtsConfig,
        device: Device,
        max_upper: usize,
    ) -> Result<Self> {
        let weights_path = model_dir.join(crate::download::TTS_WEIGHTS_FILE);
        let weights = load_weight_map(&weights_path)?;
        Self::from_weights(cfg, weights, device, max_upper)
    }

    pub fn from_weights(
        cfg: KyutaiTtsConfig,
        weights: WeightMap,
        device: Device,
        max_upper: usize,
    ) -> Result<Self> {
        let t_cross = MAX_SPEAKER_CROSS_FRAMES;
        let dims = TtsDims::from_cfg(&cfg, t_cross);
        let text_emb = TextEmbedding::load(&weights, cfg.text_card)?;
        let mut audio_embs = Vec::with_capacity(cfg.n_q);
        for q in 0..cfg.n_q {
            let key = format!("emb.{q}.weight");
            if weights.contains_key(&key) {
                audio_embs.push(Embedding {
                    weight: take_mat2(&weights, &key)?,
                });
            }
        }
        let depformer = DepformerStream::load(&cfg, &weights)?;
        let conditioners = ConditionerBundle::load(&cfg, &weights)?;
        let cross_ctx = vec![0.0f32; t_cross * cfg.dim];
        Ok(Self {
            cfg: cfg.clone(),
            dims,
            text_emb,
            audio_embs,
            depformer,
            conditioners,
            weights,
            device,
            temporal_pruned: false,
            sum_offset: None,
            cross_ctx,
            kv: Vec::new(),
            seq_len: 0,
            temporal_compiled: None,
            max_upper,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn prune_temporal_weights(&mut self) {
        self.weights.retain(|k, _| {
            !(k.starts_with("transformer.layers.")
                || k == "out_norm.alpha"
                || k == "text_linear.weight")
        });
        self.temporal_pruned = true;
    }

    fn compile_temporal_bucketed(&self) -> Result<CompiledGraph> {
        anyhow::ensure!(
            !self.temporal_pruned,
            "temporal weights pruned; open a fresh model to recompile"
        );
        let mut c = Session::new(self.device).compile(build_temporal_decode_graph_bucketed(
            &self.dims,
            self.max_upper,
        ));
        set_temporal_params(&mut c, &self.dims, &self.weights)?;
        Ok(c)
    }

    pub fn reset_state(&mut self) {
        self.kv.clear();
        self.seq_len = 0;
        self.depformer.reset();
    }

    pub fn set_generation_conditions(
        &mut self,
        cfg_alpha: f32,
        speaker: Option<&Array2<f32>>,
    ) -> Result<()> {
        let key = format!("{cfg_alpha:.1}");
        let fused = self.conditioners.fused(&self.cfg, &key, speaker)?;
        self.sum_offset = fused.sum.vector;
        if let Some(ctx) = fused.cross {
            let t = ctx.nrows().min(MAX_SPEAKER_CROSS_FRAMES);
            let mut padded = Array2::<f32>::zeros((MAX_SPEAKER_CROSS_FRAMES, self.cfg.dim));
            for ti in 0..t {
                for di in 0..self.cfg.dim {
                    padded[[ti, di]] = ctx[[ti, di]];
                }
            }
            self.cross_ctx = prepare_cross_ctx(
                &padded,
                self.cfg.fuser.cross_attention_pos_emb,
                self.cfg.fuser.cross_attention_pos_emb_scale,
                self.cfg.max_period as f32,
            );
        } else {
            self.cross_ctx.fill(0.0);
        }
        Ok(())
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.cfg.audio_pad_token()
    }

    pub fn lm_zero_token(&self) -> u32 {
        u32::MAX
    }

    pub fn forward_step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut StreamSampler,
    ) -> Result<(u32, Array1<f32>)> {
        let zero = self.lm_zero_token();
        let mut emb = self.text_emb.forward_multiplex(text_token);
        for (q, &tok) in audio_delayed.iter().enumerate() {
            if tok == zero {
                continue;
            }
            if let Some(ae) = self.audio_embs.get(q) {
                let e = ae.forward_one(tok);
                for (i, v) in e.iter().enumerate() {
                    emb[i] += v;
                }
            }
        }
        if let Some(sum) = &self.sum_offset {
            for (h, &v) in emb.iter_mut().zip(sum.iter()) {
                *h += v;
            }
        }

        if self.temporal_compiled.is_none() {
            self.temporal_compiled = Some(self.compile_temporal_bucketed()?);
            self.prune_temporal_weights();
        }
        let mut compiled = self.temporal_compiled.take().unwrap();
        let (text_logits, hidden, new_kv) = decode_bucketed_run(
            &mut compiled,
            &self.dims,
            emb.as_slice().unwrap(),
            &self.cross_ctx,
            &self.kv,
            self.seq_len,
            self.max_upper,
        )?;
        self.temporal_compiled = Some(compiled);

        if self.kv.len() != self.dims.n_layers {
            self.kv = (0..self.dims.n_layers)
                .map(|_| (Vec::new(), Vec::new()))
                .collect();
        }
        for (li, (k, v)) in new_kv.iter().enumerate() {
            self.kv[li].0.extend_from_slice(k);
            self.kv[li].1.extend_from_slice(v);
        }
        self.seq_len += 1;

        let sampled_text = sampler.sample_text(&Array1::from_vec(text_logits));
        Ok((sampled_text, Array1::from_vec(hidden)))
    }

    pub fn depformer_step(
        &mut self,
        hidden: &Array1<f32>,
        depformer_text: u32,
        sampler: &mut StreamSampler,
    ) -> Result<Vec<u32>> {
        self.depformer.reset();
        let mut audio_tokens = Vec::with_capacity(self.depformer.dep_q());
        let mut prev = depformer_text;
        for cb in 0..self.depformer.dep_q() {
            let logits = self.depformer.forward_codebook(cb, hidden, prev)?;
            let tok = sampler.sample_audio(&logits);
            audio_tokens.push(tok);
            prev = tok;
        }
        Ok(audio_tokens)
    }

    pub fn step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut StreamSampler,
    ) -> Result<(u32, Vec<u32>)> {
        let (sampled_text, hidden) = self.forward_step(text_token, audio_delayed, sampler)?;
        let audio_tokens = self.depformer_step(&hidden, sampled_text, sampler)?;
        Ok((sampled_text, audio_tokens))
    }
}

impl crate::model::KyutaiLm for RlxKyutaiTtsModel {
    fn config(&self) -> &KyutaiTtsConfig {
        &self.cfg
    }

    fn reset_state(&mut self) {
        RlxKyutaiTtsModel::reset_state(self);
    }

    fn set_generation_conditions(
        &mut self,
        cfg_alpha: f32,
        speaker: Option<&Array2<f32>>,
    ) -> Result<()> {
        RlxKyutaiTtsModel::set_generation_conditions(self, cfg_alpha, speaker)
    }

    fn forward_step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut StreamSampler,
    ) -> Result<(u32, Array1<f32>)> {
        RlxKyutaiTtsModel::forward_step(self, text_token, audio_delayed, sampler)
    }

    fn depformer_step(
        &mut self,
        hidden: &Array1<f32>,
        depformer_text: u32,
        sampler: &mut StreamSampler,
    ) -> Result<Vec<u32>> {
        RlxKyutaiTtsModel::depformer_step(self, hidden, depformer_text, sampler)
    }

    fn audio_pad_token(&self) -> u32 {
        RlxKyutaiTtsModel::audio_pad_token(self)
    }

    fn lm_zero_token(&self) -> u32 {
        u32::MAX
    }
}
