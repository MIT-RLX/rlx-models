//! Kyutai TTS LM — weight loading + eager forward (backbone + DepFormer + conditioners).

use crate::conditioner::{LutConditioner, TensorConditioner};
use crate::config::KyutaiTtsConfig;
use crate::depformer_stream::DepformerStream;
use crate::fuser::{ConditionerOutputs, fuse};
use crate::nn::{Embedding, linear, rms_norm};
use crate::sampling::StreamSampler;
use crate::transformer::StreamingTransformer;
use crate::util::{take_mat2, take_rms_alpha, take_vec1};
use crate::weights::{WeightMap, load_weight_map};
use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2};
use rlx_runtime::Device;
use std::path::Path;

/// Demuxed text embedding (`demux_second_stream = true`).
#[derive(Debug)]
pub(crate) struct TextEmbedding {
    table: Embedding,
    out1: Array2<f32>,
    out2: Array2<f32>,
    card: usize,
}

impl TextEmbedding {
    pub(crate) fn load(weights: &WeightMap, card: usize) -> Result<Self> {
        Ok(Self {
            table: Embedding {
                weight: take_mat2(weights, "text_emb.weight")?,
            },
            out1: take_mat2(weights, "text_emb.out1.weight")?,
            out2: take_mat2(weights, "text_emb.out2.weight")?,
            card,
        })
    }

    pub(crate) fn forward_multiplex(&self, token: u32) -> Array1<f32> {
        if token == u32::MAX {
            return Array1::zeros(self.table.weight.ncols());
        }
        let card = (self.card + 1) as u32;
        let main_id = token % card;
        let second_slot = token / card;
        let main_emb = self.table.forward_one(main_id);
        let mut y = linear(main_emb.view().insert_axis(ndarray::Axis(0)), &self.out1)
            .row(0)
            .to_owned();
        if second_slot > 0 {
            let r = second_slot - 1;
            if r < card {
                let right_emb = self.table.forward_one(r);
                let r_row = linear(right_emb.view().insert_axis(ndarray::Axis(0)), &self.out2);
                for (a, &b) in y.iter_mut().zip(r_row.row(0).iter()) {
                    *a += b;
                }
            }
        }
        y
    }
}

/// Frames per voice slot in cross-attention (`kyutai/tts-voices` embeddings).
pub const SPEAKER_FRAMES_PER_SLOT: usize = 125;
/// Multi-speaker cross-attention slots (Moshi `TTSModel.max_speakers`).
pub const MAX_SPEAKER_SLOTS: usize = 5;
/// Total cross-attention context length (`5 × 125`).
pub const MAX_SPEAKER_CROSS_FRAMES: usize = SPEAKER_FRAMES_PER_SLOT * MAX_SPEAKER_SLOTS;

#[derive(Debug)]
pub(crate) struct ConditionerBundle {
    cfg: LutConditioner,
    cfg_proj: Array2<f32>,
    control: LutConditioner,
    control_proj: Array2<f32>,
    speaker: TensorConditioner,
    speaker_learnt_padding: Array1<f32>,
}

impl ConditionerBundle {
    pub(crate) fn load(cfg: &KyutaiTtsConfig, weights: &WeightMap) -> Result<Self> {
        let pfx = "condition_provider.conditioners.";
        let cfg_lut = LutConditioner {
            table: Embedding {
                weight: take_mat2(weights, &format!("{pfx}cfg.embed.weight"))?,
            },
            possible_values: match &cfg.conditioners["cfg"] {
                crate::config::ConditionerKind::Lut { lut } => lut.possible_values.clone(),
                _ => vec![],
            },
        };
        let control = LutConditioner {
            table: Embedding {
                weight: take_mat2(weights, &format!("{pfx}control.embed.weight"))?,
            },
            possible_values: vec!["ok".into()],
        };
        let speaker = TensorConditioner {
            input_dim: 512,
            output_proj: Some(take_mat2(
                weights,
                &format!("{pfx}speaker_wavs.output_proj.weight"),
            )?),
        };
        let speaker_learnt_padding =
            take_vec1(weights, &format!("{pfx}speaker_wavs.learnt_padding"))?;
        Ok(Self {
            cfg: cfg_lut,
            cfg_proj: take_mat2(weights, &format!("{pfx}cfg.output_proj.weight"))?,
            control,
            control_proj: take_mat2(weights, &format!("{pfx}control.output_proj.weight"))?,
            speaker,
            speaker_learnt_padding,
        })
    }

    /// Pack one voice into the 5×125 Moshi cross-attention layout with learnt padding.
    fn speaker_cross_sequence(&self, speaker: Option<&Array2<f32>>) -> Result<Array2<f32>> {
        let mut raw = Array2::<f32>::zeros((MAX_SPEAKER_CROSS_FRAMES, self.speaker.input_dim));
        if let Some(spk) = speaker {
            let t = spk.nrows().min(SPEAKER_FRAMES_PER_SLOT);
            let d = spk.ncols().min(self.speaker.input_dim);
            for ti in 0..t {
                for di in 0..d {
                    raw[[ti, di]] = spk[[ti, di]];
                }
            }
        }
        let mut proj = self.speaker.forward_seq(&raw)?;
        let d = proj.ncols();
        ensure!(
            self.speaker_learnt_padding.len() == d,
            "speaker learnt_padding dim {} != cross dim {d}",
            self.speaker_learnt_padding.len()
        );
        for ti in SPEAKER_FRAMES_PER_SLOT..MAX_SPEAKER_CROSS_FRAMES {
            for di in 0..d {
                proj[[ti, di]] = self.speaker_learnt_padding[di];
            }
        }
        Ok(proj)
    }

    pub(crate) fn fused(
        &self,
        cfg: &KyutaiTtsConfig,
        cfg_key: &str,
        speaker: Option<&Array2<f32>>,
    ) -> Result<crate::fuser::FusedConditioning> {
        let mut outs = ConditionerOutputs::new();
        let cfg_vec = self.cfg.forward_value(cfg_key)?;
        outs.insert_vector(
            "cfg",
            linear(cfg_vec.view().insert_axis(ndarray::Axis(0)), &self.cfg_proj)
                .row(0)
                .to_owned(),
        );
        let ctrl_vec = self.control.forward_value("ok")?;
        outs.insert_vector(
            "control",
            linear(
                ctrl_vec.view().insert_axis(ndarray::Axis(0)),
                &self.control_proj,
            )
            .row(0)
            .to_owned(),
        );
        let proj = self.speaker_cross_sequence(speaker)?;
        outs.insert_sequence("speaker_wavs", proj);
        fuse(&cfg.fuser, &outs, cfg.dim)
    }
}

/// Native eager Kyutai TTS LM.
pub struct KyutaiTtsModel {
    pub cfg: KyutaiTtsConfig,
    text_emb: TextEmbedding,
    audio_embs: Vec<Embedding>,
    text_linear: Array2<f32>,
    out_norm_alpha: Array1<f32>,
    transformer: StreamingTransformer,
    depformer: DepformerStream,
    conditioners: ConditionerBundle,
    sum_offset: Option<Array1<f32>>,
    device: Device,
}

impl KyutaiTtsModel {
    pub fn open(model_dir: &Path, cfg: KyutaiTtsConfig, device: Device) -> Result<Self> {
        let weights_path = model_dir.join(crate::download::TTS_WEIGHTS_FILE);
        let weights = load_weight_map(&weights_path)?;
        Self::from_weights(cfg, weights, device)
    }

    pub fn from_weights(cfg: KyutaiTtsConfig, weights: WeightMap, device: Device) -> Result<Self> {
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
        let backbone_cfg = cfg.backbone_runtime();
        let layers = StreamingTransformer::load_layers(
            &backbone_cfg,
            &weights,
            cfg.fuser.cross_attention_pos_emb,
            cfg.fuser.cross_attention_pos_emb_scale,
        )?;
        let transformer = StreamingTransformer::new(backbone_cfg, layers)?;
        let depformer = DepformerStream::load(&cfg, &weights)?;
        let conditioners = ConditionerBundle::load(&cfg, &weights)?;
        Ok(Self {
            cfg: cfg.clone(),
            text_emb,
            audio_embs,
            text_linear: take_mat2(&weights, "text_linear.weight")?,
            out_norm_alpha: take_rms_alpha(&weights, "out_norm.alpha")?,
            transformer,
            depformer,
            conditioners,
            sum_offset: None,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn reset_state(&mut self) {
        self.transformer.reset_state();
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
        self.transformer.set_cross_context(fused.cross.as_ref())?;
        Ok(())
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.cfg.audio_pad_token()
    }

    pub fn lm_zero_token(&self) -> u32 {
        u32::MAX
    }

    pub fn delay_steps(&self) -> usize {
        self.cfg.audio_delay_frames()
    }

    /// Backbone forward for one DSM step: sum embeddings → transformer → sample text.
    pub fn forward_step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut StreamSampler,
    ) -> Result<(u32, Array1<f32>)> {
        let d = self.cfg.dim;
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
        let x = Array2::from_shape_vec((1, d), emb.to_vec())?;
        let h = self.transformer.forward(&x)?;
        let normed = rms_norm(h.view(), &self.out_norm_alpha);
        let text_logits = linear(normed.view(), &self.text_linear);
        let row = text_logits.row(0).to_owned();
        if std::env::var_os("RLX_KYUTAI_TTS_TRACE").is_some() {
            let mut top: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "logits top5 {:?} pad={:.3} new_word={:.3}",
                &top[..5],
                row[3],
                row[0]
            );
        }
        let sampled_text = sampler.sample_text(&row);
        Ok((sampled_text, normed.row(0).to_owned()))
    }

    /// DepFormer depth decode for one temporal frame (`depformer_text` is state-machine output).
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

    /// One DSM step: sum embeddings → backbone → text logits + depformer audio codes.
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

/// Shared LM interface for eager CPU and native RLX backends.
pub trait KyutaiLm {
    fn config(&self) -> &KyutaiTtsConfig;
    fn reset_state(&mut self);
    fn set_generation_conditions(
        &mut self,
        cfg_alpha: f32,
        speaker: Option<&Array2<f32>>,
    ) -> Result<()>;
    fn forward_step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut StreamSampler,
    ) -> Result<(u32, Array1<f32>)>;
    fn depformer_step(
        &mut self,
        hidden: &Array1<f32>,
        depformer_text: u32,
        sampler: &mut StreamSampler,
    ) -> Result<Vec<u32>>;
    fn step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut StreamSampler,
    ) -> Result<(u32, Vec<u32>)> {
        let (sampled_text, hidden) = self.forward_step(text_token, audio_delayed, sampler)?;
        let audio_tokens = self.depformer_step(&hidden, sampled_text, sampler)?;
        Ok((sampled_text, audio_tokens))
    }
    fn audio_pad_token(&self) -> u32;
    /// Audio/text LM sentinel — contributes zero to embeddings (Moshi `zero_token_id = -1`).
    fn lm_zero_token(&self) -> u32;
}

impl KyutaiLm for KyutaiTtsModel {
    fn config(&self) -> &KyutaiTtsConfig {
        &self.cfg
    }

    fn reset_state(&mut self) {
        KyutaiTtsModel::reset_state(self);
    }

    fn set_generation_conditions(
        &mut self,
        cfg_alpha: f32,
        speaker: Option<&Array2<f32>>,
    ) -> Result<()> {
        KyutaiTtsModel::set_generation_conditions(self, cfg_alpha, speaker)
    }

    fn forward_step(
        &mut self,
        text_token: u32,
        audio_delayed: &[u32],
        sampler: &mut StreamSampler,
    ) -> Result<(u32, Array1<f32>)> {
        KyutaiTtsModel::forward_step(self, text_token, audio_delayed, sampler)
    }

    fn depformer_step(
        &mut self,
        hidden: &Array1<f32>,
        depformer_text: u32,
        sampler: &mut StreamSampler,
    ) -> Result<Vec<u32>> {
        KyutaiTtsModel::depformer_step(self, hidden, depformer_text, sampler)
    }

    fn audio_pad_token(&self) -> u32 {
        KyutaiTtsModel::audio_pad_token(self)
    }

    fn lm_zero_token(&self) -> u32 {
        u32::MAX
    }
}

/// Load `speaker_wavs` from a `kyutai/tts-voices` file → `[T, 512]`.
pub fn load_voice_speaker_wavs(path: &Path) -> Result<Array2<f32>> {
    let map = load_weight_map(path)?;
    let (data, shape) = map
        .get("speaker_wavs")
        .context("voice file missing `speaker_wavs` tensor")?;
    match shape.as_slice() {
        [1, d, t] => {
            let (d, t) = (*d, *t);
            let mut out = Array2::<f32>::zeros((t, d));
            for di in 0..d {
                for ti in 0..t {
                    out[[ti, di]] = data[di * t + ti];
                }
            }
            Ok(out)
        }
        [t, d] => {
            let (t, d) = (*t, *d);
            let mut out = Array2::<f32>::zeros((t, d));
            for ti in 0..t {
                for di in 0..d {
                    out[[ti, di]] = data[ti * d + di];
                }
            }
            Ok(out)
        }
        [d] => {
            let d = *d;
            let mut out = Array2::<f32>::zeros((1, d));
            for di in 0..d {
                out[[0, di]] = data[di];
            }
            Ok(out)
        }
        other => bail!("unsupported speaker_wavs shape {other:?}"),
    }
}

/// Load speaker embedding from a `kyutai/tts-voices` safetensors file (512-D mean pool).
pub fn load_speaker_embedding(path: &Path) -> Result<Array1<f32>> {
    let seq = load_voice_speaker_wavs(path)?;
    if seq.nrows() == 1 {
        return Ok(seq.row(0).to_owned());
    }
    let mut mean = Array1::<f32>::zeros(seq.ncols());
    for row in seq.rows() {
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += v;
        }
    }
    mean.mapv_inplace(|v| v / seq.nrows() as f32);
    Ok(mean)
}
