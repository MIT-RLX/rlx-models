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

//! End-to-end runner: `.nemo` → log-mel → Conformer encoder → CTC → text.
//!
//! The encoder graph is compiled once per mel-length bucket and reused via
//! [`CompileCache`] (same pattern as Whisper). Call [`ConformerCtc::warm`] to
//! precompile before the first utterance, or let [`ConformerCtc::transcribe`]
//! populate the cache on demand.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::{built_from_hir, compile_cache_ensure_built};
use rlx_nemo::NemoModel;
use rlx_runtime::Device;
use rlx_runtime::compile_cache::CompileCache;

use crate::config::AsrConfig;
use crate::ctc::{self, CtcHead};
use crate::encoder::build_encoder_hir;
use crate::mel::{self, Frontend, MelSpectrogram};
use crate::tokenizer::SpmTokenizer;
use crate::weights::NemoWeights;

const ENC_CACHE_CAPACITY: usize = 8;

/// A loaded Conformer-CTC model ready to transcribe.
///
/// Holds the `.nemo` weights, host-side CTC head / SentencePiece tokenizer,
/// and a mel-bucketed encoder [`CompileCache`]. Methods that run the encoder
/// take `&mut self` so the cache can be updated.
pub struct ConformerCtc {
    model: NemoModel,
    cfg: AsrConfig,
    device: Device,
    ctc: CtcHead,
    frontend: Option<Frontend>,
    tokenizer: Option<SpmTokenizer>,
    enc_cache: CompileCache,
    /// Params for each cached encoder key (re-bound before every run — required
    /// for reliable WGPU / some GPU backends after the first execute).
    enc_params: HashMap<u64, Arc<HashMap<String, Vec<f32>>>>,
}

impl ConformerCtc {
    /// Open a `.nemo` checkpoint and prepare the CTC head, mel filterbank, and
    /// SentencePiece tokenizer (when present in the archive).
    ///
    /// Hyperparameters are read from the embedded `model_config.yaml`; no
    /// architecture options are hard-coded beyond NeMo defaults used as
    /// fallbacks.
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let model = NemoModel::open(path).with_context(|| format!("open {}", path.display()))?;
        let cfg = AsrConfig::from_nemo(model.config())?;

        let ctc = {
            let mut w = NemoWeights::new(&model);
            CtcHead::from_weights(&mut w)?
        };
        ensure!(
            ctc.num_classes == cfg.num_classes,
            "CTC head classes {} != config {}",
            ctc.num_classes,
            cfg.num_classes
        );

        let frontend = Frontend::from_model(&model, &cfg).ok();
        let tokenizer = model
            .tokenizers()
            .iter()
            .find(|t| t.name.ends_with(".model"))
            .and_then(|t| SpmTokenizer::from_model_bytes(&t.bytes).ok());

        Ok(Self {
            model,
            cfg,
            device,
            ctc,
            frontend,
            tokenizer,
            enc_cache: CompileCache::new(device, ENC_CACHE_CAPACITY),
            enc_params: HashMap::new(),
        })
    }

    /// Hyperparameters parsed from the checkpoint YAML.
    pub fn config(&self) -> &AsrConfig {
        &self.cfg
    }

    /// Execution device passed to [`Self::open`].
    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of compiled encoder graphs currently held in the cache.
    pub fn cached_encoder_count(&self) -> usize {
        self.enc_cache.len()
    }

    /// Precompile the encoder for the mel-length bucket that covers
    /// `mel_frames` (no audio run). Subsequent [`Self::transcribe`] calls of
    /// similar duration skip graph compile.
    pub fn warm(&mut self, mel_frames: usize) -> Result<()> {
        let bucket = mel::bucket_mel_frames(mel_frames.max(1));
        self.ensure_encoder(bucket)
    }

    fn ensure_encoder(&mut self, bucket_frames: usize) -> Result<()> {
        let key = bucket_frames as u64;
        if self.enc_cache.contains(key) {
            return Ok(());
        }
        let mut w = NemoWeights::new(&self.model);
        let (hir, params, _t) = build_encoder_hir(&self.cfg, &mut w, bucket_frames)?;
        self.enc_params.insert(key, Arc::new(params.clone()));
        let built = built_from_hir(hir, params)?;
        compile_cache_ensure_built(&mut self.enc_cache, key, built)?;
        Ok(())
    }

    /// Run the encoder; returns `[real_t, d_model]` (padded frames sliced off).
    fn encode(&mut self, mel: &MelSpectrogram) -> Result<(Vec<f32>, usize)> {
        ensure!(mel.n_frames > 0, "empty mel");
        let real_frames = mel.n_frames;
        let bucket = mel::bucket_mel_frames(real_frames);
        let padded = mel::pad_mel_to_frames(mel, bucket);

        self.ensure_encoder(bucket)?;

        let key = bucket as u64;
        let params = Arc::clone(
            self.enc_params
                .get(&key)
                .context("encoder params missing after ensure")?,
        );
        let cg = self
            .enc_cache
            .get_or_compile(key, || panic!("encoder cache missing after ensure"));
        for (name, data) in params.iter() {
            cg.set_param(name, data);
        }
        let out = cg
            .run(&[("mel", padded.data.as_slice())])
            .into_iter()
            .next()
            .context("encoder produced no output")?;
        self.slice_encoder_out(out, real_frames, bucket)
    }

    fn slice_encoder_out(
        &self,
        out: Vec<f32>,
        real_frames: usize,
        bucket: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let real_t = self.cfg.enc_frames(real_frames);
        let bucket_t = self.cfg.enc_frames(bucket);
        ensure!(
            out.len() == bucket_t * self.cfg.d_model,
            "encoder output {} != bucket_t*d {}",
            out.len(),
            bucket_t * self.cfg.d_model
        );
        ensure!(
            real_t <= bucket_t,
            "real enc frames {real_t} > bucket {bucket_t}"
        );
        let keep = real_t * self.cfg.d_model;
        Ok((out[..keep].to_vec(), real_t))
    }

    /// Transcribe mono PCM at the model's sample rate (see [`AsrConfig::sample_rate`]).
    ///
    /// Resample with [`crate::wav::resample`] when the source rate differs.
    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<String> {
        let ids = self.transcribe_ids(pcm)?;
        Ok(self.decode_text(&ids))
    }

    /// Transcribe to CTC token ids (repeats collapsed, blank removed).
    pub fn transcribe_ids(&mut self, pcm: &[f32]) -> Result<Vec<u32>> {
        ensure!(!pcm.is_empty(), "empty audio");
        let mel = mel::log_mel(&self.cfg, pcm, self.frontend.as_ref());
        let (enc, t) = self.encode(&mel)?;
        ensure!(
            enc.len() == t * self.cfg.d_model,
            "encoder output {} != t*d {}",
            enc.len(),
            t * self.cfg.d_model
        );
        let logits = self.ctc.logits(&enc, t)?;
        Ok(ctc::greedy_decode(
            &logits,
            t,
            self.cfg.num_classes,
            self.cfg.blank_id,
        ))
    }

    /// Detokenize CTC piece ids with the checkpoint SentencePiece model.
    ///
    /// Falls back to space-separated numeric ids when no `.model` was found
    /// inside the `.nemo`.
    pub fn decode_text(&self, ids: &[u32]) -> String {
        match &self.tokenizer {
            Some(tok) => tok.decode(ids, true),
            None => ids
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::{bucket_mel_frames, pad_mel_to_frames};

    #[test]
    fn bucket_ladder() {
        assert_eq!(bucket_mel_frames(1), 256);
        assert_eq!(bucket_mel_frames(744), 768);
        assert_eq!(bucket_mel_frames(768), 768);
        assert_eq!(bucket_mel_frames(769), 1024);
    }

    #[test]
    fn pad_preserves_prefix() {
        let m = MelSpectrogram {
            n_mels: 2,
            n_frames: 3,
            data: vec![1., 2., 3., 4., 5., 6.],
        };
        let p = pad_mel_to_frames(&m, 5);
        assert_eq!(p.n_frames, 5);
        assert_eq!(&p.data[0..5], &[1., 2., 3., 0., 0.]);
        assert_eq!(&p.data[5..10], &[4., 5., 6., 0., 0.]);
    }
}
