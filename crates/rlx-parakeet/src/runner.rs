// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! End-to-end Parakeet-TDT runner: `.nemo` → log-mel → FastConformer encoder graph
//! → host TDT greedy decode → text. Mirrors [`rlx_nemotron_asr::NemotronAsr`] but
//! swaps the RNN-T joint/decode for the **Token-and-Duration Transducer**: the
//! joint grows a duration head ([`crate::TdtJoint`]) and the decode skips a learned
//! number of encoder frames per emitted token
//! ([`crate::transducer::tdt_greedy_decode`]). The FastConformer encoder, mel
//! frontend, LSTM prediction net, tokenizer, and `.nemo` weight bridge are all
//! reused from `rlx-nemotron-asr` (Parakeet-TDT shares that stack).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_flow::WeightSource;
use rlx_nemo::NemoModel;
use rlx_runtime::Device;

use rlx_nemotron_asr::config::AsrConfig;
use rlx_nemotron_asr::decoder::{LstmCell, PredictionNet};
use rlx_nemotron_asr::encoder::build_encoder_hir;
use rlx_nemotron_asr::mel::{self, Frontend};
use rlx_nemotron_asr::tokenizer::SpmTokenizer;
use rlx_nemotron_asr::weights::{NemoWeights, keys};

use crate::joint::TdtJoint;
use crate::transducer::tdt_greedy_decode;

/// Canonical NeMo Parakeet-TDT duration table, used when the checkpoint config
/// does not spell it out (the duration head indexes into this).
const DEFAULT_TDT_DURATIONS: &[i32] = &[0, 1, 2, 3, 4];

/// A loaded Parakeet-TDT model ready to transcribe.
pub struct Parakeet {
    model: NemoModel,
    cfg: AsrConfig,
    device: Device,
    pred: PredictionNet,
    joint: TdtJoint,
    durations: Vec<i32>,
    frontend: Option<Frontend>,
    tokenizer: Option<SpmTokenizer>,
}

impl Parakeet {
    /// Open a `.nemo` Parakeet-TDT checkpoint and build the host-side decode stack.
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let model = NemoModel::open(path).with_context(|| format!("open {}", path.display()))?;
        let cfg = AsrConfig::from_nemo(model.config())?;

        // TDT duration table (indexes the joint's duration head). NeMo exports it
        // under one of a few keys; fall back to the canonical [0,1,2,3,4].
        let durations: Vec<i32> = [
            "model.model_defaults.tdt_durations",
            "model.joint.jointnet.tdt_durations",
            "model.joint.durations",
            "model.durations",
            "model.tdt_durations",
        ]
        .iter()
        .find_map(|k| model.config().get_i64_vec(k))
        .map(|v| v.into_iter().map(|d| d as i32).collect())
        .unwrap_or_else(|| DEFAULT_TDT_DURATIONS.to_vec());
        let num_durations = durations.len();

        let (pred, joint) = {
            let mut w = NemoWeights::new(&model);
            let pred = build_prediction_net(&mut w)?;
            let joint = TdtJoint::from_weights(&mut w, num_durations)?;
            (pred, joint)
        };

        // Prefer the model's own mel filterbank + window for exact parity.
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
            pred,
            joint,
            durations,
            frontend,
            tokenizer,
        })
    }

    pub fn config(&self) -> &AsrConfig {
        &self.cfg
    }

    /// The resolved TDT duration table.
    pub fn durations(&self) -> &[i32] {
        &self.durations
    }

    /// Run the FastConformer encoder over mel features, returning the
    /// `[enc_frames, d_model]` hidden states (row-major) and the frame count.
    fn encode(&self, mel_data: &[f32], mel_frames: usize) -> Result<(Vec<f32>, usize)> {
        let mut w = NemoWeights::new(&self.model);
        let (hir, params, t) = build_encoder_hir(&self.cfg, &mut w, mel_frames)?;
        let built = built_from_hir(hir, params)?;
        let saved = built.params().clone();
        let mut cg = compile_built(built, self.device)?;
        for (n, d) in &saved {
            cg.set_param(n, d);
        }
        let out = cg
            .run(&[("mel", mel_data)])
            .into_iter()
            .next()
            .context("encoder produced no output")?;
        Ok((out, t))
    }

    /// Transcribe a full utterance of mono PCM at the model's sample rate.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        let ids = self.transcribe_ids(pcm)?;
        Ok(self.decode_text(&ids))
    }

    /// Transcribe to raw TDT token ids (blank excluded).
    pub fn transcribe_ids(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        ensure!(!pcm.is_empty(), "empty audio");
        let mel = mel::log_mel(&self.cfg, pcm, self.frontend.as_ref());
        let (enc, t) = self.encode(&mel.data, mel.n_frames)?;
        // The encoder graph emits [1, t, d_model] row-major == [t, d_model].
        ensure!(
            enc.len() == t * self.cfg.d_model,
            "encoder output {} != t*d {}",
            enc.len(),
            t * self.cfg.d_model
        );
        let res = tdt_greedy_decode(
            &self.pred,
            &self.joint,
            &enc,
            &self.durations,
            self.cfg.max_symbols_per_step,
        )?;
        Ok(res.token_ids.iter().map(|&i| i as u32).collect())
    }

    /// Map TDT ids to text using the bundled SentencePiece model, if any.
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

/// Build the LSTM prediction net from a `.nemo` [`WeightSource`] (embedding +
/// stacked LSTM layers). Same layout as Nemotron ASR — Parakeet-TDT shares the
/// prediction network verbatim; only the joint and decode differ.
fn build_prediction_net(w: &mut dyn WeightSource) -> Result<PredictionNet> {
    let (embed, embed_sh) = w.take(keys::PRED_EMBED, false)?;
    let vocab = embed_sh[0];
    let embed_dim = embed_sh[1];

    let mut lstms = Vec::new();
    let mut layer = 0usize;
    while let Ok((ih, ih_sh)) = w.take(&keys::pred_lstm("weight_ih", layer), false) {
        let (hh, _) = w.take(&keys::pred_lstm("weight_hh", layer), false)?;
        let (bih, _) = w.take(&keys::pred_lstm("bias_ih", layer), false)?;
        let (bhh, _) = w.take(&keys::pred_lstm("bias_hh", layer), false)?;
        let hidden = ih_sh[0] / 4;
        let input = ih_sh[1];
        lstms.push(LstmCell::new(hidden, input, ih, hh, bih, bhh)?);
        layer += 1;
    }
    PredictionNet::new(embed, embed_dim, vocab, lstms)
}
