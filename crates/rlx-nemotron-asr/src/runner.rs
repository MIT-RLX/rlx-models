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

//! End-to-end runner: `.nemo` → log-mel → FastConformer encoder graph →
//! host RNN-T greedy decode → text.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_flow::WeightSource;
use rlx_nemo::NemoModel;
use rlx_runtime::Device;

use crate::config::AsrConfig;
use crate::decoder::{self, Joint, LstmCell, PredictionNet, PromptKernel};
use crate::encoder::build_encoder_hir;
use crate::mel::{self, Frontend};
use crate::tokenizer::SpmTokenizer;
use crate::weights::{NemoWeights, keys};

/// A loaded Nemotron ASR model ready to transcribe.
pub struct NemotronAsr {
    model: NemoModel,
    cfg: AsrConfig,
    device: Device,
    pred: PredictionNet,
    joint: Joint,
    prompt: Option<PromptKernel>,
    frontend: Option<Frontend>,
    tokenizer: Option<SpmTokenizer>,
    /// `prompt_dictionary`: language code → prompt index.
    lang_map: std::collections::HashMap<String, usize>,
    lang_index: usize,
}

impl NemotronAsr {
    /// Open a `.nemo` and build the host-side decoder pieces.
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let model = NemoModel::open(path).with_context(|| format!("open {}", path.display()))?;
        let cfg = AsrConfig::from_nemo(model.config())?;

        let (pred, joint, prompt) = {
            let mut w = NemoWeights::new(&model);
            let pred = build_prediction_net(&mut w)?;
            let joint = Joint::from_weights(&mut w, cfg.d_model)?;
            let prompt = PromptKernel::from_weights(&mut w, cfg.d_model)?;
            (pred, joint, prompt)
        };

        // Prefer the model's own mel filterbank + window for exact parity.
        let frontend = Frontend::from_model(&model, &cfg).ok();

        // Language code → prompt index (from the checkpoint's dictionary).
        let lang_map: std::collections::HashMap<String, usize> = model
            .config()
            .get_str_i64_map("model_defaults.prompt_dictionary")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(k, v)| usize::try_from(v).ok().map(|u| (k, u)))
            .collect();

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
            prompt,
            frontend,
            tokenizer,
            lang_map,
            lang_index: 0, // en-US
        })
    }

    pub fn config(&self) -> &AsrConfig {
        &self.cfg
    }

    /// Select the target-language prompt index directly.
    pub fn set_language_index(&mut self, idx: usize) {
        self.lang_index = idx;
    }

    /// Select the target language by code (e.g. `en-US`, `fr`, `ja-JP`).
    /// Returns `false` if the code is not in the checkpoint's dictionary.
    pub fn set_language(&mut self, code: &str) -> bool {
        match self.lang_map.get(code) {
            Some(&idx) => {
                self.lang_index = idx;
                true
            }
            None => false,
        }
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
        Ok(self.decode_text(&ids, true))
    }

    /// Transcribe to raw RNN-T token ids (blank excluded).
    pub fn transcribe_ids(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        ensure!(!pcm.is_empty(), "empty audio");
        let mel = mel::log_mel(&self.cfg, pcm, self.frontend.as_ref());
        let (enc, t) = self.encode(&mel.data, mel.n_frames)?;
        // The graph emits [1, t, d_model] row-major == [t, d_model].
        ensure!(
            enc.len() == t * self.cfg.d_model,
            "encoder output {} != t*d {}",
            enc.len(),
            t * self.cfg.d_model
        );
        let lang = self.language_vector();
        Ok(decoder::greedy_decode(
            &self.cfg,
            &self.pred,
            &self.joint,
            self.prompt.as_ref(),
            &enc,
            self.cfg.d_model,
            &lang,
        ))
    }

    /// Map RNN-T ids to text using the bundled SentencePiece model, if any.
    pub fn decode_text(&self, ids: &[u32], strip_lang_tags: bool) -> String {
        match &self.tokenizer {
            Some(tok) => tok.decode(ids, strip_lang_tags),
            None => ids
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// One-hot language conditioning vector for the `prompt_kernel`. The
    /// model is prompt-conditioned (`prompt_field: target_lang`), so a valid
    /// one-hot is required; defaults to en-US (index 0).
    fn language_vector(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.cfg.num_languages];
        let idx = self
            .lang_index
            .min(self.cfg.num_languages.saturating_sub(1));
        v[idx] = 1.0;
        v
    }
}

fn build_prediction_net(w: &mut dyn WeightSource) -> Result<PredictionNet> {
    let (embed, embed_sh) = w.take(keys::PRED_EMBED, false)?;
    let vocab = embed_sh[0];
    let embed_dim = embed_sh[1];

    // Discover stacked LSTM layers (`weight_ih_l0`, `_l1`, …) by probing
    // with `take` (reliably forwarded), so an absent layer simply ends it.
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
