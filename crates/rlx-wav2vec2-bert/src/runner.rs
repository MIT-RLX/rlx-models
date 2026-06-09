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

use anyhow::{Context, Result, anyhow, bail, ensure};
use rlx_core::validate_standard_device;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct Wav2Vec2BertRunnerBuilder {
    weights: Option<PathBuf>,
    config_path: Option<PathBuf>,
    preprocessor_config_path: Option<PathBuf>,
    config: Option<crate::Wav2Vec2BertConfig>,
    device: Option<Device>,
    batch: Option<usize>,
    seq: Option<usize>,
}

impl Wav2Vec2BertRunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.weights = Some(path.into());
        self
    }
    pub fn config_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config_path = Some(path.into());
        self
    }
    pub fn preprocessor_config_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.preprocessor_config_path = Some(path.into());
        self
    }
    pub fn config(mut self, cfg: crate::Wav2Vec2BertConfig) -> Self {
        self.config = Some(cfg);
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    pub fn seq(mut self, n: usize) -> Self {
        self.seq = Some(n);
        self
    }

    pub fn build(self) -> Result<Wav2Vec2BertRunner> {
        use crate::{LogMelExtractor, Wav2Vec2BertConfig, Wav2Vec2BertPreprocessConfig};

        let weights = self
            .weights
            .ok_or_else(|| anyhow!("weights path required"))?;
        if !weights.exists() {
            bail!("weights file not found: {weights:?}");
        }
        let _wt_str = weights
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
        let weights_dir = weights
            .parent()
            .ok_or_else(|| anyhow!("weights path has no parent dir"))?;

        let cfg_path = self
            .config_path
            .unwrap_or_else(|| weights_dir.join("config.json"));
        let cfg = match self.config {
            Some(c) => c,
            None => Wav2Vec2BertConfig::from_file(&cfg_path)
                .with_context(|| format!("reading config {cfg_path:?}"))?,
        };

        let pre_cfg_path = self
            .preprocessor_config_path
            .unwrap_or_else(|| weights_dir.join("preprocessor_config.json"));
        let pre_cfg = if pre_cfg_path.exists() {
            Wav2Vec2BertPreprocessConfig::from_file(&pre_cfg_path)
                .with_context(|| format!("reading preprocessor config {pre_cfg_path:?}"))?
        } else {
            Wav2Vec2BertPreprocessConfig::w2v_bert_2_0()
        };
        ensure!(
            pre_cfg.feature_dim() == cfg.feature_projection_input_dim,
            "preprocessor feature_dim {} != model feature_projection_input_dim {}",
            pre_cfg.feature_dim(),
            cfg.feature_projection_input_dim
        );

        let batch = self.batch.unwrap_or(1);
        let seq = self.seq.unwrap_or(128);
        let device = self.device.unwrap_or(Device::Cpu);
        validate_standard_device("wav2vec2-bert", device)?;

        let mut wm = rlx_core::load_weight_map(&weights, rlx_core::W2V_BERT_GGUF_ARCHES)?;
        let built = crate::build_wav2vec2_bert_built(&cfg, &mut wm, batch, seq)?;
        let params = built.params().clone();
        let mut compiled = rlx_core::flow_util::compile_built(built, device)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }

        Ok(Wav2Vec2BertRunner {
            compiled,
            cfg,
            extractor: LogMelExtractor::new(pre_cfg),
            batch,
            seq,
        })
    }
}

/// Wav2Vec2-BERT speech encoder runner.
pub struct Wav2Vec2BertRunner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: crate::Wav2Vec2BertConfig,
    extractor: crate::LogMelExtractor,
    batch: usize,
    seq: usize,
}

impl Wav2Vec2BertRunner {
    pub fn builder() -> Wav2Vec2BertRunnerBuilder {
        Wav2Vec2BertRunnerBuilder::default()
    }

    pub fn config(&self) -> &crate::Wav2Vec2BertConfig {
        &self.cfg
    }

    /// Run the encoder on log-mel features `[batch, seq, feature_dim]`.
    /// Pass `attention_mask` as `[batch, seq]` (1 = valid); `None` uses all-ones.
    pub fn encode_features(
        &mut self,
        input_features: &[f32],
        attention_mask: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let feat_dim = self.cfg.feature_projection_input_dim;
        let expected_feat = self.batch * self.seq * feat_dim;
        if input_features.len() != expected_feat {
            bail!(
                "input_features: expected {expected_feat} f32 ({feat_dim}-dim mel x batch x seq), got {}",
                input_features.len()
            );
        }
        let mask: Vec<f32> = match attention_mask {
            Some(m) => {
                if m.len() != self.batch * self.seq {
                    bail!(
                        "attention_mask: expected {} f32, got {}",
                        self.batch * self.seq,
                        m.len()
                    );
                }
                m.to_vec()
            }
            None => vec![1.0; self.batch * self.seq],
        };
        let outputs = self.compiled.run(&[
            ("input_features", input_features),
            ("attention_mask", &mask),
        ]);
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("wav2vec2_bert forward returned no output"))
    }

    pub fn preprocess_config(&self) -> &crate::Wav2Vec2BertPreprocessConfig {
        self.extractor.config()
    }

    /// Extract log-mel features from mono PCM samples in [-1, 1] at
    /// [`Self::preprocess_config`]'s sampling rate (16 kHz for W2v-BERT 2.0).
    pub fn extract_log_mel(&self, waveform: &[f32]) -> crate::LogMelFeatures {
        let feats = self.extractor.extract(waveform);
        self.extractor.pad_to_seq(feats, self.seq)
    }

    /// End-to-end: mono waveform → encoder hidden states `[batch, seq, hidden]`.
    pub fn encode_waveform(&mut self, waveform: &[f32]) -> Result<Vec<f32>> {
        if self.batch != 1 {
            bail!(
                "encode_waveform supports batch=1 only (compiled batch={})",
                self.batch
            );
        }
        let mel = self.extract_log_mel(waveform);
        self.encode_features(&mel.features, Some(&mel.attention_mask))
    }

    /// Load a 16-bit PCM WAV and run the encoder. Requires sample rate to
    /// match the preprocessor config (16 kHz for W2v-BERT 2.0).
    pub fn encode_wav(&mut self, path: &Path) -> Result<Vec<f32>> {
        use crate::load_wav_mono_f32;
        let (samples, sr) = load_wav_mono_f32(path)?;
        let expected = self.extractor.config().sampling_rate;
        if sr != expected {
            bail!("wav sample rate {sr} Hz != model expectation {expected} Hz (resample first)");
        }
        self.encode_waveform(&samples)
    }
}
