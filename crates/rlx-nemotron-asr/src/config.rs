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

//! Hyperparameters for the Nemotron 3.5 ASR model, populated from the
//! `.nemo`'s `model_config.yaml` — nothing here is hard-coded magic; the
//! constants are only fallbacks used when a field is absent.

use anyhow::{Result, ensure};
use rlx_nemo::NemoConfig;

/// FastConformer encoder + RNN-T decoder hyperparameters.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    // ── preprocessor / mel frontend ──
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    /// `per_feature` (NeMo default) normalizes each mel bin over time.
    pub normalize: String,

    // ── FastConformer encoder ──
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ff_expansion: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    /// Default `[left, right]` attention context in encoder frames.
    pub att_context_size: [usize; 2],
    /// Switchable streaming presets `[left, right]` (cache-aware).
    pub att_context_presets: Vec<[usize; 2]>,

    // ── RNN-T prediction net + joint ──
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub joint_hidden: usize,
    /// Acoustic vocabulary size (text tokens, excluding the blank).
    pub vocab_size: usize,
    /// Blank index for the transducer (NeMo convention: == vocab_size).
    pub blank_id: usize,
    /// Max non-blank symbols emitted per encoder frame in greedy decode.
    pub max_symbols_per_step: usize,

    // ── language conditioning (Nemotron 3.5) ──
    /// Width of the one-hot language vector fused into the decoder input.
    pub num_languages: usize,
}

impl AsrConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
    pub fn ff_dim(&self) -> usize {
        self.d_model * self.ff_expansion
    }
    /// Number of encoder output frames for `mel_frames` input frames.
    pub fn enc_frames(&self, mel_frames: usize) -> usize {
        mel_frames.div_ceil(self.subsampling_factor)
    }

    /// Build from a parsed `model_config.yaml`. Reads canonical NeMo
    /// paths with documented fallbacks so a slightly different layout
    /// still yields a usable config.
    pub fn from_nemo(cfg: &NemoConfig) -> Result<Self> {
        let sample_rate = cfg
            .get_usize("preprocessor.sample_rate")
            .or_else(|| cfg.get_usize("sample_rate"))
            .unwrap_or(16_000);
        let n_mels = cfg
            .get_usize("preprocessor.features")
            .or_else(|| cfg.get_usize("encoder.feat_in"))
            .unwrap_or(80);
        let n_fft = cfg.get_usize("preprocessor.n_fft").unwrap_or(512);
        let win_size_s = cfg.get_f64("preprocessor.window_size").unwrap_or(0.025);
        let win_stride_s = cfg.get_f64("preprocessor.window_stride").unwrap_or(0.01);
        let win_length = (win_size_s * sample_rate as f64).round() as usize;
        let hop_length = (win_stride_s * sample_rate as f64).round() as usize;
        let normalize = cfg
            .get_str("preprocessor.normalize")
            .unwrap_or("per_feature")
            .to_string();

        let d_model = cfg.get_usize("encoder.d_model").unwrap_or(1024);
        let n_layers = cfg.get_usize("encoder.n_layers").unwrap_or(24);
        let n_heads = cfg.get_usize("encoder.n_heads").unwrap_or(8);
        let ff_expansion = cfg.get_usize("encoder.ff_expansion_factor").unwrap_or(4);
        let conv_kernel = cfg.get_usize("encoder.conv_kernel_size").unwrap_or(9);
        let subsampling_factor = cfg.get_usize("encoder.subsampling_factor").unwrap_or(8);
        let subsampling_conv_channels = cfg
            .get_usize("encoder.subsampling_conv_channels")
            .unwrap_or(256);

        let (att_context_size, att_context_presets) = parse_att_context(cfg);

        let pred_hidden = cfg
            .get_usize("decoder.prednet.pred_hidden")
            .or_else(|| cfg.get_usize("model_defaults.pred_hidden"))
            .unwrap_or(640);
        let pred_rnn_layers = cfg
            .get_usize("decoder.prednet.pred_rnn_layers")
            .unwrap_or(1);
        let joint_hidden = cfg
            .get_usize("joint.jointnet.joint_hidden")
            .or_else(|| cfg.get_usize("model_defaults.joint_hidden"))
            .unwrap_or(640);

        // Vocabulary: prefer the explicit class count; otherwise the
        // labels list length. `num_classes` in NeMo joints already
        // includes the blank, so the text vocab is one fewer.
        let vocab_size = cfg
            .get_usize("decoder.vocab_size")
            .or_else(|| cfg.seq_len("decoder.vocabulary"))
            .or_else(|| {
                cfg.get_usize("joint.num_classes")
                    .map(|n| n.saturating_sub(1))
            })
            .unwrap_or(0);
        let blank_id = vocab_size;

        let num_languages = cfg
            .get_usize("decoder.num_languages")
            .or_else(|| cfg.get_usize("encoder.num_languages"))
            .unwrap_or(128);

        let c = Self {
            sample_rate,
            n_mels,
            n_fft,
            win_length,
            hop_length,
            normalize,
            d_model,
            n_layers,
            n_heads,
            ff_expansion,
            conv_kernel,
            subsampling_factor,
            subsampling_conv_channels,
            att_context_size,
            att_context_presets,
            pred_hidden,
            pred_rnn_layers,
            joint_hidden,
            vocab_size,
            blank_id,
            max_symbols_per_step: 10,
            num_languages,
        };
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.d_model > 0, "d_model must be > 0");
        ensure!(self.n_heads > 0, "n_heads must be > 0");
        ensure!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model {} not divisible by n_heads {}",
            self.d_model,
            self.n_heads
        );
        ensure!(self.n_layers > 0, "n_layers must be > 0");
        ensure!(
            self.subsampling_factor.is_power_of_two(),
            "subsampling_factor {} must be a power of two (dw_striding)",
            self.subsampling_factor
        );
        Ok(())
    }
}

/// Parse `encoder.att_context_size`, which is either a single `[left, right]`
/// or a list of `[left, right]` presets (cache-aware streaming modes).
fn parse_att_context(cfg: &NemoConfig) -> ([usize; 2], Vec<[usize; 2]>) {
    let default = [70usize, 13usize];
    // Single pair: [left, right].
    if let Some(v) = cfg.get_i64_vec("encoder.att_context_size") {
        let lr = pair_from(&v).unwrap_or(default);
        return (lr, vec![lr]);
    }
    // List of pairs.
    if let Some(m) = cfg.get_i64_matrix("encoder.att_context_size") {
        let presets: Vec<[usize; 2]> = m.iter().filter_map(|r| pair_from(r)).collect();
        if !presets.is_empty() {
            return (presets[0], presets);
        }
    }
    (default, vec![default])
}

fn pair_from(seq: &[i64]) -> Option<[usize; 2]> {
    let l = *seq.first()?;
    let r = *seq.get(1)?;
    Some([usize::try_from(l).ok()?, usize::try_from(r).ok()?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_minimal_yaml() {
        let yaml = br#"
sample_rate: 16000
preprocessor:
  features: 80
  n_fft: 512
  window_size: 0.025
  window_stride: 0.01
  normalize: per_feature
encoder:
  d_model: 1024
  n_layers: 24
  n_heads: 8
  ff_expansion_factor: 4
  conv_kernel_size: 9
  subsampling_factor: 8
  subsampling_conv_channels: 256
  att_context_size:
    - [56, 0]
    - [56, 13]
decoder:
  prednet:
    pred_hidden: 640
    pred_rnn_layers: 1
  vocabulary: ["a", "b", "c", "d"]
joint:
  jointnet:
    joint_hidden: 640
  num_classes: 5
"#;
        let nemo = NemoConfig::from_yaml_bytes(yaml).unwrap();
        let c = AsrConfig::from_nemo(&nemo).unwrap();
        assert_eq!(c.d_model, 1024);
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.ff_dim(), 4096);
        assert_eq!(c.hop_length, 160);
        assert_eq!(c.win_length, 400);
        assert_eq!(c.subsampling_factor, 8);
        assert_eq!(c.att_context_size, [56, 0]);
        assert_eq!(c.att_context_presets.len(), 2);
        assert_eq!(c.vocab_size, 4);
        assert_eq!(c.blank_id, 4);
        assert_eq!(c.enc_frames(80), 10);
    }
}
