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

//! Hyperparameters for NeMo EncDecCTC Conformer models, populated from the
//! `.nemo`'s `model_config.yaml`.

use anyhow::{Result, ensure};
use rlx_nemo::NemoConfig;

/// Conformer encoder + CTC decoder hyperparameters.
///
/// Built with [`AsrConfig::from_nemo`] from the checkpoint YAML. Values match
/// NeMo `EncDecCTCModelBPE` field names under `preprocessor` / `encoder` /
/// `decoder`.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    // ── preprocessor / mel frontend ──
    /// Expected PCM sample rate (Hz).
    pub sample_rate: usize,
    /// Mel filterbank size (`preprocessor.features`).
    pub n_mels: usize,
    /// STFT FFT size.
    pub n_fft: usize,
    /// Analysis window length in samples.
    pub win_length: usize,
    /// STFT hop length in samples.
    pub hop_length: usize,
    /// `per_feature` (NeMo default) normalizes each mel bin over time.
    pub normalize: String,

    // ── Conformer encoder ──
    /// Model / hidden width.
    pub d_model: usize,
    /// Number of Conformer blocks.
    pub n_layers: usize,
    /// Attention heads per block.
    pub n_heads: usize,
    /// Feed-forward expansion factor (`ff_dim = d_model * ff_expansion`).
    pub ff_expansion: usize,
    /// Depthwise conv kernel size inside each ConvModule.
    pub conv_kernel: usize,
    /// `striding` (classic, implemented) or `dw_striding` (FastConformer).
    pub subsampling: String,
    /// Time-reduction factor (power of two; classic small model uses 4).
    pub subsampling_factor: usize,
    /// Intermediate channels in the striding Conv2d stack.
    pub subsampling_conv_channels: usize,

    // ── CTC decoder (ConvASRDecoder) ──
    /// Text vocabulary size (SentencePiece pieces, excluding blank).
    pub vocab_size: usize,
    /// CTC blank index (NeMo: last class = `vocab_size`).
    pub blank_id: usize,
    /// Logits width = `vocab_size + 1` (blank included).
    pub num_classes: usize,
}

impl AsrConfig {
    /// Attention head dimension (`d_model / n_heads`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
    /// Feed-forward hidden size (`d_model * ff_expansion`).
    pub fn ff_dim(&self) -> usize {
        self.d_model * self.ff_expansion
    }
    /// Encoder time steps for `mel_frames` input frames after subsampling.
    pub fn enc_frames(&self, mel_frames: usize) -> usize {
        let mut t = mel_frames;
        let n_stages = (self.subsampling_factor as f64).log2().round() as usize;
        for _ in 0..n_stages {
            // Symmetric pad=1, k=3, stride=2: (t + 2 - 3) / 2 + 1.
            t = (t + 2 - 3) / 2 + 1;
        }
        t
    }

    /// Frequency bins after striding subsampling (symmetric pad=1).
    pub fn freq_after_subsample(&self) -> usize {
        let mut f = self.n_mels;
        let n_stages = (self.subsampling_factor as f64).log2().round() as usize;
        for _ in 0..n_stages {
            f = (f + 2 - 3) / 2 + 1;
        }
        f
    }

    /// Parse hyperparameters from a [`NemoConfig`] (embedded YAML).
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

        let d_model = cfg.get_usize("encoder.d_model").unwrap_or(176);
        let n_layers = cfg.get_usize("encoder.n_layers").unwrap_or(16);
        let n_heads = cfg.get_usize("encoder.n_heads").unwrap_or(4);
        let ff_expansion = cfg.get_usize("encoder.ff_expansion_factor").unwrap_or(4);
        let conv_kernel = cfg.get_usize("encoder.conv_kernel_size").unwrap_or(31);
        let subsampling = cfg
            .get_str("encoder.subsampling")
            .unwrap_or("striding")
            .to_string();
        let subsampling_factor = cfg.get_usize("encoder.subsampling_factor").unwrap_or(4);
        let subsampling_conv_channels = cfg
            .get_usize("encoder.subsampling_conv_channels")
            .unwrap_or(d_model);

        // ConvASRDecoder: `num_classes` is the text vocab; the linear head
        // emits `num_classes + 1` logits (blank last). Prefer vocabulary
        // length when present.
        let vocab_size = cfg
            .get_usize("decoder.num_classes")
            .or_else(|| cfg.seq_len("decoder.vocabulary"))
            .unwrap_or(1024);
        let blank_id = vocab_size;
        let num_classes = vocab_size + 1;

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
            subsampling,
            subsampling_factor,
            subsampling_conv_channels,
            vocab_size,
            blank_id,
            num_classes,
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
            "subsampling_factor {} must be a power of two",
            self.subsampling_factor
        );
        ensure!(
            self.subsampling == "striding" || self.subsampling == "dw_striding",
            "unsupported encoder.subsampling {:?}; need striding or dw_striding",
            self.subsampling
        );
        ensure!(self.vocab_size > 0, "vocab_size must be > 0");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_small_ctc_yaml() {
        let yaml = br#"
sample_rate: 16000
preprocessor:
  features: 80
  n_fft: 512
  window_size: 0.025
  window_stride: 0.01
  normalize: per_feature
encoder:
  d_model: 176
  n_layers: 16
  n_heads: 4
  ff_expansion_factor: 4
  conv_kernel_size: 31
  subsampling: striding
  subsampling_factor: 4
  subsampling_conv_channels: 176
decoder:
  feat_in: 176
  num_classes: 1024
"#;
        let nemo = NemoConfig::from_yaml_bytes(yaml).unwrap();
        let c = AsrConfig::from_nemo(&nemo).unwrap();
        assert_eq!(c.d_model, 176);
        assert_eq!(c.head_dim(), 44);
        assert_eq!(c.ff_dim(), 704);
        assert_eq!(c.hop_length, 160);
        assert_eq!(c.subsampling_factor, 4);
        assert_eq!(c.subsampling, "striding");
        assert_eq!(c.vocab_size, 1024);
        assert_eq!(c.blank_id, 1024);
        assert_eq!(c.num_classes, 1025);
        assert_eq!(c.freq_after_subsample(), 20);
        assert_eq!(c.enc_frames(100), 25);
    }
}
