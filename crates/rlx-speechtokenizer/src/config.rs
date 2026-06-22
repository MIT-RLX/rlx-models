// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SpeechTokenizer config (fnlp/SpeechTokenizer, speechtokenizer_hubert_avg).

#[derive(Debug, Clone)]
pub struct SpeechTokenizerConfig {
    pub sampling_rate: u32,
    pub audio_channels: usize,
    pub n_filters: usize,            // 64
    pub ratios: Vec<usize>,          // [8,5,4,2] (encoder uses reversed)
    pub dimension: usize,            // 1024 (latent = lstm dim)
    pub kernel_size: usize,          // 7
    pub last_kernel_size: usize,     // 7
    pub residual_kernel_size: usize, // 3
    pub dilation_base: usize,        // 2
    pub n_residual_layers: usize,    // 1
    pub lstm_layers: usize,          // 2
    pub codebook_size: usize,        // 1024
    pub n_q: usize,                  // 8
    pub compress: usize,             // 2
}

impl SpeechTokenizerConfig {
    pub fn default_16khz() -> Self {
        Self {
            sampling_rate: 16_000,
            audio_channels: 1,
            n_filters: 64,
            ratios: vec![8, 5, 4, 2],
            dimension: 1024,
            kernel_size: 7,
            last_kernel_size: 7,
            residual_kernel_size: 3,
            dilation_base: 2,
            n_residual_layers: 1,
            lstm_layers: 2,
            codebook_size: 1024,
            n_q: 8,
            compress: 2,
        }
    }

    pub fn encoder_ratios(&self) -> Vec<usize> {
        self.ratios.iter().rev().copied().collect()
    }

    pub fn hop_length(&self) -> usize {
        self.ratios.iter().product()
    }
}
