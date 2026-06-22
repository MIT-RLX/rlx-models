// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// EnCodec config (facebook/encodec_24khz).

#[derive(Debug, Clone)]
pub struct EncodecConfig {
    pub sampling_rate: u32,
    pub audio_channels: usize,
    pub num_filters: usize,            // 32
    pub upsampling_ratios: Vec<usize>, // [8,5,4,2] (encoder uses reversed)
    pub kernel_size: usize,            // 7
    pub last_kernel_size: usize,       // 7
    pub residual_kernel_size: usize,   // 3
    pub dilation_growth_rate: usize,   // 2
    pub num_residual_layers: usize,    // 1
    pub num_lstm_layers: usize,        // 2
    pub hidden_size: usize,            // 128 (latent dim)
    pub codebook_size: usize,          // 1024
    pub codebook_dim: usize,           // 128
    pub trim_right_ratio: f32,         // 1.0
    pub compress: usize,               // 2 (resnet hidden = dim/compress)
}

impl EncodecConfig {
    pub fn encodec_24khz() -> Self {
        Self {
            sampling_rate: 24_000,
            audio_channels: 1,
            num_filters: 32,
            upsampling_ratios: vec![8, 5, 4, 2],
            kernel_size: 7,
            last_kernel_size: 7,
            residual_kernel_size: 3,
            dilation_growth_rate: 2,
            num_residual_layers: 1,
            num_lstm_layers: 2,
            hidden_size: 128,
            codebook_size: 1024,
            codebook_dim: 128,
            trim_right_ratio: 1.0,
            compress: 2,
        }
    }

    /// Encoder downsampling ratios (reversed upsampling ratios).
    pub fn encoder_ratios(&self) -> Vec<usize> {
        self.upsampling_ratios.iter().rev().copied().collect()
    }

    /// Latent dim entering the LSTM (= num_filters * 2^num_stages).
    pub fn lstm_dim(&self) -> usize {
        self.num_filters * 2usize.pow(self.upsampling_ratios.len() as u32)
    }

    /// Total hop (product of ratios).
    pub fn hop_length(&self) -> usize {
        self.upsampling_ratios.iter().product()
    }
}
