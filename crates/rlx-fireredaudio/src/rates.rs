//! Sample rates and RedAE / patch length helpers.

/// Mel / understanding encoder sample rate.
pub const UNDERSTAND_SAMPLE_RATE: usize = 16_000;

/// RedAE generation pathway sample rate.
pub const GENERATION_SAMPLE_RATE: usize = 24_000;

/// Waveform samples per 25 Hz RedAE latent (`480 * 2`).
pub const VAE_DOWNSAMPLE_RATE: usize = 960;

/// Waveform samples per backbone patch token (`VAE_DOWNSAMPLE_RATE * patch_size`).
pub const PATCH_ENCODER_DOWNSAMPLE_RATE: usize = VAE_DOWNSAMPLE_RATE * 4;

/// Pad a 24 kHz mono length up to a multiple of [`PATCH_ENCODER_DOWNSAMPLE_RATE`].
pub fn pad_generation_len(num_samples: usize) -> usize {
    let step = PATCH_ENCODER_DOWNSAMPLE_RATE;
    num_samples.div_ceil(step) * step
}

/// Number of 25 Hz VAE frames for a (already patch-aligned) waveform length.
pub fn vae_latent_frames(num_samples: usize) -> usize {
    num_samples / VAE_DOWNSAMPLE_RATE
}

/// Number of 6.25 Hz patch tokens for a (already patch-aligned) waveform length.
pub fn patch_token_frames(num_samples: usize) -> usize {
    num_samples / PATCH_ENCODER_DOWNSAMPLE_RATE
}

/// Whisper-style audio-encoder output length from mel frame count
/// (three `(L-1)//2+1` halvings: conv2 + adapter.conv3 + adapter.conv4).
pub fn audio_encoder_output_len(mel_frames: usize) -> usize {
    let after_conv2 = mel_frames.saturating_sub(1) / 2 + 1;
    let after_conv3 = after_conv2.saturating_sub(1) / 2 + 1;
    after_conv3.saturating_sub(1) / 2 + 1
}
