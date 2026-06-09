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

//! Native 12Hz speech tokenizer (decode path).

mod compiled_pt;
pub mod decode;
#[cfg(feature = "incremental-decode")]
pub mod decode_streaming;
pub mod encoder;
pub mod encoder_transformer;
mod gpu_conv;
mod gpu_matmul;
mod layer_scale;
mod ops;
pub mod rvq;

pub use compiled_pt::{speech_pt_backend_label, speech_pt_use_compiled};
pub use decode::St12HzDecoder;

pub fn speech_conv_backend_label(device: Device) -> &'static str {
    if crate::gpu_pipeline::speech_conv_use_gpu(device) {
        "GPU conv+matmul"
    } else {
        "CPU eager"
    }
}

use anyhow::Result;
use rlx_runtime::Device;
use std::path::Path;

/// Open speech decoder and warm GPU `pre_transformer` compile cache (no-op on CPU).
pub fn open_speech_decoder(model_dir: &Path, device: Device) -> Result<St12HzDecoder> {
    let mut dec = St12HzDecoder::open(model_dir)?;
    dec.warmup(device, None)?;
    Ok(dec)
}

/// Open speech decoder and pre-compile GPU conv matmuls for `n_codec_frames`.
pub fn open_speech_decoder_for_frames(
    model_dir: &Path,
    device: Device,
    n_codec_frames: usize,
) -> Result<St12HzDecoder> {
    let mut dec = St12HzDecoder::open(model_dir)?;
    dec.warmup(device, Some(n_codec_frames))?;
    Ok(dec)
}

/// Decode talker codec frames → mono PCM f32 @ 24 kHz.
pub fn decode_codec_frames(
    model_dir: &Path,
    frames: &[Vec<u32>],
    device: Device,
) -> Result<Vec<f32>> {
    open_speech_decoder(model_dir, device)?.decode(frames, device)
}

/// Encode reference WAV → codec frames (Base model encoder path).
///
/// Runs the SEANet conv encoder + Mimi encoder transformer + stride-2 downsample
/// + Split RVQ over a 24 kHz mono WAV. Returns `[T_codes][num_quantizers]`
///   matching the HF reference layout the talker's ICL prompt builder expects.
pub fn encode_wav_to_codec_frames(model_dir: &Path, wav: &Path) -> Result<Vec<Vec<u32>>> {
    use crate::speaker_encoder::parse_wav_mono_f32;
    let bytes =
        std::fs::read(wav).map_err(|e| anyhow::anyhow!("read wav {}: {e}", wav.display()))?;
    let pcm = parse_wav_mono_f32(&bytes, 24_000)?;
    encode_pcm_to_codec_frames(model_dir, &pcm)
}

/// Same as [`encode_wav_to_codec_frames`] but for an already-loaded 24 kHz
/// mono PCM waveform.
pub fn encode_pcm_to_codec_frames(model_dir: &Path, pcm: &[f32]) -> Result<Vec<Vec<u32>>> {
    use crate::speech_tokenizer::encoder::{MimiDownsample, open_conv_encoder};
    use crate::speech_tokenizer::encoder_transformer::open_encoder_transformer;
    use crate::speech_tokenizer::rvq::open_split_rvq;
    use ndarray::Array2;

    let tok_dir = model_dir.join("speech_tokenizer");
    let conv = open_conv_encoder(&tok_dir)?;
    let tf = open_encoder_transformer(&tok_dir)?;
    let down = MimiDownsample::open(&tok_dir, tf.cfg.hidden_size)?;
    let rvq = open_split_rvq(&tok_dir)?;

    // PCM → [audio_channels=1, T_in]
    let t_in = pcm.len();
    let mut input = Array2::<f32>::zeros((1, t_in));
    for (i, &s) in pcm.iter().enumerate() {
        input[[0, i]] = s;
    }
    // SEANet conv stack → [hidden=512, T_conv]
    let conv_out = conv.forward(input.view());
    // → [T_conv, hidden] for the transformer
    let (h, t_conv) = conv_out.dim();
    let mut pre_tf = Array2::<f32>::zeros((t_conv, h));
    for ci in 0..h {
        for ti in 0..t_conv {
            pre_tf[[ti, ci]] = conv_out[[ci, ti]];
        }
    }
    let post_tf = tf.forward(pre_tf.view());
    // → [hidden, T_conv]
    let mut pre_ds = Array2::<f32>::zeros((h, t_conv));
    for ci in 0..h {
        for ti in 0..t_conv {
            pre_ds[[ci, ti]] = post_tf[[ti, ci]];
        }
    }
    let ds_out = down.forward(pre_ds.view());
    // Split RVQ over [hidden, T_codes]. Use only the encoder's valid quantizers
    // (16 for Qwen3-TTS) to match the talker's ICL prompt shape.
    let num_q = Some(rvq.cfg.encoder_valid_num_quantizers);
    let frames = rvq.encode_frames(&ds_out, num_q);
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_one_golden_frame() {
        let dir = match std::env::var("RLX_QWEN3_TTS_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        if !dir.join("speech_tokenizer/model.safetensors").is_file() {
            return;
        }
        let frame = vec![
            1995, 1642, 988, 1088, 246, 1543, 1579, 437, 1356, 86, 1042, 248, 1555, 781, 1772, 374,
        ];
        let pcm = decode_codec_frames(&dir, &[frame], Device::Cpu).expect("decode");
        assert!(pcm.len() > 1000);
    }
}
