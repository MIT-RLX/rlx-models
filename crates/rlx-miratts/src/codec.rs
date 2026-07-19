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

//! MiraTTS FastBiCodec decoder (STEP 3), native on RLX.
//!
//! The `detokenizer.onnx` is a **self-contained** BiCodec decoder: acoustic
//! `speech_tokens [1, 249]` (i64) + reference `context_tokens [1, 1, 32]` (i32)
//! → `output_waveform [1, 1, 79680]` — a 16 kHz low-res waveform (the 1.7 MB
//! FASR `upsampler.pth` optionally lifts it to 48 kHz; 16 kHz is already what
//! Whisper consumes). It imports cleanly into rlx-ir (params=554) and runs on
//! any RLX backend via rlx-tiny-tts's `compile_named`/`run_typed` harness.
//! (`processer.onnx` is only for the safetensors path; the ONNX detokenizer
//! bundles it.)

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

/// Fixed acoustic-token length of `detokenizer.onnx`.
pub const SPEECH_LEN: usize = 249;
/// Fixed reference/global context-token length.
pub const CONTEXT_LEN: usize = 32;
/// Sample rate of the detokenizer's low-res output (pre-upsampler).
pub const SAMPLE_RATE: u32 = 16_000;

/// Native BiCodec decoder (`detokenizer.onnx`) over RLX.
pub struct MiraCodec {
    model: TinyModel,
    device: Device,
}

impl MiraCodec {
    /// Load the decoder from the `decoders/` directory (holds `detokenizer.onnx`).
    pub fn load(decoders_dir: &Path, device: Device) -> Result<Self> {
        anyhow::ensure!(
            decoders_dir.join("detokenizer.onnx").is_file(),
            "detokenizer.onnx missing in {}",
            decoders_dir.display()
        );
        let cfg = BundleConfig {
            model: String::new(),
            sample_rate: SAMPLE_RATE,
            add_blank: false,
            language: "EN".into(),
            speakers: Default::default(),
            default_speaker: None,
            noise_scale: 0.0,
            noise_scale_w: 0.0,
            length_scale: 1.0,
            inter_channels: 0,
            gin_channels: 0,
        };
        Ok(Self {
            model: TinyModel::new(decoders_dir.to_path_buf(), cfg),
            device,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Decode acoustic `speech_codes` + reference `context_codes` → 16 kHz mono.
    /// Both are padded/truncated to the decoder's fixed lengths (249 / 32).
    pub fn decode(&self, speech_codes: &[u32], context_codes: &[u32]) -> Result<Vec<f32>> {
        let speech: Vec<i64> = fit(speech_codes, SPEECH_LEN)
            .into_iter()
            .map(|c| c as i64)
            .collect();
        let context: Vec<i32> = fit(context_codes, CONTEXT_LEN)
            .into_iter()
            .map(|c| c as i32)
            .collect();
        let mut g = self
            .model
            .compile_named("detokenizer", self.device, SPEECH_LEN, &[])
            .map_err(|e| anyhow::anyhow!("compile detokenizer: {e:#}"))?;
        let out = g.run_typed(&[
            ("speech_tokens", &i64_bytes(&speech), DType::I64),
            ("context_tokens", &i32_bytes(&context), DType::I32),
        ]);
        if std::env::var("RLX_MIRA_DBG").is_ok() {
            for (i, (b, dt)) in out.iter().enumerate() {
                let elems = b.len() / if *dt == DType::F16 { 2 } else { 4 };
                eprintln!("[mira-codec] out[{i}] elems={elems} dtype={dt:?}");
            }
        }
        let (bytes, dt) = out
            .into_iter()
            .next()
            .context("detokenizer produced no output")?;
        Ok(bytes_to_f32(&bytes, dt))
    }
}

/// Pad with 0 / truncate to exactly `n`.
fn fit(codes: &[u32], n: usize) -> Vec<u32> {
    let mut v = codes.to_vec();
    v.resize(n, 0);
    v
}

fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F16 => b
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

/// IEEE-754 half → f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal → normalize
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127 + 14) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        assert_eq!(f16_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0xc000), -2.0); // -2.0
        assert!((f16_to_f32(0x3555) - 0.333).abs() < 1e-3); // ~1/3
    }

    #[test]
    fn fit_pads_and_truncates() {
        assert_eq!(fit(&[1, 2, 3], 5), vec![1, 2, 3, 0, 0]);
        assert_eq!(fit(&[1, 2, 3, 4, 5, 6], 4), vec![1, 2, 3, 4]);
    }
}
