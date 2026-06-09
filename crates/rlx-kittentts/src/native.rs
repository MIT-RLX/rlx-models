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

//! Native RLX inference via [`kitten_tts_mini_rlx`] (decomposed ONNX graph).

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};

/// Compiled Kitten graph with per-sequence compile cache (ORT-style `[1, seq]`).
pub struct NativeEngine {
    cache: kitten_tts_mini_rlx::bundle_compile::SeqCompileCache,
    pub sequence_length: usize,
    pub max_waveform_samples: usize,
    pub style_dim: usize,
    style_bytes: Mutex<Vec<u8>>,
    speed_bytes: Mutex<Vec<u8>>,
}

impl NativeEngine {
    pub fn load(
        weights_dir: &Path,
        device: Device,
        sequence_length: usize,
        max_waveform_samples: usize,
    ) -> Result<Self> {
        let bundle = kitten_tts_mini_rlx::bundle_compile::bundle_dir_near_weights(weights_dir)
            .with_context(|| format!("find rlx_bundle near {weights_dir:?}"))?;
        let cache = kitten_tts_mini_rlx::bundle_compile::SeqCompileCache::new(
            device,
            bundle,
            sequence_length,
            max_waveform_samples,
            16,
        );
        Ok(Self {
            cache,
            sequence_length,
            max_waveform_samples,
            style_dim: 256,
            style_bytes: Mutex::new(vec![0u8; 256 * 4]),
            speed_bytes: Mutex::new(1.0f32.to_le_bytes().to_vec()),
        })
    }

    /// Run the decomposed graph: `input_ids` `[1, seq]`, `style` `[1, 256]`, `speed` `[1]`.
    pub fn infer(&self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>> {
        if ids.len() > self.sequence_length {
            anyhow::bail!(
                "IPA length {} exceeds compiled sequence_length {} (re-load with a larger --seq-len)",
                ids.len(),
                self.sequence_length
            );
        }
        if style.len() != self.style_dim {
            anyhow::bail!("style dim {} != expected {}", style.len(), self.style_dim);
        }

        let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();

        {
            let mut style_bytes = self.style_bytes.lock().expect("style_bytes mutex");
            for (i, &v) in style.iter().enumerate() {
                style_bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        {
            let mut speed_bytes = self.speed_bytes.lock().expect("speed_bytes mutex");
            speed_bytes.copy_from_slice(&speed.to_le_bytes());
        }

        unsafe {
            kitten_tts_mini_rlx::opts::set_compile_sequence_length(ids.len());
        }

        let style_bytes = self.style_bytes.lock().expect("style_bytes mutex");
        let speed_bytes = self.speed_bytes.lock().expect("speed_bytes mutex");

        let mut graph = self
            .cache
            .graph_for_seq(ids.len())
            .with_context(|| format!("compile kitten graph for seq={}", ids.len()))?;
        let outputs = graph.run_typed(&[
            ("input_ids", ids_bytes.as_slice(), DType::I64),
            ("style", style_bytes.as_slice(), DType::F32),
            ("speed", speed_bytes.as_slice(), DType::F32),
        ]);

        let (waveform_bytes, waveform_dt) = outputs
            .first()
            .context("native Kitten graph returned no outputs")?;
        if waveform_dt != &DType::F32 {
            anyhow::bail!("expected waveform f32, got {waveform_dt:?}");
        }

        let mut audio = decode_f32_waveform(waveform_bytes);
        if let Some((dur_bytes, dur_dt)) = outputs.get(1) {
            if *dur_dt == DType::I64 {
                trim_waveform_from_duration(&mut audio, dur_bytes, ids.len());
            }
        }
        Ok(audio)
    }
}

fn decode_f32_waveform(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_i64_tensor(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("i64 chunk")))
        .collect()
}

/// ONNX emits `sum(duration[:seq]) * 600` waveform samples (24 kHz vocoder hop).
fn trim_waveform_from_duration(audio: &mut Vec<f32>, dur_bytes: &[u8], token_len: usize) -> bool {
    let duration = decode_i64_tensor(dur_bytes);
    let Some(len) = crate::infer_opts::waveform_samples_from_duration(&duration, token_len) else {
        return false;
    };
    if len < audio.len() {
        audio.truncate(len);
    }
    true
}
