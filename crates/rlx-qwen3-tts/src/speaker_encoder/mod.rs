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

//! ECAPA-TDNN speaker embedding for Qwen3-TTS Base voice clone.
//!
//! Mirrors `Qwen3TTSSpeakerEncoder` from the HF `qwen_tts` package: log-mel
//! front-end (24 kHz / 128 mels / 1024 FFT / 256 hop) feeds a small ECAPA-TDNN
//! that emits a 1024-d x-vector consumed by [`crate::voice_clone`].

pub mod config;
pub mod ecapa;
pub mod mel;

pub use config::{MelParams, SpeakerEncoderConfig};
pub use ecapa::SpeakerEncoder;

use crate::load::Qwen3TtsWeightStore;
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Mel → speaker x-vector for voice clone (`ref_spk_embedding`).
pub fn encode_reference_wav(
    model_dir: &Path,
    store: &Qwen3TtsWeightStore,
    ref_wav: &Path,
) -> Result<Vec<f32>> {
    let cfg = SpeakerEncoderConfig::from_model_dir(model_dir)?;
    let pcm = load_wav_mono_f32(ref_wav, cfg.sample_rate)?;
    encode_pcm(model_dir, store, &pcm)
}

/// Same as [`encode_reference_wav`] but the WAV has already been decoded to
/// mono `f32` at the speaker encoder's sample rate.
pub fn encode_pcm(model_dir: &Path, store: &Qwen3TtsWeightStore, pcm: &[f32]) -> Result<Vec<f32>> {
    let debug = std::env::var("RLX_QWEN3_TTS_SPK_TIMING").ok().as_deref() == Some("1");
    let t0 = std::time::Instant::now();
    let cfg = SpeakerEncoderConfig::from_model_dir(model_dir)?;
    if debug {
        eprintln!("[spk] cfg load: {:.3}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let enc = open_speaker_encoder(store, &cfg)?;
    if debug {
        eprintln!("[spk] weights load: {:.3}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    let mel = mel::log_mel(pcm, &cfg.mel_params())?;
    if debug {
        eprintln!("[spk] mel: {:.3}s", t2.elapsed().as_secs_f64());
    }
    let t3 = std::time::Instant::now();
    let x = enc.forward(mel.view());
    if debug {
        eprintln!("[spk] ecapa forward: {:.3}s", t3.elapsed().as_secs_f64());
    }
    Ok(x)
}

/// Load all `speaker_encoder.*` tensors and build the forward module.
pub fn open_speaker_encoder(
    store: &Qwen3TtsWeightStore,
    cfg: &SpeakerEncoderConfig,
) -> Result<SpeakerEncoder> {
    let keys: Vec<String> = store
        .keys()
        .iter()
        .filter(|k| k.starts_with("speaker_encoder."))
        .cloned()
        .collect();
    if keys.is_empty() {
        bail!("no speaker_encoder.* tensors in checkpoint — Base model required");
    }
    let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let snap = store.tensor_snapshot(&refs)?;
    // tensor_snapshot returns owned (data, shape); rewrap as HashMap<String, _>.
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::with_capacity(snap.len());
    for (k, v) in snap {
        raw.insert(k, v);
    }
    ecapa::build_speaker_encoder(cfg, raw)
}

/// Tiny RIFF/WAVE PCM parser; supports mono int16 or float32 at the expected SR.
pub fn load_wav_mono_f32(path: &Path, expected_sr: u32) -> Result<Vec<f32>> {
    let bytes = fs::read(path).map_err(|e| anyhow!("read wav {}: {e}", path.display()))?;
    parse_wav_mono_f32(&bytes, expected_sr)
}

pub fn parse_wav_mono_f32(bytes: &[u8], expected_sr: u32) -> Result<Vec<f32>> {
    if bytes.len() < 44 {
        bail!("wav too small");
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut off = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data_chunk: Option<&[u8]> = None;
    while off + 8 <= bytes.len() {
        let tag = &bytes[off..off + 4];
        let len = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if off + len > bytes.len() {
            break;
        }
        match tag {
            b"fmt " => {
                if len < 16 {
                    bail!("wav fmt chunk too small");
                }
                let audio_format = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
                let channels = u16::from_le_bytes(bytes[off + 2..off + 4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
                let bits_per_sample =
                    u16::from_le_bytes(bytes[off + 14..off + 16].try_into().unwrap());
                fmt = Some((audio_format, channels, sample_rate, bits_per_sample));
            }
            b"data" => {
                data_chunk = Some(&bytes[off..off + len]);
            }
            _ => {}
        }
        off += (len + 1) & !1;
        if fmt.is_some() && data_chunk.is_some() {
            break;
        }
    }
    let (audio_format, channels, sr, bps) = fmt.ok_or_else(|| anyhow!("wav missing fmt chunk"))?;
    if channels != 1 {
        bail!("wav: expected mono, got {channels} channels");
    }
    if sr != expected_sr {
        bail!("wav: expected {expected_sr} Hz, got {sr}");
    }
    let data = data_chunk.ok_or_else(|| anyhow!("wav missing data chunk"))?;
    match (audio_format, bps) {
        (1, 16) => {
            if data.len() % 2 != 0 {
                bail!("wav data chunk not aligned (16-bit PCM)");
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for i in (0..data.len()).step_by(2) {
                out.push(i16::from_le_bytes([data[i], data[i + 1]]) as f32 / 32768.0);
            }
            Ok(out)
        }
        (3, 32) => {
            if data.len() % 4 != 0 {
                bail!("wav data chunk not aligned (32-bit float)");
            }
            let mut out = Vec::with_capacity(data.len() / 4);
            for i in (0..data.len()).step_by(4) {
                out.push(f32::from_le_bytes([
                    data[i],
                    data[i + 1],
                    data[i + 2],
                    data[i + 3],
                ]));
            }
            Ok(out)
        }
        _ => bail!("wav: unsupported format=0x{audio_format:x} bps={bps}"),
    }
}
