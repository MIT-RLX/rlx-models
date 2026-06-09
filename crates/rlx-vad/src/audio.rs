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

//! Audio I/O helpers for VAD (16 kHz mono f32).

use anyhow::{Result, anyhow, bail};
use std::fs;
use std::path::Path;

pub const SAMPLE_RATE_16K: usize = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    pub start: usize,
    pub end: usize,
}

/// Parse a minimal WAV (PCM16 mono) into normalized f32 samples.
pub fn parse_wav_mono_f32(bytes: &[u8]) -> Result<(usize, Vec<f32>)> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut pos = 12usize;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut channels = 0u16;
    let mut data_off = None;
    let mut data_len = 0usize;
    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let payload = pos + 8;
        if tag == b"fmt " && sz >= 16 {
            channels = u16::from_le_bytes(bytes[payload + 2..payload + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(bytes[payload + 4..payload + 8].try_into().unwrap());
            bits = u16::from_le_bytes(bytes[payload + 14..payload + 16].try_into().unwrap());
        } else if tag == b"data" {
            data_off = Some(payload);
            data_len = sz;
        }
        pos = payload + sz + (sz & 1);
    }
    let off = data_off.ok_or_else(|| anyhow!("WAV missing data chunk"))?;
    if channels != 1 {
        bail!("expected mono WAV, got {channels} channels");
    }
    if bits != 16 {
        bail!("expected 16-bit PCM, got {bits}-bit");
    }
    if off + data_len > bytes.len() {
        bail!("truncated WAV data");
    }
    let mut pcm = Vec::with_capacity(data_len / 2);
    for chunk in bytes[off..off + data_len].chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        pcm.push(s as f32 / i16::MAX as f32);
    }
    Ok((sample_rate as usize, pcm))
}

pub fn load_wav_mono_f32(path: &Path) -> Result<(usize, Vec<f32>)> {
    parse_wav_mono_f32(&fs::read(path)?)
}

/// Simple linear resample to `target_hz` (mono f32).
pub fn resample_linear(pcm: &[f32], src_hz: usize, target_hz: usize) -> Vec<f32> {
    if src_hz == target_hz || pcm.is_empty() {
        return pcm.to_vec();
    }
    let out_len = pcm.len() * target_hz / src_hz;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * src_hz as f64 / target_hz as f64;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(pcm.len().saturating_sub(1));
        let t = (src - i0 as f64) as f32;
        out.push(pcm[i0] * (1.0 - t) + pcm[i1] * t);
    }
    out
}
