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

//! Audio I/O helpers for AEC (16 kHz mono f32).

use anyhow::{Result, anyhow, bail};
use std::fs;
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;

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

/// Load mono f32 at [`SAMPLE_RATE`], resampling if needed.
pub fn load_wav_16k(path: &Path) -> Result<Vec<f32>> {
    let (sr, pcm) = load_wav_mono_f32(path)?;
    let pcm = resample_linear(&pcm, sr, SAMPLE_RATE);
    Ok(pcm)
}

/// Simple linear resample to `target_hz` (mono f32).
pub fn resample_linear(pcm: &[f32], src_hz: usize, target_hz: usize) -> Vec<f32> {
    if src_hz == target_hz || pcm.is_empty() {
        return pcm.to_vec();
    }
    let out_len = (pcm.len() as f64 * target_hz as f64 / src_hz as f64).round() as usize;
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len.max(1) {
        let src = i as f64 * src_hz as f64 / target_hz as f64;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = pcm.get(idx).copied().unwrap_or(0.0);
        let b = pcm.get(idx + 1).copied().unwrap_or(a);
        out.push(a + frac * (b - a));
    }
    out
}

pub fn write_wav_mono_16(path: &Path, pcm: &[f32]) -> Result<()> {
    let mut bytes = Vec::new();
    let data_len = pcm.len() * 2;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE as u32 * 2).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| anyhow!("write wav {path:?}: {e}"))?;
    Ok(())
}
