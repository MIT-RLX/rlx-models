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

//! Minimal RIFF/WAVE reader: PCM-16 and IEEE-float-32, mono or multi-channel
//! (downmixed to mono). Returns `f32` samples in `[-1, 1]` plus the sample rate.

use anyhow::{Result, anyhow, bail};

/// Decoded mono PCM with its sample rate.
pub struct Wav {
    /// Mono samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Sample rate (Hz).
    pub sample_rate: u32,
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Parse a WAV file from raw bytes.
pub fn parse(bytes: &[u8]) -> Result<Wav> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32le(bytes, pos + 4) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(bytes.len());
        match id {
            b"fmt " => {
                let b = &bytes[body_start..body_end];
                if b.len() < 16 {
                    bail!("short fmt chunk");
                }
                fmt = Some((u16le(b, 0), u16le(b, 2), u32le(b, 4), u16le(b, 14)));
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        pos = body_start + size + (size & 1); // chunks are word-aligned
    }
    let (format, channels, rate, bits) = fmt.ok_or_else(|| anyhow!("missing fmt chunk"))?;
    let data = data.ok_or_else(|| anyhow!("missing data chunk"))?;
    let ch = channels.max(1) as usize;

    let mono: Vec<f32> = match (format, bits) {
        (1, 16) => data
            .chunks_exact(2 * ch)
            .map(|frame| {
                let s: i32 = (0..ch)
                    .map(|c| i16::from_le_bytes([frame[2 * c], frame[2 * c + 1]]) as i32)
                    .sum();
                (s as f32 / ch as f32) / 32768.0
            })
            .collect(),
        (3, 32) => data
            .chunks_exact(4 * ch)
            .map(|frame| {
                let s: f32 = (0..ch)
                    .map(|c| {
                        f32::from_le_bytes([
                            frame[4 * c],
                            frame[4 * c + 1],
                            frame[4 * c + 2],
                            frame[4 * c + 3],
                        ])
                    })
                    .sum();
                s / ch as f32
            })
            .collect(),
        _ => bail!("unsupported WAV format {format} / {bits}-bit (need PCM16 or float32)"),
    };

    Ok(Wav {
        samples: mono,
        sample_rate: rate,
    })
}

/// Linear-interpolation resample to `target` Hz (adequate for ASR frontends).
pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let i0 = src.floor() as usize;
        let frac = (src - i0 as f64) as f32;
        let a = samples.get(i0).copied().unwrap_or(0.0);
        let b = samples.get(i0 + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}
