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

//! Reference-audio prep matching official `f5_tts.infer.utils_infer`
//! (`remove_silence_edges` + 50 ms pad).

use crate::config::SAMPLE_RATE;

/// Trim leading/trailing silence (~−42 dBFS) and append 50 ms of silence —
/// same recipe as SWivid/F5-TTS before duration estimation.
pub fn preprocess_ref_audio(pcm: &[f32], sample_rate: u32) -> Vec<f32> {
    let sr = if sample_rate == 0 {
        SAMPLE_RATE
    } else {
        sample_rate
    };
    let thresh = 10f32.powf(-42.0 / 20.0); // pydub default silence_threshold=-42
    let start = pcm.iter().position(|&s| s.abs() >= thresh).unwrap_or(0);
    let end = pcm
        .iter()
        .rposition(|&s| s.abs() >= thresh)
        .map(|i| i + 1)
        .unwrap_or(pcm.len());
    let mut out = if start < end {
        pcm[start..end].to_vec()
    } else {
        pcm.to_vec()
    };
    let pad = (sr as usize) / 20; // 50 ms
    out.extend(std::iter::repeat_n(0.0, pad));
    out
}

/// Soft peak limit so Vocos overshoot does not hard-clip the WAV.
pub fn soft_peak_limit(pcm: &mut [f32], ceiling: f32) {
    let peak = pcm.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak > ceiling && peak > 0.0 {
        let g = ceiling / peak;
        for s in pcm.iter_mut() {
            *s *= g;
        }
    }
}
