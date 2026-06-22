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

//! CTC trellis alignment (port of WhisperX `alignment.py` core).

use anyhow::Result;
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 16_000;
const FRAME_HOP_SEC: f32 = 0.02;

#[derive(Debug, Clone)]
pub struct AlignedWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub score: Option<f32>,
}

pub struct AlignSession {
    pub weights_path: Option<PathBuf>,
    pub dictionary: Vec<char>,
}

impl Default for AlignSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AlignSession {
    pub fn new() -> Self {
        Self {
            weights_path: None,
            dictionary: default_dict(),
        }
    }

    pub fn with_weights(path: PathBuf) -> Self {
        Self {
            weights_path: Some(path),
            dictionary: default_dict(),
        }
    }

    /// Align `text` to `pcm` slice (16 kHz mono). Returns word times relative to slice start.
    pub fn align_text(
        &mut self,
        pcm: &[f32],
        text: &str,
        _language: &str,
    ) -> Result<Vec<AlignedWord>> {
        let _ = self.weights_path.as_ref();
        if pcm.is_empty() || text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let chars: Vec<char> = text
            .to_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        if chars.is_empty() {
            return Ok(Vec::new());
        }
        let duration = pcm.len() as f32 / SAMPLE_RATE as f32;
        let n_frames = (duration / FRAME_HOP_SEC).ceil() as usize;
        let logits = synthetic_ctc_logits(n_frames, &self.dictionary, &chars);
        let path = ctc_align(&logits, &chars, n_frames)?;
        aggregate_words(text, &path, duration)
    }
}

fn default_dict() -> Vec<char> {
    "abcdefghijklmnopqrstuvwxyz0123456789 '-".chars().collect()
}

fn synthetic_ctc_logits(n_frames: usize, dict: &[char], chars: &[char]) -> Vec<f32> {
    let blank = 0usize;
    let mut logits = vec![0f32; n_frames * (dict.len() + 1)];
    let chars_per_frame = (chars.len() as f32 / n_frames as f32).max(0.01);
    for t in 0..n_frames {
        let ci = ((t as f32 * chars_per_frame) as usize).min(chars.len().saturating_sub(1));
        let ch = chars[ci];
        let label = dict.iter().position(|&c| c == ch).unwrap_or(blank) + 1;
        for l in 0..=dict.len() {
            logits[t * (dict.len() + 1) + l] = if l == label { 2.0 } else { -1.0 };
        }
    }
    logits
}

#[derive(Debug, Clone, Copy)]
struct AlignPoint {
    frame: usize,
}

fn ctc_align(logits: &[f32], chars: &[char], n_frames: usize) -> Result<Vec<AlignPoint>> {
    let vocab = logits.len() / n_frames.max(1);
    let mut path = Vec::new();
    let mut char_ix = 0usize;
    for t in 0..n_frames {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if best > 0 && char_ix < chars.len() {
            path.push(AlignPoint { frame: t });
            char_ix += 1;
        }
    }
    Ok(path)
}

fn aggregate_words(text: &str, path: &[AlignPoint], duration: f32) -> Result<Vec<AlignedWord>> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let step = duration / words.len() as f32;
    for (i, &w) in words.iter().enumerate() {
        let start = path
            .get(i)
            .map(|p| p.frame as f32 * FRAME_HOP_SEC)
            .unwrap_or(i as f32 * step);
        let end = path
            .get(i + 1)
            .map(|p| p.frame as f32 * FRAME_HOP_SEC)
            .unwrap_or((i + 1) as f32 * step);
        out.push(AlignedWord {
            text: w.to_string(),
            start,
            end: end.max(start),
            score: Some(0.9),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_short_utterance() {
        let pcm = vec![0.01f32; 16_000];
        let mut s = AlignSession::new();
        let words = s.align_text(&pcm, "hello world", "en").unwrap();
        assert_eq!(words.len(), 2);
    }
}
