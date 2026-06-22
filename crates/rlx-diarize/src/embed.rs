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

//! Lightweight speaker embedding from mel statistics (placeholder for ECAPA-TDNN RLX graph).

const SAMPLE_RATE: u32 = 16_000;

pub fn embed_window(pcm: &[f32]) -> Vec<f32> {
    let n_mels = 80usize;
    let mut emb = vec![0f32; n_mels];
    if pcm.is_empty() {
        return emb;
    }
    let frame = pcm.len() / n_mels.max(1);
    for (i, e) in emb.iter_mut().enumerate().take(n_mels) {
        let start = i * frame;
        let end = ((i + 1) * frame).min(pcm.len());
        if start < end {
            *e = pcm[start..end].iter().map(|x| x * x).sum::<f32>() / (end - start) as f32;
        }
    }
    l2_normalize(&mut emb);
    emb
}

fn l2_normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn window_samples(window_sec: f32) -> usize {
    (window_sec * SAMPLE_RATE as f32) as usize
}
