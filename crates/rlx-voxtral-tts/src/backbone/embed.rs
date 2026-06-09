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

//! Text + multi-codebook audio embeddings.

use crate::config::{AudioModelArgs, TextConfig};
use crate::load::PREFIX_MM_EMBED;
use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub const DEFAULT_AUDIO_TOKEN_ID: u32 = 24;

pub struct EmbeddingTables {
    tok: Array2<f32>,
    codebook: Array2<f32>,
    codebook_offsets: Vec<usize>,
    hidden: usize,
    pub audio_token_id: u32,
}

impl EmbeddingTables {
    pub fn from_tensors(
        tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        text: &TextConfig,
        audio: &AudioModelArgs,
    ) -> Result<Self> {
        let tok_key = format!("{PREFIX_MM_EMBED}tok_embeddings.weight");
        let cb_key = format!("{PREFIX_MM_EMBED}audio_codebook_embeddings.embeddings.weight");
        let tok = take2d(tensors, &tok_key)?;
        let codebook = take2d(tensors, &cb_key)?;
        let hidden = text.hidden_size;
        ensure!(tok.dim().1 == hidden, "tok hidden mismatch");
        let sizes = codebook_sizes(audio);
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut acc = 0usize;
        for s in &sizes {
            offsets.push(acc);
            acc += s;
        }
        ensure!(
            codebook.dim().0 >= acc,
            "codebook table rows {} < vocab {}",
            codebook.dim().0,
            acc
        );
        Ok(Self {
            tok,
            codebook,
            codebook_offsets: offsets,
            hidden,
            audio_token_id: DEFAULT_AUDIO_TOKEN_ID,
        })
    }

    pub fn embed_tokens(&self, ids: &[u32]) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((ids.len(), self.hidden));
        for (i, &id) in ids.iter().enumerate() {
            if (id as usize) < self.tok.dim().0 {
                for j in 0..self.hidden {
                    out[[i, j]] = self.tok[[id as usize, j]];
                }
            }
        }
        out
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Scatter preset voice rows into audio placeholder positions.
    pub fn inject_voice(&self, embeds: &mut Array2<f32>, token_ids: &[u32], voice_rows: &[&[f32]]) {
        let mut vi = 0usize;
        for (ti, &tid) in token_ids.iter().enumerate() {
            if tid == self.audio_token_id && vi < voice_rows.len() {
                let row = voice_rows[vi];
                for j in 0..self.hidden.min(row.len()) {
                    embeds[[ti, j]] = row[j];
                }
                vi += 1;
            }
        }
    }

    /// Sum multi-codebook embeddings for one generated frame (`[37]` vLLM layout).
    pub fn embed_audio_frame(&self, frame: &[u32]) -> Array1<f32> {
        let mut out = Array1::<f32>::zeros(self.hidden);
        for (ci, &code) in frame.iter().enumerate().take(37) {
            let table_idx = self.codebook_offsets[ci] + code as usize;
            if table_idx < self.codebook.dim().0 {
                for j in 0..self.hidden {
                    out[j] += self.codebook[[table_idx, j]];
                }
            }
        }
        out
    }
}

fn codebook_sizes(audio: &AudioModelArgs) -> Vec<usize> {
    let sem = audio.semantic_codebook_size + 2;
    let ac = audio.acoustic_codebook_size + 2;
    std::iter::once(sem)
        .chain(std::iter::repeat_n(ac, 36))
        .collect()
}

fn take2d(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Array2<f32>> {
    let (data, shape) = map.get(key).with_context(|| format!("missing {key}"))?;
    ensure!(shape.len() == 2, "{key}: rank 2 expected");
    Array2::from_shape_vec((shape[0], shape[1]), data.clone()).with_context(|| key.to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    fn codebook_size_layout() {
        assert_eq!(8192 + 2, 8194);
    }
}
