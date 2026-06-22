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

//! Host-side token embedding + audio-placeholder fusion.

use crate::config::Qwen3AsrConfig;
use anyhow::{Result, ensure};

/// Look up text token embeddings and overwrite the `audio_token_id` slots with
/// the projected audio vectors (in order), producing `inputs_embeds`.
pub fn fuse_inputs_embeds(
    cfg: &Qwen3AsrConfig,
    embed: &[f32],
    token_ids: &[u32],
    audio_embeds: &[f32],
) -> Result<Vec<f32>> {
    let h = cfg.text.hidden_size;
    let vocab = cfg.text.vocab_size;
    ensure!(
        embed.len() == vocab * h,
        "embed_tokens len {} != vocab*hidden {}",
        embed.len(),
        vocab * h
    );

    let seq = token_ids.len();
    let n_slots = token_ids
        .iter()
        .filter(|&&t| t == cfg.audio_token_id)
        .count();
    let n_vecs = audio_embeds.len() / h;
    ensure!(
        n_slots == n_vecs,
        "audio placeholders ({n_slots}) != audio vectors ({n_vecs})"
    );

    let mut out = vec![0f32; seq * h];
    let mut a = 0usize;
    for (pos, &tok) in token_ids.iter().enumerate() {
        if tok == cfg.audio_token_id {
            out[pos * h..(pos + 1) * h].copy_from_slice(&audio_embeds[a * h..(a + 1) * h]);
            a += 1;
        } else {
            ensure!((tok as usize) < vocab, "token id {tok} >= vocab {vocab}");
            out[pos * h..(pos + 1) * h]
                .copy_from_slice(&embed[tok as usize * h..(tok as usize + 1) * h]);
        }
    }
    Ok(out)
}

/// Greedy argmax over a `[vocab]` logits row.
pub fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Number of `<|audio_pad|>` placeholders in a prompt.
pub fn count_audio_placeholders(cfg: &Qwen3AsrConfig, prompt: &[u32]) -> usize {
    prompt.iter().filter(|&&t| t == cfg.audio_token_id).count()
}
