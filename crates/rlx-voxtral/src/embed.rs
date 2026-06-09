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

//! Host-side token embedding + audio placeholder fusion.

use crate::config::VoxtralConfig;
use crate::weights::VoxtralWeightPrefix;
use anyhow::{Result, bail, ensure};
use rlx_core::weight_map::WeightMap;
use std::path::Path;

const BOS_TOKEN_ID: u32 = 1;
const INST_TOKEN_ID: u32 = 3;
const BEGIN_AUDIO_TOKEN_ID: u32 = 25;
const END_INST_TOKEN_ID: u32 = 4;
const LANG_TOKEN_ID: u32 = 9909;
const COLON_TOKEN_ID: u32 = 1058;
const TRANSCRIBE_TOKEN_ID: u32 = 34;

/// Lookup token embeddings and scatter projected audio vectors at `audio_token_id` slots.
pub fn fuse_inputs_embeds(
    cfg: &VoxtralConfig,
    weights: &WeightMap,
    token_ids: &[u32],
    audio_embeds: &[f32],
) -> Result<Vec<f32>> {
    let h = cfg.text_config.hidden_size;
    let vocab = cfg.text_config.vocab_size;
    let embed_key = VoxtralWeightPrefix::lm_embed_tokens();
    let (embed, shape) = weights
        .get(embed_key)
        .ok_or_else(|| anyhow::anyhow!("missing {embed_key}"))?;
    ensure!(
        shape == [vocab, h],
        "unexpected embed shape {shape:?}, expected [{vocab}, {h}]"
    );
    let embed = embed.to_vec();

    let seq = token_ids.len();
    let n_audio_slots = token_ids
        .iter()
        .filter(|&&id| id == cfg.audio_token_id)
        .count();
    let n_audio_vecs = audio_embeds.len() / h;
    ensure!(
        n_audio_slots == n_audio_vecs,
        "audio token placeholders ({n_audio_slots}) != audio vectors ({n_audio_vecs})"
    );

    let mut out = vec![0f32; seq * h];
    let mut audio_idx = 0usize;
    for (pos, &tok) in token_ids.iter().enumerate() {
        if tok == cfg.audio_token_id {
            let src = &audio_embeds[audio_idx * h..(audio_idx + 1) * h];
            out[pos * h..(pos + 1) * h].copy_from_slice(src);
            audio_idx += 1;
            continue;
        }
        ensure!(
            (tok as usize) < vocab,
            "token id {tok} out of vocab range {vocab}"
        );
        let row = &embed[tok as usize * h..(tok as usize + 1) * h];
        out[pos * h..(pos + 1) * h].copy_from_slice(row);
    }
    Ok(out)
}

/// Greedy argmax on a logits row `[vocab]`.
pub fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn language_code_token_id(model_dir: Option<&Path>, lang: &str) -> Result<u32> {
    #[cfg(feature = "tokenizer")]
    {
        if let Some(dir) = model_dir {
            let tekken = dir.join("tekken.json");
            if tekken.is_file() {
                let tok = tokenizers::Tokenizer::from_file(&tekken)
                    .map_err(|e| anyhow::anyhow!("load tekken tokenizer {tekken:?}: {e}"))?;
                let enc = tok
                    .encode(lang, false)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let ids = enc.get_ids();
                ensure!(!ids.is_empty(), "empty tokenization for language {lang:?}");
                return Ok(ids[0]);
            }
        }
    }
    match lang {
        "en" => Ok(1262),
        "fr" => Ok(7064),
        "de" => Ok(1558),
        "es" => Ok(4613),
        "pt" => Ok(8551),
        "hi" => Ok(6797),
        "nl" => Ok(7371),
        "it" => Ok(6360),
        other => bail!(
            "unknown language code {other:?}; place tekken.json next to weights or use en/fr/de/es/pt/hi/nl/it"
        ),
    }
}

/// Build a Mistral transcription prompt matching `apply_transcription_request`.
pub fn transcription_prompt_ids(
    cfg: &VoxtralConfig,
    n_audio: usize,
    language: Option<&str>,
    model_dir: Option<&Path>,
) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(3 + n_audio + 6);
    out.extend([BOS_TOKEN_ID, INST_TOKEN_ID, BEGIN_AUDIO_TOKEN_ID]);
    out.extend(std::iter::repeat_n(cfg.audio_token_id, n_audio));
    out.push(END_INST_TOKEN_ID);
    if let Some(lang) = language {
        out.push(LANG_TOKEN_ID);
        out.push(COLON_TOKEN_ID);
        out.push(language_code_token_id(model_dir, lang)?);
    }
    out.push(TRANSCRIBE_TOKEN_ID);
    Ok(out)
}

pub fn decode_token_ids(model_dir: Option<&Path>, ids: &[u32]) -> Result<String> {
    #[cfg(feature = "tokenizer")]
    if let Some(dir) = model_dir {
        let tekken = dir.join("tekken.json");
        if tekken.is_file() {
            let tok = tokenizers::Tokenizer::from_file(&tekken)
                .map_err(|e| anyhow::anyhow!("load tekken tokenizer {tekken:?}: {e}"))?;
            return tok.decode(ids, true).map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    Ok(format!("{ids:?}"))
}

pub fn validate_prompt_audio_match(
    cfg: &VoxtralConfig,
    prompt: &[u32],
    n_audio: usize,
) -> Result<()> {
    let placeholders = prompt
        .iter()
        .filter(|&&id| id == cfg.audio_token_id)
        .count();
    if placeholders != n_audio {
        bail!(
            "prompt has {placeholders} audio placeholders (token {}), need {n_audio}",
            cfg.audio_token_id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcription_prompt_matches_hf_template() {
        let cfg = VoxtralConfig::tiny_synthetic();
        let prompt = transcription_prompt_ids(&cfg, 4, Some("en"), None).unwrap();
        assert_eq!(
            prompt,
            vec![1, 3, 25, 24, 24, 24, 24, 4, 9909, 1058, 1262, 34]
        );
        let auto = transcription_prompt_ids(&cfg, 4, None, None).unwrap();
        assert_eq!(auto, vec![1, 3, 25, 24, 24, 24, 24, 4, 34]);
    }
}
