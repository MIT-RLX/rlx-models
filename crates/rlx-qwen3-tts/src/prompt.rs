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

//! CustomVoice prompt embeddings (non-streaming, batch=1) — aligned with HF `generate()`.

use crate::config::{Qwen3TtsConfig, TalkerConfig};
use crate::load::Qwen3TtsWeightStore;
use crate::text_embed::TextEmbedder;
use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct CustomVoicePrompt {
    pub embeds: Array2<f32>,
    /// `tts_pad` projection — HF `trailing_text_hidden` for non-streaming (length 1).
    pub tts_pad_embed: Vec<f32>,
}

pub fn load_text_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    if path.is_file() {
        return Tokenizer::from_file(&path).map_err(|e| anyhow::anyhow!("{e}"));
    }
    anyhow::bail!(
        "missing tokenizer.json under {} — re-run `just fetch-qwen3-tts`",
        model_dir.display()
    )
}

/// Matches `Qwen3TTSModel._build_assistant_text`.
pub fn build_assistant_text(user_text: &str) -> String {
    format!("<|im_start|>assistant\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
}

pub fn build_custom_voice_prompt(
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    text_embedder: &TextEmbedder,
    tokenizer: &Tokenizer,
    user_text: &str,
    speaker: &str,
    language: &str,
) -> Result<CustomVoicePrompt> {
    build_custom_voice_prompt_inner(
        cfg,
        store,
        text_embedder,
        tokenizer,
        user_text,
        Some(speaker),
        language,
        None,
    )
}

/// Same prompt skeleton as `build_custom_voice_prompt`, but uses a supplied
/// 1024-d speaker embedding (e.g. ECAPA x-vector) in place of the codec
/// speaker-ID lookup. Used by the voice-clone XVectorOnly path.
pub fn build_custom_voice_prompt_with_speaker_embed(
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    text_embedder: &TextEmbedder,
    tokenizer: &Tokenizer,
    user_text: &str,
    speaker_embed: &[f32],
    language: &str,
) -> Result<CustomVoicePrompt> {
    build_custom_voice_prompt_inner(
        cfg,
        store,
        text_embedder,
        tokenizer,
        user_text,
        None,
        language,
        Some(speaker_embed),
    )
}

fn build_custom_voice_prompt_inner(
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    text_embedder: &TextEmbedder,
    tokenizer: &Tokenizer,
    user_text: &str,
    speaker: Option<&str>,
    language: &str,
    speaker_embed_override: Option<&[f32]>,
) -> Result<CustomVoicePrompt> {
    let talker = cfg.talker();
    let assistant = build_assistant_text(user_text);
    let encoding = tokenizer
        .encode(assistant.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    ensure!(
        input_ids.len() >= 8,
        "prompt too short ({} tokens)",
        input_ids.len()
    );

    let codec_table = load_codec_embedding(store, talker)?;
    let speaker_row: Vec<f32> = match speaker_embed_override {
        Some(emb) => {
            ensure!(
                emb.len() == talker.hidden_size,
                "speaker_embed_override len {} != hidden {}",
                emb.len(),
                talker.hidden_size
            );
            emb.to_vec()
        }
        None => {
            let name = speaker.context("speaker name required when no embedding override")?;
            let spk_id = talker
                .spk_id
                .get(&name.to_lowercase())
                .copied()
                .with_context(|| format!("unknown speaker {name:?}"))?;
            codec_row(&codec_table, spk_id)
        }
    };
    let language_id = if language.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(
            talker
                .codec_language_id
                .get(&language.to_lowercase())
                .copied()
                .with_context(|| format!("unknown language {language:?}"))?,
        )
    };

    let text = text_embedder.embed_project_ids(&input_ids)?;

    let tts_tokens = [
        cfg.tts_bos_token_id,
        cfg.tts_eos_token_id,
        cfg.tts_pad_token_id,
    ];
    let tts_chunks = text_embedder.embed_project_ids(&tts_tokens)?;
    let tts_bos = tts_chunks[0].clone();
    let tts_eos = tts_chunks[1].clone();
    let tts_pad = tts_chunks[2].clone();

    let codec_prefill: Vec<u32> = match language_id {
        None => vec![
            talker.codec_nothink_id,
            talker.codec_think_bos_id,
            talker.codec_think_eos_id,
        ],
        Some(lang) => vec![
            talker.codec_think_id,
            talker.codec_think_bos_id,
            lang,
            talker.codec_think_eos_id,
        ],
    };
    let codec_tail = [talker.codec_pad_id, talker.codec_bos_id];

    let mut codec_rows: Vec<Vec<f32>> = Vec::new();
    for &cid in &codec_prefill {
        codec_rows.push(codec_row(&codec_table, cid));
    }
    codec_rows.push(speaker_row);
    for &cid in &codec_tail {
        codec_rows.push(codec_row(&codec_table, cid));
    }

    let n_codec = codec_rows.len();
    let hidden = talker.hidden_size;
    let pad_codec = codec_row(&codec_table, talker.codec_pad_id);

    let mut seq: Vec<Vec<f32>> = text.iter().take(3).cloned().collect();

    // HF: `tts_pad.expand(-1, n_codec - 2, -1)` + `tts_bos` + `codec[:, :-1]`
    for (i, piece) in codec_rows.iter().take(n_codec - 1).enumerate() {
        let mut row = piece.clone();
        let add = if i < n_codec - 2 { &tts_pad } else { &tts_bos };
        for (j, v) in row.iter_mut().enumerate() {
            *v += add[j];
        }
        seq.push(row);
    }

    // Non-streaming: full text body `input_id[:, 3:-5]` + codec pad, then eos + pad, then bos row.
    let text_end = input_ids.len().saturating_sub(5);
    for ti in 3..text_end {
        let mut row = text[ti].clone();
        for (j, v) in row.iter_mut().enumerate() {
            *v += pad_codec[j];
        }
        seq.push(row);
    }
    let mut eos_row = tts_eos.clone();
    for (j, v) in eos_row.iter_mut().enumerate() {
        *v += pad_codec[j];
    }
    seq.push(eos_row);

    let mut bos_row = tts_pad.clone();
    let bos_codec = codec_row(&codec_table, talker.codec_bos_id);
    for (j, v) in bos_row.iter_mut().enumerate() {
        *v += bos_codec[j];
    }
    seq.push(bos_row);

    let rows = seq.len();
    let mut embeds = Array2::<f32>::zeros((rows, hidden));
    for (i, row) in seq.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            embeds[[i, j]] = v;
        }
    }

    Ok(CustomVoicePrompt {
        embeds,
        tts_pad_embed: tts_pad,
    })
}

fn load_codec_embedding(
    store: &Qwen3TtsWeightStore,
    _talker: &TalkerConfig,
) -> Result<Vec<Vec<f32>>> {
    let snap = store.tensor_snapshot(&["talker.model.codec_embedding.weight"])?;
    let (data, shape) = snap
        .get("talker.model.codec_embedding.weight")
        .context("codec_embedding")?;
    ensure!(shape.len() == 2);
    let vocab = shape[0];
    let hidden = shape[1];
    let mut table = vec![vec![0f32; hidden]; vocab];
    for v in 0..vocab {
        let off = v * hidden;
        table[v].copy_from_slice(&data[off..off + hidden]);
    }
    Ok(table)
}

fn codec_row(table: &[Vec<f32>], id: u32) -> Vec<f32> {
    table[id as usize].clone()
}
