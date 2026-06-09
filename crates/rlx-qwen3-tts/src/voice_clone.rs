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

//! Voice clone prompts (Base model) — ICL and x-vector modes.

use crate::config::Qwen3TtsConfig;
use crate::load::Qwen3TtsWeightStore;
use crate::prompt::{build_assistant_text, load_text_tokenizer};
use crate::text_embed::TextEmbedder;
use anyhow::{Context, Result, bail, ensure};
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceCloneMode {
    /// Reference text + reference codec (in-context learning).
    Icl,
    /// Speaker embedding only (no `ref_code` in talker prefill).
    XVectorOnly,
}

/// HF `VoiceClonePromptItem` — built offline or via [`create_voice_clone_prompt`].
#[derive(Debug, Clone)]
pub struct VoiceClonePrompt {
    pub mode: VoiceCloneMode,
    pub ref_spk_embedding: Vec<f32>,
    /// Per-frame codec groups from 12Hz encoder (`[T, num_code_groups]`).
    pub ref_code: Option<Vec<Vec<u32>>>,
    pub ref_text: Option<String>,
    pub x_vector_only_mode: bool,
    pub icl_mode: bool,
    /// Talker prefill rows (after HF `generate_icl_prompt` / x-vector layout).
    pub embeds: Array2<f32>,
    pub tts_pad_embed: Vec<f32>,
}

pub fn ensure_base_model(cfg: &Qwen3TtsConfig) -> Result<()> {
    if cfg.tts_model_type != "base" {
        bail!(
            "voice clone requires a Base checkpoint (tts_model_type=base); got {:?}",
            cfg.tts_model_type
        );
    }
    Ok(())
}

/// HF `_build_assistant_text` for clone target text (same as CustomVoice).
pub fn build_clone_target_text(user_text: &str) -> String {
    build_assistant_text(user_text)
}

/// Build ICL talker prefill from reference transcript + codec + speaker embedding.
pub fn build_icl_prompt(
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    text_embedder: &TextEmbedder,
    tokenizer: &Tokenizer,
    target_text: &str,
    ref_text: &str,
    ref_code: &[Vec<u32>],
    ref_spk_embedding: &[f32],
) -> Result<VoiceClonePrompt> {
    ensure_base_model(cfg)?;
    ensure!(!ref_code.is_empty(), "ref_code must not be empty");
    let talker = cfg.talker();
    let hidden = talker.hidden_size;
    ensure!(
        ref_spk_embedding.len() == hidden,
        "ref_spk_embedding len {} != hidden {}",
        ref_spk_embedding.len(),
        hidden
    );

    let mut rows: Vec<Vec<f32>> = Vec::new();
    let ref_ids = tokenizer
        .encode(ref_text, false)
        .map_err(|e| anyhow::anyhow!("tokenize ref: {e}"))?;
    let ref_token_ids: Vec<u32> = ref_ids.get_ids().to_vec();
    for &tid in &ref_token_ids {
        rows.push(text_embedder.embed_token(tid)?);
    }
    for frame in ref_code {
        ensure!(
            frame.len() == talker.num_code_groups,
            "ref_code frame needs {} groups",
            talker.num_code_groups
        );
        rows.push(sum_codec_row(store, cfg, frame)?);
    }
    let target = build_clone_target_text(target_text);
    let target_ids = tokenizer
        .encode(target.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize target: {e}"))?;
    for &tid in target_ids.get_ids() {
        rows.push(text_embedder.embed_token(tid)?);
    }
    rows.push(ref_spk_embedding.to_vec());

    let n = rows.len();
    let mut embeds = Array2::<f32>::zeros((n, hidden));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            embeds[[i, j]] = v;
        }
    }
    let tts_pad_embed = codec_embed(store, talker.codec_pad_id, talker.hidden_size)?;

    Ok(VoiceClonePrompt {
        mode: VoiceCloneMode::Icl,
        ref_spk_embedding: ref_spk_embedding.to_vec(),
        ref_code: Some(ref_code.to_vec()),
        ref_text: Some(ref_text.to_string()),
        x_vector_only_mode: false,
        icl_mode: true,
        embeds,
        tts_pad_embed,
    })
}

/// X-vector-only: target text + speaker embedding inserted at the codec
/// speaker slot of the CustomVoice prompt skeleton (HF `generate()` path with
/// `x_vector_only_mode=True`). Language is hardcoded to "auto" — pass
/// [`build_x_vector_prompt_lang`] to override.
pub fn build_x_vector_prompt(
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    text_embedder: &TextEmbedder,
    tokenizer: &Tokenizer,
    target_text: &str,
    ref_spk_embedding: &[f32],
) -> Result<VoiceClonePrompt> {
    build_x_vector_prompt_lang(
        cfg,
        store,
        text_embedder,
        tokenizer,
        target_text,
        ref_spk_embedding,
        "english",
    )
}

pub fn build_x_vector_prompt_lang(
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    text_embedder: &TextEmbedder,
    tokenizer: &Tokenizer,
    target_text: &str,
    ref_spk_embedding: &[f32],
    language: &str,
) -> Result<VoiceClonePrompt> {
    ensure_base_model(cfg)?;
    let talker = cfg.talker();
    let hidden = talker.hidden_size;
    ensure!(
        ref_spk_embedding.len() == hidden,
        "ref_spk_embedding len {} != hidden {}",
        ref_spk_embedding.len(),
        hidden
    );
    let cv = crate::prompt::build_custom_voice_prompt_with_speaker_embed(
        cfg,
        store,
        text_embedder,
        tokenizer,
        target_text,
        ref_spk_embedding,
        language,
    )?;
    Ok(VoiceClonePrompt {
        mode: VoiceCloneMode::XVectorOnly,
        ref_spk_embedding: ref_spk_embedding.to_vec(),
        ref_code: None,
        ref_text: None,
        x_vector_only_mode: true,
        icl_mode: false,
        embeds: cv.embeds,
        tts_pad_embed: cv.tts_pad_embed,
    })
}

/// Reference WAV → mel → speaker encoder + speech tokenizer encode (requires Base weights + `speech_tokenizer/`).
pub fn create_voice_clone_prompt(
    model_dir: &Path,
    cfg: &Qwen3TtsConfig,
    store: &Qwen3TtsWeightStore,
    ref_wav: &Path,
    ref_text: Option<&str>,
    mode: VoiceCloneMode,
) -> Result<VoiceClonePrompt> {
    ensure_base_model(cfg)?;
    let spk = crate::speaker_encoder::encode_reference_wav(model_dir, store, ref_wav)?;
    let text_embedder = TextEmbedder::open(store)?;
    let tokenizer = load_text_tokenizer(model_dir)?;
    match mode {
        VoiceCloneMode::Icl => {
            let ref_txt = ref_text.context("ICL clone requires --ref-text")?;
            let ref_code = crate::speech_tokenizer::encode_wav_to_codec_frames(model_dir, ref_wav)?;
            build_icl_prompt(
                cfg,
                store,
                &text_embedder,
                &tokenizer,
                "",
                ref_txt,
                &ref_code,
                &spk,
            )
        }
        VoiceCloneMode::XVectorOnly => {
            build_x_vector_prompt(cfg, store, &text_embedder, &tokenizer, "", &spk)
        }
    }
}

fn codec_embed(store: &Qwen3TtsWeightStore, token: u32, hidden: usize) -> Result<Vec<f32>> {
    let key = "talker.model.codec_embedding.weight";
    let snap = store.tensor_snapshot(&[key])?;
    let (data, shape) = snap.get(key).context("codec embed")?;
    ensure!(shape[1] == hidden);
    let table = Array2::from_shape_vec((shape[0], shape[1]), data.clone())?;
    Ok(table.row(token as usize).to_vec())
}

fn sum_codec_row(
    store: &Qwen3TtsWeightStore,
    cfg: &Qwen3TtsConfig,
    frame: &[u32],
) -> Result<Vec<f32>> {
    let hidden = cfg.talker().hidden_size;
    let mut emb = vec![0f32; hidden];
    let key0 = "talker.model.codec_embedding.weight";
    let snap = store.tensor_snapshot(&[key0])?;
    let (data, shape) = snap.get(key0).context("codec embed")?;
    let table = Array2::from_shape_vec((shape[0], shape[1]), data.clone())?;
    for (gi, &tok) in frame.iter().enumerate() {
        let row = if gi == 0 {
            table.row(tok as usize).to_vec()
        } else {
            let key = format!(
                "talker.code_predictor.model.codec_embedding.{}.weight",
                gi - 1
            );
            let (d, sh) = store.tensor_snapshot(&[&key])?[&key].clone();
            let t = Array2::from_shape_vec((sh[0], sh[1]), d)?;
            t.row(tok as usize).to_vec()
        };
        for (j, v) in row.iter().enumerate() {
            emb[j] += *v;
        }
    }
    Ok(emb)
}
