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

//! HuggingFace `LocateAnythingProcessor` chat + `<image-1>` expansion layout.
//!
//! Matches `processing_locateanything.py` (`py_apply_chat_template` + `replace_media_placeholder`
//! + single-pass tokenizer encode). Distinct from [`crate::tokenizer::build_user_prompt_ids`]
//!   (RLX CLI path: no system message, raw `image_token_index` placeholders).

use crate::config::LocateAnythingConfig;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorPromptConfig {
    pub image_start_token: String,
    pub image_end_token: String,
    pub image_token: String,
}

impl ProcessorPromptConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("processor_config.json");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?;
        serde_json::from_str(&raw).with_context(|| format!("parse {path:?}"))
    }

    /// Replace `<image-1>` with `<image 1><img>{image_token×N}</img>` (HF `replace_media_placeholder`).
    pub fn expand_image_placeholder(&self, text: &str, n_image_tokens: usize) -> String {
        let span = format!(
            "<image 1>{}{}{}",
            self.image_start_token,
            self.image_token.repeat(n_image_tokens),
            self.image_end_token
        );
        text.replace("<image-1>", &span)
    }

    /// Full string passed to the HF processor tokenizer (system + user + assistant prompt).
    pub fn build_chat_prompt_string(
        &self,
        user_body_with_placeholder: &str,
        n_image_tokens: usize,
    ) -> String {
        let user_expanded =
            self.expand_image_placeholder(user_body_with_placeholder, n_image_tokens);
        format!(
            "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n\
             <|im_start|>user\n\
             {user_expanded}\
             <|im_end|>\n\
             <|im_start|>assistant\n"
        )
    }
}

/// Token ids from HF processor layout; `user_text` should include the `<image-1>` prefix.
///
/// Vision placeholders are inserted as [`LocateAnythingConfig::image_token_index`] ids
/// (same as HF `replace_media_placeholder`), not by encoding a long `<IMG_CONTEXT>` run
/// (BPE would merge adjacent specials when using `vocab.json`+`merges.txt` alone).
#[cfg(feature = "tokenizer")]
pub fn build_processor_prompt_ids(
    model_dir: &Path,
    cfg: &LocateAnythingConfig,
    tokenizer: &tokenizers::Tokenizer,
    user_text_with_placeholder: &str,
    n_image_tokens: usize,
) -> Result<Vec<u32>> {
    let proc_cfg = ProcessorPromptConfig::from_model_dir(model_dir)?;
    let user_body = user_text_with_placeholder
        .strip_prefix("<image-1>")
        .unwrap_or(user_text_with_placeholder);

    let mut ids = crate::tokenizer::encode(
        tokenizer,
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n",
    )?;
    ids.extend(crate::tokenizer::encode(tokenizer, "<|im_start|>user\n")?);
    ids.extend(crate::tokenizer::encode(tokenizer, "<image 1>")?);
    ids.extend(crate::tokenizer::encode(
        tokenizer,
        &proc_cfg.image_start_token,
    )?);
    ids.extend(std::iter::repeat_n(cfg.image_token_index, n_image_tokens));
    ids.extend(crate::tokenizer::encode(
        tokenizer,
        &proc_cfg.image_end_token,
    )?);
    ids.extend(crate::tokenizer::encode(tokenizer, user_body)?);
    ids.extend(crate::tokenizer::encode(
        tokenizer,
        "<|im_end|>\n<|im_start|>assistant\n",
    )?);
    Ok(ids)
}

#[cfg(feature = "tokenizer")]
pub fn ground_single_with_image_placeholder(phrase: &str) -> String {
    format!("<image-1>{}", crate::prompts::ground_single(phrase))
}
