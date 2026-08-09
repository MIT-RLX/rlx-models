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

//! Multimodal turn assembly (PLAN.md M7).
//!
//! Replaces `llama-cpp-4`'s `MtmdContext` end-to-end. The runner
//! receives a list of [`MtmdTurn`]s — text + images + audio — and
//! produces an [`AssembledTurn`] the per-family VL/Omni runner
//! consumes via the `rlx_vlm_base` traits.
//!
//! **Status:** TYPE SKELETON. The shape is in place so `skill` can
//! write code against `MtmdContext::build_turn(..)` today; the
//! actual image-loading / audio-resampling implementations land
//! alongside the per-family runners in M7.

use anyhow::{Result, bail};
use std::path::PathBuf;

type TokenizerFn<'a> = dyn Fn(&str) -> Result<Vec<u32>> + 'a;

/// Where one image / audio chunk lives.
#[derive(Debug, Clone)]
pub enum MediaSource {
    /// Read from a file path on disk.
    FilePath(PathBuf),
    /// Decoded bytes (e.g. base64 from a chat client).
    Bytes(Vec<u8>),
}

/// One turn in a multimodal conversation. `text` is rendered through
/// the same `ChatTemplate` as the text-only path; `images` / `audio`
/// are interleaved into the LM stream by the per-family runner.
#[derive(Debug, Clone)]
pub struct MtmdTurn {
    pub role: String,
    pub text: String,
    pub images: Vec<MediaSource>,
    pub audio: Vec<MediaSource>,
}

impl MtmdTurn {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            text: text.into(),
            images: Vec::new(),
            audio: Vec::new(),
        }
    }
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            text: text.into(),
            images: Vec::new(),
            audio: Vec::new(),
        }
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            text: text.into(),
            images: Vec::new(),
            audio: Vec::new(),
        }
    }
    pub fn with_image_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.images.push(MediaSource::FilePath(path.into()));
        self
    }
    pub fn with_image_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.images.push(MediaSource::Bytes(bytes));
        self
    }
    pub fn with_audio_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.audio.push(MediaSource::FilePath(path.into()));
        self
    }
    pub fn with_audio_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.audio.push(MediaSource::Bytes(bytes));
        self
    }

    pub fn has_media(&self) -> bool {
        !self.images.is_empty() || !self.audio.is_empty()
    }
}

/// Result of assembling a turn list into something the per-family
/// runner can feed into prefill. `text_tokens` is the chat-template
/// output run through the tokenizer; `image_refs` / `audio_refs`
/// retain order so the runner knows where to insert the embeddings.
#[derive(Debug, Clone, Default)]
pub struct AssembledTurn {
    pub text_tokens: Vec<u32>,
    pub image_refs: Vec<MediaSource>,
    pub audio_refs: Vec<MediaSource>,
}

/// Context for assembling multimodal turns. Holds the chat template
/// and (eventually) the tokenizer; per-family runners hand the
/// resulting [`AssembledTurn`] into their prefill path.
pub struct MtmdContext {
    template_source: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl MtmdContext {
    /// Build a context from a Jinja chat template (typically loaded
    /// from a GGUF via [`crate::ChatTemplate::from_gguf`]).
    pub fn from_template_source(src: impl Into<String>) -> Self {
        Self {
            template_source: src.into(),
            bos_token: None,
            eos_token: None,
        }
    }

    pub fn with_tokens(mut self, bos: Option<String>, eos: Option<String>) -> Self {
        self.bos_token = bos;
        self.eos_token = eos;
        self
    }

    pub fn template_source(&self) -> &str {
        &self.template_source
    }
    pub fn bos_token(&self) -> Option<&str> {
        self.bos_token.as_deref()
    }
    pub fn eos_token(&self) -> Option<&str> {
        self.eos_token.as_deref()
    }

    /// Assemble one turn list into [`AssembledTurn`].
    ///
    /// Renders the text using the registered chat template, replaces
    /// each `<|image|>` / `<|audio|>` marker with the per-family
    /// placeholder token id (resolved from the optional tokenizer
    /// vocabulary), and records the order of media in `image_refs` /
    /// `audio_refs` so the runner can insert the corresponding
    /// embeddings at decode time.
    ///
    /// `tokenizer_fn` lets the caller plug in a per-family text→ids
    /// encoder (typically `auto_tokenize`). Passing `None` populates
    /// `text_tokens` with an empty vec — useful for callers that
    /// own tokenization separately.
    pub fn build_turn(
        &self,
        turns: &[MtmdTurn],
        tokenizer_fn: Option<&TokenizerFn<'_>>,
    ) -> Result<AssembledTurn> {
        if turns.is_empty() {
            bail!("MtmdContext::build_turn: empty turn list");
        }
        let mut text = String::new();
        let mut image_refs = Vec::new();
        let mut audio_refs = Vec::new();

        // Minimal ChatML-style assembly. Real chat-template rendering
        // (Jinja) lives in `crate::ChatTemplate` — when present,
        // callers should pre-render and pass a single turn.
        if let Some(bos) = self.bos_token.as_deref() {
            text.push_str(bos);
        }
        for t in turns {
            text.push_str("<|im_start|>");
            text.push_str(&t.role);
            text.push('\n');
            text.push_str(&t.text);
            // Insert image/audio markers after the text so the runner
            // can interleave embeddings in order.
            for img in &t.images {
                text.push_str("<|image|>");
                image_refs.push(img.clone());
            }
            for au in &t.audio {
                text.push_str("<|audio|>");
                audio_refs.push(au.clone());
            }
            text.push_str("<|im_end|>\n");
        }
        if let Some(eos) = self.eos_token.as_deref() {
            text.push_str(eos);
        }

        let text_tokens = match tokenizer_fn {
            Some(f) => f(&text)?,
            None => Vec::new(),
        };
        Ok(AssembledTurn {
            text_tokens,
            image_refs,
            audio_refs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_turn_records_media_order() {
        let ctx = MtmdContext::from_template_source("").with_tokens(None, None);
        let turn = MtmdTurn::user("describe")
            .with_image_path("/tmp/a.png")
            .with_audio_path("/tmp/b.wav")
            .with_image_path("/tmp/c.png");
        let out = ctx.build_turn(&[turn], None).unwrap();
        assert_eq!(out.image_refs.len(), 2);
        assert_eq!(out.audio_refs.len(), 1);
        // Tokenizer absent → text_tokens empty.
        assert!(out.text_tokens.is_empty());
    }

    #[test]
    fn build_turn_invokes_tokenizer_callback() {
        let ctx = MtmdContext::from_template_source("");
        let counter = std::cell::Cell::new(0u32);
        let tokenize = |s: &str| -> Result<Vec<u32>> {
            counter.set(s.len() as u32);
            Ok(vec![1, 2, 3])
        };
        let turn = MtmdTurn::user("hello");
        let out = ctx
            .build_turn(
                &[turn],
                Some(&tokenize as &dyn Fn(&str) -> Result<Vec<u32>>),
            )
            .unwrap();
        assert_eq!(out.text_tokens, vec![1, 2, 3]);
        assert!(counter.get() > 0, "tokenizer must see the rendered text");
    }

    #[test]
    fn build_turn_rejects_empty() {
        let ctx = MtmdContext::from_template_source("");
        let err = ctx.build_turn(&[], None).unwrap_err();
        assert!(format!("{err}").contains("empty turn list"));
    }
}
