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

//! The Gemma 4 canonical chat template, plus the tokenizer wrapper that turns
//! its output into ids.
//!
//! Turns are delimited by `<|turn>role\n … <turn|>\n`. A system block is only
//! emitted when there is a system message or thinking is enabled, and it always
//! comes first. `add_generation_prompt` opens a trailing `<|turn>model\n` for
//! the canvas to complete.
//!
//! Images are the subtle part. The template emits a bare `<|image|>` placeholder
//! and the *processor* then expands each one to
//! `<|image>` + `<|image|>` × n + `<image|>` — begin-of-image, `n` soft-token
//! slots, end-of-image. Note the three strings differ only in where the bars
//! sit. Crucially `n` is **per image**: it is that image's own
//! `patches / pooling_kernel²`, not the padded budget, so a small image
//! contributes fewer slots than a large one.
//!
//! DiffusionGemma supports text and images only — the reference raises
//! `NotImplementedError` for both audio and video — so no `<|audio|>` or
//! `<|video|>` handling exists here.

use anyhow::{Result, anyhow};

/// `<bos>`.
pub const BOS: &str = "<bos>";
/// Opens a turn; the role and a newline follow.
pub const TURN_OPEN: &str = "<|turn>";
/// Closes a turn.
pub const TURN_CLOSE: &str = "<turn|>";
/// Opens the thinking channel inside the first system turn.
pub const THINK: &str = "<|think|>";
/// Begin-of-image.
pub const BOI: &str = "<|image>";
/// End-of-image.
pub const EOI: &str = "<image|>";
/// One image soft-token slot.
pub const IMAGE: &str = "<|image|>";

/// Who is speaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    /// The template renders `assistant` as `model`.
    Model,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Model => "model",
        }
    }
}

/// One piece of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    Text(String),
    /// An image, expanded to `boi + image·n + eoi` with that image's own soft
    /// token count.
    Image,
}

/// One chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl ChatMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::Text(text.into())],
        }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::Text(text.into())],
        }
    }
    pub fn model(text: impl Into<String>) -> Self {
        Self {
            role: Role::Model,
            content: vec![ContentPart::Text(text.into())],
        }
    }
    /// A user turn with images before the text, the usual layout.
    pub fn user_with_images(images: usize, text: impl Into<String>) -> Self {
        let mut content: Vec<ContentPart> = (0..images).map(|_| ContentPart::Image).collect();
        content.push(ContentPart::Text(text.into()));
        Self {
            role: Role::User,
            content,
        }
    }

    fn image_count(&self) -> usize {
        self.content
            .iter()
            .filter(|p| matches!(p, ContentPart::Image))
            .count()
    }
}

/// Template switches.
#[derive(Debug, Clone, Copy)]
pub struct ChatOptions {
    /// Open a trailing `<|turn>model\n` for generation.
    pub add_generation_prompt: bool,
    /// Emit `<|think|>` at the top of the system turn (which forces a system
    /// turn to exist even without a system message).
    pub enable_thinking: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
        }
    }
}

fn render_content(
    msg: &ChatMessage,
    soft_tokens: &[usize],
    next_image: &mut usize,
    out: &mut String,
) -> Result<()> {
    for part in &msg.content {
        match part {
            ContentPart::Text(t) => out.push_str(t.trim()),
            ContentPart::Image => {
                let n = *soft_tokens.get(*next_image).ok_or_else(|| {
                    anyhow!(
                        "prompt has more images than the {} soft-token counts supplied",
                        soft_tokens.len()
                    )
                })?;
                anyhow::ensure!(n > 0, "image {} contributes 0 soft tokens", *next_image);
                out.push_str(BOI);
                for _ in 0..n {
                    out.push_str(IMAGE);
                }
                out.push_str(EOI);
                *next_image += 1;
            }
        }
    }
    Ok(())
}

/// Render a conversation into the exact string the tokenizer should see.
///
/// `soft_tokens` gives the per-image slot count, in the order the images appear
/// — [`crate::preprocess::PreprocessedImage::num_soft_tokens`] for each.
pub fn format_chat(
    messages: &[ChatMessage],
    opts: ChatOptions,
    soft_tokens: &[usize],
) -> Result<String> {
    let total_images: usize = messages.iter().map(|m| m.image_count()).sum();
    anyhow::ensure!(
        total_images == soft_tokens.len(),
        "prompt contains {total_images} images but {} soft-token counts were supplied",
        soft_tokens.len()
    );

    let mut out = String::from(BOS);
    let mut next_image = 0usize;

    // A leading system message is folded into the system block; the reference
    // only treats index 0 that way.
    let leading_system = messages
        .first()
        .filter(|m| m.role == Role::System)
        .map(|m| m as &ChatMessage);
    anyhow::ensure!(
        messages.iter().skip(1).all(|m| m.role != Role::System),
        "a system message may only appear first"
    );

    if opts.enable_thinking || leading_system.is_some() {
        out.push_str(TURN_OPEN);
        out.push_str("system\n");
        if opts.enable_thinking {
            out.push_str(THINK);
            out.push('\n');
        }
        if let Some(sys) = leading_system {
            render_content(sys, soft_tokens, &mut next_image, &mut out)?;
        }
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    for msg in messages.iter().skip(usize::from(leading_system.is_some())) {
        out.push_str(TURN_OPEN);
        out.push_str(msg.role.as_str());
        out.push('\n');
        render_content(msg, soft_tokens, &mut next_image, &mut out)?;
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    if opts.add_generation_prompt {
        out.push_str(TURN_OPEN);
        out.push_str("model\n");
    }
    Ok(out)
}

/// Tokenizer wrapper that knows DiffusionGemma's special ids.
#[cfg(feature = "tokenizer")]
pub struct DiffusionGemmaTokenizer {
    inner: tokenizers::Tokenizer,
    image_token_id: u32,
}

#[cfg(feature = "tokenizer")]
impl DiffusionGemmaTokenizer {
    /// Load `tokenizer.json`, checking that the image token resolves.
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
        cfg: &crate::config::DiffusionGemmaConfig,
    ) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow!("loading tokenizer: {e}"))?;
        let image_token_id = inner
            .token_to_id(IMAGE)
            .ok_or_else(|| anyhow!("tokenizer has no `{IMAGE}` token"))?;
        anyhow::ensure!(
            image_token_id == cfg.image_token_id,
            "tokenizer maps `{IMAGE}` to {image_token_id} but the config says {}",
            cfg.image_token_id
        );
        Ok(Self {
            inner,
            image_token_id,
        })
    }

    /// Encode a rendered prompt. Special tokens are already written out by
    /// [`format_chat`], so none are added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encoding prompt: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("decoding: {e}"))
    }

    pub fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    /// Positions of the image soft-token slots, in order — the slots
    /// [`crate::vision::merge_multimodal_embeds`] overwrites.
    pub fn image_positions(&self, ids: &[u32]) -> Vec<usize> {
        ids.iter()
            .enumerate()
            .filter(|(_, t)| **t == self.image_token_id)
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_user_turn() {
        let msgs = [ChatMessage::user("  Why is the sky blue?  ")];
        let s = format_chat(&msgs, ChatOptions::default(), &[]).unwrap();
        assert_eq!(
            s,
            "<bos><|turn>user\nWhy is the sky blue?<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn system_message_becomes_the_leading_system_turn() {
        let msgs = [
            ChatMessage::system("You are terse."),
            ChatMessage::user("Hi"),
        ];
        let s = format_chat(&msgs, ChatOptions::default(), &[]).unwrap();
        assert_eq!(
            s,
            "<bos><|turn>system\nYou are terse.<turn|>\n\
             <|turn>user\nHi<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn thinking_forces_a_system_turn_even_without_a_system_message() {
        let msgs = [ChatMessage::user("Hi")];
        let opts = ChatOptions {
            add_generation_prompt: true,
            enable_thinking: true,
        };
        let s = format_chat(&msgs, opts, &[]).unwrap();
        assert_eq!(
            s,
            "<bos><|turn>system\n<|think|>\n<turn|>\n\
             <|turn>user\nHi<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn thinking_precedes_the_system_text() {
        let msgs = [ChatMessage::system("Be brief."), ChatMessage::user("Hi")];
        let opts = ChatOptions {
            add_generation_prompt: false,
            enable_thinking: true,
        };
        let s = format_chat(&msgs, opts, &[]).unwrap();
        assert!(s.starts_with("<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n"));
        assert!(!s.ends_with("<|turn>model\n"), "no generation prompt asked");
    }

    #[test]
    fn multi_turn_history_alternates_and_renames_assistant_to_model() {
        let msgs = [
            ChatMessage::user("2+2?"),
            ChatMessage::model("4"),
            ChatMessage::user("and 3+3?"),
        ];
        let s = format_chat(&msgs, ChatOptions::default(), &[]).unwrap();
        assert_eq!(
            s,
            "<bos><|turn>user\n2+2?<turn|>\n\
             <|turn>model\n4<turn|>\n\
             <|turn>user\nand 3+3?<turn|>\n\
             <|turn>model\n"
        );
    }

    #[test]
    fn an_image_expands_to_boi_slots_eoi() {
        let msgs = [ChatMessage::user_with_images(1, "What is this?")];
        let s = format_chat(&msgs, ChatOptions::default(), &[3]).unwrap();
        assert_eq!(
            s,
            "<bos><|turn>user\n<|image><|image|><|image|><|image|><image|>\
             What is this?<turn|>\n<|turn>model\n"
        );
        // Exactly three slots, wrapped once.
        assert_eq!(s.matches(IMAGE).count(), 3);
        assert_eq!(s.matches(BOI).count(), 1);
        assert_eq!(s.matches(EOI).count(), 1);
    }

    /// Each image gets its *own* slot count — a small image contributes fewer
    /// soft tokens than a large one, so a single shared count would misalign
    /// every later image.
    #[test]
    fn multiple_images_use_their_own_slot_counts() {
        let msgs = [ChatMessage::user_with_images(2, "Compare these.")];
        let s = format_chat(&msgs, ChatOptions::default(), &[2, 4]).unwrap();
        let first = s.find(BOI).unwrap();
        let second = s[first + 1..].find(BOI).unwrap() + first + 1;
        let img1 = &s[first..second];
        let img2 = &s[second..s.find("Compare").unwrap()];
        assert_eq!(img1.matches(IMAGE).count(), 2, "first image: {img1}");
        assert_eq!(img2.matches(IMAGE).count(), 4, "second image: {img2}");
        assert_eq!(s.matches(IMAGE).count(), 6);
    }

    #[test]
    fn images_across_separate_turns_consume_counts_in_order() {
        let msgs = [
            ChatMessage::user_with_images(1, "first"),
            ChatMessage::model("ok"),
            ChatMessage::user_with_images(1, "second"),
        ];
        let s = format_chat(&msgs, ChatOptions::default(), &[1, 5]).unwrap();
        let turns: Vec<&str> = s.split("<|turn>").collect();
        let user1 = turns.iter().find(|t| t.contains("first")).unwrap();
        let user2 = turns.iter().find(|t| t.contains("second")).unwrap();
        assert_eq!(user1.matches(IMAGE).count(), 1);
        assert_eq!(user2.matches(IMAGE).count(), 5);
    }

    #[test]
    fn image_count_mismatch_is_an_error() {
        let msgs = [ChatMessage::user_with_images(2, "x")];
        let err = format_chat(&msgs, ChatOptions::default(), &[4]).unwrap_err();
        assert!(
            format!("{err}").contains("2 images but 1"),
            "unhelpful: {err}"
        );
        // And a zero-slot image is rejected rather than silently dropped.
        assert!(format_chat(&msgs, ChatOptions::default(), &[4, 0]).is_err());
    }

    #[test]
    fn a_system_message_must_come_first() {
        let msgs = [
            ChatMessage::user("hi"),
            ChatMessage::system("late system prompt"),
        ];
        assert!(format_chat(&msgs, ChatOptions::default(), &[]).is_err());
    }

    /// The three image markers differ only in bar placement; a mix-up would put
    /// the soft tokens in the wrong slots.
    #[test]
    fn image_markers_are_distinct() {
        assert_ne!(BOI, IMAGE);
        assert_ne!(EOI, IMAGE);
        assert_ne!(BOI, EOI);
        assert_eq!((BOI, IMAGE, EOI), ("<|image>", "<|image|>", "<image|>"));
    }
}
