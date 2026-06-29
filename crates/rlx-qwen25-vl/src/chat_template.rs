// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Qwen2.5-VL ChatML template (VLMEvalKit-compatible user turn layout).

use crate::multimodal::{IMAGE_PAD, MEDIA_MARKER, VISION_END, VISION_START};

/// Default system preamble for Qwen2.5-VL instruct checkpoints.
pub const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// Build the user turn text with a single image placeholder marker for [`MultimodalPrompt`].
pub fn user_turn_with_media(question: &str) -> String {
    format!("{MEDIA_MARKER}{question}")
}

/// Full ChatML prompt string before tokenization (image pads expanded at assemble time).
pub fn qwen25_vl_chatml(user_text: &str, system: &str) -> String {
    format!(
        "<|im_start|>system\n{system}\n\
         <|im_start|>user\n{user_text}\n\
         <|im_start|>assistant\n"
    )
}

/// VLMEvalKit-style VQA user turn (image placeholder + question).
pub fn vlmevalkit_user_text(question: &str) -> String {
    format!("{MEDIA_MARKER}{question}")
}

/// Expand [`MEDIA_MARKER`] to the vision token wrapper used in HF processor output.
pub fn expand_media_marker(prompt: &str) -> String {
    prompt.replace(
        MEDIA_MARKER,
        &format!("{VISION_START}{IMAGE_PAD}{VISION_END}"),
    )
}

/// ChatML prompt for a single-image VQA item (VLMEvalKit default layout).
pub fn vlmevalkit_chat_prompt(question: &str, system: Option<&str>) -> String {
    let sys = system.unwrap_or(DEFAULT_SYSTEM);
    let user = vlmevalkit_user_text(question);
    qwen25_vl_chatml(&user, sys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_contains_assistant_turn() {
        let p = qwen25_vl_chatml("hello", DEFAULT_SYSTEM);
        assert!(p.contains("<|im_start|>assistant"));
        assert!(p.contains("hello"));
    }

    #[test]
    fn expand_media_marker_inserts_vision_tokens() {
        let p = expand_media_marker("Look at <__media__> please");
        assert!(p.contains("<|vision_start|>"));
        assert!(p.contains("<|vision_end|>"));
        assert!(!p.contains(MEDIA_MARKER));
    }
}
