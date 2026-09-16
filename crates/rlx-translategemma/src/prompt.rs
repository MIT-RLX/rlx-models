// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! TranslateGemma-oriented prompts.
//!
//! TranslateGemma GGUFs ship a specialized chat template that requires structured
//! `content` (lang codes + text). For subtitle dubbing we use Google's documented
//! plain-text prompt wrapped in Gemma 3 turn markers instead.

use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TranslatePrompt {
    pub source_text: String,
    pub target_language: String,
    pub source_language: Option<String>,
}

impl TranslatePrompt {
    pub fn new(source_text: impl Into<String>, target_language: impl Into<String>) -> Self {
        Self {
            source_text: source_text.into(),
            target_language: target_language.into(),
            source_language: None,
        }
    }

    pub fn with_source_lang(mut self, lang: impl Into<String>) -> Self {
        self.source_language = Some(lang.into());
        self
    }

    /// Official TranslateGemma instruction block (Figure 3 in the tech report).
    pub fn render_user(&self) -> String {
        let tgt_name = target_language_name(&self.target_language);
        let tgt_code = language_code(&self.target_language);
        let (src_name, src_code) = match self.source_language.as_deref() {
            Some(s) => (target_language_name(s), language_code(s)),
            None => ("English", "en"),
        };
        format_official_translate_prompt(src_name, src_code, tgt_name, tgt_code, &self.source_text)
    }
}

/// ISO 639-1 code for TranslateGemma prompts.
pub fn language_code(code_or_name: &str) -> &'static str {
    match code_or_name.trim().to_ascii_lowercase().as_str() {
        "en" | "eng" | "english" => "en",
        "fr" | "fra" | "french" => "fr",
        "de" | "deu" | "german" => "de",
        "es" | "spa" | "spanish" => "es",
        "zh" | "zho" | "chinese" => "zh",
        "ja" | "jpn" | "japanese" => "ja",
        "it" | "ita" | "italian" => "it",
        "pt" | "por" | "portuguese" => "pt",
        "ru" | "rus" | "russian" => "ru",
        other if other.len() == 2 => "en", // fallback; regional codes handled by name
        _ => "en",
    }
}

/// Google TranslateGemma preferred plain-text prompt (two blank lines before source).
pub fn format_official_translate_prompt(
    source_lang_name: &str,
    source_code: &str,
    target_lang_name: &str,
    target_code: &str,
    source_text: &str,
) -> String {
    format!(
        "You are a professional {source_lang_name} ({source_code}) to {target_lang_name} ({target_code}) translator. \
Your goal is to accurately convey the meaning and nuances of the original {source_lang_name} text \
while adhering to {target_lang_name} grammar, vocabulary, and cultural sensitivities.\n\
Produce only the {target_lang_name} translation, without any additional explanations or commentary. \
Please translate the following {source_lang_name} text into {target_lang_name}:\n\n\n{source_text}"
    )
}

/// Gemma 3 turn markers — do **not** pass through TranslateGemma's structured chat template.
pub fn wrap_gemma_user_turn(user_body: &str) -> String {
    format!("<start_of_turn>user\n{user_body}<end_of_turn>\n<start_of_turn>model\n")
}

/// Tokenize an official TranslateGemma dubbing prompt (BOS + turns, no HF chat template).
pub fn encode_translate_prompt(
    weights: &Path,
    tokenizer: Option<&Path>,
    source_lang: &str,
    target_lang: &str,
    source_text: &str,
) -> Result<Vec<u32>> {
    let prompt = TranslatePrompt::new(source_text, target_language_name(target_lang))
        .with_source_lang(target_language_name(source_lang))
        .render_user();
    let wrapped = wrap_gemma_user_turn(&prompt);
    rlx_gemma::encode_chat_prompt_auto(weights, tokenizer, None, &wrapped, false)
}

#[deprecated(note = "use format_official_translate_prompt / TranslatePrompt::render_user")]
pub fn format_translate_prompt(
    source_language: Option<&str>,
    target_language: &str,
    source_text: &str,
) -> String {
    let tgt_name = target_language_name(target_language);
    let tgt_code = language_code(target_language);
    let (src_name, src_code) = match source_language {
        Some(s) => (s, language_code(s)),
        None => ("English", "en"),
    };
    format_official_translate_prompt(src_name, src_code, tgt_name, tgt_code, source_text)
}

pub fn target_language_name(code_or_name: &str) -> &str {
    // Reuse the same ISO map as HY-MT for subtitle dubbing.
    match code_or_name.trim().to_ascii_lowercase().as_str() {
        "en" | "eng" | "english" => "English",
        "fr" | "fra" | "french" => "French",
        "de" | "deu" | "german" => "German",
        "es" | "spa" | "spanish" => "Spanish",
        "zh" | "zho" | "chinese" => "Chinese",
        "ja" | "jpn" | "japanese" => "Japanese",
        "it" | "ita" | "italian" => "Italian",
        "pt" | "por" | "portuguese" => "Portuguese",
        "ru" | "rus" | "russian" => "Russian",
        other if !other.is_empty() => code_or_name.trim(),
        _ => "English",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_fr() {
        let s = TranslatePrompt::new("Hello", target_language_name("fr"))
            .with_source_lang("English")
            .render_user();
        assert!(s.contains("French (fr)"));
        assert!(s.contains("English (en)"));
        assert!(s.contains("Hello"));
        assert!(s.contains("\n\n\nHello"));
    }

    #[test]
    fn gemma_turn_wrap() {
        let w = wrap_gemma_user_turn("ping");
        assert!(w.starts_with("<start_of_turn>user"));
        assert!(w.contains("<start_of_turn>model"));
    }
}
