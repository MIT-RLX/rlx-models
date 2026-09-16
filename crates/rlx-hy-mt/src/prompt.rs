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

//! Official HY-MT1.5 prompt templates (from the Hugging Face model card).

/// Structured translate request.
#[derive(Debug, Clone)]
pub struct TranslatePrompt {
    pub source_text: String,
    pub target_language: String,
    /// When true, use the Chinese-instruction ZH↔XX template.
    pub zh_pair: bool,
}

impl TranslatePrompt {
    pub fn xx_to_xx(source_text: impl Into<String>, target_language: impl Into<String>) -> Self {
        Self {
            source_text: source_text.into(),
            target_language: target_language.into(),
            zh_pair: false,
        }
    }

    pub fn zh_to_xx(source_text: impl Into<String>, target_language: impl Into<String>) -> Self {
        Self {
            source_text: source_text.into(),
            target_language: target_language.into(),
            zh_pair: true,
        }
    }

    /// Render the card's user message (no chat wrappers).
    pub fn render_user(&self) -> String {
        if self.zh_pair {
            format_zh_to_xx_prompt(&self.target_language, &self.source_text)
        } else {
            format_xx_to_xx_prompt(&self.target_language, &self.source_text)
        }
    }
}

/// Hunyuan / HY-MT chat control tokens (must match `tokenizer.json`).
pub const HY_BOS: &str = "<｜hy_begin▁of▁sentence｜>";
pub const HY_USER: &str = "<｜hy_User｜>";
pub const HY_ASSISTANT: &str = "<｜hy_Assistant｜>";
pub const HY_EOT: &str = "<｜hy_EOT｜>";
/// Token id for [`HY_EOT`].
pub const HY_EOT_TOKEN_ID: u32 = 120_008;

/// Wrap a plain user prompt in HY-MT1.5 chat turns (required for GGUF inference).
pub fn wrap_hy_mt_user_turn(user_body: &str) -> String {
    format!("{HY_BOS}{HY_USER}{user_body}{HY_EOT}{HY_ASSISTANT}")
}

/// XX↔XX template (also used for EN→FR, etc.).
pub fn format_xx_to_xx_prompt(target_language: &str, source_text: &str) -> String {
    format!(
        "Translate the following segment into {target_language}, without additional explanation.\n\n{source_text}"
    )
}

/// ZH↔XX template from the model card.
pub fn format_zh_to_xx_prompt(target_language: &str, source_text: &str) -> String {
    format!(
        "将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"
    )
}

/// Map common ISO-ish codes to the English language names HY-MT prompts expect.
pub fn target_language_name(code_or_name: &str) -> &str {
    match code_or_name.trim().to_ascii_lowercase().as_str() {
        "en" | "eng" | "english" => "English",
        "fr" | "fra" | "french" => "French",
        "zh" | "zho" | "chinese" | "zh-cn" | "zh-hans" => "Chinese",
        "de" | "deu" | "german" => "German",
        "es" | "spa" | "spanish" => "Spanish",
        "pt" | "por" | "portuguese" => "Portuguese",
        "ja" | "jpn" | "japanese" => "Japanese",
        "ko" | "kor" | "korean" => "Korean",
        "it" | "ita" | "italian" => "Italian",
        "ru" | "rus" | "russian" => "Russian",
        "ar" | "ara" | "arabic" => "Arabic",
        "vi" | "vie" | "vietnamese" => "Vietnamese",
        "th" | "tha" | "thai" => "Thai",
        "tr" | "tur" | "turkish" => "Turkish",
        "nl" | "nld" | "dutch" => "Dutch",
        "pl" | "pol" | "polish" => "Polish",
        "cs" | "ces" | "czech" => "Czech",
        "hi" | "hin" | "hindi" => "Hindi",
        "uk" | "ukr" | "ukrainian" => "Ukrainian",
        other if !other.is_empty() => code_or_name.trim(),
        _ => "English",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_fr_prompt() {
        let p = TranslatePrompt::xx_to_xx("Hello world", target_language_name("fr"));
        let s = p.render_user();
        assert!(s.contains("French"));
        assert!(s.contains("Hello world"));
        assert!(s.contains("without additional explanation"));
    }

    #[test]
    fn hy_chat_wrap() {
        let w = wrap_hy_mt_user_turn("ping");
        assert!(w.contains(HY_USER));
        assert!(w.contains(HY_ASSISTANT));
    }
}
