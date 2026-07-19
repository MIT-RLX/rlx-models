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

//! Chat / special-token helpers matching Inkling's HF chat template.
//!
//! Full Jinja tool-calling lives in the checkpoint's `chat_template.jinja`.
//! This module covers the text turn shape + `reasoning_effort` header used
//! by almost every prompt path.

use anyhow::{Result, bail};

/// Special tokens from `tokenizer_config.json` (thinkingmachines/Inkling).
pub mod tokens {
    pub const MESSAGE_USER: &str = "<|message_user|>";
    pub const MESSAGE_MODEL: &str = "<|message_model|>";
    pub const MESSAGE_SYSTEM: &str = "<|message_system|>";
    pub const MESSAGE_TOOL: &str = "<|message_tool|>";
    pub const CONTENT_TEXT: &str = "<|content_text|>";
    pub const CONTENT_IMAGE: &str = "<|content_image|>";
    pub const CONTENT_AUDIO: &str = "<|content_audio_input|>";
    pub const CONTENT_THINKING: &str = "<|content_thinking|>";
    pub const CONTENT_MODEL_END: &str = "<|content_model_end_sampling|>";
    pub const END_MESSAGE: &str = "<|end_message|>";
    pub const AUDIO_END: &str = "<|audio_end|>";
    pub const IMAGE_PLACEHOLDER: &str = "<|unused_200054|>";
    pub const AUDIO_PLACEHOLDER: &str = "<|unused_200053|>";
    pub const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
}

/// Reasoning effort for the system header (`Thinking effort level: …`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    /// Alias used in some HF docs (`xhigh` ≈ 0.95).
    XHigh,
    Max,
    /// Explicit numeric in `[0.0, 0.99]`.
    Custom(f32),
}

impl ReasoningEffort {
    pub fn parse(s: &str) -> Result<Self> {
        let key = s.trim().to_ascii_lowercase();
        Ok(match key.as_str() {
            "none" => Self::None,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "max" => Self::Max,
            other => {
                let v: f32 = other
                    .parse()
                    .map_err(|_| anyhow::anyhow!("unknown reasoning_effort: {s}"))?;
                if !(0.0..=0.99).contains(&v) {
                    bail!("reasoning_effort must be in [0.0, 0.99], got {v}");
                }
                Self::Custom(v)
            }
        })
    }

    pub fn as_f32(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Minimal => 0.1,
            Self::Low => 0.2,
            Self::Medium => 0.7,
            Self::High => 0.9,
            Self::XHigh => 0.95,
            Self::Max => 0.99,
            Self::Custom(v) => v,
        }
    }
}

/// Emit the thinking-effort system message (matches HF `emit_thinking_effort`).
pub fn thinking_effort_header(effort: ReasoningEffort) -> String {
    let num = effort.as_f32();
    let num_s = if num == 0.0 {
        "0".to_string()
    } else {
        // Trim trailing zeros for a stable render (0.9 not 0.90).
        let s = format!("{num}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    format!(
        "{sys}{text}Thinking effort level: {num_s}{end}",
        sys = tokens::MESSAGE_SYSTEM,
        text = tokens::CONTENT_TEXT,
        end = tokens::END_MESSAGE,
    )
}

/// One user text turn + generation prompt (`<|message_model|>`).
pub fn format_user_turn(user: &str, effort: ReasoningEffort) -> String {
    format!(
        "{effort_hdr}{user_tok}{text}{user}{end}{model}",
        effort_hdr = thinking_effort_header(effort),
        user_tok = tokens::MESSAGE_USER,
        text = tokens::CONTENT_TEXT,
        end = tokens::END_MESSAGE,
        model = tokens::MESSAGE_MODEL,
    )
}

/// Optional system text, then user turn (effort inserted before first non-system role).
pub fn format_chat(system: Option<&str>, user: &str, effort: ReasoningEffort) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str(tokens::MESSAGE_SYSTEM);
        out.push_str(tokens::CONTENT_TEXT);
        out.push_str(sys);
        out.push_str(tokens::END_MESSAGE);
    }
    out.push_str(&thinking_effort_header(effort));
    out.push_str(tokens::MESSAGE_USER);
    out.push_str(tokens::CONTENT_TEXT);
    out.push_str(user);
    out.push_str(tokens::END_MESSAGE);
    out.push_str(tokens::MESSAGE_MODEL);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_parse_and_header() {
        assert!((ReasoningEffort::parse("high").unwrap().as_f32() - 0.9).abs() < 1e-6);
        let h = thinking_effort_header(ReasoningEffort::Max);
        assert!(h.contains("Thinking effort level: 0.99"));
        assert!(h.contains(tokens::MESSAGE_SYSTEM));
    }

    #[test]
    fn user_turn_has_generation_prompt() {
        let s = format_user_turn("hi", ReasoningEffort::None);
        assert!(s.ends_with(tokens::MESSAGE_MODEL));
        assert!(s.contains("Thinking effort level: 0"));
        assert!(s.contains("hi"));
    }
}
