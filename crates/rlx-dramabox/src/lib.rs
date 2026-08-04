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

//! # rlx-dramabox
//!
//! **DramaBox** — expressive TTS + voice cloning on RLX. An LM-style backbone emits
//! neural-codec tokens conditioned on text, an inline **expressive** style/emotion
//! track, and an optional reference voice; a neural codec renders the waveform.
//!
//! Native Rust, composing rlx pieces:
//!
//! - **Backbone** → Llama-shaped LM (`rlx-llama32`).
//! - **Codec** → a neural audio codec (`rlx-dac` / `rlx-snac`).
//!
//! The checkpoint-free core here is the config plus an **inline expressive-tag
//! parser** ([`parse_expressive`]) that splits `"…[happy]…[sad]…"` markup into
//! styled spans the backbone conditions on. LM + codec graph wiring is next.

use anyhow::{Result, ensure};

/// DramaBox model config. Dimensional fields carry plausible values; exact widths
/// come from the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct DramaBoxConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    // LM backbone.
    pub backbone_hidden: usize,
    pub backbone_layers: usize,
    pub backbone_heads: usize,
    // Neural codec.
    pub num_codebooks: usize,
    pub codebook_size: usize,
    /// Expressive style/emotion conditioning width.
    pub emotion_dim: usize,
    pub supports_cloning: bool,
}

impl Default for DramaBoxConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            hop_length: 320,
            backbone_hidden: 1536,
            backbone_layers: 24,
            backbone_heads: 16,
            num_codebooks: 4,
            codebook_size: 1024,
            emotion_dim: 512,
            supports_cloning: true,
        }
    }
}

impl DramaBoxConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_codebooks > 0, "num_codebooks must be > 0");
        ensure!(self.codebook_size > 0, "codebook_size must be > 0");
        Ok(())
    }

    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }
}

/// A run of text carrying a single expressive style (`None` = the default/neutral
/// style up to the first tag).
#[derive(Debug, Clone, PartialEq)]
pub struct StyledSpan {
    pub style: Option<String>,
    pub text: String,
}

/// Parse inline expressive markup into styled spans. A `[tag]` sets the style for
/// all following text until the next tag; text before the first tag is neutral.
/// An unclosed `[` is treated as literal text.
pub fn parse_expressive(input: &str) -> Vec<StyledSpan> {
    let mut spans: Vec<StyledSpan> = Vec::new();
    let mut cur_style: Option<String> = None;
    let mut cur_text = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' {
            // Read the tag first so an unclosed '[' stays part of the text run
            // (no premature span split).
            let mut tag = String::new();
            let mut closed = false;
            for tc in chars.by_ref() {
                if tc == ']' {
                    closed = true;
                    break;
                }
                tag.push(tc);
            }
            if closed {
                // A real tag: flush the pending text under the *old* style, then
                // switch style.
                if !cur_text.is_empty() {
                    spans.push(StyledSpan {
                        style: cur_style.clone(),
                        text: std::mem::take(&mut cur_text),
                    });
                }
                cur_style = Some(tag.trim().to_string());
            } else {
                // Unclosed bracket → literal text, contiguous with the run.
                cur_text.push('[');
                cur_text.push_str(&tag);
            }
        } else {
            cur_text.push(c);
        }
    }
    if !cur_text.is_empty() {
        spans.push(StyledSpan {
            style: cur_style,
            text: cur_text,
        });
    }
    spans
}

/// The spoken text with all expressive tags removed.
pub fn strip_tags(input: &str) -> String {
    parse_expressive(input)
        .into_iter()
        .map(|s| s.text)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_validate() {
        let c = DramaBoxConfig::default();
        assert!(c.supports_cloning);
        c.validate().unwrap();
    }

    #[test]
    fn parses_inline_style_tags() {
        let spans = parse_expressive("Hello [happy]world [sad]bye");
        assert_eq!(
            spans,
            vec![
                StyledSpan {
                    style: None,
                    text: "Hello ".into()
                },
                StyledSpan {
                    style: Some("happy".into()),
                    text: "world ".into()
                },
                StyledSpan {
                    style: Some("sad".into()),
                    text: "bye".into()
                },
            ]
        );
    }

    #[test]
    fn plain_text_is_one_neutral_span() {
        let spans = parse_expressive("just talking");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].style, None);
        assert_eq!(strip_tags("Hello [happy]world [sad]bye"), "Hello world bye");
    }

    #[test]
    fn unclosed_bracket_is_literal() {
        let spans = parse_expressive("oops [broken text");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "oops [broken text");
        assert_eq!(spans[0].style, None);
    }
}
