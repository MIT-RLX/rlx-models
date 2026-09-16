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

//! S1-mini's input protocol: the fixed system prompt, the three-axis control
//! line, and the ChatML wrapper with thinking **off**.
//!
//! The model card is unusually strict about this, and for a reason — S1-mini is
//! not a chat model, it is a single transformation trained on exactly one input
//! shape. Change the system prompt's wording, drop the control line, or send an
//! axis value outside the trained set and the output degrades or hallucinates.
//! So the wording lives here as a `const`, the axes are enums (unrepresentable
//! values can't be built), and the assistant turn is always primed with the
//! empty think block.
//!
//! Everything in this module is pure string work with no tokenizer and no
//! weights, so the wire format is unit-testable on its own — see the tests at
//! the bottom, which pin [`render_prompt`] against the literal prompt printed
//! in the model card.

use anyhow::Result;
use std::fmt;
use std::str::FromStr;

/// The system prompt S1-mini was trained with. Reproduce it **exactly**; the
/// model card calls out re-wording as a way to get garbled output.
pub const SYSTEM_PROMPT: &str = "You are a text normalizer for speech-to-text transcripts. \
The input begins with a control line specifying the styling, structure, and context settings; \
clean the transcript to match those settings and output only the cleaned text.";

/// ChatML turn delimiters (stock Qwen3).
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// The `enable_thinking=False` assistant prefix: `<think>\n\n</think>\n\n`.
/// S1-mini was trained with this exact prefix in front of every completion.
/// Leave it out and the model emits an empty think block and stops — the single
/// most common way to get a blank result out of it.
pub const EMPTY_THINK_BLOCK: &str = "<think>\n\n</think>\n\n";

/// Register — how much of the speaker's voice survives into the written text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Styling {
    /// All lowercase, apostrophes stripped, colloquialisms kept, final period
    /// usually omitted.
    Casual,
    /// Speaker's phrasing kept; `I` and its contractions capitalized, sentence
    /// starts stay lowercase.
    SemiCasual,
    /// Standard written English: full capitalization and punctuation,
    /// contractions kept, colloquialisms smoothed. The model card's default.
    #[default]
    SemiFormal,
    /// Like [`Styling::SemiFormal`] with contractions expanded (`I am`, `cannot`).
    Formal,
}

/// Whether the model may break enumerable content into Markdown bullets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Structure {
    /// Sentences and paragraphs only.
    #[default]
    Prose,
    /// Bullets permitted. The model is conservative: it wants ≥3 items, and
    /// anything that isn't clearly an enumeration stays prose.
    Lists,
}

/// Destination conventions for the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Context {
    /// Flowing text.
    #[default]
    General,
    /// Greeting line, body, and sign-off block separated by blank lines.
    Email,
}

macro_rules! axis_strings {
    ($ty:ty, $axis:literal, $( $variant:ident => $wire:literal ),+ $(,)?) => {
        impl $ty {
            /// Every trained value for this axis.
            pub const ALL: &'static [Self] = &[ $( Self::$variant ),+ ];

            /// The wire spelling used inside the control line.
            pub fn as_str(self) -> &'static str {
                match self { $( Self::$variant => $wire ),+ }
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $ty {
            type Err = anyhow::Error;
            fn from_str(s: &str) -> Result<Self> {
                // Accept `semi_formal` / `SemiFormal` as well as the wire
                // `semi-formal`, so CLI and config callers aren't fighting
                // punctuation. Only trained values are reachable either way.
                let key: String = s
                    .trim()
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric())
                    .map(|c| c.to_ascii_lowercase())
                    .collect();
                $(
                    if key == $wire.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>() {
                        return Ok(Self::$variant);
                    }
                )+
                anyhow::bail!(
                    "unknown {} value {:?} — S1-mini only accepts {}",
                    $axis,
                    s,
                    Self::ALL.iter().map(|v| v.as_str()).collect::<Vec<_>>().join(", "),
                )
            }
        }
    };
}

axis_strings!(
    Styling, "Styling",
    Casual => "casual",
    SemiCasual => "semi-casual",
    SemiFormal => "semi-formal",
    Formal => "formal",
);
axis_strings!(Structure, "Structure", Prose => "prose", Lists => "lists");
axis_strings!(Context, "Context", General => "general", Email => "email");

/// The three independent control axes. Every combination was trained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Controls {
    pub styling: Styling,
    pub structure: Structure,
    pub context: Context,
}

impl Controls {
    /// `[Styling: semi-formal] [Structure: prose] [Context: general]` — the
    /// model card's recommended default.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn styling(mut self, v: Styling) -> Self {
        self.styling = v;
        self
    }

    pub fn structure(mut self, v: Structure) -> Self {
        self.structure = v;
        self
    }

    pub fn context(mut self, v: Context) -> Self {
        self.context = v;
        self
    }

    /// What to put between the outputs of two chunks of one transcript.
    ///
    /// Deliberately dumb: chunking is a fallback for transcripts past the
    /// model's design point, and a single pass always produces better structure
    /// than any join can recover. Email layout wants blank lines between blocks,
    /// a bulleted list wants one newline per item, prose wants a space.
    pub fn chunk_separator(&self) -> &'static str {
        match (self.context, self.structure) {
            (Context::Email, _) => "\n\n",
            (_, Structure::Lists) => "\n",
            _ => " ",
        }
    }

    /// The literal control line, without the trailing newline.
    pub fn control_line(&self) -> String {
        format!(
            "[Styling: {}] [Structure: {}] [Context: {}]",
            self.styling, self.structure, self.context
        )
    }
}

impl fmt::Display for Controls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.control_line())
    }
}

/// The user turn: control line, newline, raw transcript.
///
/// The transcript is passed through untouched apart from trimming — S1-mini was
/// trained on raw ASR output, so lowercasing or stripping punctuation here would
/// move the input away from the training distribution, not toward it.
pub fn render_user_turn(controls: Controls, transcript: &str) -> String {
    format!("{}\n{}", controls.control_line(), transcript.trim())
}

/// The complete prompt string, ready to tokenize: ChatML system + user turns
/// plus an assistant turn primed with the empty think block.
///
/// Byte-identical to HF `apply_chat_template(messages, add_generation_prompt=True,
/// enable_thinking=False)` against the repo's `chat_template.jinja` (stock
/// Qwen3), which the tests below pin against the literal in the model card.
pub fn render_prompt(controls: Controls, transcript: &str) -> String {
    let mut out = String::with_capacity(SYSTEM_PROMPT.len() + transcript.len() + 128);
    out.push_str(IM_START);
    out.push_str("system\n");
    out.push_str(SYSTEM_PROMPT);
    out.push_str(IM_END);
    out.push('\n');
    out.push_str(IM_START);
    out.push_str("user\n");
    out.push_str(&render_user_turn(controls, transcript));
    out.push_str(IM_END);
    out.push('\n');
    out.push_str(IM_START);
    out.push_str("assistant\n");
    out.push_str(EMPTY_THINK_BLOCK);
    out
}

/// The fixed prefix of every prompt — system turn plus the `<|im_start|>user\n`
/// opener. Identical across calls, so a runner can prefill it once and reuse the
/// KV cache (see `S1Runner`'s prefix cache).
pub fn shared_prefix() -> String {
    format!("{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n{IM_START}user\n")
}

/// The model card's `max_new_tokens` ceiling: output length closely tracks input
/// length, so `1.3 × input + 32` is safe and far cheaper than parking at 1024.
pub fn recommended_max_new_tokens(prompt_tokens: usize) -> usize {
    ((prompt_tokens as f64 * 1.3).ceil() as usize) + 32
}

// ─── Chunking ─────────────────────────────────────────────────────────
//
// S1-mini is built for dictation-length input and the card asks for single
// passes under ~1,000 tokens, chunked at sentence boundaries. Raw ASR output
// often has no sentence punctuation at all, so `sentence_units` degrades to
// whole-input and the packer falls back to word boundaries.

/// Characters that can end a sentence.
const TERMINATORS: [char; 4] = ['.', '?', '!', '…'];
/// Closing punctuation that may trail a terminator (`he said "no!"`).
const CLOSERS: [char; 6] = ['"', '\'', ')', ']', '\u{201d}', '\u{2019}'];

/// Split into sentence-ish units. Trailing whitespace stays attached to its
/// unit, so concatenating the result reproduces the input exactly.
///
/// A terminator only breaks when followed by whitespace or end-of-input, which
/// keeps `3.15`, `e.g`, and `support@x.com` intact — exactly the tokens this
/// model exists to produce.
pub fn sentence_units(text: &str) -> Vec<&str> {
    let mut units = Vec::new();
    let mut start = 0usize;
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let newline = c == '\n';
        if !newline && !TERMINATORS.contains(&c) {
            continue;
        }
        let mut end = i + c.len_utf8();
        if !newline {
            while let Some(&(j, c2)) = it.peek() {
                if TERMINATORS.contains(&c2) || CLOSERS.contains(&c2) {
                    end = j + c2.len_utf8();
                    it.next();
                } else {
                    break;
                }
            }
        }
        let mut boundary = newline;
        while let Some(&(j, c2)) = it.peek() {
            if c2.is_whitespace() {
                boundary = true;
                end = j + c2.len_utf8();
                it.next();
            } else {
                break;
            }
        }
        // Mid-token punctuation (`3.15`) — not a boundary.
        if !boundary && end < text.len() {
            continue;
        }
        units.push(&text[start..end]);
        start = end;
    }
    if start < text.len() {
        units.push(&text[start..]);
    }
    units
}

/// Split into whitespace-delimited words, trailing whitespace attached. The
/// fallback when a "sentence" is itself over budget (unpunctuated dictation).
pub fn word_units(text: &str) -> Vec<&str> {
    let mut units = Vec::new();
    let mut start = 0usize;
    let mut in_ws = false;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            in_ws = true;
        } else if in_ws {
            units.push(&text[start..i]);
            start = i;
            in_ws = false;
        }
    }
    if start < text.len() {
        units.push(&text[start..]);
    }
    units
}

/// Greedily pack `text` into chunks of at most `max_tokens`, measured by
/// `count`, breaking at sentence boundaries and falling back to word boundaries
/// for oversized sentences.
///
/// `count` measures the **transcript** only; the caller is responsible for
/// leaving headroom for the ~60-token system + control prefix. A single word
/// that exceeds the budget on its own is emitted as its own oversized chunk
/// rather than being cut mid-word — truncating it would corrupt the one thing
/// the model is supposed to preserve.
pub fn chunk_transcript(
    text: &str,
    max_tokens: usize,
    count: impl Fn(&str) -> Result<usize>,
) -> Result<Vec<String>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if max_tokens == 0 || count(trimmed)? <= max_tokens {
        return Ok(vec![trimmed.to_string()]);
    }

    // Sentence units, exploded into words when a single sentence is over budget.
    let mut units: Vec<&str> = Vec::new();
    for unit in sentence_units(trimmed) {
        if count(unit)? > max_tokens {
            units.extend(word_units(unit));
        } else {
            units.push(unit);
        }
    }

    let mut chunks = Vec::new();
    let mut cur = String::new();
    for unit in units {
        if cur.is_empty() {
            cur.push_str(unit);
            continue;
        }
        let mut candidate = String::with_capacity(cur.len() + unit.len());
        candidate.push_str(&cur);
        candidate.push_str(unit);
        if count(candidate.trim())? <= max_tokens {
            cur = candidate;
        } else {
            chunks.push(std::mem::take(&mut cur).trim().to_string());
            cur.push_str(unit);
        }
    }
    if !cur.trim().is_empty() {
        chunks.push(cur.trim().to_string());
    }
    chunks.retain(|c| !c.is_empty());
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal prompt printed in the model card's "Set `enable_thinking=False`"
    /// section. If this ever drifts, every downstream normalization silently
    /// leaves the training distribution — so pin the bytes, not the shape.
    #[test]
    fn prompt_matches_model_card_literal() {
        let expected = concat!(
            "<|im_start|>system\n",
            "You are a text normalizer for speech-to-text transcripts. The input begins with a ",
            "control line specifying the styling, structure, and context settings; clean the ",
            "transcript to match those settings and output only the cleaned text.<|im_end|>\n",
            "<|im_start|>user\n",
            "[Styling: semi-formal] [Structure: prose] [Context: general]\n",
            "<raw transcript><|im_end|>\n",
            "<|im_start|>assistant\n",
            "<think>\n\n</think>\n\n",
        );
        assert_eq!(render_prompt(Controls::new(), "<raw transcript>"), expected);
    }

    #[test]
    fn assistant_prefix_is_empty_think_block() {
        let p = render_prompt(Controls::new(), "hi");
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        // Two newlines inside the think block and two more after it.
        assert_eq!(EMPTY_THINK_BLOCK, "<think>\n\n</think>\n\n");
    }

    #[test]
    fn shared_prefix_is_a_real_prefix_of_every_prompt() {
        let pre = shared_prefix();
        for &s in Styling::ALL {
            for &st in Structure::ALL {
                for &c in Context::ALL {
                    let ctl = Controls {
                        styling: s,
                        structure: st,
                        context: c,
                    };
                    assert!(render_prompt(ctl, "whatever").starts_with(&pre));
                }
            }
        }
    }

    #[test]
    fn control_line_spellings() {
        assert_eq!(
            Controls::new().control_line(),
            "[Styling: semi-formal] [Structure: prose] [Context: general]"
        );
        let ctl = Controls::new()
            .styling(Styling::SemiCasual)
            .structure(Structure::Lists)
            .context(Context::Email);
        assert_eq!(
            ctl.control_line(),
            "[Styling: semi-casual] [Structure: lists] [Context: email]"
        );
    }

    #[test]
    fn axis_parsing_accepts_variants_and_rejects_untrained() {
        assert_eq!(
            "semi-formal".parse::<Styling>().unwrap(),
            Styling::SemiFormal
        );
        assert_eq!(
            "semi_formal".parse::<Styling>().unwrap(),
            Styling::SemiFormal
        );
        assert_eq!(
            "SemiFormal".parse::<Styling>().unwrap(),
            Styling::SemiFormal
        );
        assert_eq!("  Formal ".parse::<Styling>().unwrap(), Styling::Formal);
        assert_eq!("lists".parse::<Structure>().unwrap(), Structure::Lists);
        assert_eq!("email".parse::<Context>().unwrap(), Context::Email);
        // Untrained values must not reach the model.
        assert!("business".parse::<Styling>().is_err());
        assert!("bullets".parse::<Structure>().is_err());
        assert!("slack".parse::<Context>().is_err());
    }

    #[test]
    fn chunk_separator_per_axis() {
        assert_eq!(Controls::new().chunk_separator(), " ");
        assert_eq!(
            Controls::new()
                .structure(Structure::Lists)
                .chunk_separator(),
            "\n"
        );
        assert_eq!(
            Controls::new().context(Context::Email).chunk_separator(),
            "\n\n"
        );
        // Email layout wins over list layout — blank-line blocks are the
        // coarser structure.
        let both = Controls::new()
            .structure(Structure::Lists)
            .context(Context::Email);
        assert_eq!(both.chunk_separator(), "\n\n");
    }

    #[test]
    fn user_turn_trims_transcript_but_keeps_content() {
        let t = render_user_turn(Controls::new(), "  so um send it  \n");
        assert_eq!(
            t,
            "[Styling: semi-formal] [Structure: prose] [Context: general]\nso um send it"
        );
    }

    #[test]
    fn max_new_tokens_heuristic() {
        assert_eq!(recommended_max_new_tokens(0), 32);
        assert_eq!(recommended_max_new_tokens(100), 162);
        assert_eq!(recommended_max_new_tokens(1000), 1332);
    }

    #[test]
    fn sentence_units_round_trip_and_split() {
        let text = "One. Two? Three!\nFour";
        let units = sentence_units(text);
        assert_eq!(units, vec!["One. ", "Two? ", "Three!\n", "Four"]);
        assert_eq!(units.concat(), text);
    }

    #[test]
    fn sentence_units_keep_decimals_and_emails_intact() {
        for text in [
            "meet at 3.15 today",
            "send it to support@superwhisper.com now",
            "e.g. this one",
        ] {
            let units = sentence_units(text);
            assert_eq!(units.concat(), text);
        }
        // `3.15` must not split mid-number.
        assert_eq!(
            sentence_units("meet at 3.15 today"),
            vec!["meet at 3.15 today"]
        );
        // `e.g. ` legitimately looks like a boundary; that's fine, we only
        // promise not to split *inside* a token.
        assert!(
            sentence_units("e.g. this one")
                .iter()
                .all(|u| !u.is_empty())
        );
    }

    #[test]
    fn word_units_round_trip() {
        let text = "so um   i need\nto send";
        assert_eq!(word_units(text).concat(), text);
        assert_eq!(word_units(text).len(), 6);
    }

    /// Word-count stand-in for the tokenizer so chunking is testable without
    /// weights. Real callers pass the BPE.
    fn words(s: &str) -> Result<usize> {
        Ok(s.split_whitespace().count())
    }

    #[test]
    fn chunking_short_input_is_a_single_chunk() {
        let out = chunk_transcript("so um send the report", 100, words).unwrap();
        assert_eq!(out, vec!["so um send the report"]);
        assert!(chunk_transcript("   ", 100, words).unwrap().is_empty());
    }

    #[test]
    fn chunking_breaks_at_sentence_boundaries() {
        let text = "One two three. Four five six. Seven eight nine.";
        let out = chunk_transcript(text, 6, words).unwrap();
        assert_eq!(
            out,
            vec!["One two three. Four five six.", "Seven eight nine."]
        );
        for c in &out {
            assert!(words(c).unwrap() <= 6);
        }
    }

    #[test]
    fn chunking_falls_back_to_words_when_unpunctuated() {
        // Raw ASR: no sentence punctuation anywhere.
        let text = "so um i need to like send the report by friday no wait thursday";
        let out = chunk_transcript(text, 5, words).unwrap();
        assert!(out.len() > 1);
        for c in &out {
            assert!(words(c).unwrap() <= 5, "chunk over budget: {c:?}");
        }
        // Nothing lost, nothing duplicated.
        assert_eq!(out.join(" "), text);
    }

    #[test]
    fn chunking_emits_oversized_single_word_rather_than_truncating() {
        let text = "supercalifragilistic";
        let out = chunk_transcript(text, 0, words).unwrap();
        assert_eq!(out, vec![text]);
    }
}
