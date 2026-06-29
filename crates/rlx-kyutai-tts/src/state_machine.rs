//! TTS state machine — feeds words from a prepared script into the text stream.
//!
//! Ported from upstream `moshi.models.tts.StateMachine`.

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct TokenIds {
    /// Multiplex stride (`text_card + 1` in the published checkpoint).
    pub card: u32,
    pub new_word: u32,
    pub pad: u32,
    /// Start-of-turn marker for the main speaker (Kyutai TTS training).
    pub main: u32,
    pub other: u32,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub tokens: Vec<u32>,
    pub text: String,
    pub padding: usize,
}

#[derive(Debug)]
pub struct TtsState {
    pub entries: VecDeque<Entry>,
    pub remaining_padding: usize,
    pub forced_padding: usize,
    pub queued: VecDeque<u32>,
    pub lookahead_queued: VecDeque<u32>,
    pub end_step: Option<usize>,
    pub transcript: Vec<(String, usize)>,
}

#[derive(Debug, Clone)]
pub struct StateMachine {
    pub token_ids: TokenIds,
    pub second_stream_ahead: usize,
    pub max_padding: usize,
    pub initial_padding: usize,
}

impl StateMachine {
    pub fn for_config(text_card: u32, second_stream_ahead: usize) -> Self {
        Self {
            token_ids: TokenIds {
                card: text_card + 1,
                new_word: 0,
                pad: 3,
                main: 1,
                other: 2,
            },
            second_stream_ahead,
            max_padding: 8,
            initial_padding: 2,
        }
    }

    pub fn new_state(&self, entries: Vec<Entry>) -> TtsState {
        TtsState {
            entries: entries.into(),
            remaining_padding: self.initial_padding,
            forced_padding: self.initial_padding,
            queued: VecDeque::new(),
            lookahead_queued: VecDeque::new(),
            end_step: None,
            transcript: Vec::new(),
        }
    }

    /// Map a sampled text token to the multiplexed LM text input for the next step.
    pub fn process(&self, step: usize, state: &mut TtsState, token: u32) -> u32 {
        let ids = &self.token_ids;
        let mut token = if token == ids.new_word || token == ids.pad {
            token
        } else {
            ids.pad
        };

        if !state.queued.is_empty() {
            token = ids.pad;
        } else if state.forced_padding > 0 {
            token = ids.pad;
        } else if state.remaining_padding == 0 {
            token = ids.new_word;
        }

        if token == ids.new_word {
            if let Some(entry) = state.entries.pop_front() {
                if !entry.tokens.is_empty() {
                    state.transcript.push((entry.text.clone(), step));
                }
                state.queued.extend(entry.tokens);
                if self.second_stream_ahead > 0 {
                    state
                        .lookahead_queued
                        .extend(tokens_ahead(&state.entries, self.second_stream_ahead));
                }
                state.remaining_padding = self.max_padding;
                state.forced_padding = entry.padding;
            } else {
                token = ids.pad;
                if self.second_stream_ahead > 0 && state.end_step.is_none() {
                    token = ids.new_word;
                }
                if state.end_step.is_none() {
                    state.end_step = Some(step);
                }
            }
        }

        let mut output = if token == ids.pad {
            if state.remaining_padding > 0 {
                state.remaining_padding -= 1;
            }
            if state.forced_padding > 0 {
                state.forced_padding -= 1;
            }
            state.queued.pop_front().unwrap_or(ids.pad)
        } else if token == ids.new_word {
            ids.new_word
        } else {
            ids.pad
        };

        if self.second_stream_ahead > 0 {
            let mut second: i32 = -1;
            if output == ids.new_word {
                second = ids.new_word as i32;
                output = state.queued.pop_front().unwrap_or(ids.pad);
            } else if let Some(tok) = state.lookahead_queued.pop_front() {
                second = tok as i32;
            }
            output = (second + 1) as u32 * ids.card + output;
        }

        output
    }
}

fn tokens_ahead(entries: &VecDeque<Entry>, lookahead: usize) -> Vec<u32> {
    let mut rem = lookahead;
    for entry in entries {
        if !entry.tokens.is_empty() {
            rem -= 1;
            if rem == 0 {
                return entry.tokens.clone();
            }
        }
    }
    Vec::new()
}

pub fn script_to_entries(
    tokenizer: &crate::tokenizer::KyutaiTokenizer,
    prompt: &str,
) -> anyhow::Result<Vec<Entry>> {
    script_to_entries_with_options(tokenizer, prompt, true, 1)
}

/// Build script entries (Moshi `prepare_script` / `script_to_entries`).
pub fn script_to_entries_with_options(
    tokenizer: &crate::tokenizer::KyutaiTokenizer,
    prompt: &str,
    multi_speaker: bool,
    padding_between: usize,
) -> anyhow::Result<Vec<Entry>> {
    let normalized = prompt
        .replace('’', "'")
        .replace(':', " ")
        .replace('(', "")
        .replace(')', "");
    let words = tokenizer.encode_prompt_words(&normalized)?;
    let mut first_content = true;
    Ok(words
        .into_iter()
        .enumerate()
        .map(|(i, mut tokens)| {
            let text = normalized
                .split_whitespace()
                .nth(i)
                .unwrap_or("")
                .to_string();
            if multi_speaker && first_content && !tokens.is_empty() {
                tokens.insert(0, 1);
            }
            if !tokens.is_empty() {
                first_content = false;
            }
            let padding = if padding_between > 0 {
                padding_between
                    .saturating_add(tokens.len())
                    .saturating_sub(1)
            } else {
                0
            };
            Entry {
                tokens,
                text,
                padding,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplex_second_minus_one_is_identity() {
        let sm = StateMachine::for_config(8000, 2);
        let mut st = sm.new_state(vec![]);
        // Sampled pad, no lookahead → second stays -1, output passes through.
        let out = sm.process(0, &mut st, 3);
        assert_eq!(out, 3);
    }
}
