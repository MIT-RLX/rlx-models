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

//! Prompt rows: the `[main token, code 0, …, code N-1]` sequence both fish
//! models read.
//!
//! A text row is `[token, 0, …, 0]`; an audio row is
//! `[semantic_begin + c0, c0, …, cN-1]`. Every literal string is tokenized
//! *on its own* — the reference encodes each part separately, and
//! tokenizing the joined string gives different ids at the seams.

use anyhow::{Result, ensure};

/// One piece of a prompt.
#[derive(Clone, Debug)]
pub enum Segment {
    Text(String),
    /// Codes `[num_codebooks][frames]`.
    Codes(Vec<Vec<u32>>),
}

/// A prompt row.
pub type Row = Vec<u32>;

/// Text → token ids, special tokens (`<|im_start|>`, `<|speaker:0|>`, …)
/// parsed inline, no BOS/EOS added.
pub trait TextEncoder {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>>;
    fn special_id(&self, token: &str) -> Option<u32>;
}

#[cfg(feature = "tokenizer")]
impl TextEncoder for tokenizers::Tokenizer {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn special_id(&self, token: &str) -> Option<u32> {
        self.token_to_id(token)
    }
}

/// Load `tokenizer.json` from a model directory.
#[cfg(feature = "tokenizer")]
pub fn load_tokenizer(dir: &std::path::Path) -> Result<tokenizers::Tokenizer> {
    let path = dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))
}

/// Encode segments into rows `1 + num_codebooks` wide.
pub fn rows(
    segments: &[Segment],
    enc: &dyn TextEncoder,
    num_codebooks: usize,
    semantic_begin: u32,
) -> Result<Vec<Row>> {
    let mut out = Vec::new();
    for seg in segments {
        match seg {
            Segment::Text(t) => {
                for id in enc.encode_text(t)? {
                    let mut r = vec![0u32; 1 + num_codebooks];
                    r[0] = id;
                    out.push(r);
                }
            }
            Segment::Codes(c) => {
                ensure!(
                    c.len() == num_codebooks,
                    "codes have {} codebooks, want {num_codebooks}",
                    c.len()
                );
                let frames = c[0].len();
                ensure!(c.iter().all(|r| r.len() == frames), "ragged code matrix");
                for t in 0..frames {
                    let mut r = Vec::with_capacity(1 + num_codebooks);
                    r.push(semantic_begin + c[0][t]);
                    r.extend(c.iter().map(|cb| cb[t]));
                    out.push(r);
                }
            }
        }
    }
    Ok(out)
}

/// Whether `text` holds a `<|speaker:N|>` tag anywhere.
pub fn has_speaker_tag(text: &str) -> bool {
    speaker_tags(text).next().is_some()
}

/// Byte ranges of every well-formed `<|speaker:N|>` tag.
fn speaker_tags(text: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    text.match_indices("<|speaker:").filter_map(move |(i, m)| {
        let rest = &text[i + m.len()..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        (digits > 0 && rest[digits..].starts_with("|>")).then_some((i, i + m.len() + digits + 2))
    })
}

/// S2's `split_text_by_speaker`: one turn per `<|speaker:N|>` tag, each
/// starting with its tag and trimmed. Text before the first tag is dropped,
/// exactly like the reference (callers fall back to the whole text when no
/// tag is present).
pub fn split_by_speaker(text: &str) -> Vec<String> {
    let tags: Vec<(usize, usize)> = speaker_tags(text).collect();
    let mut turns = Vec::with_capacity(tags.len());
    for (i, &(start, end)) in tags.iter().enumerate() {
        let stop = tags.get(i + 1).map_or(text.len(), |t| t.0);
        let tag = text[start..end].trim();
        let body = &text[end..stop];
        turns.push(format!("{tag}{body}").trim().to_string());
    }
    turns
}

/// S2's `group_turns_into_batches`: consecutive turns joined by `\n` while a
/// batch has fewer than `max_speakers` turns and stays within `max_bytes`.
pub fn group_turns(turns: &[String], max_speakers: usize, max_bytes: usize) -> Vec<String> {
    let mut batches = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut bytes = 0;
    for t in turns {
        let n = t.len();
        if cur.len() >= max_speakers || (!cur.is_empty() && bytes + n > max_bytes) {
            batches.push(cur.join("\n"));
            cur.clear();
            bytes = 0;
        }
        cur.push(t);
        bytes += n;
    }
    if !cur.is_empty() {
        batches.push(cur.join("\n"));
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Chars;
    impl TextEncoder for Chars {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.chars().map(|c| c as u32).collect())
        }
        fn special_id(&self, _: &str) -> Option<u32> {
            None
        }
    }

    #[test]
    fn rows_mix_text_and_codes() {
        let segs = [
            Segment::Text("ab".into()),
            Segment::Codes(vec![vec![5, 6], vec![1, 2], vec![3, 4]]),
        ];
        let r = rows(&segs, &Chars, 3, 1000).unwrap();
        assert_eq!(
            r,
            vec![
                vec![97, 0, 0, 0],
                vec![98, 0, 0, 0],
                vec![1005, 5, 1, 3],
                vec![1006, 6, 2, 4]
            ]
        );
    }

    #[test]
    fn speaker_split_and_group() {
        let t = "intro <|speaker:0|>Hi there. <|speaker:1|> Hello!";
        let turns = split_by_speaker(t);
        assert_eq!(
            turns,
            vec!["<|speaker:0|>Hi there.", "<|speaker:1|> Hello!"]
        );
        assert_eq!(
            group_turns(&turns, 5, 200),
            vec!["<|speaker:0|>Hi there.\n<|speaker:1|> Hello!"]
        );
        assert_eq!(group_turns(&turns, 1, 200).len(), 2);
        assert!(split_by_speaker("no tags").is_empty());
        assert!(!has_speaker_tag("<|speaker:|>"));
    }
}
