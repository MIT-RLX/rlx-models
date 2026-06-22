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

//! Token → text reconstruction for FunASR vocabularies.
//!
//! A single ordered piece list (from `tokens.json` / `tokens.txt`) covers both
//! the Paraformer **char** tokenizer (CJK pieces joined without spaces, `@@`
//! continuation for latin sub-words) and the SenseVoice **SentencePiece**
//! pieces (`▁` marks word starts). SenseVoice rich tags (`<|zh|>`, `<|HAPPY|>`,
//! …) can be split off or mapped to emoji.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

const SP_SPACE: char = '\u{2581}'; // ▁

/// An ordered piece vocabulary.
pub struct Tokenizer {
    pieces: Vec<String>,
    rev: HashMap<String, u32>,
}

impl Tokenizer {
    /// Build a tokenizer from an ordered piece list.
    pub fn new(pieces: Vec<String>) -> Self {
        let rev = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        Self { pieces, rev }
    }

    /// Id of a surface piece, if present.
    pub fn id_of(&self, piece: &str) -> Option<u32> {
        self.rev.get(piece).copied()
    }

    /// Number of pieces in the vocabulary.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }
    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Load from `tokens.json` (a JSON array, or `{"piece": id}` object),
    /// `tokens.txt` (one piece per line), or a SentencePiece `*.bpe.model` /
    /// `*.model` (the proto is parsed natively) in a model directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let j = dir.join("tokens.json");
        if j.is_file() {
            let text =
                std::fs::read_to_string(&j).with_context(|| format!("read {}", j.display()))?;
            return Self::from_json(&text);
        }
        let t = dir.join("tokens.txt");
        if t.is_file() {
            let text =
                std::fs::read_to_string(&t).with_context(|| format!("read {}", t.display()))?;
            let pieces = text
                .lines()
                .map(|l| l.split_whitespace().next().unwrap_or("").to_string())
                .collect();
            return Ok(Self::new(pieces));
        }
        // SentencePiece model
        if let Ok(rd) = std::fs::read_dir(dir) {
            let mut sp: Option<std::path::PathBuf> = None;
            for e in rd.flatten() {
                let p = e.path();
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if name.ends_with(".bpe.model") || name.ends_with(".model") {
                        sp = Some(p);
                        break;
                    }
                }
            }
            if let Some(p) = sp {
                let bytes = std::fs::read(&p).with_context(|| format!("read {}", p.display()))?;
                return Ok(Self::new(parse_sentencepiece(&bytes)));
            }
        }
        anyhow::bail!("no tokens.json / tokens.txt / *.model in {}", dir.display())
    }

    /// Parse a `tokens.json` array or `{piece: id}` object.
    pub fn from_json(text: &str) -> Result<Self> {
        let val: serde_json::Value = serde_json::from_str(text).context("parse tokens.json")?;
        let pieces = match val {
            serde_json::Value::Array(a) => a
                .into_iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            serde_json::Value::Object(map) => {
                // {"piece": id} → place each piece at its id
                let mut pairs: Vec<(usize, String)> = map
                    .into_iter()
                    .filter_map(|(k, v)| v.as_u64().map(|id| (id as usize, k)))
                    .collect();
                pairs.sort_by_key(|(id, _)| *id);
                let max = pairs.last().map(|(id, _)| *id).unwrap_or(0);
                let mut out = vec![String::new(); max + 1];
                for (id, k) in pairs {
                    out[id] = k;
                }
                out
            }
            _ => anyhow::bail!("tokens.json: unexpected JSON shape"),
        };
        Ok(Self::new(pieces))
    }

    fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|s| s.as_str())
    }

    fn is_structural(p: &str) -> bool {
        matches!(
            p,
            "<blank>" | "<s>" | "</s>" | "<sos/eos>" | "<sos>" | "<eos>" | "" | "<pad>"
        )
    }

    fn is_tag(p: &str) -> bool {
        p.starts_with("<|") && p.ends_with("|>")
    }

    /// Decode token ids to text. `strip_tags` removes `<|...|>` markers.
    pub fn decode(&self, ids: &[u32], strip_tags: bool) -> String {
        let mut kept: Vec<&str> = Vec::new();
        for &id in ids {
            let Some(p) = self.piece(id) else { continue };
            if Self::is_structural(p) {
                continue;
            }
            if Self::is_tag(p) && strip_tags {
                continue;
            }
            kept.push(p);
        }
        let joined: String = kept.concat();
        if joined.contains(SP_SPACE) {
            // SentencePiece detokenization
            joined.replace(SP_SPACE, " ").trim().to_string()
        } else {
            // char tokenizer: `@@` marks a continuation (join with no space)
            joined.replace("@@", "")
        }
    }

    /// Return only the leading `<|...|>` rich tags (SenseVoice language / event
    /// / emotion / text-norm), in order.
    pub fn tags(&self, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .filter_map(|&id| self.piece(id))
            .filter(|p| Self::is_tag(p))
            .map(|p| p.to_string())
            .collect()
    }

    /// SenseVoice `rich_transcription_postprocess`: emotion / audio-event tags
    /// rendered as emoji, prepended to the cleaned text. Language and text-norm
    /// tags are dropped.
    pub fn rich(&self, ids: &[u32]) -> String {
        let mut emojis = String::new();
        for t in self.tags(ids) {
            if let Some(e) = tag_emoji(&t) {
                if !emojis.contains(e) {
                    emojis.push_str(e);
                }
            }
        }
        let text = self.decode(ids, true);
        if emojis.is_empty() {
            text
        } else {
            format!("{emojis} {text}")
        }
    }
}

/// Map a SenseVoice emotion / audio-event tag to an emoji (others → `None`).
fn tag_emoji(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "<|HAPPY|>" => "😊",
        "<|SAD|>" => "😔",
        "<|ANGRY|>" => "😡",
        "<|FEARFUL|>" => "😰",
        "<|DISGUSTED|>" => "🤢",
        "<|SURPRISED|>" => "😮",
        "<|BGM|>" => "🎼",
        "<|Applause|>" => "👏",
        "<|Laughter|>" => "😀",
        "<|Cry|>" => "😭",
        "<|Sneeze|>" => "🤧",
        "<|Cough|>" => "😷",
        _ => return None, // NEUTRAL / Speech / language / text-norm → no emoji
    })
}

/// Read a protobuf base-128 varint.
fn read_varint(b: &[u8], i: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0u32;
    while *i < b.len() {
        let x = b[*i];
        *i += 1;
        v |= ((x & 0x7f) as u64) << shift;
        if x & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    v
}

/// Parse a SentencePiece `ModelProto`: top-level field 1 is the repeated
/// `pieces` message; each piece's field 1 is the surface string. The list order
/// is the token-id order.
pub fn parse_sentencepiece(data: &[u8]) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let tag = read_varint(data, &mut i);
        let field = tag >> 3;
        let wt = tag & 7;
        if field == 1 && wt == 2 {
            let len = read_varint(data, &mut i) as usize;
            let end = (i + len).min(data.len());
            let msg = &data[i..end];
            i = end;
            let mut j = 0;
            let mut piece = String::new();
            while j < msg.len() {
                let t2 = read_varint(msg, &mut j);
                let f2 = t2 >> 3;
                match t2 & 7 {
                    2 => {
                        let l2 = read_varint(msg, &mut j) as usize;
                        let e2 = (j + l2).min(msg.len());
                        if f2 == 1 {
                            piece = String::from_utf8_lossy(&msg[j..e2]).into_owned();
                        }
                        j = e2;
                    }
                    0 => {
                        read_varint(msg, &mut j);
                    }
                    5 => j += 4,
                    1 => j += 8,
                    _ => break,
                }
            }
            pieces.push(piece);
        } else {
            match wt {
                2 => {
                    let l = read_varint(data, &mut i) as usize;
                    i = (i + l).min(data.len());
                }
                0 => {
                    read_varint(data, &mut i);
                }
                5 => i += 4,
                1 => i += 8,
                _ => break,
            }
        }
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentencepiece_decode() {
        let tok = Tokenizer::new(
            ["<blank>", "<s>", "</s>", "▁hello", "▁world", "!"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        let text = tok.decode(&[1, 3, 4, 5, 2], true);
        assert_eq!(text, "hello world!");
    }

    #[test]
    fn char_decode_cjk() {
        let tok = Tokenizer::new(
            ["<blank>", "<s>", "</s>", "你", "好", "世", "界"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        let text = tok.decode(&[3, 4, 5, 6], true);
        assert_eq!(text, "你好世界");
    }

    #[test]
    fn tags_extracted() {
        let tok = Tokenizer::new(
            ["<blank>", "<|zh|>", "<|HAPPY|>", "▁hi"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        assert_eq!(tok.tags(&[1, 2, 3]), vec!["<|zh|>", "<|HAPPY|>"]);
        assert_eq!(tok.decode(&[1, 2, 3], true), "hi");
    }
}
