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

//! Token ids → text. Recognition only ever decodes, so this is the
//! id → string half of the two tokenizers TrOCR checkpoints use:
//!
//! - **small**: `XLMRobertaTokenizer` over `sentencepiece.bpe.model`, with
//!   fairseq's ids — `<s> <pad> </s> <unk>` at 0–3, sentencepiece piece `i`
//!   at `i + 1`, `<mask>` after the last piece;
//! - **base / large**: `RobertaTokenizer`, byte-level BPE from `vocab.json`.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Tokenizer {
    SentencePiece {
        /// Piece text indexed by sentencepiece id.
        pieces: Vec<String>,
    },
    ByteBpe {
        tokens: HashMap<u32, String>,
        byte_of: HashMap<char, u8>,
    },
}

/// fairseq's four leading special ids.
const FAIRSEQ_SPECIALS: u32 = 4;

impl Tokenizer {
    /// Load from a model directory (`sentencepiece.bpe.model` or
    /// `vocab.json`).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let spm = dir.join("sentencepiece.bpe.model");
        if spm.is_file() {
            let b = std::fs::read(&spm).with_context(|| format!("reading {}", spm.display()))?;
            return Self::from_sentencepiece(&b);
        }
        let vocab = dir.join("vocab.json");
        if vocab.is_file() {
            let s = std::fs::read_to_string(&vocab)
                .with_context(|| format!("reading {}", vocab.display()))?;
            return Self::from_vocab_json(&s);
        }
        bail!(
            "{}: no sentencepiece.bpe.model or vocab.json",
            dir.display()
        )
    }

    /// Pieces of a sentencepiece `ModelProto` (field 1, each a message whose
    /// field 1 is the piece string).
    pub fn from_sentencepiece(model: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        for (field, body) in proto_fields(model)? {
            if field != 1 {
                continue;
            }
            let piece = proto_fields(body)?
                .into_iter()
                .find(|(f, _)| *f == 1)
                .map(|(_, b)| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default();
            pieces.push(piece);
        }
        if pieces.is_empty() {
            bail!("sentencepiece model has no pieces");
        }
        Ok(Tokenizer::SentencePiece { pieces })
    }

    pub fn from_vocab_json(text: &str) -> Result<Self> {
        let v: HashMap<String, u32> = serde_json::from_str(text).context("parsing vocab.json")?;
        let tokens = v.into_iter().map(|(t, i)| (i, t)).collect();
        let byte_of = bytes_to_unicode()
            .into_iter()
            .enumerate()
            .map(|(b, c)| (c, b as u8))
            .collect();
        Ok(Tokenizer::ByteBpe { tokens, byte_of })
    }

    /// Text for `ids`, special tokens skipped (`skip_special_tokens=True`).
    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Tokenizer::SentencePiece { pieces } => {
                let s: String = ids
                    .iter()
                    .filter(|&&i| i >= FAIRSEQ_SPECIALS)
                    .filter_map(|&i| pieces.get((i - 1) as usize))
                    .filter(|p| !(p.starts_with('<') && p.ends_with('>')))
                    .map(String::as_str)
                    .collect();
                s.replace('\u{2581}', " ").trim().to_string()
            }
            Tokenizer::ByteBpe { tokens, byte_of } => {
                let mut bytes = Vec::new();
                for t in ids.iter().filter_map(|i| tokens.get(i)) {
                    if is_special(t) {
                        continue;
                    }
                    bytes.extend(t.chars().filter_map(|c| byte_of.get(&c)));
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }
        }
    }
}

fn is_special(t: &str) -> bool {
    matches!(t, "<s>" | "</s>" | "<pad>" | "<unk>" | "<mask>")
}

/// GPT-2's byte → printable-char table (index = byte).
fn bytes_to_unicode() -> Vec<char> {
    let mut printable: Vec<u32> = (u32::from(b'!')..=u32::from(b'~'))
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars = printable.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    let mut table = vec!['\0'; 256];
    for (b, c) in printable.into_iter().zip(chars) {
        table[b as usize] = char::from_u32(c).expect("valid scalar");
    }
    table
}

/// Length-delimited fields of one protobuf message level (other wire types
/// skipped).
fn proto_fields(buf: &[u8]) -> Result<Vec<(u32, &[u8])>> {
    fn varint(b: &[u8], p: &mut usize) -> Result<u64> {
        let mut out = 0u64;
        for shift in (0..64).step_by(7) {
            let x = *b.get(*p).ok_or_else(|| anyhow!("truncated varint"))?;
            *p += 1;
            out |= u64::from(x & 0x7f) << shift;
            if x & 0x80 == 0 {
                return Ok(out);
            }
        }
        bail!("varint too long")
    }
    let (mut p, mut out) = (0, Vec::new());
    while p < buf.len() {
        let key = varint(buf, &mut p)?;
        let field = (key >> 3) as u32;
        match key & 7 {
            0 => {
                varint(buf, &mut p)?;
            }
            1 => p += 8,
            5 => p += 4,
            2 => {
                let n = varint(buf, &mut p)? as usize;
                let end = p
                    .checked_add(n)
                    .filter(|&e| e <= buf.len())
                    .ok_or_else(|| anyhow!("field {field} past the end"))?;
                out.push((field, &buf[p..end]));
                p = end;
            }
            w => bail!("wire type {w}"),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_round_trip() {
        let table = bytes_to_unicode();
        assert_eq!(table[b'a' as usize], 'a');
        // Space is not printable in GPT-2's table: it maps to 'Ġ'.
        assert_eq!(table[b' ' as usize], 'Ġ');
        let vocab = r#"{"<s>": 0, "</s>": 2, "Hello": 5, "Ġwor": 6, "ld": 7, "Ã©": 8}"#;
        let t = Tokenizer::from_vocab_json(vocab).unwrap();
        assert_eq!(t.decode(&[0, 5, 6, 7, 8, 2]), "Hello world\u{e9}");
    }

    #[test]
    fn sentencepiece_fairseq_offset() {
        let t = Tokenizer::SentencePiece {
            pieces: ["<unk>", "<s>", "</s>", "\u{2581}the", "\u{2581}cat", "s"]
                .map(String::from)
                .to_vec(),
        };
        // ids 4, 5, 6 are pieces 3, 4, 5; 0 and 2 are specials.
        assert_eq!(t.decode(&[0, 4, 5, 6, 2]), "the cats");
    }
}
