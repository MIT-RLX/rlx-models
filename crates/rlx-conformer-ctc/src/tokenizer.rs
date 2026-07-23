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

//! SentencePiece detokenization for CTC symbol ids. NeMo bundles a raw
//! SentencePiece `*.model` (protobuf) inside the `.nemo`; we parse its
//! ordered piece list directly (no extra dependency) so token ids map to
//! text. The U+2581 "▁" meta-symbol marks word boundaries.

use anyhow::{Result, bail};

/// SentencePiece word-boundary marker (`▁`, U+2581).
const SPACE_MARK: char = '\u{2581}';

/// A decoder backed by a SentencePiece piece table (`id -> piece`).
pub struct SpmTokenizer {
    pieces: Vec<String>,
}

impl SpmTokenizer {
    /// Parse a SentencePiece `ModelProto` from raw `.model` bytes.
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self> {
        let pieces = parse_spm_pieces(bytes)?;
        if pieces.is_empty() {
            bail!("SentencePiece model contained no pieces");
        }
        Ok(Self { pieces })
    }

    /// Number of pieces in the table (includes blank / specials as stored).
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Piece string for `id`, if in range.
    pub fn piece(&self, id: usize) -> Option<&str> {
        self.pieces.get(id).map(String::as_str)
    }

    /// Detokenize ids to text. Control/`<…>`-style pieces are dropped when
    /// `strip_specials` is set (used to remove `<en-US>` language tags).
    pub fn decode(&self, ids: &[u32], strip_specials: bool) -> String {
        let mut s = String::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            if strip_specials && is_special(p) {
                continue;
            }
            for ch in p.chars() {
                if ch == SPACE_MARK {
                    s.push(' ');
                } else {
                    s.push(ch);
                }
            }
        }
        s.trim().to_string()
    }
}

fn is_special(piece: &str) -> bool {
    piece.starts_with('<') && piece.ends_with('>')
}

// ── minimal protobuf reader for the SentencePiece ModelProto ──
//
// ModelProto { repeated SentencePiece pieces = 1; ... }
// SentencePiece { optional string piece = 1; optional float score = 2; ... }

fn read_varint(b: &[u8], pos: &mut usize) -> Option<u64> {
    let mut shift = 0u32;
    let mut out = 0u64;
    loop {
        let byte = *b.get(*pos)?;
        *pos += 1;
        out |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(out);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

/// Parse the top-level `pieces` (field 1) and return their `piece` strings
/// in id order.
fn parse_spm_pieces(b: &[u8]) -> Result<Vec<String>> {
    let mut pos = 0;
    let mut pieces = Vec::new();
    while pos < b.len() {
        let Some(tag) = read_varint(b, &mut pos) else {
            break;
        };
        let field = tag >> 3;
        let wire = tag & 7;
        match wire {
            2 => {
                let Some(len) = read_varint(b, &mut pos) else {
                    break;
                };
                let len = len as usize;
                let end = (pos + len).min(b.len());
                let body = &b[pos..end];
                pos = end;
                if field == 1 {
                    // a SentencePiece sub-message: extract its field-1 string.
                    if let Some(piece) = parse_piece(body) {
                        pieces.push(piece);
                    } else {
                        pieces.push(String::new());
                    }
                }
            }
            0 => {
                read_varint(b, &mut pos);
            }
            5 => pos += 4,
            1 => pos += 8,
            _ => bail!("unsupported protobuf wire type {wire}"),
        }
    }
    Ok(pieces)
}

fn parse_piece(b: &[u8]) -> Option<String> {
    let mut pos = 0;
    while pos < b.len() {
        let tag = read_varint(b, &mut pos)?;
        let field = tag >> 3;
        let wire = tag & 7;
        match wire {
            2 => {
                let len = read_varint(b, &mut pos)? as usize;
                let end = (pos + len).min(b.len());
                let body = &b[pos..end];
                pos = end;
                if field == 1 {
                    return Some(String::from_utf8_lossy(body).into_owned());
                }
            }
            0 => {
                read_varint(b, &mut pos)?;
            }
            5 => pos += 4,
            1 => pos += 8,
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-encode a tiny ModelProto with three pieces.
    fn encode_piece(piece: &str) -> Vec<u8> {
        let mut m = Vec::new();
        // field 1, wire 2: piece string
        m.push((1 << 3) | 2);
        m.push(piece.len() as u8);
        m.extend_from_slice(piece.as_bytes());
        // field 2, wire 5: score 0.0
        m.push((2 << 3) | 5);
        m.extend_from_slice(&0f32.to_le_bytes());
        m
    }
    fn encode_model(pieces: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for p in pieces {
            let body = encode_piece(p);
            out.push((1 << 3) | 2); // field 1, wire 2: SentencePiece
            out.push(body.len() as u8);
            out.extend_from_slice(&body);
        }
        out
    }

    #[test]
    fn parse_and_decode() {
        let model = encode_model(&["<unk>", "\u{2581}hello", "\u{2581}world", "<en-US>"]);
        let tok = SpmTokenizer::from_model_bytes(&model).unwrap();
        assert_eq!(tok.vocab_size(), 4);
        assert_eq!(tok.decode(&[1, 2], false), "hello world");
        // strip specials drops the language tag.
        assert_eq!(tok.decode(&[1, 2, 3], true), "hello world");
        assert_eq!(tok.decode(&[1, 2, 3], false), "hello world<en-US>");
    }
}
