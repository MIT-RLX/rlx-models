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

//! SentencePiece vocabulary reader for `MT/spm.model`.
//!
//! Only the piece table is decoded, which is all that is needed to map ids to
//! surface forms (and to size-check the embedding: the shipped model has
//! **168 000** pieces, exactly the `nRow` of the vocabulary `quantized_gather`).
//!
//! The file is a `sentencepiece.ModelProto`. Rather than take a protobuf
//! dependency for two fields, this walks the wire format directly:
//!
//! ```text
//!   ModelProto      field 1, length-delimited, repeated  -> SentencePiece
//!   SentencePiece   field 1, length-delimited            -> piece  (UTF-8)
//!                   field 2, 32-bit                      -> score  (f32)
//!                   field 3, varint                      -> type
//! ```
//!
//! Piece index is the token id, so `<unk>` is 0, `<s>` is 1, `</s>` is 2.

use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::path::Path;

/// SentencePiece's own type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Unused,
    Other(i64),
}

impl PieceKind {
    fn from_i64(v: i64) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            6 => Self::Byte,
            5 => Self::Unused,
            other => Self::Other(other),
        }
    }
}

/// One vocabulary entry.
#[derive(Debug, Clone)]
pub struct Piece {
    pub piece: String,
    pub score: f32,
    pub kind: PieceKind,
}

/// The piece table.
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    pieces: Vec<Piece>,
    index: HashMap<String, u32>,
    /// Longest piece in bytes, bounding the Viterbi inner loop.
    max_piece_len: usize,
}

impl Vocab {
    /// Reads a `spm.model`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parsing {}", path.display()))
    }

    /// Parses a `ModelProto`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut i = 0usize;
        while i < bytes.len() {
            let (key, ni) = varint(bytes, i)?;
            i = ni;
            let (field, wire) = ((key >> 3) as u32, (key & 7) as u8);
            match wire {
                2 => {
                    let (len, ni) = varint(bytes, i)?;
                    i = ni;
                    let end = i
                        .checked_add(len as usize)
                        .filter(|e| *e <= bytes.len())
                        .ok_or_else(|| anyhow::anyhow!("length-delimited field overruns"))?;
                    if field == 1 {
                        pieces.push(parse_piece(&bytes[i..end])?);
                    }
                    i = end;
                }
                0 => i = varint(bytes, i)?.1,
                5 => i += 4,
                1 => i += 8,
                other => bail!("unsupported protobuf wire type {other}"),
            }
        }
        ensure!(!pieces.is_empty(), "model contains no pieces");
        let index = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.piece.clone(), i as u32))
            .collect();
        let max_piece_len = pieces.iter().map(|p| p.piece.len()).max().unwrap_or(1);
        Ok(Self {
            pieces,
            index,
            max_piece_len,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Surface form of a token id.
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.piece.as_str())
    }

    pub fn entry(&self, id: u32) -> Option<&Piece> {
        self.pieces.get(id as usize)
    }

    /// Token id of a surface form.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.index.get(piece).copied()
    }

    /// Id of a whole word, i.e. one preceded by SentencePiece's `▁` marker.
    pub fn word_id(&self, word: &str) -> Option<u32> {
        self.id(&format!("\u{2581}{word}"))
    }

    /// Segments text the way SentencePiece's unigram model does.
    ///
    /// The `score` on each piece is its log-probability, so the tokenization is
    /// the path through the string maximising the summed score \u{2014} a Viterbi
    /// walk over byte positions. Spaces become the word marker and a leading one
    /// is added (SentencePiece's `add_dummy_prefix`).
    ///
    /// Anything unmatched falls back to the `<0xNN>` byte pieces rather than a
    /// single `<unk>`, which is what this model ships them for.
    ///
    /// [`Vocab::word_id`] is only a lookup, not a segmentation; use this to
    /// tokenize text.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm: String = std::iter::once('\u{2581}')
            .chain(text.chars().map(|c| if c == ' ' { '\u{2581}' } else { c }))
            .collect();
        let b = norm.as_bytes();
        let n = b.len();

        // best[i] = (score of the best segmentation of the first i bytes, start
        // of its last piece, id of that piece). Unreachable positions stay None.
        let mut best: Vec<Option<(f32, usize, u32)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, 0));
        for i in 0..n {
            let Some((base, _, _)) = best[i] else {
                continue;
            };
            if !norm.is_char_boundary(i) {
                continue;
            }
            for j in (i + 1)..=n.min(i + self.max_piece_len) {
                if !norm.is_char_boundary(j) {
                    continue;
                }
                let Some(cand) = norm.get(i..j) else { continue };
                let Some(id) = self.index.get(cand).copied() else {
                    continue;
                };
                // Control pieces are never produced by segmentation.
                if !matches!(
                    self.pieces[id as usize].kind,
                    PieceKind::Normal | PieceKind::UserDefined | PieceKind::Byte
                ) {
                    continue;
                }
                let s = base + self.pieces[id as usize].score;
                if best[j].is_none_or(|(prev, _, _)| s > prev) {
                    best[j] = Some((s, i, id));
                }
            }
            // Byte fallback: a character no piece covers costs one token per
            // byte instead of losing the whole word. Penalised so it is only
            // taken when nothing else reaches this position.
            let cw = norm[i..].chars().next().map_or(1, char::len_utf8);
            let j = i + cw;
            if j <= n && best[j].is_none() {
                let mut s = base - 1e3;
                let mut ok = true;
                for k in i..j {
                    match self.id(&format!("<0x{:02X}>", b[k])) {
                        Some(id) => s += self.pieces[id as usize].score,
                        None => ok = false,
                    }
                }
                if ok && let Some(id) = self.id(&format!("<0x{:02X}>", b[j - 1])) {
                    best[j] = Some((s, i, id));
                }
            }
        }

        let mut out = Vec::new();
        let mut i = n;
        while i > 0 {
            let Some((_, start, id)) = best[i] else { break };
            if start + 1 < i && self.pieces[id as usize].piece.starts_with("<0x") {
                for k in (start..i).rev() {
                    if let Some(bid) = self.id(&format!("<0x{:02X}>", b[k])) {
                        out.push(bid);
                    }
                }
            } else {
                out.push(id);
            }
            i = start;
        }
        out.reverse();
        out
    }

    /// Renders token ids back to text, reassembling byte fallback.
    ///
    /// A piece the model has no entry for is spelled as `<0xNN>` bytes, and
    /// joining those as *text* leaves the markers in the string. That is not
    /// cosmetic: the stage graph re-renders `spm_encode`'s tokens and hands
    /// the result to the translator, so a Thai source containing SARA AM —
    /// which falls back — reached the NMT as `\u{0e19}<0xE0><0xB8><0xB3>` and
    /// came back as `ADecasting,ORTHESSORTING,ADSORTHSORTINGDEIED`. It cost
    /// `th_TH-en_US` 0.15 chrF against the OS.
    ///
    /// Bytes are collected and decoded together, because one fallback
    /// character is several `<0xNN>` pieces and no single one is valid UTF-8.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for id in ids {
            let Some(p) = self.piece(*id) else { continue };
            if let Some(hex) = p.strip_prefix("<0x").and_then(|r| r.strip_suffix('>'))
                && hex.len() == 2
                && let Ok(b) = u8::from_str_radix(hex, 16)
            {
                bytes.push(b);
                continue;
            }
            let mut buf = [0u8; 4];
            for ch in p.chars() {
                if ch == '\u{2581}' {
                    bytes.push(b' ');
                } else {
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }

    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }
}

fn parse_piece(b: &[u8]) -> Result<Piece> {
    let (mut piece, mut score, mut kind) = (String::new(), 0.0f32, PieceKind::Normal);
    let mut i = 0usize;
    while i < b.len() {
        let (key, ni) = varint(b, i)?;
        i = ni;
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u8);
        match wire {
            2 => {
                let (len, ni) = varint(b, i)?;
                i = ni;
                let end = i
                    .checked_add(len as usize)
                    .filter(|e| *e <= b.len())
                    .ok_or_else(|| anyhow::anyhow!("piece field overruns"))?;
                if field == 1 {
                    piece = String::from_utf8_lossy(&b[i..end]).into_owned();
                }
                i = end;
            }
            5 => {
                ensure!(i + 4 <= b.len(), "piece score overruns");
                if field == 2 {
                    score = f32::from_le_bytes(b[i..i + 4].try_into().expect("four bytes"));
                }
                i += 4;
            }
            0 => {
                let (v, ni) = varint(b, i)?;
                i = ni;
                if field == 3 {
                    kind = PieceKind::from_i64(v as i64);
                }
            }
            1 => i += 8,
            other => bail!("unsupported wire type {other} inside a piece"),
        }
    }
    Ok(Piece { piece, score, kind })
}

fn varint(b: &[u8], mut i: usize) -> Result<(u64, usize)> {
    let (mut out, mut shift) = (0u64, 0u32);
    loop {
        let byte = *b
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("varint runs past the end"))?;
        i += 1;
        out |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((out, i));
        }
        shift += 7;
        ensure!(shift < 64, "varint is longer than 64 bits");
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn decode_reassembles_byte_fallback_into_characters() {
        // A vocabulary with no entry for SARA AM, so it can only be spelled as
        // bytes — which is exactly the Thai case that broke the stage graph.
        let mut pieces: Vec<(String, f32, i64)> = vec![
            ("<unk>".to_string(), 0.0, 2),
            ("\u{2581}\u{0e19}".to_string(), -1.0, 1),
            ("\u{0e0a}\u{0e32}\u{0e22}".to_string(), -1.0, 1),
        ];
        for b in [0xE0u8, 0xB8, 0xB3] {
            pieces.push((format!("<0x{b:02X}>"), -20.0, 6));
        }
        let refs: Vec<(&str, f32, i64)> = pieces
            .iter()
            .map(|(p, s, t)| (p.as_str(), *s, *t))
            .collect();
        let v = Vocab::parse(&model(&refs)).expect("parses");

        let text = "\u{0e19}\u{0e33}\u{0e0a}\u{0e32}\u{0e22}";
        let ids = v.encode(text);
        // The fallback really is in play, or the test proves nothing.
        assert!(
            ids.iter()
                .any(|i| v.piece(*i).is_some_and(|p| p.starts_with("<0x"))),
            "expected byte fallback, got {:?}",
            ids.iter().filter_map(|i| v.piece(*i)).collect::<Vec<_>>()
        );
        let back = v.decode(&ids);
        assert_eq!(back, text, "byte fallback did not round-trip");
        assert!(!back.contains("<0x"), "byte markers survived in {back:?}");
    }

    use super::*;

    /// Builds a minimal ModelProto with the given pieces.
    fn model(pieces: &[(&str, f32, i64)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (p, s, t) in pieces {
            let mut sub = Vec::new();
            sub.push(0x0a); // field 1, length-delimited
            sub.push(p.len() as u8);
            sub.extend_from_slice(p.as_bytes());
            sub.push(0x15); // field 2, 32-bit
            sub.extend_from_slice(&s.to_le_bytes());
            sub.push(0x18); // field 3, varint
            sub.push(*t as u8);
            out.push(0x0a); // ModelProto field 1
            out.push(sub.len() as u8);
            out.extend_from_slice(&sub);
        }
        out
    }

    #[test]
    fn parses_pieces_in_id_order() {
        let v = Vocab::parse(&model(&[
            ("<unk>", 0.0, 2),
            ("<s>", 0.0, 3),
            ("</s>", 0.0, 3),
            ("\u{2581}dog", -7.5, 1),
        ]))
        .expect("parses");
        assert_eq!(v.len(), 4);
        assert_eq!(v.piece(0), Some("<unk>"));
        assert_eq!(v.piece(2), Some("</s>"));
        assert_eq!(v.id("</s>"), Some(2));
        assert_eq!(v.word_id("dog"), Some(3));
        assert_eq!(v.piece(99), None);
    }

    #[test]
    fn scores_and_kinds_round_trip() {
        let v =
            Vocab::parse(&model(&[("<unk>", 0.0, 2), ("\u{2581}a", -1.25, 1)])).expect("parses");
        assert_eq!(v.entry(0).expect("entry").kind, PieceKind::Unknown);
        let e = v.entry(1).expect("entry");
        assert_eq!(e.kind, PieceKind::Normal);
        assert!((e.score + 1.25).abs() < 1e-6, "score {}", e.score);
    }

    #[test]
    fn an_empty_model_is_rejected() {
        assert!(Vocab::parse(&[]).is_err());
    }

    #[test]
    fn a_truncated_field_is_rejected_not_silently_dropped() {
        let mut b = model(&[("\u{2581}dog", 0.0, 1)]);
        b.truncate(b.len() - 3);
        assert!(Vocab::parse(&b).is_err());
    }

    #[test]
    fn varints_decode_multibyte_values() {
        let (v, i) = varint(&[0xAC, 0x02], 0).expect("varint");
        assert_eq!((v, i), (300, 2));
        assert!(varint(&[0x80], 0).is_err(), "unterminated varint");
    }
}
