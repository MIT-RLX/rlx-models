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

//! SentencePiece in Rust: reads a `.model` (`ModelProto`) and encodes /
//! decodes the way the C++ `SentencePieceProcessor` does, with no C++ build.
//!
//! What is reproduced:
//!
//! - **Normalizer** (`normalizer.cc`): the precompiled charsmap (a darts-clone
//!   double array + replacement strings, longest match), user-defined symbols
//!   kept whole, `add_dummy_prefix`, `remove_extra_whitespaces`,
//!   `escape_whitespaces` (`' '` → `▁`), `treat_whitespace_as_suffix`.
//! - **Models**: unigram (the Viterbi of `EncodeOptimized`, f32 path scores,
//!   ties to the earliest start, unknown characters at `min_score - 10`,
//!   user-defined pieces at `len · max_score - 0.1`), BPE (score-ordered
//!   merges, ties to the leftmost pair, `UNUSED` pieces re-split), word, char.
//! - **Processor**: byte fallback (`<0xXX>` per byte of an unknown piece),
//!   runs of unknown pieces merged, decode with control pieces dropped, the
//!   dummy-prefix space stripped, byte pieces reassembled as UTF-8 (`U+FFFD`
//!   per invalid byte), `unk_surface` for the unknown piece.
//!
//! Sampling (`enable_sampling`, n-best) and `extra_options` (`bos:eos`,
//! `reverse`) are not implemented; add BOS/EOS ids at the call site.
//!
//! A Hugging Face `tokenizer.json` holding a `Unigram` model converted from
//! SentencePiece also loads ([`SentencePiece::from_tokenizer_json`]), with the
//! normalizer settings SentencePiece's defaults give that conversion.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// The whitespace marker, `U+2581`.
pub const SPACE_SYMBOL: &str = "\u{2581}";
/// Score penalty of an unknown character below the lowest piece score.
const UNK_PENALTY: f32 = 10.0;
const REPLACEMENT_CHAR: &str = "\u{FFFD}";

/// `ModelProto.SentencePiece.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceType {
    fn from_proto(v: u64) -> Result<Self> {
        Ok(match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            v => bail!("piece type {v} in a SentencePiece model"),
        })
    }
}

/// `TrainerSpec.ModelType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Unigram,
    Bpe,
    Word,
    Char,
}

/// One vocabulary entry.
#[derive(Debug, Clone)]
pub struct Piece {
    pub piece: String,
    pub score: f32,
    pub kind: PieceType,
}

/// `NormalizerSpec` (the fields encoding reads).
#[derive(Debug, Clone)]
pub struct NormalizerSpec {
    pub name: String,
    pub precompiled_charsmap: Vec<u8>,
    pub add_dummy_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
}

impl Default for NormalizerSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            precompiled_charsmap: Vec::new(),
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

/// A loaded SentencePiece model.
pub struct SentencePiece {
    pieces: Vec<Piece>,
    model_type: ModelType,
    normalizer: NormalizerSpec,
    treat_whitespace_as_suffix: bool,
    byte_fallback: bool,
    unk_surface: String,
    unk_id: u32,
    /// `NORMAL`, `USER_DEFINED` and `UNUSED` pieces (`pieces_` in C++).
    vocab: HashMap<String, u32>,
    /// `CONTROL`, `UNKNOWN` and `BYTE` pieces (`reserved_id_map_`).
    reserved: HashMap<String, u32>,
    /// Longest `vocab` key in bytes, bounding the prefix search.
    max_piece_bytes: usize,
    /// User-defined symbols, longest first per prefix (`PrefixMatcher`).
    user_defined: HashMap<String, ()>,
    max_user_defined_bytes: usize,
    charsmap: Option<Charsmap>,
    min_score: f32,
    max_score: f32,
}

impl SentencePiece {
    /// Load a `.model` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        if path.extension().is_some_and(|e| e == "json") {
            let text = String::from_utf8(bytes).context("tokenizer.json is not UTF-8")?;
            return Self::from_tokenizer_json(&text)
                .with_context(|| format!("load {}", path.display()));
        }
        Self::from_bytes(&bytes).with_context(|| format!("load {}", path.display()))
    }

    /// Parse a serialized `ModelProto`.
    pub fn from_bytes(model: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut model_type = ModelType::Unigram;
        let mut treat_whitespace_as_suffix = false;
        let mut byte_fallback = false;
        let mut unk_surface = " \u{2047} ".to_string();
        let mut normalizer = NormalizerSpec::default();
        for f in proto::fields(model)? {
            match (f.field, f.wire) {
                (1, 2) => {
                    let (mut piece, mut score, mut kind) = (String::new(), 0f32, 1u64);
                    for g in proto::fields(f.bytes)? {
                        match (g.field, g.wire) {
                            (1, 2) => piece = utf8(g.bytes, "piece")?,
                            (2, 5) => score = f32::from_le_bytes(g.bytes.try_into()?),
                            (3, 0) => kind = g.varint,
                            _ => {}
                        }
                    }
                    pieces.push(Piece {
                        piece,
                        score,
                        kind: PieceType::from_proto(kind)?,
                    });
                }
                (2, 2) => {
                    for g in proto::fields(f.bytes)? {
                        match (g.field, g.wire) {
                            (3, 0) => {
                                model_type = match g.varint {
                                    1 => ModelType::Unigram,
                                    2 => ModelType::Bpe,
                                    3 => ModelType::Word,
                                    4 => ModelType::Char,
                                    v => bail!("model type {v} in a SentencePiece model"),
                                }
                            }
                            (24, 0) => treat_whitespace_as_suffix = g.varint != 0,
                            (35, 0) => byte_fallback = g.varint != 0,
                            (44, 2) => unk_surface = utf8(g.bytes, "unk_surface")?,
                            _ => {}
                        }
                    }
                }
                (3, 2) => {
                    for g in proto::fields(f.bytes)? {
                        match (g.field, g.wire) {
                            (1, 2) => normalizer.name = utf8(g.bytes, "normalizer name")?,
                            (2, 2) => normalizer.precompiled_charsmap = g.bytes.to_vec(),
                            (3, 0) => normalizer.add_dummy_prefix = g.varint != 0,
                            (4, 0) => normalizer.remove_extra_whitespaces = g.varint != 0,
                            (5, 0) => normalizer.escape_whitespaces = g.varint != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        Self::new(
            pieces,
            model_type,
            normalizer,
            treat_whitespace_as_suffix,
            byte_fallback,
            unk_surface,
        )
    }

    /// A Hugging Face `tokenizer.json` with a `Unigram` model (the
    /// SentencePiece conversion): pieces and scores from `model.vocab`,
    /// special added tokens as control pieces, `<0xXX>` as byte pieces when
    /// `byte_fallback` is set, and SentencePiece's normalizer for that
    /// conversion — dummy prefix, whitespace escaped, spaces kept as they are.
    pub fn from_tokenizer_json(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json).context("parse tokenizer.json")?;
        let model = &v["model"];
        if model["type"].as_str() != Some("Unigram") {
            bail!(
                "tokenizer.json model type {:?}: only Unigram (SentencePiece) is supported",
                model["type"]
            );
        }
        let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);
        let unk_id = model["unk_id"].as_u64().map(|u| u as usize);
        let special: std::collections::HashSet<&str> = v["added_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|t| t["special"].as_bool().unwrap_or(false))
            .filter_map(|t| t["content"].as_str())
            .collect();
        let vocab = model["vocab"]
            .as_array()
            .context("tokenizer.json model.vocab is not a list")?;
        let mut pieces = Vec::with_capacity(vocab.len());
        for (i, e) in vocab.iter().enumerate() {
            let piece = e[0].as_str().context("vocab entry without a piece")?;
            let score = e[1].as_f64().context("vocab entry without a score")? as f32;
            let kind = if Some(i) == unk_id {
                PieceType::Unknown
            } else if special.contains(piece) {
                PieceType::Control
            } else if byte_fallback && byte_piece_value(piece).is_some() {
                PieceType::Byte
            } else {
                PieceType::Normal
            };
            pieces.push(Piece {
                piece: piece.to_string(),
                score,
                kind,
            });
        }
        let normalizer = NormalizerSpec {
            name: "identity".into(),
            remove_extra_whitespaces: false,
            ..NormalizerSpec::default()
        };
        Self::new(
            pieces,
            ModelType::Unigram,
            normalizer,
            false,
            byte_fallback,
            " \u{2047} ".into(),
        )
    }

    fn new(
        pieces: Vec<Piece>,
        model_type: ModelType,
        normalizer: NormalizerSpec,
        treat_whitespace_as_suffix: bool,
        byte_fallback: bool,
        unk_surface: String,
    ) -> Result<Self> {
        if pieces.is_empty() {
            bail!("no pieces: not a SentencePiece model");
        }
        let mut vocab = HashMap::new();
        let mut reserved = HashMap::new();
        let mut user_defined = HashMap::new();
        let mut unk_id = None;
        let (mut min_score, mut max_score) = (f32::MAX, f32::MIN_POSITIVE);
        for (i, p) in pieces.iter().enumerate() {
            let id = i as u32;
            if p.piece.is_empty() {
                bail!("piece {i} is empty");
            }
            let map = match p.kind {
                PieceType::Normal | PieceType::UserDefined | PieceType::Unused => &mut vocab,
                _ => &mut reserved,
            };
            if map.insert(p.piece.clone(), id).is_some() {
                bail!("piece {:?} appears twice", p.piece);
            }
            match p.kind {
                PieceType::Normal => {
                    min_score = min_score.min(p.score);
                    max_score = max_score.max(p.score);
                }
                PieceType::UserDefined => {
                    user_defined.insert(p.piece.clone(), ());
                }
                PieceType::Unknown => {
                    if unk_id.replace(id).is_some() {
                        bail!("several unknown pieces");
                    }
                }
                PieceType::Byte if !byte_fallback => {
                    bail!("byte piece {:?} without byte_fallback", p.piece)
                }
                _ => {}
            }
        }
        let unk_id = unk_id.context("no unknown piece")?;
        if byte_fallback {
            for b in 0..=255u8 {
                let name = byte_piece(b);
                if !reserved.contains_key(&name) {
                    bail!("byte_fallback is set but piece {name} is missing");
                }
            }
        }
        let charsmap = if normalizer.precompiled_charsmap.is_empty() {
            None
        } else {
            Some(Charsmap::parse(&normalizer.precompiled_charsmap)?)
        };
        let max_piece_bytes = vocab.keys().map(String::len).max().unwrap_or(0);
        let max_user_defined_bytes = user_defined.keys().map(String::len).max().unwrap_or(0);
        Ok(Self {
            pieces,
            model_type,
            normalizer,
            treat_whitespace_as_suffix,
            byte_fallback,
            unk_surface,
            unk_id,
            vocab,
            reserved,
            max_piece_bytes,
            user_defined,
            max_user_defined_bytes,
            charsmap,
            min_score,
            max_score,
        })
    }

    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn normalizer_spec(&self) -> &NormalizerSpec {
        &self.normalizer
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// The id of `piece`, or the unknown id.
    pub fn piece_to_id(&self, piece: &str) -> u32 {
        self.reserved
            .get(piece)
            .or_else(|| self.vocab.get(piece))
            .copied()
            .unwrap_or(self.unk_id)
    }

    pub fn id_to_piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.piece.as_str())
    }

    fn kind(&self, id: u32) -> Option<PieceType> {
        self.pieces.get(id as usize).map(|p| p.kind)
    }

    /// Token ids of `text` (no BOS / EOS).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_pieces(text)
            .into_iter()
            .map(|(_, id)| id)
            .collect()
    }

    /// `(piece surface, id)` of each token of `text`. Byte-fallback tokens
    /// carry their `<0xXX>` name.
    pub fn encode_pieces(&self, text: &str) -> Vec<(String, u32)> {
        let normalized = self.normalize(text);
        let mut out: Vec<(String, u32)> = Vec::new();
        for (w, id) in self.encode_normalized(&normalized) {
            let is_unk = id == self.unk_id;
            if is_unk && self.byte_fallback {
                for &b in w.as_bytes() {
                    let name = byte_piece(b);
                    let id = self.piece_to_id(&name);
                    out.push((name, id));
                }
                continue;
            }
            match out.last_mut() {
                Some((prev, prev_id)) if is_unk && *prev_id == self.unk_id => prev.push_str(w),
                _ => out.push((w.to_string(), id)),
            }
        }
        out
    }

    /// The normalized form of `text` the model segments.
    pub fn normalize(&self, text: &str) -> String {
        let spec = &self.normalizer;
        let mut input = text;
        let mut out = String::with_capacity(text.len() * 3 / 2);
        if spec.remove_extra_whitespaces {
            while !input.is_empty() {
                let (s, n) = self.normalize_prefix(input);
                if s != " " {
                    break;
                }
                input = &input[n..];
            }
        }
        if input.is_empty() {
            return out;
        }
        let ws = if spec.escape_whitespaces {
            SPACE_SYMBOL
        } else {
            " "
        };
        let mut is_prev_space = spec.remove_extra_whitespaces;
        if !self.treat_whitespace_as_suffix && spec.add_dummy_prefix {
            out.push_str(ws);
        }
        while !input.is_empty() {
            let (s, n) = self.normalize_prefix(input);
            let mut s: &str = &s;
            if is_prev_space {
                s = s.trim_start_matches(' ');
            }
            if !s.is_empty() {
                if spec.escape_whitespaces {
                    out.push_str(&s.replace(' ', SPACE_SYMBOL));
                } else {
                    out.push_str(s);
                }
                is_prev_space = s.ends_with(' ');
            }
            input = &input[n..];
            if !spec.remove_extra_whitespaces {
                is_prev_space = false;
            }
        }
        if spec.remove_extra_whitespaces {
            while out.ends_with(ws) {
                out.truncate(out.len() - ws.len());
            }
        }
        if self.treat_whitespace_as_suffix && spec.add_dummy_prefix {
            out.push_str(ws);
        }
        out
    }

    /// `Normalizer::NormalizePrefix`: the replacement of the longest
    /// normalizable prefix of `input` and how many bytes it consumed.
    fn normalize_prefix<'a>(&'a self, input: &'a str) -> (std::borrow::Cow<'a, str>, usize) {
        if let Some(n) = self.match_user_defined(input) {
            return (input[..n].into(), n);
        }
        if let Some(map) = &self.charsmap
            && let Some((n, repl)) = map.longest_prefix(input.as_bytes())
        {
            // A replacement that is not UTF-8 is a corrupt charsmap; keep the
            // input rather than emit invalid text.
            if let Ok(r) = std::str::from_utf8(repl)
                && input.is_char_boundary(n)
            {
                return (r.into(), n);
            }
        }
        let n = char_len(input);
        (input[..n].into(), n)
    }

    /// Length of the longest user-defined symbol `s` starts with.
    fn match_user_defined(&self, s: &str) -> Option<usize> {
        if self.user_defined.is_empty() {
            return None;
        }
        let limit = self.max_user_defined_bytes.min(s.len());
        (1..=limit)
            .rev()
            .filter(|&n| s.is_char_boundary(n))
            .find(|&n| self.user_defined.contains_key(&s[..n]))
    }

    /// The model's segmentation of already-normalized text.
    fn encode_normalized<'a>(&self, s: &'a str) -> Vec<(&'a str, u32)> {
        if s.is_empty() {
            return Vec::new();
        }
        match self.model_type {
            ModelType::Unigram => self.encode_unigram(s),
            ModelType::Bpe => self.encode_bpe(s),
            ModelType::Word => self.encode_word(s),
            ModelType::Char => self.encode_char(s),
        }
    }

    /// `unigram::Model::EncodeOptimized`.
    fn encode_unigram<'a>(&self, s: &'a str) -> Vec<(&'a str, u32)> {
        #[derive(Clone, Copy)]
        struct Node {
            id: u32,
            score: f32,
            start: Option<usize>,
        }
        let n = s.len();
        let unk_score = self.min_score - UNK_PENALTY;
        let mut best = vec![
            Node {
                id: 0,
                score: 0.0,
                start: None,
            };
            n + 1
        ];
        let mut start = 0;
        while start < n {
            let here = best[start].score;
            let mblen = char_len(&s[start..]);
            let mut has_single = false;
            let limit = self.max_piece_bytes.min(n - start);
            for len in 1..=limit {
                let end = start + len;
                if !s.is_char_boundary(end) {
                    continue;
                }
                let Some(&id) = self.vocab.get(&s[start..end]) else {
                    continue;
                };
                let p = &self.pieces[id as usize];
                let score = match p.kind {
                    PieceType::Unused => continue,
                    PieceType::UserDefined => (f64::from(len as f32 * self.max_score) - 0.1) as f32,
                    _ => p.score,
                };
                let cand = score + here;
                let t = &mut best[end];
                if t.start.is_none() || cand > t.score {
                    *t = Node {
                        id,
                        score: cand,
                        start: Some(start),
                    };
                }
                if len == mblen {
                    has_single = true;
                }
            }
            if !has_single {
                let cand = unk_score + here;
                let t = &mut best[start + mblen];
                if t.start.is_none() || cand > t.score {
                    *t = Node {
                        id: self.unk_id,
                        score: cand,
                        start: Some(start),
                    };
                }
            }
            start += mblen;
        }
        let mut out = Vec::new();
        let mut end = n;
        while end > 0 {
            let node = best[end];
            let start = node.start.expect("every character boundary is reachable");
            out.push((&s[start..end], node.id));
            end = start;
        }
        out.reverse();
        out
    }

    /// `bpe::Model::Encode`.
    fn encode_bpe<'a>(&self, s: &'a str) -> Vec<(&'a str, u32)> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        struct Symbol {
            start: usize,
            end: usize,
            prev: Option<usize>,
            next: Option<usize>,
            freeze: bool,
        }
        struct Pair {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl PartialEq for Pair {
            fn eq(&self, o: &Self) -> bool {
                self.cmp(o) == Ordering::Equal
            }
        }
        impl Eq for Pair {}
        impl PartialOrd for Pair {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Pair {
            // Highest score first, then the leftmost pair.
            fn cmp(&self, o: &Self) -> Ordering {
                self.score
                    .total_cmp(&o.score)
                    .then_with(|| o.left.cmp(&self.left))
            }
        }

        let mut symbols: Vec<Symbol> = Vec::new();
        let mut at = 0;
        while at < s.len() {
            let (len, freeze) = match self.match_user_defined(&s[at..]) {
                Some(n) => (n, true),
                None => (char_len(&s[at..]), false),
            };
            let i = symbols.len();
            symbols.push(Symbol {
                start: at,
                end: at + len,
                prev: i.checked_sub(1),
                next: (at + len < s.len()).then_some(i + 1),
                freeze,
            });
            at += len;
        }

        let mut agenda = BinaryHeap::new();
        let mut rev_merge: HashMap<&'a str, (&'a str, &'a str)> = HashMap::new();
        let mut add_pair = |symbols: &[Symbol],
                            agenda: &mut BinaryHeap<Pair>,
                            left: Option<usize>,
                            right: Option<usize>| {
            let (Some(l), Some(r)) = (left, right) else {
                return;
            };
            if symbols[l].freeze || symbols[r].freeze {
                return;
            }
            let piece = &s[symbols[l].start..symbols[r].end];
            let Some(&id) = self.vocab.get(piece) else {
                return;
            };
            let p = &self.pieces[id as usize];
            agenda.push(Pair {
                score: p.score,
                left: l,
                right: r,
                size: piece.len(),
            });
            if p.kind == PieceType::Unused {
                rev_merge.insert(
                    piece,
                    (
                        &s[symbols[l].start..symbols[l].end],
                        &s[symbols[r].start..symbols[r].end],
                    ),
                );
            }
        };
        for i in 1..symbols.len() {
            add_pair(&symbols, &mut agenda, Some(i - 1), Some(i));
        }
        while let Some(top) = agenda.pop() {
            let (l, r) = (top.left, top.right);
            let (ls, rs) = (&symbols[l], &symbols[r]);
            if ls.start == ls.end
                || rs.start == rs.end
                || (ls.end - ls.start) + (rs.end - rs.start) != top.size
            {
                continue;
            }
            let (r_end, r_next) = (rs.end, rs.next);
            symbols[l].end = r_end;
            symbols[l].next = r_next;
            if let Some(nn) = r_next {
                symbols[nn].prev = Some(l);
            }
            symbols[r].start = symbols[r].end;
            let prev = symbols[l].prev;
            add_pair(&symbols, &mut agenda, prev, Some(l));
            let next = symbols[l].next;
            add_pair(&symbols, &mut agenda, Some(l), next);
        }

        let mut out = Vec::new();
        let mut cur = (!symbols.is_empty()).then_some(0);
        while let Some(i) = cur {
            self.bpe_resegment(&s[symbols[i].start..symbols[i].end], &rev_merge, &mut out);
            cur = symbols[i].next;
        }
        out
    }

    fn bpe_resegment<'a>(
        &self,
        w: &'a str,
        rev_merge: &HashMap<&'a str, (&'a str, &'a str)>,
        out: &mut Vec<(&'a str, u32)>,
    ) {
        let id = self.piece_to_id(w);
        if self.kind(id) != Some(PieceType::Unused) {
            out.push((w, id));
            return;
        }
        match rev_merge.get(w) {
            Some(&(a, b)) => {
                self.bpe_resegment(a, rev_merge, out);
                self.bpe_resegment(b, rev_merge, out);
            }
            None => out.push((w, id)),
        }
    }

    /// `word::Model::Encode`: one piece per whitespace-delimited word.
    fn encode_word<'a>(&self, s: &'a str) -> Vec<(&'a str, u32)> {
        let mut words = Vec::new();
        let mut begin = 0;
        for (i, _) in s.match_indices(SPACE_SYMBOL) {
            if self.treat_whitespace_as_suffix {
                let end = i + SPACE_SYMBOL.len();
                words.push(&s[begin..end]);
                begin = end;
            } else if i > begin {
                words.push(&s[begin..i]);
                begin = i;
            }
        }
        if begin < s.len() {
            words.push(&s[begin..]);
        }
        words
            .into_iter()
            .map(|w| (w, self.piece_to_id(w)))
            .collect()
    }

    /// `character::Model::Encode`.
    fn encode_char<'a>(&self, s: &'a str) -> Vec<(&'a str, u32)> {
        s.char_indices()
            .map(|(i, c)| {
                let w = &s[i..i + c.len_utf8()];
                (w, self.piece_to_id(w))
            })
            .collect()
    }

    /// Text of `ids`. Ids past the vocabulary are an error.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut text = String::new();
        // The first non-control piece loses its dummy-prefix marker; with
        // `remove_extra_whitespaces`, every piece does until one decodes to
        // something (control pieces never count).
        let mut at_bos = true;
        let mut i = 0;
        while i < ids.len() {
            let id = ids[i];
            let Some(p) = self.pieces.get(id as usize) else {
                bail!("token id {id} past the vocabulary ({})", self.pieces.len());
            };
            if p.kind != PieceType::Byte {
                let surface = self.decode_piece(p, at_bos);
                text.push_str(&surface);
                at_bos &= p.kind == PieceType::Control
                    || (self.normalizer.remove_extra_whitespaces && surface.is_empty());
                i += 1;
                continue;
            }
            let mut bytes = Vec::new();
            while i < ids.len() {
                match self.pieces.get(ids[i] as usize) {
                    Some(q) if q.kind == PieceType::Byte => {
                        bytes.push(byte_piece_value(&q.piece).expect("checked at load"));
                        i += 1;
                    }
                    _ => break,
                }
            }
            push_utf8_lossy(&mut text, &bytes);
            at_bos = false;
        }
        Ok(text)
    }

    fn decode_piece(&self, p: &Piece, is_bos: bool) -> String {
        match p.kind {
            PieceType::Control => return String::new(),
            PieceType::Unknown => return self.unk_surface.clone(),
            _ => {}
        }
        let spec = &self.normalizer;
        let strip = spec.add_dummy_prefix || spec.remove_extra_whitespaces;
        let mut piece = p.piece.as_str();
        // Only leading markers go: with `treat_whitespace_as_suffix` the
        // trailing one stays a space, as in the C++.
        if strip && is_bos {
            piece = if spec.remove_extra_whitespaces {
                piece.trim_start_matches(SPACE_SYMBOL)
            } else {
                piece.strip_prefix(SPACE_SYMBOL).unwrap_or(piece)
            };
        }
        piece.replace(SPACE_SYMBOL, " ")
    }
}

/// `<0xXX>`, the byte-fallback piece of `b`.
pub fn byte_piece(b: u8) -> String {
    format!("<0x{b:02X}>")
}

fn byte_piece_value(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok())?
}

fn char_len(s: &str) -> usize {
    s.chars().next().map_or(0, char::len_utf8)
}

fn utf8(b: &[u8], what: &str) -> Result<String> {
    String::from_utf8(b.to_vec()).with_context(|| format!("{what} is not UTF-8"))
}

/// Decode `bytes` as UTF-8, one `U+FFFD` per byte that starts no valid
/// character (SentencePiece's byte-piece surface rule).
fn push_utf8_lossy(out: &mut String, mut bytes: &[u8]) {
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(s) => {
                out.push_str(s);
                return;
            }
            Err(e) => {
                let good = e.valid_up_to();
                out.push_str(std::str::from_utf8(&bytes[..good]).expect("valid prefix"));
                out.push_str(REPLACEMENT_CHAR);
                bytes = &bytes[good + 1..];
            }
        }
    }
}

/// `NormalizerSpec.precompiled_charsmap`: a little-endian `u32` trie size,
/// the darts-clone double array, then NUL-terminated replacement strings the
/// trie values index.
struct Charsmap {
    units: Vec<u32>,
    normalized: Vec<u8>,
}

impl Charsmap {
    fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() < 4 {
            bail!("precompiled charsmap shorter than its header");
        }
        let size = u32::from_le_bytes(blob[..4].try_into()?) as usize;
        let rest = &blob[4..];
        if size > rest.len() || !size.is_multiple_of(4) {
            bail!("precompiled charsmap trie size {size} does not fit the blob");
        }
        let units = rest[..size]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect::<Vec<_>>();
        if units.is_empty() {
            bail!("precompiled charsmap with an empty trie");
        }
        Ok(Self {
            units,
            normalized: rest[size..].to_vec(),
        })
    }

    /// Longest key that prefixes `key`: its length and replacement bytes
    /// (darts-clone `commonPrefixSearch`, longest result).
    fn longest_prefix(&self, key: &[u8]) -> Option<(usize, &[u8])> {
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let label = |u: u32| u & ((1 << 31) | 0xFF);
        let has_leaf = |u: u32| (u >> 8) & 1 == 1;
        let value = |u: u32| (u & ((1 << 31) - 1)) as usize;

        let mut found = None;
        let mut pos = offset(self.units[0]);
        for (i, &c) in key.iter().enumerate() {
            pos ^= c as usize;
            let unit = *self.units.get(pos)?;
            if label(unit) != u32::from(c) {
                break;
            }
            pos ^= offset(unit);
            if has_leaf(unit) {
                let v = value(*self.units.get(pos)?);
                found = Some((i + 1, v));
            }
        }
        let (len, v) = found?;
        let tail = self.normalized.get(v..)?;
        let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
        Some((len, &tail[..end]))
    }
}

/// The protobuf wire format, as far as `ModelProto` needs it.
mod proto {
    use anyhow::{Result, bail};

    pub struct Field<'a> {
        pub field: u64,
        pub wire: u64,
        /// Payload of a length-delimited or fixed-width field.
        pub bytes: &'a [u8],
        /// Value of a varint field.
        pub varint: u64,
    }

    fn varint(b: &[u8], mut at: usize) -> Result<(u64, usize)> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let Some(&byte) = b.get(at) else {
                bail!("a varint runs past the end");
            };
            at += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok((v, at));
            }
        }
        bail!("a varint longer than 64 bits")
    }

    fn take(b: &[u8], at: usize, n: usize) -> Result<&[u8]> {
        match b.get(at..at + n) {
            Some(s) => Ok(s),
            None => bail!("a field runs past the end"),
        }
    }

    /// Each field of a message, in order.
    pub fn fields(b: &[u8]) -> Result<Vec<Field<'_>>> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < b.len() {
            let (key, next) = varint(b, at)?;
            at = next;
            let (field, wire) = (key >> 3, key & 7);
            let (bytes, varint_value) = match wire {
                0 => {
                    let (v, next) = varint(b, at)?;
                    at = next;
                    (&b[0..0], v)
                }
                1 => {
                    let s = take(b, at, 8)?;
                    at += 8;
                    (s, 0)
                }
                2 => {
                    let (len, next) = varint(b, at)?;
                    let s = take(b, next, len as usize)?;
                    at = next + len as usize;
                    (s, 0)
                }
                5 => {
                    let s = take(b, at, 4)?;
                    at += 4;
                    (s, 0)
                }
                w => bail!("wire type {w} in a SentencePiece model"),
            };
            out.push(Field {
                field,
                wire,
                bytes,
                varint: varint_value,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(
        pieces: &[(&str, f32, PieceType)],
        ty: ModelType,
        byte_fallback: bool,
    ) -> SentencePiece {
        let mut all: Vec<Piece> = pieces
            .iter()
            .map(|&(p, s, k)| Piece {
                piece: p.into(),
                score: s,
                kind: k,
            })
            .collect();
        if byte_fallback {
            all.extend((0..=255u8).map(|b| Piece {
                piece: byte_piece(b),
                score: 0.0,
                kind: PieceType::Byte,
            }));
        }
        let normalizer = NormalizerSpec {
            name: "identity".into(),
            ..NormalizerSpec::default()
        };
        SentencePiece::new(
            all,
            ty,
            normalizer,
            false,
            byte_fallback,
            " \u{2047} ".into(),
        )
        .unwrap()
    }

    fn base() -> Vec<(&'static str, f32, PieceType)> {
        vec![
            ("<unk>", 0.0, PieceType::Unknown),
            ("<s>", 0.0, PieceType::Control),
            ("</s>", 0.0, PieceType::Control),
            ("▁", -2.0, PieceType::Normal),
            ("a", -3.0, PieceType::Normal),
            ("b", -3.0, PieceType::Normal),
            ("ab", -1.0, PieceType::Normal),
            ("▁ab", -1.5, PieceType::Normal),
            ("c", -4.0, PieceType::Normal),
        ]
    }

    #[test]
    fn normalizer_collapses_and_escapes_whitespace() {
        let sp = model(&base(), ModelType::Unigram, false);
        assert_eq!(sp.normalize("  ab   c  "), "▁ab▁c");
        assert_eq!(sp.normalize(""), "");
        assert_eq!(sp.normalize("   "), "");
    }

    #[test]
    fn unigram_picks_the_best_path() {
        let sp = model(&base(), ModelType::Unigram, false);
        // "▁ab" (-1.5) beats "▁"+"ab" (-3.0).
        assert_eq!(sp.encode("ab"), vec![7]);
        assert_eq!(sp.encode("ab c"), vec![7, 3, 8]);
        assert_eq!(sp.decode(&[7, 3, 8]).unwrap(), "ab c");
    }

    #[test]
    fn unknowns_merge_or_fall_back_to_bytes() {
        let sp = model(&base(), ModelType::Unigram, false);
        assert_eq!(
            sp.encode_pieces("aé€"),
            vec![("▁".into(), 3), ("a".into(), 4), ("é€".into(), 0)]
        );
        assert_eq!(sp.decode(&[3, 4, 0]).unwrap(), "a \u{2047} ");

        let sp = model(&base(), ModelType::Unigram, true);
        let ids = sp.encode("aé");
        let bytes: Vec<u32> = "é"
            .bytes()
            .map(|b| sp.piece_to_id(&byte_piece(b)))
            .collect();
        assert_eq!(ids, [vec![3, 4], bytes].concat());
        assert_eq!(sp.decode(&ids).unwrap(), "aé");
        // A lone continuation byte decodes to U+FFFD.
        assert_eq!(sp.decode(&[sp.piece_to_id("<0xA9>")]).unwrap(), "\u{FFFD}");
    }

    #[test]
    fn decode_drops_control_pieces() {
        let sp = model(&base(), ModelType::Unigram, false);
        assert_eq!(sp.decode(&[1, 7, 2]).unwrap(), "ab");
    }

    #[test]
    fn bpe_merges_by_score() {
        let sp = model(&base(), ModelType::Bpe, false);
        // ▁ a b → (ab: -1.0) → ▁ ab → (▁ab: -1.5) → ▁ab.
        assert_eq!(sp.encode("ab"), vec![7]);
        assert_eq!(sp.encode("abc"), vec![7, 8]);
    }

    #[test]
    fn user_defined_symbols_stay_whole() {
        let mut p = base();
        p.push(("<sep>", 0.0, PieceType::UserDefined));
        let sp = model(&p, ModelType::Unigram, false);
        let sep = sp.piece_to_id("<sep>");
        assert_eq!(sp.encode("ab<sep>c"), vec![7, sep, 8]);
    }

    #[test]
    fn proto_round_trip() {
        fn varint(mut v: u64, out: &mut Vec<u8>) {
            loop {
                let b = (v & 0x7f) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(b);
                    return;
                }
                out.push(b | 0x80);
            }
        }
        fn bytes_field(f: u64, b: &[u8], out: &mut Vec<u8>) {
            varint(f << 3 | 2, out);
            varint(b.len() as u64, out);
            out.extend_from_slice(b);
        }
        let mut m = Vec::new();
        for (p, s, k) in [("<unk>", 0.0f32, 2u64), ("▁", -1.0, 1), ("x", -2.0, 1)] {
            let mut e = Vec::new();
            bytes_field(1, p.as_bytes(), &mut e);
            varint(2 << 3 | 5, &mut e);
            e.extend_from_slice(&s.to_le_bytes());
            varint(3 << 3, &mut e);
            varint(k, &mut e);
            bytes_field(1, &e, &mut m);
        }
        let mut norm = Vec::new();
        bytes_field(1, b"identity", &mut norm);
        varint(4 << 3, &mut norm);
        varint(0, &mut norm);
        bytes_field(3, &norm, &mut m);
        let sp = SentencePiece::from_bytes(&m).unwrap();
        assert_eq!(sp.len(), 3);
        assert!(!sp.normalizer_spec().remove_extra_whitespaces);
        assert_eq!(sp.encode("x  x"), vec![1, 2, 1, 1, 2]);
    }
}
