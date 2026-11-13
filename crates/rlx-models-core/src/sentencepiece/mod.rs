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

//! SentencePiece in Rust: reads a `.model` (`ModelProto`) and does what the
//! C++ `SentencePieceProcessor` does, with no C++ build.
//!
//! - **Normalizer**: the precompiled charsmap (darts-clone double array,
//!   longest match), user-defined symbols kept whole, `add_dummy_prefix`,
//!   `remove_extra_whitespaces`, `escape_whitespaces`,
//!   `treat_whitespace_as_suffix`, and the alignment back to the input
//!   ([`SentencePiece::normalize_with_offsets`]). The denormalizer, when the
//!   model has one, runs on decoded text.
//! - **Models**: unigram (the optimized Viterbi for encoding, f32 path
//!   scores; a lattice for n-best and sampling; user-defined pieces at
//!   `0.1 · (characters − 1)`, unknown characters at `min_score − 10`), BPE (score-ordered merges,
//!   `UNUSED` pieces re-split, BPE-dropout sampling), word, char.
//! - **Processor**: byte fallback, runs of unknown pieces merged, pieces with
//!   their surface and byte span in the input ([`SentencePiece::encode_full`]),
//!   `bos` / `eos` / `reverse` / `unk` extra options ([`EncodeOptions`],
//!   [`ExtraOption`]), n-best ([`SentencePiece::nbest_encode`]), sampling
//!   ([`SentencePiece::sample_encode`]), decoding of ids or piece strings with
//!   per-piece spans ([`SentencePiece::decode_full`]).
//!
//! The random streams differ from the C++ (`std::mt19937`), so sampled
//! segmentations agree in distribution, not draw for draw.
//!
//! A Hugging Face `tokenizer.json` holding a `Unigram` model converted from
//! SentencePiece also loads ([`SentencePiece::from_tokenizer_json`]).

mod model;
mod normalizer;
mod proto;

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

use normalizer::{Normalizer, PrefixMatcher};

/// The whitespace marker, `U+2581`.
pub const SPACE_SYMBOL: &str = "\u{2581}";
const REPLACEMENT_CHAR: &str = "\u{FFFD}";
/// Largest `nbest_size` the C++ accepts (`kMaxNBestSize`).
const MAX_NBEST: usize = 512;

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

/// The `TrainerSpec` fields encoding and decoding read.
#[derive(Debug, Clone)]
pub struct TrainerSpec {
    pub model_type: ModelType,
    pub treat_whitespace_as_suffix: bool,
    pub byte_fallback: bool,
    pub unk_surface: String,
    pub unk_piece: String,
    pub bos_piece: String,
    pub eos_piece: String,
    pub pad_piece: String,
}

impl Default for TrainerSpec {
    fn default() -> Self {
        Self {
            model_type: ModelType::Unigram,
            treat_whitespace_as_suffix: false,
            byte_fallback: false,
            unk_surface: " \u{2047} ".into(),
            unk_piece: "<unk>".into(),
            bos_piece: "<s>".into(),
            eos_piece: "</s>".into(),
            pad_piece: "<pad>".into(),
        }
    }
}

/// One token of an [`Encoding`] (`SentencePieceText.SentencePiece`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPiece {
    /// The piece: its vocabulary string, the normalized text of an unknown
    /// piece, or `<0xXX>` for a byte-fallback token.
    pub piece: String,
    pub id: u32,
    /// The text this token stands for: a slice of the input for encoding, of
    /// the output for decoding. Empty for control tokens, the dummy prefix,
    /// and all but the last byte token of a character.
    pub surface: String,
    /// Byte span of `surface` in that text.
    pub begin: usize,
    pub end: usize,
}

/// A tokenized text (`SentencePieceText`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Encoding {
    /// The input for encoding; the decoded text for decoding.
    pub text: String,
    pub pieces: Vec<EncodedPiece>,
    /// Total segmentation score for n-best results; `0` otherwise.
    pub score: f32,
}

impl Encoding {
    pub fn ids(&self) -> Vec<u32> {
        self.pieces.iter().map(|p| p.id).collect()
    }

    pub fn piece_strs(&self) -> Vec<&str> {
        self.pieces.iter().map(|p| p.piece.as_str()).collect()
    }
}

/// One `extra_options` step (`SetEncodeExtraOptions("bos:eos")`), applied in
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtraOption {
    /// Prepend the BOS token.
    Bos,
    /// Append the EOS token.
    Eos,
    /// Reverse the token order.
    Reverse,
    /// Unknown tokens carry the unknown piece (`<unk>`) as their piece.
    Unk,
}

/// `bos`, `eos`, `reverse`, `unk` separated by `:`.
pub fn parse_extra_options(s: &str) -> Result<Vec<ExtraOption>> {
    s.split(':')
        .filter(|o| !o.is_empty())
        .map(|o| {
            Ok(match o {
                "bos" => ExtraOption::Bos,
                "eos" => ExtraOption::Eos,
                "reverse" => ExtraOption::Reverse,
                "unk" => ExtraOption::Unk,
                o => bail!("option {o:?} is not one of bos, eos, reverse, unk"),
            })
        })
        .collect()
}

/// The keyword options of Python's `SentencePieceProcessor.encode`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
    pub reverse: bool,
    pub emit_unk_piece: bool,
}

impl EncodeOptions {
    /// As extra options, in the order the Python binding applies them.
    pub fn extra_options(&self) -> Vec<ExtraOption> {
        [
            (self.reverse, ExtraOption::Reverse),
            (self.add_bos, ExtraOption::Bos),
            (self.add_eos, ExtraOption::Eos),
            (self.emit_unk_piece, ExtraOption::Unk),
        ]
        .into_iter()
        .filter_map(|(on, o)| on.then_some(o))
        .collect()
    }
}

/// A small seeded generator for [`SentencePiece::sample_encode`]
/// (SplitMix64); any `FnMut() -> f64` in `[0, 1)` works.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// A uniform draw in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A loaded SentencePiece model.
pub struct SentencePiece {
    pieces: Vec<Piece>,
    trainer: TrainerSpec,
    normalizer: Normalizer,
    denormalizer: Option<Normalizer>,
    unk_id: u32,
    /// `NORMAL`, `USER_DEFINED` and `UNUSED` pieces (`pieces_` in C++).
    vocab: HashMap<String, u32>,
    /// `CONTROL`, `UNKNOWN` and `BYTE` pieces (`reserved_id_map_`).
    reserved: HashMap<String, u32>,
    /// Longest `vocab` key in bytes, bounding the prefix search.
    max_piece_bytes: usize,
    user_defined: PrefixMatcher,
    min_score: f32,
}

impl SentencePiece {
    /// Load a `.model` file, or a `tokenizer.json` (by extension).
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
        let mut trainer = TrainerSpec::default();
        let mut normalizer = NormalizerSpec::default();
        let mut denormalizer = None;
        for f in proto::fields(model)? {
            match (f.field, f.wire) {
                (1, 2) => {
                    let (mut piece, mut score, mut kind) = (String::new(), 0f32, 1u64);
                    for g in proto::fields(f.bytes)? {
                        match (g.field, g.wire) {
                            (1, 2) => piece = g.string("piece")?,
                            (2, 5) => score = g.f32()?,
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
                (2, 2) => trainer = parse_trainer(f.bytes)?,
                (3, 2) => normalizer = parse_normalizer(f.bytes)?,
                (5, 2) => denormalizer = Some(parse_normalizer(f.bytes)?),
                _ => {}
            }
        }
        let denormalizer = denormalizer.filter(|d| !d.precompiled_charsmap.is_empty());
        Self::new(pieces, trainer, normalizer, denormalizer)
    }

    /// A Hugging Face `tokenizer.json` with a `Unigram` model (the
    /// SentencePiece conversion): pieces and scores from `model.vocab`,
    /// special added tokens as control pieces, `<0xXX>` as byte pieces when
    /// `byte_fallback` is set. The normalizer settings are read back from
    /// what the conversion writes for them: `Precompiled` is the charsmap,
    /// `Replace(" {2,}", " ")` is `remove_extra_whitespaces`, `Prepend("▁")`
    /// or the `Metaspace` prefix is `add_dummy_prefix`. Other normalizers
    /// are an error rather than silently skipped.
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
        let trainer = TrainerSpec {
            byte_fallback,
            unk_piece: unk_id
                .and_then(|i| pieces.get(i))
                .map_or_else(|| "<unk>".into(), |p| p.piece.clone()),
            ..TrainerSpec::default()
        };
        let normalizer = json_normalizer(&v["normalizer"], &v["pre_tokenizer"])?;
        Self::new(pieces, trainer, normalizer, None)
    }

    fn new(
        pieces: Vec<Piece>,
        trainer: TrainerSpec,
        normalizer: NormalizerSpec,
        denormalizer: Option<NormalizerSpec>,
    ) -> Result<Self> {
        if pieces.is_empty() {
            bail!("no pieces: not a SentencePiece model");
        }
        let mut vocab = HashMap::new();
        let mut reserved = HashMap::new();
        let mut user_defined = Vec::new();
        let mut unk_id = None;
        let mut min_score = f32::MAX;
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
                }
                PieceType::UserDefined => user_defined.push(p.piece.clone()),
                PieceType::Unknown => {
                    if unk_id.replace(id).is_some() {
                        bail!("several unknown pieces");
                    }
                }
                PieceType::Byte if !trainer.byte_fallback => {
                    bail!("byte piece {:?} without byte_fallback", p.piece)
                }
                _ => {}
            }
        }
        let unk_id = unk_id.context("no unknown piece")?;
        if trainer.byte_fallback {
            for b in 0..=255u8 {
                let name = byte_piece(b);
                if !reserved.contains_key(&name) {
                    bail!("byte_fallback is set but piece {name} is missing");
                }
            }
        }
        let max_piece_bytes = vocab.keys().map(String::len).max().unwrap_or(0);
        let normalizer = Normalizer::new(normalizer, trainer.treat_whitespace_as_suffix)?;
        let denormalizer = denormalizer
            .map(|d| Normalizer::new(d, false))
            .transpose()?;
        Ok(Self {
            pieces,
            trainer,
            normalizer,
            denormalizer,
            unk_id,
            vocab,
            reserved,
            max_piece_bytes,
            user_defined: PrefixMatcher::new(user_defined),
            min_score,
        })
    }

    // ── vocabulary ──────────────────────────────────────────────────────

    /// Vocabulary size (`GetPieceSize`).
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
        self.trainer.model_type
    }

    pub fn trainer_spec(&self) -> &TrainerSpec {
        &self.trainer
    }

    pub fn normalizer_spec(&self) -> &NormalizerSpec {
        &self.normalizer.spec
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

    pub fn score(&self, id: u32) -> Option<f32> {
        self.pieces.get(id as usize).map(|p| p.score)
    }

    fn kind(&self, id: u32) -> Option<PieceType> {
        self.pieces.get(id as usize).map(|p| p.kind)
    }

    pub fn is_unknown(&self, id: u32) -> bool {
        self.kind(id) == Some(PieceType::Unknown)
    }

    pub fn is_control(&self, id: u32) -> bool {
        self.kind(id) == Some(PieceType::Control)
    }

    pub fn is_unused(&self, id: u32) -> bool {
        self.kind(id) == Some(PieceType::Unused)
    }

    pub fn is_byte(&self, id: u32) -> bool {
        self.kind(id) == Some(PieceType::Byte)
    }

    pub fn is_user_defined(&self, id: u32) -> bool {
        self.kind(id) == Some(PieceType::UserDefined)
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// The id of a control piece named `piece`, as `bos_id()` & co. find it.
    fn control_id(&self, piece: &str) -> Option<u32> {
        let id = self.piece_to_id(piece);
        self.is_control(id).then_some(id)
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.control_id(&self.trainer.bos_piece)
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.control_id(&self.trainer.eos_piece)
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.control_id(&self.trainer.pad_piece)
    }

    // ── normalization ───────────────────────────────────────────────────

    /// The normalized form of `text` the model segments.
    pub fn normalize(&self, text: &str) -> String {
        self.normalizer.normalize(text, &self.user_defined).0
    }

    /// The normalized text and, for each of its bytes plus one past the end,
    /// the byte offset in `text` it came from.
    pub fn normalize_with_offsets(&self, text: &str) -> (String, Vec<usize>) {
        self.normalizer.normalize(text, &self.user_defined)
    }

    // ── encoding ────────────────────────────────────────────────────────

    /// Token ids of `text` (no BOS / EOS).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_full(text).ids()
    }

    /// Piece strings of `text` (`encode(…, out_type=str)`).
    pub fn encode_as_pieces(&self, text: &str) -> Vec<String> {
        self.encode_full(text)
            .pieces
            .into_iter()
            .map(|p| p.piece)
            .collect()
    }

    /// Every token of `text` with its surface and byte span in `text`.
    pub fn encode_full(&self, text: &str) -> Encoding {
        let (normalized, n2o) = self.normalize_with_offsets(text);
        let segments = self.segment(&normalized);
        self.populate(text, &normalized, &n2o, &segments)
    }

    /// [`encode_full`](Self::encode_full) with the Python binding's options.
    pub fn encode_with(&self, text: &str, options: &EncodeOptions) -> Result<Encoding> {
        let mut e = self.encode_full(text);
        self.apply_extra_options(&mut e, &options.extra_options())?;
        Ok(e)
    }

    /// Apply `options` in order (`SetEncodeExtraOptions`).
    pub fn apply_extra_options(&self, e: &mut Encoding, options: &[ExtraOption]) -> Result<()> {
        for o in options {
            match o {
                ExtraOption::Reverse => e.pieces.reverse(),
                ExtraOption::Unk => {
                    let unk = self.pieces[self.unk_id as usize].piece.clone();
                    for p in e.pieces.iter_mut().filter(|p| p.id == self.unk_id) {
                        p.piece.clone_from(&unk);
                    }
                }
                ExtraOption::Bos | ExtraOption::Eos => {
                    let (name, at_end) = match o {
                        ExtraOption::Bos => (&self.trainer.bos_piece, false),
                        _ => (&self.trainer.eos_piece, true),
                    };
                    let Some(id) = self.control_id(name) else {
                        bail!("id for `{name}` is not defined");
                    };
                    let at = if at_end { e.text.len() } else { 0 };
                    let p = EncodedPiece {
                        piece: name.clone(),
                        id,
                        surface: String::new(),
                        begin: at,
                        end: at,
                    };
                    if at_end {
                        e.pieces.push(p);
                    } else {
                        e.pieces.insert(0, p);
                    }
                }
            }
        }
        Ok(())
    }

    /// Up to `nbest_size` segmentations of `text`, best first, each with its
    /// total score (`NBestEncode`; unigram models only; at most 512).
    pub fn nbest_encode(&self, text: &str, nbest_size: usize) -> Result<Vec<Encoding>> {
        if self.trainer.model_type != ModelType::Unigram {
            bail!("NBestEncode is not available for the current model");
        }
        if nbest_size > MAX_NBEST {
            bail!("nbest_size must be <= {MAX_NBEST}");
        }
        let n = nbest_size.max(1);
        let (normalized, n2o) = self.normalize_with_offsets(text);
        Ok(self
            .nbest_unigram(&normalized, n)
            .into_iter()
            .map(|(seg, score)| {
                let mut e = self.populate(text, &normalized, &n2o, &seg);
                e.score = score;
                e
            })
            .collect())
    }

    /// One segmentation drawn at random (`SampleEncode`, `enable_sampling`).
    ///
    /// Unigram: with `nbest_size < 0`, from every segmentation with
    /// probability `∝ exp(alpha · score)`; with `nbest_size > 1`, from the
    /// `nbest_size` best the same way; `0` or `1` is plain
    /// [`encode_full`](Self::encode_full). BPE: BPE-dropout, each merge
    /// skipped with probability `alpha`; `nbest_size` is ignored. `uniform`
    /// returns draws in `[0, 1)` ([`SplitMix64::uniform`]).
    pub fn sample_encode(
        &self,
        text: &str,
        nbest_size: i32,
        alpha: f32,
        uniform: &mut dyn FnMut() -> f64,
    ) -> Result<Encoding> {
        if nbest_size > MAX_NBEST as i32 {
            bail!("nbest_size must be <= {MAX_NBEST}");
        }
        let (normalized, n2o) = self.normalize_with_offsets(text);
        let segments = match self.trainer.model_type {
            ModelType::Unigram if nbest_size < 0 => {
                self.sample_unigram(&normalized, alpha, uniform)
            }
            ModelType::Unigram if nbest_size <= 1 => self.encode_unigram(&normalized),
            ModelType::Unigram => {
                let nbests = self.nbest_unigram(&normalized, nbest_size as usize);
                let top = nbests
                    .iter()
                    .map(|(_, s)| alpha * s)
                    .fold(f32::MIN, f32::max);
                let w: Vec<f64> = nbests
                    .iter()
                    .map(|(_, s)| f64::from(alpha * s - top).exp())
                    .collect();
                let i = model::pick(&w, uniform);
                nbests
                    .into_iter()
                    .nth(i)
                    .map(|(seg, _)| seg)
                    .unwrap_or_default()
            }
            ModelType::Bpe => {
                let mut skip = || alpha >= 1.0 || (alpha > 0.0 && uniform() < f64::from(alpha));
                self.encode_bpe(&normalized, &mut skip)
            }
            ModelType::Word | ModelType::Char => {
                bail!("SampleEncode is not available for the current model")
            }
        };
        Ok(self.populate(text, &normalized, &n2o, &segments))
    }

    fn segment<'a>(&self, normalized: &'a str) -> model::Segments<'a> {
        if normalized.is_empty() {
            return Vec::new();
        }
        match self.trainer.model_type {
            ModelType::Unigram => self.encode_unigram(normalized),
            ModelType::Bpe => self.encode_bpe(normalized, &mut || false),
            ModelType::Word => self.encode_word(normalized),
            ModelType::Char => self.encode_char(normalized),
        }
    }

    /// `PopulateSentencePieceText`: byte fallback, unknown runs merged, each
    /// token's surface and span in `input`.
    fn populate(
        &self,
        input: &str,
        normalized: &str,
        n2o: &[usize],
        segments: &[(&str, u32)],
    ) -> Encoding {
        let mut out: Vec<EncodedPiece> = Vec::with_capacity(segments.len());
        let mut consumed = 0;
        for &(w, id) in segments {
            let (begin, end) = (consumed, consumed + w.len());
            consumed = end;
            let (ob, oe) = (n2o[begin], n2o[end]);
            let surface = input.get(ob..oe).unwrap_or_default();
            let is_unk = id == self.unk_id;
            if is_unk && self.trainer.byte_fallback {
                for (i, &b) in w.as_bytes().iter().enumerate() {
                    let name = byte_piece(b);
                    let last = i + 1 == w.len();
                    out.push(EncodedPiece {
                        id: self.piece_to_id(&name),
                        piece: name,
                        surface: if last {
                            surface.to_string()
                        } else {
                            String::new()
                        },
                        begin: ob,
                        end: if last { oe } else { ob },
                    });
                }
                continue;
            }
            match out.last_mut() {
                Some(prev) if is_unk && prev.id == self.unk_id => {
                    prev.piece.push_str(w);
                    prev.surface.push_str(surface);
                    prev.end = oe;
                }
                _ => out.push(EncodedPiece {
                    piece: w.to_string(),
                    id,
                    surface: surface.to_string(),
                    begin: ob,
                    end: oe,
                }),
            }
        }
        debug_assert_eq!(consumed, normalized.len());
        Encoding {
            text: input.to_string(),
            pieces: out,
            score: 0.0,
        }
    }

    // ── decoding ────────────────────────────────────────────────────────

    /// Text of `ids`. Ids past the vocabulary are an error.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.decode_full(ids)?.text)
    }

    /// Text of piece strings (`DecodePieces`); a piece outside the vocabulary
    /// stands for itself.
    pub fn decode_pieces<S: AsRef<str>>(&self, pieces: &[S]) -> String {
        let items: Vec<(Cow<'_, str>, u32)> = pieces
            .iter()
            .map(|p| (Cow::Borrowed(p.as_ref()), self.piece_to_id(p.as_ref())))
            .collect();
        self.decode_items(&items).text
    }

    /// The decoded text with each token's surface and span in it.
    pub fn decode_full(&self, ids: &[u32]) -> Result<Encoding> {
        let mut items = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                bail!("token id {id} past the vocabulary ({})", self.pieces.len());
            };
            items.push((Cow::Borrowed(p.piece.as_str()), id));
        }
        Ok(self.decode_items(&items))
    }

    fn decode_items(&self, items: &[(Cow<'_, str>, u32)]) -> Encoding {
        let mut text = String::new();
        let mut pieces = Vec::with_capacity(items.len());
        // The first non-control piece loses its dummy-prefix marker; with
        // `remove_extra_whitespaces`, every piece does until one decodes to
        // something (control pieces never count).
        let mut at_bos = true;
        let mut i = 0;
        while i < items.len() {
            let (piece, id) = (&items[i].0, items[i].1);
            if !self.is_byte(id) {
                let surface = self.decode_piece(piece, id, at_bos);
                at_bos &= self.is_control(id)
                    || (self.normalizer.spec.remove_extra_whitespaces && surface.is_empty());
                let begin = text.len();
                text.push_str(&surface);
                pieces.push(EncodedPiece {
                    piece: piece.to_string(),
                    id,
                    surface,
                    begin,
                    end: text.len(),
                });
                i += 1;
                continue;
            }
            // A run of byte pieces: each character's surface goes on the
            // piece holding its last byte; an invalid byte becomes U+FFFD.
            let start = i;
            while i < items.len() && self.is_byte(items[i].1) {
                i += 1;
            }
            let bytes: Vec<u8> = items[start..i]
                .iter()
                .map(|(p, _)| byte_piece_value(p).expect("byte pieces are named <0xXX>"))
                .collect();
            let mut k = 0;
            while k < bytes.len() {
                let (surface, n) = utf8_prefix(&bytes[k..]);
                let begin = text.len();
                for j in 0..n {
                    let (p, id) = &items[start + k + j];
                    let last = j + 1 == n;
                    if last {
                        text.push_str(surface);
                    }
                    pieces.push(EncodedPiece {
                        piece: p.to_string(),
                        id: *id,
                        surface: if last {
                            surface.to_string()
                        } else {
                            String::new()
                        },
                        begin,
                        end: if last { text.len() } else { begin },
                    });
                }
                k += n;
            }
            at_bos = false;
        }
        let mut e = Encoding {
            text,
            pieces,
            score: 0.0,
        };
        if let Some(d) = &self.denormalizer {
            self.denormalize(d, &mut e);
        }
        e
    }

    fn decode_piece(&self, piece: &str, id: u32, at_bos: bool) -> String {
        match self.kind(id) {
            Some(PieceType::Control) => return String::new(),
            Some(PieceType::Unknown) => {
                return if self.pieces[id as usize].piece == piece {
                    self.trainer.unk_surface.clone()
                } else {
                    piece.to_string()
                };
            }
            _ => {}
        }
        let spec = &self.normalizer.spec;
        let mut piece = piece;
        // Only leading markers go: with `treat_whitespace_as_suffix` the
        // trailing one stays a space, as in the C++.
        if at_bos && (spec.add_dummy_prefix || spec.remove_extra_whitespaces) {
            piece = if spec.remove_extra_whitespaces {
                piece.trim_start_matches(SPACE_SYMBOL)
            } else {
                piece.strip_prefix(SPACE_SYMBOL).unwrap_or(piece)
            };
        }
        piece.replace(SPACE_SYMBOL, " ")
    }

    /// Run the denormalizer over the decoded text. The pieces keep their
    /// surfaces and spans, as in the C++.
    fn denormalize(&self, d: &Normalizer, e: &mut Encoding) {
        e.text = d.normalize(&e.text, &PrefixMatcher::default()).0;
    }
}

/// The `NormalizerSpec` a SentencePiece → `tokenizer.json` conversion
/// started from.
fn json_normalizer(
    normalizer: &serde_json::Value,
    pre: &serde_json::Value,
) -> Result<NormalizerSpec> {
    let mut spec = NormalizerSpec {
        name: "identity".into(),
        add_dummy_prefix: false,
        remove_extra_whitespaces: false,
        ..NormalizerSpec::default()
    };
    let steps: Vec<&serde_json::Value> = match normalizer["type"].as_str() {
        None => Vec::new(),
        Some("Sequence") => normalizer["normalizers"]
            .as_array()
            .into_iter()
            .flatten()
            .collect(),
        Some(_) => vec![normalizer],
    };
    for n in steps {
        match n["type"].as_str() {
            Some("Precompiled") => {
                let b64 = n["precompiled_charsmap"].as_str().unwrap_or_default();
                spec.precompiled_charsmap = base64_decode(b64).context("precompiled_charsmap")?;
                spec.name = "precompiled".into();
            }
            Some("Replace")
                if n["pattern"]["Regex"].as_str() == Some(" {2,}") && n["content"] == " " =>
            {
                spec.remove_extra_whitespaces = true;
            }
            Some("Prepend") if n["prepend"] == SPACE_SYMBOL => spec.add_dummy_prefix = true,
            other => bail!("tokenizer.json normalizer {other:?} has no SentencePiece equivalent"),
        }
    }
    let metas: Vec<&serde_json::Value> = match pre["type"].as_str() {
        Some("Sequence") => pre["pretokenizers"]
            .as_array()
            .into_iter()
            .flatten()
            .collect(),
        Some(_) => vec![pre],
        None => Vec::new(),
    };
    for m in metas.into_iter().filter(|m| m["type"] == "Metaspace") {
        let prefix = match m["prepend_scheme"].as_str() {
            Some(scheme) => scheme != "never",
            None => m["add_prefix_space"].as_bool().unwrap_or(true),
        };
        spec.add_dummy_prefix |= prefix;
    }
    Ok(spec)
}

/// Standard base64 (with or without padding).
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes().filter(|&c| c != b'=' && !c.is_ascii_whitespace()) {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => bail!("byte {c:#x} is not base64"),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

fn parse_trainer(b: &[u8]) -> Result<TrainerSpec> {
    let mut t = TrainerSpec::default();
    for g in proto::fields(b)? {
        match (g.field, g.wire) {
            (3, 0) => {
                t.model_type = match g.varint {
                    1 => ModelType::Unigram,
                    2 => ModelType::Bpe,
                    3 => ModelType::Word,
                    4 => ModelType::Char,
                    v => bail!("model type {v} in a SentencePiece model"),
                }
            }
            (24, 0) => t.treat_whitespace_as_suffix = g.bool(),
            (35, 0) => t.byte_fallback = g.bool(),
            (44, 2) => t.unk_surface = g.string("unk_surface")?,
            (45, 2) => t.unk_piece = g.string("unk_piece")?,
            (46, 2) => t.bos_piece = g.string("bos_piece")?,
            (47, 2) => t.eos_piece = g.string("eos_piece")?,
            (48, 2) => t.pad_piece = g.string("pad_piece")?,
            _ => {}
        }
    }
    Ok(t)
}

fn parse_normalizer(b: &[u8]) -> Result<NormalizerSpec> {
    let mut n = NormalizerSpec::default();
    for g in proto::fields(b)? {
        match (g.field, g.wire) {
            (1, 2) => n.name = g.string("normalizer name")?,
            (2, 2) => n.precompiled_charsmap = g.bytes.to_vec(),
            (3, 0) => n.add_dummy_prefix = g.bool(),
            (4, 0) => n.remove_extra_whitespaces = g.bool(),
            (5, 0) => n.escape_whitespaces = g.bool(),
            _ => {}
        }
    }
    Ok(n)
}

/// `<0xXX>`, the byte-fallback piece of `b`.
pub fn byte_piece(b: u8) -> String {
    format!("<0x{b:02X}>")
}

fn byte_piece_value(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok())?
}

pub(super) fn char_len(s: &str) -> usize {
    s.chars().next().map_or(0, char::len_utf8)
}

/// The first character of `bytes` and its length, or U+FFFD and 1 for a
/// byte that starts no valid character.
fn utf8_prefix(bytes: &[u8]) -> (&str, usize) {
    let n = match bytes[0] {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 0,
    };
    match bytes.get(..n).map(std::str::from_utf8) {
        Some(Ok(s)) if n > 0 => (s, n),
        _ => (REPLACEMENT_CHAR, 1),
    }
}

#[cfg(test)]
mod tests;
