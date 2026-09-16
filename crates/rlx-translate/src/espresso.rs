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

//! Reader for `pyespresso.mdl.bin` — the NMT **manifest**.
//!
//! Despite the name and the config's `"model-type": "espresso"`, this file
//! holds no weights: it is ~77 KB of typed key/value records naming the graphs
//! and the tensors that wire them together. The weights live beside it in the
//! classic Espresso triple (`*.espresso.net` / `.shape` / `.weights`) — the
//! same container `rlx-neuralhash` reads.
//!
//! # Record framing
//!
//! One leading byte (`0x00`), then a stream of `<key><space><value>` records
//! whose value encoding depends on the key:
//!
//! ```text
//!   bool     single byte 'T' or 'F', with NO trailing separator
//!   int      tag byte 0x04 followed by a little-endian u32, no separator
//!   string   bytes up to the next space, which is consumed
//!   lang     two strings: a language code then a file name
//! ```
//!
//! Because bools run straight into the following key (`…IsEspresso TSourceInputStr…`),
//! the stream can only be walked with a schema. [`KEYS`] is that schema, and an
//! unrecognised key is an error rather than a guess — a newer manifest should
//! fail loudly instead of silently mis-parsing everything after it.
//!
//! Parsing stops at the `<InputSymbolTable>` record. Its framing is not decoded
//! yet; [`Manifest::symbol_table_offset`] reports where it starts.
//!
//! # What a shipped manifest says
//!
//! ```text
//!   BEspressoEngine CPU
//!   EncoderGraph encoder.espresso.net   EmbeddingGraph embedding.espresso.net
//!   ReadoutGraph readout.espresso.net
//!   DecoderLangGraph  {de,en,es,fr,it,nl,pt} decoder_<lang>.espresso.net
//!   HandoverLangGraph {…}                    handover_<lang>.espresso.net
//!   InputLangGraph    {…}                    input_<lang>.espresso.net
//!   SourceInputStr src_tokens   TargetInputStr prev_output_tokens
//!   EncoderValuesStr encoder.15.output   ScoresStr final_layer_output
//!   HandoverStrings decoder.{0,1,2}.encoder_attn.{key,value}_transpose
//!   StateStrings    decoder.{0,1,2}.self_attn.accum
//! ```
//!
//! So: a shared encoder and embedding, a per-target-language 3-layer decoder
//! whose cross-attention K/V are precomputed once per sentence ("handover"),
//! incremental self-attention state (`accum`), and a readout that gathers the
//! tied embedding and applies softmax.

use anyhow::{Context, Result, anyhow, bail, ensure};
use std::collections::BTreeMap;
use std::path::Path;

/// Value encoding of a manifest key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Bool,
    Int,
    Str,
    /// A language code followed by a file name; the key repeats per language.
    LangFile,
    /// Terminates the parse.
    SymbolTable,
}

/// The manifest schema: every key observed in a shipped `pyespresso.mdl.bin`.
pub const KEYS: &[(&str, ValueKind)] = &[
    ("BEspressoEngine", ValueKind::Str),
    ("EncoderGraph", ValueKind::Str),
    ("EmbeddingGraph", ValueKind::Str),
    ("ReadoutGraph", ValueKind::Str),
    ("DecoderLangGraph", ValueKind::LangFile),
    ("HandoverLangGraph", ValueKind::LangFile),
    ("InputLangGraph", ValueKind::LangFile),
    ("ReadoutStartIdx", ValueKind::Int),
    ("ReadoutWithSoftmax", ValueKind::Bool),
    ("ReadoutInputStr", ValueKind::Str),
    ("ReadoutOutputStr", ValueKind::Str),
    ("AddSrcBos", ValueKind::Bool),
    ("AddSrcEos", ValueKind::Bool),
    ("Reverse", ValueKind::Bool),
    ("IsRNN", ValueKind::Bool),
    ("Mmap", ValueKind::Bool),
    ("UseAttention", ValueKind::Bool),
    ("AddTag", ValueKind::Str),
    ("TagFormat", ValueKind::Str),
    ("IsEspresso", ValueKind::Bool),
    ("SourceInputStr", ValueKind::Str),
    ("TargetInputStr", ValueKind::Str),
    ("EncoderValuesStr", ValueKind::Str),
    ("InputNetValuesStr", ValueKind::Str),
    ("ScoresStr", ValueKind::Str),
    ("AlignmentLayerStr", ValueKind::Str),
    ("AlignmentHeads", ValueKind::Int),
    ("ShiftedAlignments", ValueKind::Bool),
    ("TwoDimSourceInput", ValueKind::Bool),
    ("HandoverStrings", ValueKind::Str),
    ("StateStrings", ValueKind::Str),
    ("StateWidth", ValueKind::Int),
    ("StateLayoutND", ValueKind::Bool),
    ("NeedsPosition", ValueKind::Bool),
    ("NeedsEncoderPositions", ValueKind::Bool),
    ("NeedsEncoderOut", ValueKind::Bool),
    ("PositionZeroBased", ValueKind::Bool),
    ("ApplyLog", ValueKind::Bool),
    ("NoSymbolTables", ValueKind::Bool),
    ("<InputSymbolTable>", ValueKind::SymbolTable),
];

fn kind_of(key: &str) -> Option<ValueKind> {
    KEYS.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// A manifest value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Bool(bool),
    Int(u32),
    Str(String),
}

impl Value {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<u32> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// A parsed `pyespresso.mdl.bin`.
#[derive(Debug, Clone, Default)]
pub struct Manifest {
    /// Single-valued records, in key order.
    pub values: BTreeMap<String, Value>,
    /// `DecoderLangGraph` etc.: key → (language → file name).
    pub lang_graphs: BTreeMap<String, BTreeMap<String, String>>,
    /// Byte offset of the undecoded `<InputSymbolTable>` payload, if reached.
    pub symbol_table_offset: Option<usize>,
}

impl Manifest {
    /// Reads and parses a manifest file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).with_context(|| format!("reading manifest {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("parsing manifest {}", path.display()))
    }

    /// Parses manifest bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(!bytes.is_empty(), "manifest is empty");
        let mut out = Self::default();
        // One unexplained leading byte; every shipped manifest has 0x00 here.
        let mut pos = usize::from(bytes[0] == 0);

        while pos < bytes.len() {
            let Some(sp) = memchr(bytes, b' ', pos) else {
                break;
            };
            let key = std::str::from_utf8(&bytes[pos..sp])
                .map_err(|_| anyhow!("non-UTF-8 manifest key at byte {pos}"))?
                .to_string();
            let kind = kind_of(&key).ok_or_else(|| {
                anyhow!("unknown manifest key {key:?} at byte {pos}; schema needs updating")
            })?;
            pos = sp + 1;

            match kind {
                ValueKind::SymbolTable => {
                    out.symbol_table_offset = Some(pos);
                    break;
                }
                ValueKind::Bool => {
                    let b = *bytes
                        .get(pos)
                        .ok_or_else(|| anyhow!("manifest ends inside bool {key:?}"))?;
                    let v = match b {
                        b'T' => true,
                        b'F' => false,
                        other => bail!(
                            "manifest key {key:?} has bool byte {other:#04x}, expected 'T' or 'F'"
                        ),
                    };
                    out.values.insert(key, Value::Bool(v));
                    pos += 1;
                }
                ValueKind::Int => {
                    ensure!(pos + 5 <= bytes.len(), "manifest ends inside int {key:?}");
                    let tag = bytes[pos];
                    ensure!(
                        tag == 0x04,
                        "manifest int {key:?} has tag {tag:#04x}, expected 0x04"
                    );
                    let raw: [u8; 4] = bytes[pos + 1..pos + 5]
                        .try_into()
                        .expect("checked four bytes");
                    out.values.insert(key, Value::Int(u32::from_le_bytes(raw)));
                    pos += 5;
                }
                ValueKind::Str => {
                    let (s, next) = read_token(bytes, pos, &key)?;
                    out.values.insert(key, Value::Str(s));
                    pos = next;
                }
                ValueKind::LangFile => {
                    let (lang, next) = read_token(bytes, pos, &key)?;
                    let (file, next) = read_token(bytes, next, &key)?;
                    out.lang_graphs.entry(key).or_default().insert(lang, file);
                    pos = next;
                }
            }
        }
        Ok(out)
    }

    /// A string record.
    pub fn str(&self, key: &str) -> Option<&str> {
        self.values.get(key).and_then(Value::as_str)
    }
    /// A bool record.
    pub fn flag(&self, key: &str) -> Option<bool> {
        self.values.get(key).and_then(Value::as_bool)
    }
    /// An int record.
    pub fn int(&self, key: &str) -> Option<u32> {
        self.values.get(key).and_then(Value::as_int)
    }

    /// Target languages with a decoder graph.
    pub fn languages(&self) -> Vec<&str> {
        self.lang_graphs
            .get("DecoderLangGraph")
            .map(|m| m.keys().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Every `.espresso.net` this manifest references, deduplicated.
    pub fn graph_files(&self) -> Vec<String> {
        let mut out: Vec<String> = ["EncoderGraph", "EmbeddingGraph", "ReadoutGraph"]
            .iter()
            .filter_map(|k| self.str(k).map(str::to_string))
            .collect();
        for m in self.lang_graphs.values() {
            out.extend(m.values().cloned());
        }
        out.sort();
        out.dedup();
        out
    }

    /// Graph files needed to translate into `lang`: the shared graphs plus that
    /// language's decoder, handover and input nets.
    pub fn graph_files_for(&self, lang: &str) -> Vec<String> {
        let mut out: Vec<String> = ["EncoderGraph", "EmbeddingGraph", "ReadoutGraph"]
            .iter()
            .filter_map(|k| self.str(k).map(str::to_string))
            .collect();
        for m in self.lang_graphs.values() {
            if let Some(f) = m.get(lang) {
                out.push(f.clone());
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Comma-separated record split into parts (`HandoverStrings`, `StateStrings`).
    pub fn csv(&self, key: &str) -> Vec<String> {
        self.str(key)
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Number of decoder layers, inferred from the distinct `decoder.<n>.`
    /// prefixes in `StateStrings`.
    pub fn decoder_layers(&self) -> usize {
        let mut idx: Vec<&str> = self
            .csv("StateStrings")
            .iter()
            .filter_map(|s| s.strip_prefix("decoder."))
            .filter_map(|s| s.split('.').next())
            .map(str::to_string)
            .collect::<Vec<_>>()
            .leak()
            .iter()
            .map(String::as_str)
            .collect();
        idx.sort_unstable();
        idx.dedup();
        idx.len()
    }
}

fn read_token(bytes: &[u8], pos: usize, key: &str) -> Result<(String, usize)> {
    let end = memchr(bytes, b' ', pos).unwrap_or(bytes.len());
    let s = std::str::from_utf8(&bytes[pos..end])
        .map_err(|_| anyhow!("non-UTF-8 manifest value for {key:?} at byte {pos}"))?
        .to_string();
    Ok((s, (end + 1).min(bytes.len())))
}

fn memchr(bytes: &[u8], needle: u8, from: usize) -> Option<usize> {
    bytes
        .get(from..)?
        .iter()
        .position(|b| *b == needle)
        .map(|i| i + from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A manifest in the shipped framing: leading 0x00, bools running straight
    /// into the next key, 0x04-tagged little-endian ints.
    fn synthetic() -> Vec<u8> {
        let mut v = vec![0x00];
        v.extend_from_slice(b"BEspressoEngine CPU ");
        v.extend_from_slice(b"EncoderGraph encoder.espresso.net ");
        v.extend_from_slice(b"DecoderLangGraph fr decoder_fr.espresso.net ");
        v.extend_from_slice(b"DecoderLangGraph en decoder_en.espresso.net ");
        v.extend_from_slice(b"StateWidth ");
        v.push(0x04);
        v.extend_from_slice(&512u32.to_le_bytes());
        v.extend_from_slice(b"AddSrcEos TAddSrcBos F");
        v.extend_from_slice(b"StateStrings decoder.0.self_attn.accum,decoder.1.self_attn.accum,decoder.2.self_attn.accum ");
        v.extend_from_slice(b"NoSymbolTables T");
        v.extend_from_slice(b"<InputSymbolTable> ");
        v.extend_from_slice(&[0x74, 0xfb, 0xb2, 0x7e]);
        v
    }

    #[test]
    fn parses_the_shipped_framing() {
        let m = Manifest::parse(&synthetic()).expect("parses");
        assert_eq!(m.str("BEspressoEngine"), Some("CPU"));
        assert_eq!(m.str("EncoderGraph"), Some("encoder.espresso.net"));
        assert_eq!(m.int("StateWidth"), Some(512));
        assert_eq!(m.flag("AddSrcEos"), Some(true));
        assert_eq!(m.flag("AddSrcBos"), Some(false));
        assert_eq!(m.flag("NoSymbolTables"), Some(true));
    }

    #[test]
    fn bools_run_into_the_following_key() {
        // `AddSrcEos TAddSrcBos F` must yield two records, not one bad string.
        let m = Manifest::parse(&synthetic()).expect("parses");
        assert!(m.values.contains_key("AddSrcEos"));
        assert!(m.values.contains_key("AddSrcBos"));
    }

    #[test]
    fn repeated_lang_keys_accumulate() {
        let m = Manifest::parse(&synthetic()).expect("parses");
        let mut langs = m.languages();
        langs.sort_unstable();
        assert_eq!(langs, vec!["en", "fr"]);
        assert_eq!(
            m.lang_graphs["DecoderLangGraph"]["fr"],
            "decoder_fr.espresso.net"
        );
    }

    #[test]
    fn graph_files_for_a_language_exclude_other_languages() {
        let m = Manifest::parse(&synthetic()).expect("parses");
        let fr = m.graph_files_for("fr");
        assert!(fr.contains(&"decoder_fr.espresso.net".to_string()));
        assert!(!fr.contains(&"decoder_en.espresso.net".to_string()));
        assert!(fr.contains(&"encoder.espresso.net".to_string()));
    }

    #[test]
    fn decoder_layer_count_comes_from_state_strings() {
        let m = Manifest::parse(&synthetic()).expect("parses");
        assert_eq!(m.decoder_layers(), 3);
    }

    #[test]
    fn symbol_table_stops_the_parse_and_records_its_offset() {
        let m = Manifest::parse(&synthetic()).expect("parses");
        assert!(m.symbol_table_offset.is_some());
    }

    #[test]
    fn an_unknown_key_fails_loudly_rather_than_desynchronising() {
        let mut v = vec![0x00];
        v.extend_from_slice(b"BEspressoEngine CPU NewKeyFromFutureOS x ");
        let err = Manifest::parse(&v).expect_err("must reject an unknown key");
        assert!(err.to_string().contains("NewKeyFromFutureOS"), "{err}");
    }

    #[test]
    fn a_bad_bool_byte_is_rejected() {
        let mut v = vec![0x00];
        v.extend_from_slice(b"AddSrcEos X");
        let err = Manifest::parse(&v).expect_err("must reject a non-T/F bool");
        assert!(err.to_string().contains("bool"), "{err}");
    }

    #[test]
    fn a_bad_int_tag_is_rejected() {
        let mut v = vec![0x00];
        v.extend_from_slice(b"StateWidth ");
        v.push(0x09);
        v.extend_from_slice(&1u32.to_le_bytes());
        let err = Manifest::parse(&v).expect_err("must reject a wrong int tag");
        assert!(err.to_string().contains("0x04"), "{err}");
    }
}
