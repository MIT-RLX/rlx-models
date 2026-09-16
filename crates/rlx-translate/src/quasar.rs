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

//! Reader for the OS's **Quasar** translation configuration — the JSON that
//! describes the whole on-device MT pipeline for a language pair.
//!
//! macOS ships one config per pair per variant as
//! `mt_app.<src>-<tgt>.<variant>.json` (and the daemon symlinks the selected
//! one in as `mt-quasar-config.json`). The config is the authoritative spec:
//! it names the SentencePiece model, the Espresso NMT, every pre/post
//! processing block, the decode hyper-parameters, and — per language pair —
//! the *graph* wiring those blocks together.
//!
//! # Layout
//!
//! ```text
//! {
//!   "version-major": 278, "version-minor": 0,
//!   "mt-model-info": { "version", "language-pairs": [...], "tasks": [...] },
//!   "text-proc": {},
//!   "mt-decoders": {
//!      "<task>": {                        // mt_app | system | stable | stable_pt
//!         "engine-type": "PDEC",
//!         "block-definitions": { "<block name>": { "block-type": ..., ... } },
//!         "language-pair-specific-settings": {
//!            "<src>-<tgt>": { "graph": { "<stage>": { "receive-from": ..., "block": ... } } }
//!         }
//!      }
//!   }
//! }
//! ```
//!
//! The pipeline is a **DAG, not a sequence**: each `graph` stage names the
//! stage(s) it receives from, and a stage may fan in from several
//! (`{"source": "pb", "in": "spm_decode", ...}`). [`Graph::topo_order`]
//! resolves it.
//!
//! Unknown fields are preserved verbatim in [`Block::attrs`] rather than
//! dropped, so a newer config still round-trips through [`Block::report`].

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A `<src>-<tgt>` locale pair, e.g. `en_US-fr_FR`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LangPair {
    pub source: String,
    pub target: String,
}

impl LangPair {
    /// Parses `en_US-fr_FR`. Locales themselves contain `_`, never `-`, so the
    /// single `-` is an unambiguous separator.
    pub fn parse(s: &str) -> Result<Self> {
        let (source, target) = s
            .split_once('-')
            .ok_or_else(|| anyhow!("not a language pair: {s:?}"))?;
        if source.is_empty() || target.is_empty() {
            bail!("not a language pair: {s:?}");
        }
        Ok(Self {
            source: source.to_string(),
            target: target.to_string(),
        })
    }

    /// The reverse pair. Configs always define both directions.
    pub fn reversed(&self) -> Self {
        Self {
            source: self.target.clone(),
            target: self.source.clone(),
        }
    }
}

impl std::fmt::Display for LangPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.source, self.target)
    }
}

/// The block types the format defines. Every one observed across the shipped
/// configs is named; anything newer lands in [`BlockKind::Other`] with its
/// attributes intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockKind {
    /// Exact-match translation memory consulted before the NMT.
    PhraseBook,
    /// SentencePiece encode or decode; see [`Block::spm_action`].
    SentencePiece,
    /// The NMT itself — beam search over the Espresso model.
    PDecTranslator,
    /// Forced alignment through the same NMT, for span projection.
    PDecForceAlign,
    /// Whitespace/locale-aware tokenizer.
    Tokenizer,
    /// Tokenizer driven by a shipped rule file.
    SimpleTokenizer,
    /// Sentence/segment splitting ahead of translation.
    Segmentation,
    /// Gender (and other) alternative selection from `<STRUCT_*>` markers.
    StructuredPrediction,
    /// Word alignment post-processing.
    AlignmentProcessor,
    /// Heuristic quality signals (`OVS`, `Repetition`).
    QualityEstimator,
    /// Marks ambiguous source spans from a dictionary.
    AmbiguityAnnotator,
    /// Script transliteration / pronunciation guide.
    Romanizer,
    /// Recasing to match source capitalisation.
    CaseMap,
    /// Drops candidates (e.g. wrong-language output).
    Filter,
    /// Routes on a metadata key.
    Select,
    /// Fan-in of two upstream stages.
    Merger,
    /// Pass-through; used to keep a graph edge without doing work.
    Null,
    /// Honours `do_not_translate` spans (the public
    /// `SkipTranslationAttribute`), copying them through verbatim.
    DoNotTranslate,
    /// Attaches alternative translations (e.g. gender variants) to the result.
    LinkAlternatives,
    /// Unrecognised — attributes preserved.
    Other(String),
}

impl BlockKind {
    fn parse(s: &str) -> Self {
        match s {
            "PhraseBookBlock" => Self::PhraseBook,
            "SentencePieceBlock" => Self::SentencePiece,
            "PDecTranslatorBlock" => Self::PDecTranslator,
            "PDecForceAlignBlock" => Self::PDecForceAlign,
            "TokenizerBlock" => Self::Tokenizer,
            "SimpleTokenizerBlock" => Self::SimpleTokenizer,
            "SegmentationBlock" => Self::Segmentation,
            "StructuredPredictionBlock" => Self::StructuredPrediction,
            "AlignmentProcessorBlock" => Self::AlignmentProcessor,
            "QualityEstimatorBlock" => Self::QualityEstimator,
            "AmbiguityAnnotatorBlock" => Self::AmbiguityAnnotator,
            "RomanizerBlock" => Self::Romanizer,
            "CaseMapBlock" => Self::CaseMap,
            "FilterBlock" => Self::Filter,
            "SelectBlock" => Self::Select,
            "MergerBlock" => Self::Merger,
            "NullBlock" => Self::Null,
            "DoNotTranslateBlock" => Self::DoNotTranslate,
            "LinkAlternativesBlock" => Self::LinkAlternatives,
            other => Self::Other(other.to_string()),
        }
    }

    /// The spelling used in the config.
    pub fn as_str(&self) -> &str {
        match self {
            Self::PhraseBook => "PhraseBookBlock",
            Self::SentencePiece => "SentencePieceBlock",
            Self::PDecTranslator => "PDecTranslatorBlock",
            Self::PDecForceAlign => "PDecForceAlignBlock",
            Self::Tokenizer => "TokenizerBlock",
            Self::SimpleTokenizer => "SimpleTokenizerBlock",
            Self::Segmentation => "SegmentationBlock",
            Self::StructuredPrediction => "StructuredPredictionBlock",
            Self::AlignmentProcessor => "AlignmentProcessorBlock",
            Self::QualityEstimator => "QualityEstimatorBlock",
            Self::AmbiguityAnnotator => "AmbiguityAnnotatorBlock",
            Self::Romanizer => "RomanizerBlock",
            Self::CaseMap => "CaseMapBlock",
            Self::Filter => "FilterBlock",
            Self::Select => "SelectBlock",
            Self::Merger => "MergerBlock",
            Self::Null => "NullBlock",
            Self::DoNotTranslate => "DoNotTranslateBlock",
            Self::LinkAlternatives => "LinkAlternativesBlock",
            Self::Other(s) => s,
        }
    }
}

/// One entry of `block-definitions`.
#[derive(Debug, Clone)]
pub struct Block {
    /// Key in `block-definitions`, e.g. `en_US-fr_FR-PDecTranslatorBlock`.
    pub name: String,
    pub kind: BlockKind,
    /// Every field verbatim, including `block-type`.
    pub attrs: Map<String, Value>,
}

impl Block {
    /// Language pair encoded in the block name, when present. Block names are
    /// `<src>-<tgt>-<Suffix>`; the locales are `xx_YY`, so taking the first
    /// two `-`-separated fields is exact.
    pub fn lang_pair(&self) -> Option<LangPair> {
        let mut it = self.name.splitn(3, '-');
        let s = it.next()?;
        let t = it.next()?;
        if is_locale(s) && is_locale(t) {
            Some(LangPair {
                source: s.to_string(),
                target: t.to_string(),
            })
        } else {
            None
        }
    }

    /// String field.
    pub fn str(&self, key: &str) -> Option<&str> {
        self.attrs.get(key).and_then(Value::as_str)
    }

    /// Integer field, tolerating a numeric string.
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.attrs.get(key)? {
            Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|v| v as i64)),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Float field, tolerating an integer or a numeric string.
    pub fn float(&self, key: &str) -> Option<f64> {
        match self.attrs.get(key)? {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Boolean field. These are JSON booleans in the block bodies
    /// but as the *strings* `"true"`/`"false"` inside graph stages, so both
    /// spellings are accepted.
    pub fn flag(&self, key: &str) -> Option<bool> {
        match self.attrs.get(key)? {
            Value::Bool(b) => Some(*b),
            Value::Number(n) => Some(n.as_i64().unwrap_or(0) != 0),
            Value::String(s) => match s.as_str() {
                "true" | "True" | "1" => Some(true),
                "false" | "False" | "0" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    /// A comma-separated list field (`pb-file-list`, `features`), with empty
    /// entries dropped — some lists carry a trailing comma.
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

    /// Array-of-strings field (`shortlist-suppress-tokens`).
    pub fn str_array(&self, key: &str) -> Vec<String> {
        self.attrs
            .get(key)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `encode` / `decode` for a [`BlockKind::SentencePiece`] block.
    pub fn spm_action(&self) -> Option<SpmAction> {
        match self.str("action")? {
            "encode" => Some(SpmAction::Encode),
            "decode" => Some(SpmAction::Decode),
            _ => None,
        }
    }

    /// The asset this block's files come from (`MT-bi-…`, `PB-en`, …).
    pub fn asset_name(&self) -> Option<&str> {
        self.str("asset-name")
    }

    /// Every asset-relative file path this block references, paired with the
    /// field it came from. Used to compute the full file manifest a pair needs.
    pub fn file_refs(&self) -> Vec<(&'static str, String)> {
        const SINGLE: &[&str] = &[
            "sentence-piece-file",
            "model-file",
            "confidence-model-file",
            "defaults-list-file",
            "regex-file",
            "src-ovs-file",
            "tgt-ovs-file",
            "tokenizer-file",
            "disambiguation-dictionary-file",
            "pron-guide-model-file",
        ];
        let mut out = Vec::new();
        for key in SINGLE {
            if let Some(v) = self.str(key)
                && !v.is_empty()
            {
                out.push((*key, v.to_string()));
            }
        }
        for v in self.csv("pb-file-list") {
            out.push(("pb-file-list", v));
        }
        out
    }

    /// Human-readable dump, including fields this crate does not interpret.
    pub fn report(&self) -> String {
        let mut keys: Vec<_> = self.attrs.keys().collect();
        keys.sort();
        let body = keys
            .iter()
            .filter(|k| k.as_str() != "block-type")
            .map(|k| format!("      {k} = {}", compact(&self.attrs[k.as_str()])))
            .collect::<Vec<_>>()
            .join("\n");
        format!("  {} [{}]\n{body}", self.name, self.kind.as_str())
    }
}

/// Direction of a [`BlockKind::SentencePiece`] block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmAction {
    Encode,
    Decode,
}

fn compact(v: &Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_default();
    if s.len() > 200 {
        format!("{}…", &s[..200])
    } else {
        s
    }
}

/// `xx_YY` / `xxx_YY` shape test, matching the daemon's own
/// `^[a-z]{2,3}_[A-Z]{2}$`.
fn is_locale(s: &str) -> bool {
    let Some((lang, region)) = s.split_once('_') else {
        return false;
    };
    (2..=3).contains(&lang.len())
        && lang.chars().all(|c| c.is_ascii_lowercase())
        && region.len() == 2
        && region.chars().all(|c| c.is_ascii_uppercase())
}

/// Where a graph stage takes its input from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiveFrom {
    /// A single upstream stage. May carry a `:port` suffix (`spm_encode:tokens`,
    /// `pb:final`) selecting a named output.
    One(String),
    /// Fan-in: input name → upstream stage (with the same optional `:port`).
    Many(BTreeMap<String, String>),
}

impl ReceiveFrom {
    /// Upstream stage names with any `:port` suffix stripped.
    pub fn deps(&self) -> Vec<&str> {
        self.named_deps().into_iter().map(|(_, d)| d).collect()
    }

    /// Upstream stages with the role each was bound to.
    ///
    /// Most blocks fan in anonymously (`in1`, `in2`) and only care that *some*
    /// input produced a value. A few name their inputs semantically —
    /// `DoNotTranslateBlock` takes `{"target": ..., "source": ...}` — and for
    /// those the role is the difference between reading the translation and
    /// reading the original text back out.
    pub fn named_deps(&self) -> Vec<(&str, &str)> {
        fn strip(s: &str) -> &str {
            s.split(':').next().unwrap_or(s)
        }
        match self {
            Self::One(s) => vec![("in", strip(s))],
            Self::Many(m) => m.iter().map(|(k, v)| (k.as_str(), strip(v))).collect(),
        }
    }

    /// Output port each dependency was read from, where one was named.
    ///
    /// A stage can expose more than one result: `pb:out` is the phrasebook's
    /// *hit* and `pb:final` is the text to carry on with, which is why the NMT
    /// can sit downstream of the phrasebook without being starved when the
    /// lookup misses.
    pub fn ports(&self) -> Vec<Option<&str>> {
        fn port(s: &str) -> Option<&str> {
            s.split_once(':').map(|(_, p)| p)
        }
        match self {
            Self::One(s) => vec![port(s)],
            Self::Many(m) => m.values().map(|s| port(s)).collect(),
        }
    }
}

/// One node of a language pair's execution graph.
///
/// Every node names a block or an inline `block-type`, except the
/// [`GRAPH_OUTPUT`] sink, which names neither.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Stage name (the key), e.g. `pdec`, `spm_encode`.
    pub name: String,
    pub receive_from: ReceiveFrom,
    /// Named block in `block-definitions`, when the stage refers to one.
    pub block: Option<String>,
    /// Inline block type, for stages defined in place (e.g. `MergerBlock`).
    pub block_type: Option<BlockKind>,
    /// Remaining per-stage overrides verbatim.
    pub attrs: Map<String, Value>,
}

impl GraphNode {
    /// True for the terminal [`GRAPH_OUTPUT`] sink.
    pub fn is_output(&self) -> bool {
        self.name == GRAPH_OUTPUT
    }
}

/// The sentinel upstream name meaning "the pipeline's input". It is only ever
/// referenced by a `receive-from`; it is never itself a stage.
pub const GRAPH_INPUT: &str = "graph-input";

/// The sentinel terminal stage. Unlike [`GRAPH_INPUT`] this *is* a node in the
/// graph, but it carries only `receive-from` — no block and no block type — so
/// it names whichever stage produces the pipeline's result.
pub const GRAPH_OUTPUT: &str = "graph-output";

/// A language pair's stage DAG.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: BTreeMap<String, GraphNode>,
}

impl Graph {
    /// Stages in dependency order (Kahn). Errors on a cycle or a dangling
    /// `receive-from`, naming the offender — a malformed graph must not
    /// silently execute in an arbitrary order.
    pub fn topo_order(&self) -> Result<Vec<&GraphNode>> {
        let mut indeg: BTreeMap<&str, usize> = self.nodes.keys().map(|k| (k.as_str(), 0)).collect();
        let mut succ: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for node in self.nodes.values() {
            for dep in node.receive_from.deps() {
                if dep == GRAPH_INPUT {
                    continue;
                }
                if !self.nodes.contains_key(dep) {
                    bail!(
                        "graph stage {:?} receives from unknown stage {dep:?}",
                        node.name
                    );
                }
                succ.entry(dep).or_default().push(node.name.as_str());
                *indeg
                    .get_mut(node.name.as_str())
                    .expect("stage is in nodes") += 1;
            }
        }
        let mut ready: Vec<&str> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(k, _)| *k)
            .collect();
        let mut out = Vec::with_capacity(self.nodes.len());
        while let Some(n) = ready.pop() {
            out.push(&self.nodes[n]);
            for &s in succ.get(n).map(Vec::as_slice).unwrap_or(&[]) {
                let d = indeg.get_mut(s).expect("successor is in nodes");
                *d -= 1;
                if *d == 0 {
                    ready.push(s);
                }
            }
        }
        if out.len() != self.nodes.len() {
            let stuck: Vec<_> = indeg
                .iter()
                .filter(|(_, d)| **d > 0)
                .map(|(k, _)| (*k).to_string())
                .collect();
            bail!("cycle in translation graph among stages {stuck:?}");
        }
        Ok(out)
    }
}

/// One task's decoder: its blocks plus per-pair graphs.
#[derive(Debug, Clone)]
pub struct Decoder {
    /// `PDEC` on every shipped config.
    pub engine_type: String,
    pub blocks: BTreeMap<String, Block>,
    pub graphs: BTreeMap<LangPair, Graph>,
}

impl Decoder {
    /// The block a graph stage resolves to, if it names one.
    pub fn block_for(&self, node: &GraphNode) -> Option<&Block> {
        node.block.as_deref().and_then(|b| self.blocks.get(b))
    }

    /// Blocks belonging to `pair`, in `block-definitions` order.
    pub fn blocks_for(&self, pair: &LangPair) -> Vec<&Block> {
        self.blocks
            .values()
            .filter(|b| b.lang_pair().as_ref() == Some(pair))
            .collect()
    }

    /// The single [`BlockKind::PDecTranslator`] block for `pair`.
    pub fn translator(&self, pair: &LangPair) -> Result<&Block> {
        self.blocks_for(pair)
            .into_iter()
            .find(|b| b.kind == BlockKind::PDecTranslator)
            .ok_or_else(|| anyhow!("no PDecTranslatorBlock for {pair}"))
    }
}

/// `mt-model-info`.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// e.g. `MT-PL-v278-20251117-8dec9e976-diff`.
    pub version: String,
    pub language_pairs: Vec<LangPair>,
    pub tasks: Vec<String>,
}

/// A parsed `mt-quasar-config.json` / `mt_app.<pair>.<variant>.json`.
#[derive(Debug, Clone)]
pub struct QuasarConfig {
    pub version_major: i64,
    pub version_minor: i64,
    pub model_info: ModelInfo,
    /// `text-proc`, verbatim (empty on every shipped config so far).
    pub text_proc: Map<String, Value>,
    /// Task name → decoder. Tasks seen: `mt_app`, `system`, `stable`, `stable_pt`.
    pub decoders: BTreeMap<String, Decoder>,
}

/// The task the public `Translation` API uses.
pub const TASK_MT_APP: &str = "mt_app";

impl QuasarConfig {
    /// Reads and parses a config file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading quasar config {}", path.display()))?;
        Self::from_slice(&bytes)
            .with_context(|| format!("parsing quasar config {}", path.display()))
    }

    /// Parses config JSON.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let root: Value = serde_json::from_slice(bytes)?;
        let root = root
            .as_object()
            .ok_or_else(|| anyhow!("quasar config is not a JSON object"))?;

        let info = root.get("mt-model-info").and_then(Value::as_object);
        let model_info = ModelInfo {
            version: info
                .and_then(|m| m.get("version"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            language_pairs: info
                .and_then(|m| m.get("language-pairs"))
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .filter_map(|s| LangPair::parse(s).ok())
                        .collect()
                })
                .unwrap_or_default(),
            tasks: info
                .and_then(|m| m.get("tasks"))
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
        };

        let decoders_json = root
            .get("mt-decoders")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("quasar config missing mt-decoders"))?;

        let mut decoders = BTreeMap::new();
        for (task, dv) in decoders_json {
            let d = dv
                .as_object()
                .ok_or_else(|| anyhow!("mt-decoders.{task} is not an object"))?;
            let defs = d
                .get("block-definitions")
                .and_then(Value::as_object)
                .ok_or_else(|| anyhow!("mt-decoders.{task} missing block-definitions"))?;

            let mut blocks = BTreeMap::new();
            for (name, bv) in defs {
                let attrs = bv
                    .as_object()
                    .ok_or_else(|| anyhow!("block {name} is not an object"))?
                    .clone();
                let kind = BlockKind::parse(
                    attrs
                        .get("block-type")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                blocks.insert(
                    name.clone(),
                    Block {
                        name: name.clone(),
                        kind,
                        attrs,
                    },
                );
            }

            let mut graphs = BTreeMap::new();
            if let Some(lps) = d
                .get("language-pair-specific-settings")
                .and_then(Value::as_object)
            {
                for (pair_s, pv) in lps {
                    let pair = LangPair::parse(pair_s)?;
                    let mut graph = Graph::default();
                    if let Some(g) = pv.get("graph").and_then(Value::as_object) {
                        for (stage, sv) in g {
                            graph.nodes.insert(stage.clone(), parse_stage(stage, sv)?);
                        }
                    }
                    graphs.insert(pair, graph);
                }
            }

            decoders.insert(
                task.clone(),
                Decoder {
                    engine_type: d
                        .get("engine-type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    blocks,
                    graphs,
                },
            );
        }

        Ok(Self {
            version_major: root
                .get("version-major")
                .and_then(Value::as_i64)
                .unwrap_or(0),
            version_minor: root
                .get("version-minor")
                .and_then(Value::as_i64)
                .unwrap_or(0),
            model_info,
            text_proc: root
                .get("text-proc")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default(),
            decoders,
        })
    }

    /// The `mt_app` decoder — what `TranslationSession.translate` drives.
    pub fn mt_app(&self) -> Result<&Decoder> {
        self.decoders
            .get(TASK_MT_APP)
            .ok_or_else(|| anyhow!("quasar config has no {TASK_MT_APP} task"))
    }

    /// Every distinct asset named anywhere in the config (`MT-bi-…`, `PB-en`, …).
    pub fn assets(&self) -> BTreeSet<String> {
        self.decoders
            .values()
            .flat_map(|d| d.blocks.values())
            .filter_map(|b| b.asset_name())
            .map(str::to_string)
            .collect()
    }

    /// Asset-relative files `pair` needs under `task`, deduplicated. Each entry
    /// is `<asset-name>/<path>` exactly as the block spells it — which is how
    /// they are stored, the asset name being the first path component.
    pub fn required_files(&self, task: &str, pair: &LangPair) -> Result<BTreeSet<String>> {
        let d = self
            .decoders
            .get(task)
            .ok_or_else(|| anyhow!("no task {task:?} in config"))?;
        Ok(d.blocks_for(pair)
            .iter()
            .flat_map(|b| b.file_refs().into_iter().map(|(_, p)| p))
            .collect())
    }
}

fn parse_stage(name: &str, v: &Value) -> Result<GraphNode> {
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow!("graph stage {name:?} is not an object"))?;
    let receive_from = match obj.get("receive-from") {
        Some(Value::String(s)) => ReceiveFrom::One(s.clone()),
        Some(Value::Object(m)) => ReceiveFrom::Many(
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect(),
        ),
        // A stage with no `receive-from` is a source; treat it as reading the
        // pipeline input rather than silently becoming unreachable.
        _ => ReceiveFrom::One(GRAPH_INPUT.to_string()),
    };
    Ok(GraphNode {
        name: name.to_string(),
        receive_from,
        block: obj.get("block").and_then(Value::as_str).map(str::to_string),
        block_type: obj
            .get("block-type")
            .and_then(Value::as_str)
            .map(BlockKind::parse),
        attrs: obj.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_pair_round_trips() {
        let p = LangPair::parse("en_US-fr_FR").expect("parses");
        assert_eq!(p.source, "en_US");
        assert_eq!(p.target, "fr_FR");
        assert_eq!(p.to_string(), "en_US-fr_FR");
        assert_eq!(p.reversed().to_string(), "fr_FR-en_US");
        assert!(LangPair::parse("en_US").is_err());
    }

    #[test]
    fn locale_shape_matches_the_shipped_regex() {
        assert!(is_locale("en_US"));
        assert!(is_locale("fil_PH"));
        assert!(!is_locale("en"));
        assert!(!is_locale("EN_us"));
        assert!(!is_locale("en-US"));
    }

    #[test]
    fn block_name_yields_pair_and_survives_plain_names() {
        let mk = |name: &str| Block {
            name: name.to_string(),
            kind: BlockKind::PDecTranslator,
            attrs: Map::new(),
        };
        assert_eq!(
            mk("en_US-fr_FR-PDecTranslatorBlock")
                .lang_pair()
                .map(|p| p.to_string()),
            Some("en_US-fr_FR".to_string())
        );
        assert_eq!(mk("some-global-block").lang_pair(), None);
    }

    #[test]
    fn flags_accept_json_bools_and_graph_strings() {
        let mut attrs = Map::new();
        attrs.insert("a".into(), Value::Bool(true));
        attrs.insert("b".into(), Value::String("false".into()));
        attrs.insert("c".into(), Value::String("true".into()));
        let b = Block {
            name: "x".into(),
            kind: BlockKind::Filter,
            attrs,
        };
        assert_eq!(b.flag("a"), Some(true));
        assert_eq!(b.flag("b"), Some(false));
        assert_eq!(b.flag("c"), Some(true));
        assert_eq!(b.flag("missing"), None);
    }

    #[test]
    fn csv_drops_trailing_empties() {
        let mut attrs = Map::new();
        attrs.insert(
            "pb-file-list".into(),
            Value::String("a.dict,b.dict,".into()),
        );
        let b = Block {
            name: "x".into(),
            kind: BlockKind::PhraseBook,
            attrs,
        };
        assert_eq!(b.csv("pb-file-list"), vec!["a.dict", "b.dict"]);
    }

    #[test]
    fn topo_order_respects_fan_in_and_rejects_cycles() {
        let node = |name: &str, rf: ReceiveFrom| GraphNode {
            name: name.into(),
            receive_from: rf,
            block: None,
            block_type: None,
            attrs: Map::new(),
        };
        let mut g = Graph::default();
        g.nodes
            .insert("a".into(), node("a", ReceiveFrom::One(GRAPH_INPUT.into())));
        g.nodes
            .insert("b".into(), node("b", ReceiveFrom::One("a:tokens".into())));
        g.nodes.insert(
            "c".into(),
            node(
                "c",
                ReceiveFrom::Many(
                    [
                        ("x".to_string(), "a".to_string()),
                        ("y".to_string(), "b".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                ),
            ),
        );
        let order: Vec<_> = g
            .topo_order()
            .expect("acyclic")
            .iter()
            .map(|n| n.name.clone())
            .collect();
        let pos = |n: &str| order.iter().position(|x| x == n).expect("present");
        assert!(pos("a") < pos("b"));
        assert!(pos("b") < pos("c"));

        let mut bad = Graph::default();
        bad.nodes
            .insert("p".into(), node("p", ReceiveFrom::One("q".into())));
        bad.nodes
            .insert("q".into(), node("q", ReceiveFrom::One("p".into())));
        assert!(bad.topo_order().is_err());
    }

    #[test]
    fn dangling_receive_from_is_an_error() {
        let mut g = Graph::default();
        g.nodes.insert(
            "a".into(),
            GraphNode {
                name: "a".into(),
                receive_from: ReceiveFrom::One("nope".into()),
                block: None,
                block_type: None,
                attrs: Map::new(),
            },
        );
        let err = g.topo_order().expect_err("dangling dep must fail");
        assert!(err.to_string().contains("nope"), "{err}");
    }
}
