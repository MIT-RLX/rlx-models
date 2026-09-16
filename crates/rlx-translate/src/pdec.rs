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

//! Decode parameters for `PDecTranslatorBlock` — the NMT stage.
//!
//! Every field here is read from the shipped Quasar config rather than guessed,
//! so a pair's decode behaviour matches the OS's exactly. The defaults are only
//! a fallback for configs that omit a field.
//!
//! # Direction is a token, not a model
//!
//! One multilingual Espresso model serves every pair in its bundle
//! (`MT-bi-en-es-de-it-fr-pt-nl-0` covers en/es/de/it/fr/pt/nl, both ways).
//! Direction comes from control tokens prepended to the source and target
//! streams, NLLB-style:
//!
//! ```text
//!   source-token = "en_US"
//!   target-token = "fr_FR> <en_US-fr_FR-optimal"
//! ```
//!
//! Note the target token is really *two* tags: the target locale plus a
//! pair-specific variant selector. They are stored pre-joined with the
//! `> <` that the surrounding `<`…`>` template supplies, so
//! [`PDecParams::target_tokens`] splits them back out.

use crate::quasar::{Block, BlockKind};
use anyhow::{Result, bail};

/// How the beam decides it is finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopMode {
    /// Stop once `stop-mode-finished-score-beam` hypotheses are complete and
    /// no live hypothesis can still beat them.
    FinishedScore,
    /// Unrecognised mode, preserved.
    Other(String),
}

/// How the phrasebook/LM bias is mixed into beam scores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LmMode {
    /// Bias only while the hypothesis still matches the biasing prefix.
    PartialBias,
    Other(String),
}

/// Overrides applied when translating a partial (still-being-typed) input.
#[derive(Debug, Clone, Default)]
pub struct PartialInputOverride {
    pub beam: Option<usize>,
    pub lm_weight: Option<f64>,
    pub source_token: Option<String>,
    pub veto_factor: Option<f64>,
}

/// Vocabulary shortlisting: restrict the softmax to a per-pair candidate set.
#[derive(Debug, Clone, Default)]
pub struct Shortlist {
    pub enabled: bool,
    pub merge: bool,
    /// Top-N by conditional probability given the source.
    pub cond_n: usize,
    /// Top-N by target-language frequency.
    pub freq_n: usize,
    /// Which shortlist table to use, e.g. `all-fr`.
    pub lang_pair: String,
    /// Tokens removed from the shortlist — the `<STRUCT_*>` gender markers on
    /// directions that must not emit them.
    pub suppress_tokens: Vec<String>,
}

/// Typed view of a `PDecTranslatorBlock`.
#[derive(Debug, Clone)]
pub struct PDecParams {
    /// Asset-relative path of `pyespresso.mdl.bin`.
    pub model_file: String,
    /// `espresso` on every shipped config.
    pub model_type: String,
    pub memory_map: bool,

    pub source_locale: String,
    pub target_locale: String,
    /// Raw `source-token` field.
    pub source_token: String,
    /// Raw `target-token` field, possibly holding two `> <`-joined tags.
    pub target_token: String,

    pub beam: usize,
    /// Relative-score beam pruning factor (`rs-beam`).
    pub rs_beam: f64,
    pub nbest: usize,
    /// Length-normalise hypothesis costs before ranking.
    pub norm_costs: bool,
    pub lm_mode: LmMode,
    pub lm_weight: f64,
    pub veto_factor: f64,
    pub stop_mode: StopMode,
    pub stop_mode_finished_score_beam: usize,

    /// Absolute cap on generated length.
    pub max_seq_length: usize,
    /// Lower bound for the length budget regardless of source length.
    pub max_seq_length_floor: usize,
    /// Length budget multiplier relative to the source token count.
    pub max_seq_length_relative: f64,

    /// Feed SentencePiece ids straight in rather than re-looking-up pieces.
    pub use_sentencepiece_ids: bool,
    /// Copy source material for `<unk>` outputs using the attention alignment.
    pub unk_replace: bool,
    pub shortlist: Shortlist,
    pub partial_input: PartialInputOverride,
}

impl Default for PDecParams {
    fn default() -> Self {
        Self {
            model_file: String::new(),
            model_type: "espresso".into(),
            memory_map: true,
            source_locale: String::new(),
            target_locale: String::new(),
            source_token: String::new(),
            target_token: String::new(),
            beam: 3,
            rs_beam: 0.66,
            nbest: 1,
            norm_costs: true,
            lm_mode: LmMode::PartialBias,
            lm_weight: 0.25,
            veto_factor: 0.0,
            stop_mode: StopMode::FinishedScore,
            stop_mode_finished_score_beam: 3,
            max_seq_length: 200,
            max_seq_length_floor: 80,
            max_seq_length_relative: 2.0,
            use_sentencepiece_ids: true,
            unk_replace: true,
            shortlist: Shortlist::default(),
            partial_input: PartialInputOverride::default(),
        }
    }
}

impl PDecParams {
    /// Reads a `PDecTranslatorBlock` (or `PDecForceAlignBlock`, which shares
    /// the model/token/shortlist fields but carries no beam settings).
    pub fn from_block(block: &Block) -> Result<Self> {
        if !matches!(
            block.kind,
            BlockKind::PDecTranslator | BlockKind::PDecForceAlign
        ) {
            bail!(
                "block {:?} is a {}, not a PDec block",
                block.name,
                block.kind.as_str()
            );
        }
        let d = Self::default();
        Ok(Self {
            model_file: block.str("model-file").unwrap_or_default().to_string(),
            model_type: block.str("model-type").unwrap_or(&d.model_type).to_string(),
            memory_map: block.flag("enable-memory-map").unwrap_or(d.memory_map),

            source_locale: block.str("source-locale").unwrap_or_default().to_string(),
            target_locale: block.str("target-locale").unwrap_or_default().to_string(),
            source_token: block.str("source-token").unwrap_or_default().to_string(),
            target_token: block.str("target-token").unwrap_or_default().to_string(),

            beam: block.int("beam").map_or(d.beam, |v| v.max(1) as usize),
            rs_beam: block.float("rs-beam").unwrap_or(d.rs_beam),
            nbest: block.int("nbest").map_or(d.nbest, |v| v.max(1) as usize),
            norm_costs: block.flag("norm-costs").unwrap_or(d.norm_costs),
            lm_mode: match block.str("lm-mode") {
                Some("partial_bias") | None => LmMode::PartialBias,
                Some(other) => LmMode::Other(other.to_string()),
            },
            lm_weight: block.float("lm-weight").unwrap_or(d.lm_weight),
            veto_factor: block.float("veto-factor").unwrap_or(d.veto_factor),
            stop_mode: match block.str("stop-mode") {
                Some("finished_score") | None => StopMode::FinishedScore,
                Some(other) => StopMode::Other(other.to_string()),
            },
            stop_mode_finished_score_beam: block
                .int("stop-mode-finished-score-beam")
                .map_or(d.stop_mode_finished_score_beam, |v| v.max(1) as usize),

            max_seq_length: block
                .int("max-seq-length")
                .map_or(d.max_seq_length, |v| v.max(1) as usize),
            max_seq_length_floor: block
                .int("max-seq-length-floor")
                .map_or(d.max_seq_length_floor, |v| v.max(1) as usize),
            max_seq_length_relative: block
                .float("max-seq-length-relative")
                .unwrap_or(d.max_seq_length_relative),

            use_sentencepiece_ids: block
                .flag("use-sentencepiece-ids")
                .unwrap_or(d.use_sentencepiece_ids),
            unk_replace: block.flag("unk-replace").unwrap_or(d.unk_replace),
            shortlist: Shortlist {
                enabled: block.flag("enable_shortlist").unwrap_or(false),
                merge: block.flag("merge_shortlist").unwrap_or(false),
                cond_n: block.int("shortlist-cond-n").unwrap_or(0).max(0) as usize,
                freq_n: block.int("shortlist-freq-n").unwrap_or(0).max(0) as usize,
                lang_pair: block
                    .str("shortlist-lang-pair")
                    .unwrap_or_default()
                    .to_string(),
                suppress_tokens: block.str_array("shortlist-suppress-tokens"),
            },
            partial_input: block
                .attrs
                .get("partial-input-override")
                .and_then(|v| v.as_object())
                .map(|o| PartialInputOverride {
                    beam: o
                        .get("beam")
                        .and_then(|v| v.as_i64())
                        .map(|v| v.max(1) as usize),
                    lm_weight: o.get("lm-weight").and_then(|v| v.as_f64()),
                    source_token: o
                        .get("source-token")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    veto_factor: o.get("veto-factor").and_then(|v| v.as_f64()),
                })
                .unwrap_or_default(),
        })
    }

    /// The source-side control tags, in order.
    pub fn source_tokens(&self) -> Vec<String> {
        split_tags(&self.source_token)
    }

    /// The target-side control tags, in order. the OS pre-joins the locale tag
    /// and the variant selector with `> <`, so `"fr_FR> <en_US-fr_FR-optimal"`
    /// is two tags, not one.
    pub fn target_tokens(&self) -> Vec<String> {
        split_tags(&self.target_token)
    }

    /// Vocabulary spellings of the source-side control tokens.
    ///
    /// The config stores a raw value which is wrapped as `<src-{v}>`; the
    /// `> <` join exists so that wrapping yields several valid pieces, and only
    /// the first is bare. So `"partial> <src-en_US"` becomes
    /// `["<src-partial>", "<src-en_US>"]`, and `"en_US"` becomes
    /// `["<src-en_US>"]`.
    pub fn source_token_pieces(&self) -> Vec<String> {
        wrap_tags(&self.source_tokens(), "src")
    }

    /// Vocabulary spellings of the target-side control tokens.
    ///
    /// `"fr_FR> <en_US-fr_FR-optimal"` becomes
    /// `["<tar-fr_FR>", "<en_US-fr_FR-optimal>"]`.
    pub fn target_token_pieces(&self) -> Vec<String> {
        wrap_tags(&self.target_tokens(), "tar")
    }

    /// Length budget for a source of `source_len` tokens:
    /// `min(max_seq_length, max(floor, ceil(relative × source_len)))`.
    pub fn length_budget(&self, source_len: usize) -> usize {
        let relative = (self.max_seq_length_relative * source_len as f64).ceil();
        let relative = if relative.is_finite() && relative >= 0.0 {
            relative as usize
        } else {
            self.max_seq_length
        };
        relative
            .max(self.max_seq_length_floor)
            .min(self.max_seq_length)
    }
}

/// Splits the OS's `> <`-joined control-tag string into individual tags,
/// tolerating stray angle brackets around the ends.
/// Wraps split control tags into their vocabulary spellings.
///
/// Only the first tag takes the `src-`/`tar-` prefix; the rest already carry
/// their own qualification (`en_US-fr_FR-optimal`).
fn wrap_tags(tags: &[String], side: &str) -> Vec<String> {
    tags.iter()
        .enumerate()
        .map(|(i, t)| {
            if i == 0 {
                format!("<{side}-{t}>")
            } else {
                format!("<{t}>")
            }
        })
        .collect()
}

fn split_tags(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split("> <")
        .map(|t| {
            t.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .trim()
        })
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, Value, json};

    fn block(pairs: &[(&str, Value)]) -> Block {
        let mut attrs = Map::new();
        attrs.insert(
            "block-type".into(),
            Value::String("PDecTranslatorBlock".into()),
        );
        for (k, v) in pairs {
            attrs.insert((*k).to_string(), v.clone());
        }
        Block {
            name: "en_US-fr_FR-PDecTranslatorBlock".into(),
            kind: BlockKind::PDecTranslator,
            attrs,
        }
    }

    #[test]
    fn target_token_splits_into_locale_and_variant() {
        let p = PDecParams::from_block(&block(&[
            ("source-token", json!("en_US")),
            ("target-token", json!("fr_FR> <en_US-fr_FR-optimal")),
        ]))
        .expect("parses");
        assert_eq!(p.source_tokens(), vec!["en_US"]);
        assert_eq!(p.target_tokens(), vec!["fr_FR", "en_US-fr_FR-optimal"]);
    }

    #[test]
    fn partial_input_source_token_also_splits() {
        let p = PDecParams::from_block(&block(&[(
            "partial-input-override",
            json!({"beam": 1, "lm-weight": 0.3, "source-token": "partial> <src-en_US", "veto-factor": 0.0}),
        )]))
        .expect("parses");
        assert_eq!(p.partial_input.beam, Some(1));
        assert_eq!(p.partial_input.lm_weight, Some(0.3));
        assert_eq!(
            split_tags(p.partial_input.source_token.as_deref().unwrap_or("")),
            vec!["partial", "src-en_US"]
        );
    }

    #[test]
    fn length_budget_clamps_between_floor_and_max() {
        let p = PDecParams {
            max_seq_length: 200,
            max_seq_length_floor: 80,
            max_seq_length_relative: 2.0,
            ..PDecParams::default()
        };
        // Short source: floor wins.
        assert_eq!(p.length_budget(3), 80);
        // Mid: 2x source.
        assert_eq!(p.length_budget(50), 100);
        // Long: absolute cap wins.
        assert_eq!(p.length_budget(500), 200);
    }

    #[test]
    fn shortlist_and_suppress_tokens_round_trip() {
        let p = PDecParams::from_block(&block(&[
            ("enable_shortlist", json!(true)),
            ("merge_shortlist", json!(true)),
            ("shortlist-cond-n", json!(100)),
            ("shortlist-freq-n", json!(100)),
            ("shortlist-lang-pair", json!("all-fr")),
            (
                "shortlist-suppress-tokens",
                json!(["<STRUCT_FEMALE_BEG>", "<STRUCT_MALE_BEG>"]),
            ),
        ]))
        .expect("parses");
        assert!(p.shortlist.enabled && p.shortlist.merge);
        assert_eq!(p.shortlist.cond_n, 100);
        assert_eq!(p.shortlist.lang_pair, "all-fr");
        assert_eq!(p.shortlist.suppress_tokens.len(), 2);
    }

    #[test]
    fn wrong_block_kind_is_rejected() {
        let mut attrs = Map::new();
        attrs.insert("block-type".into(), Value::String("CaseMapBlock".into()));
        let b = Block {
            name: "x".into(),
            kind: BlockKind::CaseMap,
            attrs,
        };
        assert!(PDecParams::from_block(&b).is_err());
    }
}
