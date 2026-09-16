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

//! Host-side machine-translation metrics for RLX dubbing / NMT bake-offs.
//!
//! Pure text metrics — no model weights, no Metal. Pair with `translator-cli`
//! or `rlx-nllb` outputs:
//!
//! - [`chrf`] — character n-gram F1 (script-agnostic; primary for FR/DE/UK/CJK)
//! - [`bleu`] — corpus BLEU-4 with brevity penalty
//! - [`ter`] — word translation edit rate (lower is better)
//! - [`entity_f1`] — glossary / named-fact recall·precision for dubbing
//! - [`score_timing`] — TTS slot fill / overrun from `result.json` cues
//! - [`score_pair`] / [`score_corpus`] — aggregate reports

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod bleu;
mod chrf;
mod entity;
mod ter;
mod timing;

pub use bleu::bleu;
pub use chrf::chrf;
pub use entity::{EntitySet, entity_f1, entity_hits};
pub use ter::ter;
pub use timing::{CueTiming, CueTimingInput, TimingScore, score_timing};

/// One hypothesis scored against one reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairScore {
    pub chrf: f64,
    pub bleu: f64,
    /// Word TER; lower is better (`0.0` = identical tokenization).
    pub ter: f64,
    pub exact: bool,
    pub entity_precision: f64,
    pub entity_recall: f64,
    pub entity_f1: f64,
    pub missing_entities: Vec<String>,
}

/// Mean scores over a parallel corpus.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusScore {
    pub n: usize,
    pub exact: usize,
    pub mean_chrf: f64,
    pub corpus_bleu: f64,
    pub mean_ter: f64,
    pub mean_entity_f1: f64,
    pub mean_entity_recall: f64,
    pub pairs: Vec<PairScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingScore>,
}

/// Score a single (hypothesis, reference) pair.
///
/// `entities` are candidate glossary substrings (case-insensitive). Only those
/// that appear in `reference` are required in the hypothesis — so cue-local
/// packs stay fair (cue 0 need not contain `950`).
pub fn score_pair(hypothesis: &str, reference: &str, entities: &[&str]) -> PairScore {
    let chrf = chrf(hypothesis, reference);
    let bleu = bleu(hypothesis, reference);
    let ter = ter(hypothesis, reference);
    let exact = normalize_cmp(hypothesis) == normalize_cmp(reference);
    let required: Vec<&str> = entities
        .iter()
        .copied()
        .filter(|e| !e.is_empty() && reference.to_lowercase().contains(&e.to_lowercase()))
        .collect();
    let ent = entity_f1(hypothesis, &required);
    PairScore {
        chrf,
        bleu,
        ter,
        exact,
        entity_precision: ent.precision,
        entity_recall: ent.recall,
        entity_f1: ent.f1,
        missing_entities: ent.missing,
    }
}

/// Aggregate chrF / entity F1 means and a single corpus BLEU over all pairs.
pub fn score_corpus<'a, I>(pairs: I, entities_per_pair: &[Vec<&str>]) -> CorpusScore
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let pairs: Vec<(&str, &str)> = pairs.into_iter().collect();
    let n = pairs.len();
    if n == 0 {
        return CorpusScore::default();
    }
    let mut out_pairs = Vec::with_capacity(n);
    let mut sum_chrf = 0.0;
    let mut sum_ter = 0.0;
    let mut sum_ef1 = 0.0;
    let mut sum_erec = 0.0;
    let mut exact = 0usize;
    let mut hyps = Vec::with_capacity(n);
    let mut refs = Vec::with_capacity(n);
    for (i, (hyp, reference)) in pairs.iter().enumerate() {
        let ents: &[&str] = entities_per_pair
            .get(i)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let s = score_pair(hyp, reference, ents);
        sum_chrf += s.chrf;
        sum_ter += s.ter;
        sum_ef1 += s.entity_f1;
        sum_erec += s.entity_recall;
        if s.exact {
            exact += 1;
        }
        hyps.push(*hyp);
        refs.push(*reference);
        out_pairs.push(s);
    }
    let corpus_bleu = bleu::corpus_bleu(&hyps, &refs);
    CorpusScore {
        n,
        exact,
        mean_chrf: sum_chrf / n as f64,
        corpus_bleu,
        mean_ter: sum_ter / n as f64,
        mean_entity_f1: sum_ef1 / n as f64,
        mean_entity_recall: sum_erec / n as f64,
        pairs: out_pairs,
        timing: None,
    }
}

fn normalize_cmp(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Built-in dubbing entity packs keyed by ISO language (motor / Helix clip).
pub fn motor_clip_entities(lang: &str) -> Vec<&'static str> {
    match lang.trim().to_ascii_lowercase().as_str() {
        "fr" => vec!["helix", "950", "chevaux", "pot", "radial"],
        "de" => vec!["helix", "950", "ps", "blumen", "radial"],
        "uk" => vec!["helix", "950", "горщик", "радіал"],
        "es" => vec!["helix", "950", "caball", "radial"],
        "it" => vec!["helix", "950", "cavall", "radial"],
        _ => vec!["helix", "950"],
    }
}

/// Flatten a language → list of (hyp, ref) for reporting.
pub fn group_by_lang(rows: &[(String, String, String)]) -> HashMap<String, Vec<(String, String)>> {
    let mut m: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (lang, hyp, reference) in rows {
        m.entry(lang.clone())
            .or_default()
            .push((hyp.clone(), reference.clone()));
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_pair_is_one() {
        let s = score_pair(
            "Il produit 950 chevaux.",
            "Il produit 950 chevaux.",
            &["950", "chevaux"],
        );
        assert!((s.chrf - 1.0).abs() < 1e-9);
        assert!(s.exact);
        assert!((s.entity_f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn entity_miss_lowers_f1() {
        let s = score_pair(
            "Il est petit.",
            "Il produit 950 chevaux.",
            &["950", "chevaux"],
        );
        assert!(s.entity_recall < 0.1);
        assert!(!s.missing_entities.is_empty());
    }

    #[test]
    fn entities_absent_from_ref_are_ignored() {
        let s = score_pair(
            "Une entreprise appelée Helix.",
            "Une entreprise appelée Helix.",
            &["helix", "950", "radial"],
        );
        assert!((s.entity_f1 - 1.0).abs() < 1e-9);
        assert!(s.missing_entities.is_empty());
    }
}
