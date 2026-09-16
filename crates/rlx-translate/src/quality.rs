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

//! the OS's `QualityEstimatorBlock`: does this translation look wrong?
//!
//! Two shipped checks, both data-driven, both aimed at the failure modes a
//! beam decoder actually has.
//!
//! **Repetition** — `qualityEstimator/repeat/repeatRegex.txt` is two regexes:
//!
//! ```text
//! source  ([^ ].*?)( \1){1}
//! target  ([^ ].*?)( \1){3}
//! ```
//!
//! A run repeated *once* is suspicious in the source (the user typed it, so a
//! repeat is probably deliberate but worth knowing); in the target it takes
//! *three* repeats to flag, because the model looping is the thing being
//! detected. This is exactly the defect seen in the shipped framework's own
//! output — `Sie sind meine Anhänger Anhänger Anhänger Anhänger Anhänger`.
//!
//! **Hallucinated profanity** — `qualityEstimator/ovs/ovs.<lang>.{src,tgt}` are
//! offensive/vulgar term lists. The target list is not a filter on its own: the
//! point is that a vulgar term appearing in the *output* with nothing
//! corresponding in the *input* is a hallucination. The shipped framework does
//! this too, rendering `Avocado platypus i love the summer` as
//! `J'adore l'été merde` — `merde` is on the French list and nothing in the
//! source licences it.
//!
//! This block *scores*; it does not rewrite. The shipped graph wires its output into
//! `merger_final`, which is one of the merge blocks still unimplemented here.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result};
use fancy_regex::Regex;

/// Why a translation was flagged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flag {
    /// A run repeats often enough to look like a decoder loop.
    RepeatedTarget(String),
    /// A run repeats in the source. Informational: the user wrote it.
    RepeatedSource(String),
    /// A vulgar term appears in the output with nothing matching in the input.
    UnlicensedProfanity(String),
}

/// The shipped quality estimator for one direction.
#[derive(Default)]
pub struct QualityEstimator {
    source_repeat: Option<Regex>,
    target_repeat: Option<Regex>,
    source_terms: BTreeSet<String>,
    target_terms: BTreeSet<String>,
}

impl QualityEstimator {
    /// Loads from a bundle's `qualityEstimator/` directory.
    ///
    /// Missing files are not an error: a bundle may ship the repeat regexes and
    /// no term list for a given language, and a partial estimator is still
    /// worth running.
    pub fn load(dir: &Path, source_lang: &str, target_lang: &str) -> Result<Self> {
        let mut qe = Self::default();
        let rx = dir.join("repeat").join("repeatRegex.txt");
        if let Ok(text) = std::fs::read_to_string(&rx) {
            for line in text.lines() {
                let Some((which, pat)) = line.split_once('\t') else {
                    continue;
                };
                let compiled = Regex::new(pat.trim())
                    .with_context(|| format!("compiling the {which} repeat regex"))?;
                match which.trim() {
                    "source" => qe.source_repeat = Some(compiled),
                    "target" => qe.target_repeat = Some(compiled),
                    _ => {}
                }
            }
        }
        qe.source_terms = read_terms(&dir.join("ovs").join(format!("ovs.{source_lang}.src")));
        qe.target_terms = read_terms(&dir.join("ovs").join(format!("ovs.{target_lang}.tgt")));
        Ok(qe)
    }

    /// Whether any check is actually armed.
    pub fn is_empty(&self) -> bool {
        self.source_repeat.is_none() && self.target_repeat.is_none() && self.target_terms.is_empty()
    }

    /// Flags on a source/translation pair. Empty means nothing looked wrong.
    pub fn check(&self, source: &str, target: &str) -> Vec<Flag> {
        let mut out = Vec::new();
        if let Some(rx) = &self.target_repeat
            && let Ok(Some(m)) = rx.captures(target)
            && let Some(g) = m.get(1)
        {
            out.push(Flag::RepeatedTarget(g.as_str().to_string()));
        }
        if let Some(rx) = &self.source_repeat
            && let Ok(Some(m)) = rx.captures(source)
            && let Some(g) = m.get(1)
        {
            out.push(Flag::RepeatedSource(g.as_str().to_string()));
        }
        // A vulgar output term is only suspicious when the input has none: the
        // user is allowed to swear, and translating it faithfully is correct.
        if !self.target_terms.is_empty() && !self.has_term(source, &self.source_terms) {
            for w in words(target) {
                if self.target_terms.contains(&w) {
                    out.push(Flag::UnlicensedProfanity(w));
                    break;
                }
            }
        }
        out
    }

    fn has_term(&self, text: &str, terms: &BTreeSet<String>) -> bool {
        !terms.is_empty() && words(text).iter().any(|w| terms.contains(w))
    }
}

/// Lowercased alphanumeric words.
fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn read_terms(path: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_lowercase)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an estimator from the shipped regexes without needing the assets.
    fn qe() -> QualityEstimator {
        QualityEstimator {
            source_repeat: Some(Regex::new(r"([^ ].*?)( \1){1}").expect("source rx")),
            target_repeat: Some(Regex::new(r"([^ ].*?)( \1){3}").expect("target rx")),
            source_terms: ["poop".to_string()].into_iter().collect(),
            target_terms: ["merde".to_string(), "salope".to_string()]
                .into_iter()
                .collect(),
        }
    }

    /// The exact loop the shipped framework produced.
    #[test]
    fn a_looping_target_is_flagged() {
        let f = qe().check(
            "Avocado platypus they are my followers",
            "Sie sind meine Anhänger Anhänger Anhänger Anhänger Anhänger",
        );
        assert!(
            f.iter()
                .any(|f| matches!(f, Flag::RepeatedTarget(w) if w == "Anhänger")),
            "{f:?}"
        );
    }

    /// Three repeats is the threshold, so two must pass.
    #[test]
    fn a_couple_of_repeats_is_not_a_loop() {
        let f = qe().check("x", "très très bien");
        assert!(
            !f.iter().any(|f| matches!(f, Flag::RepeatedTarget(_))),
            "{f:?}"
        );
    }

    /// The shipped framework's own hallucination.
    #[test]
    fn profanity_with_nothing_in_the_source_is_flagged() {
        let f = qe().check("i love the summer", "J'adore l'été merde");
        assert!(
            f.iter()
                .any(|f| matches!(f, Flag::UnlicensedProfanity(w) if w == "merde")),
            "{f:?}"
        );
    }

    /// Translating profanity the user wrote is correct, not a defect.
    #[test]
    fn profanity_licensed_by_the_source_is_not_flagged() {
        let f = qe().check("poop", "merde");
        assert!(
            !f.iter().any(|f| matches!(f, Flag::UnlicensedProfanity(_))),
            "{f:?}"
        );
    }

    #[test]
    fn a_clean_translation_is_not_flagged() {
        assert!(qe().check("i love the summer", "J'adore l'été").is_empty());
    }
}
