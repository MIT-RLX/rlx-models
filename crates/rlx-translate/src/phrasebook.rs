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

//! `PhraseBookBlock` — the exact-match translation memory consulted before the
//! NMT.
//!
//! A phrasebook is a UTF-8 text file of `|||`-separated triples:
//!
//! ```text
//!   a n other|||monsieur Untel|||{"source":"dict","gender_alternatives":{...}}
//!   i love the summer|||j’adore l’été|||{"feature_name":"quality estimation",...}
//! ```
//!
//! A source may appear more than once — that is how gendered variants are
//! carried, each entry tagged `MALE`/`FEMALE` with a `default_gender` naming
//! which to prefer. Keys are lower-cased in the shipped files (4 exceptions in
//! 4210 records of `en-fr.mt_app.dict`), so lookup tries the literal key first
//! and then a case-folded one.
//!
//! A pair's config lists several files in `pb-file-list`, most specific first
//! ([`crate::quasar::Block::csv`]); [`Phrasebook::load_all`] merges them in that
//! order and earlier files win.

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Which gendered variant an entry provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
    /// No gender annotation.
    Unspecified,
}

/// One phrasebook record.
#[derive(Debug, Clone)]
pub struct Entry {
    pub source: String,
    pub target: String,
    /// The trailing JSON object, verbatim.
    pub meta: Value,
}

impl Entry {
    /// The variant this entry provides.
    pub fn gender(&self) -> Gender {
        match self
            .meta
            .get("gender_alternatives")
            .and_then(|g| g.get("spans"))
            .and_then(Value::as_array)
            .and_then(|s| s.first())
            .and_then(|s| s.get("gender"))
            .and_then(Value::as_str)
        {
            Some("MALE") => Gender::Male,
            Some("FEMALE") => Gender::Female,
            _ => Gender::Unspecified,
        }
    }

    /// The variant the record says to prefer when several exist.
    pub fn default_gender(&self) -> Gender {
        match self
            .meta
            .get("gender_alternatives")
            .and_then(|g| g.get("spans"))
            .and_then(Value::as_array)
            .and_then(|s| s.first())
            .and_then(|s| s.get("default_gender"))
            .and_then(Value::as_str)
        {
            Some("MALE") => Gender::Male,
            Some("FEMALE") => Gender::Female,
            _ => Gender::Unspecified,
        }
    }

    /// True when this entry is the one to use absent a caller preference.
    pub fn is_default(&self) -> bool {
        let g = self.gender();
        g == Gender::Unspecified || g == self.default_gender()
    }
}

/// A loaded phrasebook.
#[derive(Debug, Default)]
pub struct Phrasebook {
    by_key: BTreeMap<String, Vec<Entry>>,
    records: usize,
}

/// Field separator in a phrasebook file.
pub const SEP: &str = "|||";

impl Phrasebook {
    /// Parses one phrasebook's text.
    ///
    /// Malformed lines are counted and skipped rather than failing the load:
    /// the framework logs "Phrasebook contains invalid record" and carries on,
    /// and a single bad line should not cost a whole language pair.
    pub fn parse(text: &str) -> (Self, usize) {
        let mut pb = Self::default();
        let mut bad = 0usize;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.splitn(3, SEP);
            let (Some(source), Some(target)) = (parts.next(), parts.next()) else {
                bad += 1;
                continue;
            };
            if source.is_empty() {
                bad += 1;
                continue;
            }
            let meta = parts
                .next()
                .and_then(|m| serde_json::from_str(m).ok())
                .unwrap_or(Value::Null);
            pb.insert(Entry {
                source: source.to_string(),
                target: target.to_string(),
                meta,
            });
        }
        (pb, bad)
    }

    fn insert(&mut self, entry: Entry) {
        self.records += 1;
        self.by_key
            .entry(lookup_key(&entry.source))
            .or_default()
            .push(entry);
    }

    /// Loads one phrasebook file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading phrasebook {}", path.display()))?;
        Ok(Self::parse(&text).0)
    }

    /// Loads and merges several phrasebooks. Earlier files win, matching the
    /// most-specific-first order of `pb-file-list`.
    pub fn load_all<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut out = Self::default();
        let mut loaded = 0usize;
        for p in paths {
            let pb = Self::load(p)?;
            loaded += 1;
            for (key, entries) in pb.by_key {
                // `or_insert_with` keeps the earlier file's entries.
                out.by_key.entry(key).or_insert(entries);
            }
            out.records += pb.records;
        }
        if loaded == 0 {
            return Err(anyhow!("no phrasebook files given"));
        }
        Ok(out)
    }

    /// Every entry for `source`, in file order.
    pub fn lookup(&self, source: &str) -> &[Entry] {
        self.by_key
            .get(&lookup_key(source))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// The translation to use for `source`, preferring the default gendered
    /// variant. `None` when the phrasebook has no entry — the caller then falls
    /// through to the NMT, exactly as the pipeline's graph does.
    pub fn translate(&self, source: &str) -> Option<&str> {
        let entries = self.lookup(source);
        entries
            .iter()
            .find(|e| e.is_default())
            .or_else(|| entries.first())
            .map(|e| e.target.as_str())
    }

    /// Translation for a specific gendered variant, when one exists.
    pub fn translate_gendered(&self, source: &str, gender: Gender) -> Option<&str> {
        self.lookup(source)
            .iter()
            .find(|e| e.gender() == gender)
            .map(|e| e.target.as_str())
    }

    /// Distinct sources.
    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    /// Total records, including repeated sources.
    pub fn records(&self) -> usize {
        self.records
    }
}

/// Lookup normalisation: trim, then case-fold. Shipped keys are lower-cased.
fn lookup_key(s: &str) -> String {
    s.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = concat!(
        "a n other|||monsieur Untel|||{\"gender_alternatives\":{\"spans\":[{\"gender\":\"MALE\",\"default_gender\":\"MALE\"}]}}\n",
        "a n other|||madame Unetelle|||{\"gender_alternatives\":{\"spans\":[{\"gender\":\"FEMALE\",\"default_gender\":\"MALE\"}]}}\n",
        "i love the summer|||j\u{2019}adore l\u{2019}\u{e9}t\u{e9}|||{\"source\":\"user\"}\n",
    );

    #[test]
    fn parses_triples_and_indexes_by_source() {
        let (pb, bad) = Phrasebook::parse(SAMPLE);
        assert_eq!(bad, 0);
        assert_eq!(pb.records(), 3);
        assert_eq!(pb.len(), 2, "two distinct sources");
        assert_eq!(pb.lookup("a n other").len(), 2);
    }

    #[test]
    fn lookup_is_case_and_whitespace_insensitive() {
        let (pb, _) = Phrasebook::parse(SAMPLE);
        assert_eq!(pb.translate("I Love The Summer"), Some("j’adore l’été"));
        assert_eq!(pb.translate("  i love the summer  "), Some("j’adore l’été"));
        assert_eq!(pb.translate("not present"), None);
    }

    #[test]
    fn default_gender_wins_when_variants_exist() {
        let (pb, _) = Phrasebook::parse(SAMPLE);
        // default_gender is MALE, so the male form is returned.
        assert_eq!(pb.translate("a n other"), Some("monsieur Untel"));
        assert_eq!(
            pb.translate_gendered("a n other", Gender::Female),
            Some("madame Unetelle")
        );
        assert_eq!(
            pb.translate_gendered("a n other", Gender::Male),
            Some("monsieur Untel")
        );
    }

    #[test]
    fn entries_without_metadata_still_load() {
        let (pb, bad) = Phrasebook::parse("hello|||bonjour\n");
        assert_eq!(bad, 0);
        assert_eq!(pb.translate("hello"), Some("bonjour"));
    }

    #[test]
    fn malformed_lines_are_counted_not_fatal() {
        let (pb, bad) = Phrasebook::parse("good|||bon|||{}\nnosep\n|||empty source|||{}\n");
        assert_eq!(bad, 2);
        assert_eq!(pb.translate("good"), Some("bon"));
    }

    #[test]
    fn targets_containing_the_separator_survive() {
        // splitn(3) keeps everything after the second separator as metadata,
        // so a target must never be split further.
        let (pb, _) = Phrasebook::parse("k|||a|||b|||c\n");
        assert_eq!(pb.translate("k"), Some("a"));
    }

    #[test]
    fn earlier_files_win_when_merging() {
        let dir = std::env::temp_dir().join(format!("rlx-tr-pb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let a = dir.join("a.dict");
        let b = dir.join("b.dict");
        std::fs::write(&a, "hello|||salut|||{}\n").expect("write a");
        std::fs::write(&b, "hello|||bonjour|||{}\nbye|||adieu|||{}\n").expect("write b");
        let pb = Phrasebook::load_all(&[&a, &b]).expect("loads");
        assert_eq!(pb.translate("hello"), Some("salut"), "first file wins");
        assert_eq!(pb.translate("bye"), Some("adieu"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
