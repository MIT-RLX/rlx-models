//! CMUdict loaders. Two dictionaries are consulted (mirroring the Python):
//!  - the repo `cmudict.rep` (checked first in `grapheme_to_phoneme`, keyed by
//!    UPPERCASE word, stored as syllables), and
//!  - the nltk `cmudict` g2p_en falls back to (lowercase key, first pron used).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

/// Repo `cmudict.rep`: UPPERCASE word → list of syllables (each a list of phones).
pub struct RepoDict {
    map: HashMap<String, Vec<Vec<String>>>,
}

impl RepoDict {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut map = HashMap::new();
        // read_dict starts at line index 49 (1-based) — skip the 48-line header.
        for line in text.lines().skip(48) {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let Some((word, rest)) = line.split_once("  ") else {
                continue;
            };
            let syllables: Vec<Vec<String>> = rest
                .split(" - ")
                .map(|syl| syl.split(' ').map(|s| s.to_string()).collect())
                .collect();
            map.insert(word.to_string(), syllables);
        }
        Ok(Self { map })
    }

    /// Lookup by the already-UPPERCASED word.
    pub fn get(&self, upper_word: &str) -> Option<&Vec<Vec<String>>> {
        self.map.get(upper_word)
    }
}

/// nltk cmudict (g2p_en's internal dict): lowercase word → list of prons.
pub struct NltkDict {
    map: HashMap<String, Vec<Vec<String>>>,
}

impl NltkDict {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut map = HashMap::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let Some((word, prons)) = line.split_once('\t') else {
                continue;
            };
            let prons: Vec<Vec<String>> = prons
                .split(" | ")
                .map(|p| p.split(' ').map(|s| s.to_string()).collect())
                .collect();
            map.insert(word.to_string(), prons);
        }
        Ok(Self { map })
    }

    pub fn contains(&self, word: &str) -> bool {
        self.map.contains_key(word)
    }

    /// First pronunciation (`self.cmu[word][0]`).
    pub fn first(&self, word: &str) -> Option<&Vec<String>> {
        self.map.get(word).and_then(|v| v.first())
    }
}
