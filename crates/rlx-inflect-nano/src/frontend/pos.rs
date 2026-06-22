//! nltk averaged-perceptron POS tagger (the `_eng` model). Ported faithfully —
//! used by g2p_en only to resolve homographs.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct TaggerJson {
    weights: HashMap<String, HashMap<String, f64>>,
    tagdict: HashMap<String, String>,
    classes: Vec<String>,
}

pub struct PerceptronTagger {
    weights: HashMap<String, HashMap<String, f64>>,
    tagdict: HashMap<String, String>,
    classes: Vec<String>,
}

fn normalize(word: &str) -> String {
    let bytes = word.as_bytes();
    if word.contains('-') && bytes.first() != Some(&b'-') {
        "!HYPHEN".to_string()
    } else if word.len() == 4 && word.chars().all(|c| c.is_ascii_digit()) {
        "!YEAR".to_string()
    } else if word.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        "!DIGITS".to_string()
    } else {
        word.to_lowercase()
    }
}

fn suffix3(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let start = chars.len().saturating_sub(3);
    chars[start..].iter().collect()
}

fn pref1(s: &str) -> String {
    s.chars().next().map(|c| c.to_string()).unwrap_or_default()
}

impl PerceptronTagger {
    pub fn load(path: &Path) -> Result<Self> {
        let j: TaggerJson = serde_json::from_str(
            &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
        )?;
        Ok(Self {
            weights: j.weights,
            tagdict: j.tagdict,
            classes: j.classes,
        })
    }

    fn predict(&self, features: &[(String, i32)]) -> String {
        let mut scores: HashMap<&str, f64> = HashMap::new();
        for (feat, value) in features {
            if *value == 0 {
                continue;
            }
            let Some(tw) = self.weights.get(feat) else {
                continue;
            };
            for (label, w) in tw {
                *scores.entry(label.as_str()).or_insert(0.0) += *value as f64 * *w;
            }
        }
        // argmax with tie-break on the larger label string (Python max key=(score,label)).
        self.classes
            .iter()
            .map(|c| (scores.get(c.as_str()).copied().unwrap_or(0.0), c))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(b.1)))
            .map(|(_, c)| c.clone())
            .unwrap_or_default()
    }

    /// Tag a single token (sufficient for g2p_en's per-word homograph use).
    pub fn tag_one(&self, word: &str) -> String {
        if let Some(t) = self.tagdict.get(word) {
            return t.clone();
        }
        // context = START + [normalize(word)] + END, i shifted by len(START)=2.
        let start = ["-START-", "-START2-"];
        let end = ["-END-", "-END2-"];
        let ctx_word = normalize(word);
        let context: Vec<&str> = vec![start[0], start[1], &ctx_word, end[0], end[1]];
        let (prev, prev2) = ("-START-", "-START2-");
        let i = 2usize; // 0 + len(START)
        let mut f: Vec<(String, i32)> = Vec::new();
        let mut add = |name: &str, args: &[&str]| {
            let mut key = String::from(name);
            for a in args {
                key.push(' ');
                key.push_str(a);
            }
            f.push((key, 1));
        };
        add("bias", &[]);
        add("i suffix", &[&suffix3(word)]);
        add("i pref1", &[&pref1(word)]);
        add("i-1 tag", &[prev]);
        add("i-2 tag", &[prev2]);
        add("i tag+i-2 tag", &[prev, prev2]);
        add("i word", &[context[i]]);
        add("i-1 tag+i word", &[prev, context[i]]);
        add("i-1 word", &[context[i - 1]]);
        add("i-1 suffix", &[&suffix3(context[i - 1])]);
        add("i-2 word", &[context[i - 2]]);
        add("i+1 word", &[context[i + 1]]);
        add("i+1 suffix", &[&suffix3(context[i + 1])]);
        add("i+2 word", &[context[i + 2]]);
        self.predict(&f)
    }
}
