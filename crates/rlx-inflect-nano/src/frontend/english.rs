//! `grapheme_to_phoneme` + the full text→ids pipeline. Mirrors
//! `tiny_tts/text/english.py` (BERT word grouping → repo CMUdict → g2p_en).

use std::path::Path;

use anyhow::Result;
use once_cell::sync::Lazy;
use std::collections::HashSet;

use super::clean::clean_tinytts_text;
use super::cmudict::RepoDict;
use super::g2p::G2p;
use super::normalize::normalize_text;
use super::symbols::{Symbols, insert_blanks};
use super::tokenize_bert::BertTokenizer;

static ARPA: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G", "AY0", "M", "AA0",
        "F", "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0", "EY1", "IH0", "K", "N", "W", "IY2",
        "T", "AA1", "ER1", "EH2", "OY0", "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2", "AA2", "ER",
        "AW0", "UW0", "R", "OW1", "EH1", "ZH", "AE0", "IH2", "IH", "Y", "JH", "P", "AY1", "EY0",
        "OY2", "TH", "HH", "D", "ER0", "CH", "AO1", "AE1", "AO2", "OY1", "AY2", "IH1", "OW0", "L",
        "SH",
    ]
    .into_iter()
    .collect()
});

/// `parse_phoneme`: trailing digit → tone = digit+1, phoneme lowercased.
fn parse_phoneme(phn: &str) -> (String, i64) {
    let bytes = phn.as_bytes();
    if let Some(&last) = bytes.last() {
        if last.is_ascii_digit() {
            let tone = (last - b'0') as i64 + 1;
            return (phn[..phn.len() - 1].to_lowercase(), tone);
        }
    }
    (phn.to_lowercase(), 0)
}

fn parse_syllables(sylls: &[Vec<String>]) -> (Vec<String>, Vec<i64>) {
    let mut phones = Vec::new();
    let mut tones = Vec::new();
    for syl in sylls {
        for phn in syl {
            let (p, t) = parse_phoneme(phn);
            phones.push(p);
            tones.push(t);
        }
    }
    (phones, tones)
}

pub struct English {
    bert: BertTokenizer,
    repo: RepoDict,
    g2p: G2p,
    symbols: Symbols,
}

impl English {
    pub fn load(frontend_dir: &Path) -> Result<Self> {
        Ok(Self {
            bert: BertTokenizer::load(&frontend_dir.join("bert/tokenizer.json"))?,
            repo: RepoDict::load(&frontend_dir.join("cmudict_rep.txt"))?,
            g2p: G2p::load(frontend_dir)?,
            symbols: Symbols::load(&frontend_dir.join("symbols.json"))?,
        })
    }

    /// `grapheme_to_phoneme(text)` with `pad_start_end=True`. Returns (phones, tones).
    pub fn grapheme_to_phoneme(&self, text: &str) -> Result<(Vec<String>, Vec<i64>)> {
        let tokens = self.bert.tokenize(text)?;
        // group wordpiece continuations (## …) back into words
        let mut groups: Vec<Vec<String>> = Vec::new();
        for t in tokens {
            if !t.starts_with('#') {
                groups.push(vec![t]);
            } else if let Some(g) = groups.last_mut() {
                g.push(t.replace('#', ""));
            }
        }

        let mut phones: Vec<String> = Vec::new();
        let mut tones: Vec<i64> = Vec::new();
        for group in &groups {
            let w: String = group.concat();
            if let Some(sylls) = self.repo.get(&w.to_uppercase()) {
                let (phns, tns) = parse_syllables(sylls);
                phones.extend(phns);
                tones.extend(tns);
            } else {
                for ph in self.g2p.call(&w) {
                    if ph == " " {
                        continue;
                    }
                    if ARPA.contains(ph.as_str()) {
                        let (p, t) = parse_phoneme(&ph);
                        phones.push(p);
                        tones.push(t);
                    } else {
                        phones.push(ph);
                        tones.push(0);
                    }
                }
            }
        }

        // map_phoneme + pad
        let mut mapped: Vec<String> = phones.iter().map(|p| self.symbols.map_phoneme(p)).collect();
        mapped.insert(0, "_".to_string());
        mapped.push("_".to_string());
        tones.insert(0, 0);
        tones.push(0);
        Ok((mapped, tones))
    }

    /// Full pipeline: raw text → (phone_ids, tone_ids, lang_ids) with blanks inserted.
    pub fn text_to_ids(
        &self,
        text: &str,
        add_blank: bool,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let cleaned = clean_tinytts_text(text);
        let normalized = normalize_text(&cleaned);
        let (phones, tones) = self.grapheme_to_phoneme(&normalized)?;
        let (mut p, mut t, mut l) = self.symbols.phonemes_to_ids(&phones, &tones, "EN");
        if add_blank {
            p = insert_blanks(&p);
            t = insert_blanks(&t);
            l = insert_blanks(&l);
        }
        Ok((p, t, l))
    }
}
