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

//! rlx-inflect-nano's `G2p` — homographs by part of speech, CMUdict, and a
//! GRU encoder-decoder for the rest — its GRU the package's graphs
//! (`g2p_encode`, `g2p_decode`: one step each), the greedy search here.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rlx_program::{Tensor, run};

use super::cmudict::NltkDict;
use super::pos::PerceptronTagger;

type Homographs = HashMap<String, (Vec<String>, Vec<String>, String)>;

const GRAPHEMES_EXTRA: usize = 3; // <pad>,<unk>,</s> then a..z
const HIDDEN: usize = 256;
const PHONEMES: [&str; 74] = [
    "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0",
    "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y",
    "Z", "ZH",
];

pub struct G2p {
    g2idx: HashMap<char, usize>,
    homographs: Homographs,
    cmu: NltkDict,
    tagger: PerceptronTagger,
}

/// One GRU step of `function` on the token `id` from `h`: its outputs.
fn step(function: &str, id: usize, h: &[f32]) -> Vec<Tensor> {
    run(
        function,
        vec![
            Tensor::new("id", "i64", vec![1], (id as i64).to_le_bytes().to_vec()),
            Tensor::f32("h", vec![1, HIDDEN], h),
        ],
    )
    .expect("the G2P's graphs run")
}

impl G2p {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut g2idx = HashMap::new();
        for (i, c) in "abcdefghijklmnopqrstuvwxyz".chars().enumerate() {
            g2idx.insert(c, GRAPHEMES_EXTRA + i);
        }
        Ok(Self {
            g2idx,
            homographs: load_homographs(&dir.join("homographs.en"))?,
            cmu: NltkDict::load(&dir.join("g2p_cmudict.txt"))?,
            tagger: PerceptronTagger::load(&dir.join("perceptron_tagger.json"))?,
        })
    }

    /// The neural guess for a word no dictionary has: ARPABET phones.
    fn predict(&self, word: &str) -> Vec<String> {
        let mut chars: Vec<usize> = word
            .chars()
            .map(|c| *self.g2idx.get(&c).unwrap_or(&1)) // <unk>
            .collect();
        chars.push(2); // </s>
        let mut h = vec![0f32; HIDDEN];
        for idx in chars {
            h = step("g2p_encode", idx, &h)[0].to_f32();
        }
        // Greedy, from <s>.
        let mut prev = 2usize;
        let mut preds = Vec::new();
        for _ in 0..20 {
            let out = step("g2p_decode", prev, &h);
            h = out[0].to_f32();
            let logits = out[1].to_f32();
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (o, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = o;
                }
            }
            if best == 3 {
                break; // </s>
            }
            preds.push(PHONEMES[best].to_string());
            prev = best;
        }
        preds
    }

    /// A word group's phones, words apart by `" "` (rlx-inflect-nano's `call`).
    pub fn call(&self, text: &str) -> Vec<String> {
        let mut t: String = text
            .to_lowercase()
            .chars()
            .filter(|c| {
                *c == ' '
                    || c.is_ascii_lowercase()
                    || matches!(c, '\'' | '.' | ',' | '?' | '!' | '-')
            })
            .collect();
        t = t.replace("i.e.", "that is").replace("e.g.", "for example");
        let mut prons: Vec<String> = Vec::new();
        for word in t.split_whitespace() {
            let pron: Vec<String> = if !word.chars().any(|c| c.is_ascii_lowercase()) {
                vec![word.to_string()]
            } else if let Some((p1, p2, pos1)) = self.homographs.get(word) {
                let pos = self.tagger.tag_one(word);
                if pos.starts_with(pos1.as_str()) {
                    p1.clone()
                } else {
                    p2.clone()
                }
            } else if self.cmu.contains(word) {
                self.cmu.first(word).unwrap().clone()
            } else {
                self.predict(word)
            };
            prons.extend(pron);
            prons.push(" ".to_string());
        }
        prons.pop();
        prons
    }
}

fn load_homographs(path: &Path) -> Result<Homographs> {
    let text = std::fs::read_to_string(path)?;
    let mut map = HashMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.trim().split('|').collect();
        if parts.len() != 4 {
            continue;
        }
        let p1: Vec<String> = parts[1].split_whitespace().map(String::from).collect();
        let p2: Vec<String> = parts[2].split_whitespace().map(String::from).collect();
        map.insert(parts[0].to_lowercase(), (p1, p2, parts[3].to_string()));
    }
    Ok(map)
}
