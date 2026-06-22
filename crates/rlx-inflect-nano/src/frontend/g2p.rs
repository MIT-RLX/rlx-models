//! g2p_en port: the OOV grapheme→phoneme seq2seq (single-layer GRU encoder +
//! decoder, greedy ≤20 steps) plus the `__call__` decision flow (homograph via
//! POS, nltk-cmudict, else neural predict). Ported from `g2p_en/g2p.py`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use super::cmudict::NltkDict;
use super::pos::PerceptronTagger;
use crate::ops::sigmoid_;
use crate::weights::Weights;

/// Homograph table: headword → (pron-if-POS-matches, fallback-pron, POS prefix).
type Homographs = HashMap<String, (Vec<String>, Vec<String>, String)>;

const GRAPHEMES_EXTRA: usize = 3; // <pad>,<unk>,</s> then a..z
const PHONEMES: [&str; 74] = [
    "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0",
    "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y",
    "Z", "ZH",
];

struct Gru {
    w_ih: Vec<f32>, // [3H, in]
    w_hh: Vec<f32>, // [3H, H]
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    hidden: usize,
    in_dim: usize,
}

impl Gru {
    fn cell(&self, x: &[f32], h: &[f32]) -> Vec<f32> {
        let hh = self.hidden;
        let mut rzn_ih = self.b_ih.clone();
        let mut rzn_hh = self.b_hh.clone();
        for o in 0..3 * hh {
            let mut a = 0.0;
            let base = o * self.in_dim;
            for j in 0..self.in_dim {
                a += x[j] * self.w_ih[base + j];
            }
            rzn_ih[o] += a;
            let mut c = 0.0;
            let baseh = o * hh;
            for j in 0..hh {
                c += h[j] * self.w_hh[baseh + j];
            }
            rzn_hh[o] += c;
        }
        let mut out = vec![0f32; hh];
        for j in 0..hh {
            let r = sigmoid_(rzn_ih[j] + rzn_hh[j]);
            let z = sigmoid_(rzn_ih[hh + j] + rzn_hh[hh + j]);
            let n = (rzn_ih[2 * hh + j] + r * rzn_hh[2 * hh + j]).tanh();
            out[j] = (1.0 - z) * n + z * h[j];
        }
        out
    }
}

pub struct G2p {
    enc: Gru,
    dec: Gru,
    enc_emb: Vec<f32>, // [29, E]
    dec_emb: Vec<f32>, // [74, E]
    fc_w: Vec<f32>,    // [74, H]
    fc_b: Vec<f32>,
    emb_dim: usize,
    hidden: usize,
    g2idx: HashMap<char, usize>,
    homographs: Homographs,
    cmu: NltkDict,
    tagger: PerceptronTagger,
}

impl G2p {
    pub fn load(dir: &Path) -> Result<Self> {
        let w = Weights::load(&dir.join("g2p_checkpoint.safetensors"))?;
        let hidden = w.shape("enc_w_hh")?[1]; // 256
        let emb_dim = w.shape("enc_emb")?[1]; // 256
        let enc = Gru {
            w_ih: w.vec1("enc_w_ih")?,
            w_hh: w.vec1("enc_w_hh")?,
            b_ih: w.vec1("enc_b_ih")?,
            b_hh: w.vec1("enc_b_hh")?,
            hidden,
            in_dim: emb_dim,
        };
        let dec = Gru {
            w_ih: w.vec1("dec_w_ih")?,
            w_hh: w.vec1("dec_w_hh")?,
            b_ih: w.vec1("dec_b_ih")?,
            b_hh: w.vec1("dec_b_hh")?,
            hidden,
            in_dim: emb_dim,
        };
        let mut g2idx = HashMap::new();
        for (i, c) in "abcdefghijklmnopqrstuvwxyz".chars().enumerate() {
            g2idx.insert(c, GRAPHEMES_EXTRA + i);
        }
        let homographs = load_homographs(&dir.join("homographs.en"))?;
        let cmu = NltkDict::load(&dir.join("g2p_cmudict.txt"))?;
        let tagger = PerceptronTagger::load(&dir.join("perceptron_tagger.json"))?;
        Ok(Self {
            enc,
            dec,
            enc_emb: w.vec1("enc_emb")?,
            dec_emb: w.vec1("dec_emb")?,
            fc_w: w.vec1("fc_w")?,
            fc_b: w.vec1("fc_b")?,
            emb_dim,
            hidden,
            g2idx,
            homographs,
            cmu,
            tagger,
        })
    }

    fn emb_row(table: &[f32], idx: usize, dim: usize) -> &[f32] {
        &table[idx * dim..(idx + 1) * dim]
    }

    /// Neural OOV prediction (`predict`): returns ARPABET phones.
    fn predict(&self, word: &str) -> Vec<String> {
        // encode
        let mut chars: Vec<usize> = word
            .chars()
            .map(|c| *self.g2idx.get(&c).unwrap_or(&1)) // <unk>=1
            .collect();
        chars.push(2); // </s>=2
        let mut h = vec![0f32; self.hidden];
        for &idx in &chars {
            let x = Self::emb_row(&self.enc_emb, idx, self.emb_dim);
            h = self.enc.cell(x, &h);
        }
        // decode (greedy, start token <s>=2)
        let mut dec_in = Self::emb_row(&self.dec_emb, 2, self.emb_dim).to_vec();
        let mut preds = Vec::new();
        for _ in 0..20 {
            h = self.dec.cell(&dec_in, &h);
            // logits = h @ fc_w.T + fc_b
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for o in 0..PHONEMES.len() {
                let base = o * self.hidden;
                let mut acc = self.fc_b[o];
                for j in 0..self.hidden {
                    acc += h[j] * self.fc_w[base + j];
                }
                if acc > best_v {
                    best_v = acc;
                    best = o;
                }
            }
            if best == 3 {
                break; // </s>
            }
            preds.push(PHONEMES[best].to_string());
            dec_in = Self::emb_row(&self.dec_emb, best, self.emb_dim).to_vec();
        }
        preds
    }

    /// `__call__` on a single word group: returns phones (no trailing space sep).
    pub fn call(&self, text: &str) -> Vec<String> {
        // preprocessing (numbers already expanded upstream; accents already
        // stripped by the bert normalizer → only filter + lower + i.e./e.g.).
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
        prons.pop(); // drop trailing separator (prons[:-1])
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
