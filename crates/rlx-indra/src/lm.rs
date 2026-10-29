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

//! Beam search with a token n-gram language model, for lines the recognizer
//! is unsure of: `score = log P_ctc + α·log P_lm + β·tokens`.
//!
//! The model is a backoff table (`lm*.safetensors`, from `lm*.npz`): per
//! history length `n`, sorted keys of seen `(history, token)` pairs with their
//! log-probabilities (`k{n}`, `v{n}`) and sorted history keys with backoff
//! weights (`hk{n}`, `hb{n}`). A key packs tokens `bits` bits each (0 = line
//! start, `t + 1` = token `t`).

use anyhow::{Context, Result, bail};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

const DEFAULT_BITS: u32 = 11;
const TAG_SHIFT: u32 = 61;

pub struct NgramLm {
    pub order: usize,
    pub vocab_size: usize,
    pub bits: u32,
    k: Vec<Vec<u64>>,
    v: Vec<Vec<f32>>,
    hk: Vec<Vec<u64>>,
    hb: Vec<Vec<f32>>,
    max_logp: f64,
}

fn scalar_i64(st: &SafeTensors<'_>, name: &str) -> Result<Option<i64>> {
    let Ok(t) = st.tensor(name) else {
        return Ok(None);
    };
    if t.dtype() != Dtype::I64 || t.data().len() != 8 {
        bail!("{name}: expected an i64 scalar");
    }
    Ok(Some(i64::from_le_bytes(t.data().try_into()?)))
}

fn vec_u64(st: &SafeTensors<'_>, name: &str) -> Result<Vec<u64>> {
    let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
    if t.dtype() != Dtype::U64 {
        bail!("{name}: expected u64");
    }
    Ok(t.data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect())
}

fn vec_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
    if t.dtype() != Dtype::F32 {
        bail!("{name}: expected f32");
    }
    Ok(t.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

impl NgramLm {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let order = scalar_i64(&st, "order")?.context("lm lacks order")? as usize;
        let vocab_size = scalar_i64(&st, "vocab_size")?.context("lm lacks vocab_size")? as usize;
        let bits = scalar_i64(&st, "bits")?.map_or(DEFAULT_BITS, |b| b as u32);
        if order as u32 * bits > TAG_SHIFT {
            bail!("order {order} × {bits} bits do not fit a key");
        }
        let (mut k, mut v, mut hk, mut hb) = (vec![], vec![], vec![], vec![]);
        for n in 0..order {
            k.push(vec_u64(&st, &format!("k{n}"))?);
            v.push(vec_f32(&st, &format!("v{n}"))?);
            hk.push(vec_u64(&st, &format!("hk{n}"))?);
            hb.push(vec_f32(&st, &format!("hb{n}"))?);
        }
        let top = v.iter().flatten().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let top = top.max(-(vocab_size as f64).ln());
        let bo: f64 = hb
            .iter()
            .filter(|h| !h.is_empty())
            .map(|h| h.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)).max(0.0) as f64)
            .sum();
        Ok(Self {
            order,
            vocab_size,
            bits,
            k,
            v,
            hk,
            hb,
            max_logp: top + bo,
        })
    }

    /// An upper bound of every [`Self::logp_key`] value.
    pub fn max_logp(&self) -> f64 {
        self.max_logp
    }

    fn find(keys: &[u64], key: u64) -> Option<usize> {
        keys.binary_search(&key).ok()
    }

    /// `log P(token | history)` for a packed key: the last `order - 1`
    /// history tokens, then the token. The value comes from the longest seen
    /// history that saw the token, plus the backoff weights of the longer seen
    /// histories; with none, all backoffs and a uniform floor.
    pub fn logp_key(&self, full: u64) -> f64 {
        let b = self.bits;
        let ct = full & ((1u64 << b) - 1);
        let hist = full >> b;
        let n_max = self.order;
        let mut hb = vec![0f64; n_max];
        let mut pv = vec![None; n_max];
        let mut m = 0;
        let mut seen = true;
        for n in 0..n_max {
            let mask = if n == 0 {
                0
            } else {
                (1u64 << (b as usize * n)) - 1
            };
            let hq = hist & mask;
            let h_hit = Self::find(&self.hk[n], hq);
            hb[n] = h_hit.map_or(0.0, |i| self.hb[n][i] as f64);
            if n > 0 {
                seen &= h_hit.is_some();
                if seen {
                    m = n;
                }
            }
            pv[n] = Self::find(&self.k[n], (hq << b) | ct).map(|i| self.v[n][i] as f64);
        }
        let chosen = (0..=m).rev().find(|&n| pv[n].is_some());
        match chosen {
            Some(c) => hb[c + 1..=m].iter().sum::<f64>() + pv[c].unwrap(),
            None => hb[..=m].iter().sum::<f64>() - (self.vocab_size as f64).ln(),
        }
    }
}

fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[derive(Clone)]
struct Beam {
    b: f64,
    nb: f64,
    s: f64,
    steps: Vec<usize>,
    hk: u64,
}

/// Insertion-ordered prefix → beam map (ties resolve by insertion order).
#[derive(Default)]
struct Beams {
    keys: Vec<Vec<u32>>,
    vals: Vec<Beam>,
    at: HashMap<Vec<u32>, usize>,
}

impl Beams {
    fn add(&mut self, prefix: &[u32], b: f64, nb: f64, s: f64, steps: &[usize], hk: u64) {
        if let Some(&i) = self.at.get(prefix) {
            let e = &mut self.vals[i];
            e.b = lse(e.b, b);
            e.nb = lse(e.nb, nb);
        } else {
            self.at.insert(prefix.to_vec(), self.keys.len());
            self.keys.push(prefix.to_vec());
            self.vals.push(Beam {
                b,
                nb,
                s,
                steps: steps.to_vec(),
                hk,
            });
        }
    }

    fn total(e: &Beam) -> f64 {
        lse(e.b, e.nb) + e.s
    }

    /// The `beam` best, stable by insertion order on ties.
    fn top(self, beam: usize) -> Beams {
        let mut idx: Vec<usize> = (0..self.keys.len()).collect();
        idx.sort_by(|&i, &j| Self::total(&self.vals[j]).total_cmp(&Self::total(&self.vals[i])));
        idx.truncate(beam);
        let mut out = Beams::default();
        for i in idx {
            let e = &self.vals[i];
            out.add(&self.keys[i], e.b, e.nb, e.s, &e.steps, e.hk);
        }
        out
    }
}

/// Search options (upstream defaults: α 0.5, β 1.5, beam 8).
#[derive(Clone, Copy, Debug)]
pub struct BeamParams {
    pub alpha: f64,
    pub beta: f64,
    pub beam: usize,
    pub blank_skip: f64,
    pub cand_floor: f64,
}

/// CTC prefix beam search over one line. `blank [T]`: blank log-probs;
/// `topv / topi [T, k]`: the k best classes per step and their log-probs.
/// Returns (token ids, the step where each token was first emitted).
pub fn beam_search(
    blank: &[f32],
    topv: &[f32],
    topi: &[u32],
    k: usize,
    lm: &NgramLm,
    p: BeamParams,
) -> (Vec<u32>, Vec<usize>) {
    let bits = lm.bits;
    let hmask = (1u64 << (bits as usize * (lm.order - 1))) - 1;
    let mut beams = Beams::default();
    beams.add(&[], 0.0, f64::NEG_INFINITY, 0.0, &[], 0);
    let log_skip = p.blank_skip.ln();
    let log_floor = p.cand_floor.ln();
    let reach = p.beta + p.alpha * lm.max_logp();
    for t in 0..blank.len() {
        let b0 = blank[t] as f64;
        if b0 > log_skip {
            for e in &mut beams.vals {
                e.b = lse(e.b, e.nb) + b0;
                e.nb = f64::NEG_INFINITY;
            }
            continue;
        }
        let cands: Vec<(u32, f64)> = (0..k)
            .map(|j| (topi[t * k + j], topv[t * k + j] as f64))
            .filter(|&(c, v)| c != 0 && v >= log_floor)
            .collect();
        let mut nxt = Beams::default();
        for (prefix, e) in beams.keys.iter().zip(&beams.vals) {
            nxt.add(
                prefix,
                lse(e.b, e.nb) + b0,
                f64::NEG_INFINITY,
                e.s,
                &e.steps,
                e.hk,
            );
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                if Some(c) == last {
                    nxt.add(prefix, f64::NEG_INFINITY, e.nb + lp, e.s, &e.steps, e.hk);
                }
            }
        }
        let floor = if beams.keys.len() >= p.beam {
            nxt.vals
                .iter()
                .map(Beams::total)
                .fold(f64::INFINITY, f64::min)
        } else {
            f64::NEG_INFINITY
        };
        let mut ext: Vec<(usize, u32, f64)> = Vec::new();
        for (bi, (prefix, e)) in beams.keys.iter().zip(&beams.vals).enumerate() {
            let tot = lse(e.b, e.nb);
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                let new_from = if Some(c) == last { e.b + lp } else { tot + lp };
                if new_from == f64::NEG_INFINITY {
                    continue;
                }
                let mut q = prefix.clone();
                q.push(c);
                if nxt.at.contains_key(&q) {
                    nxt.add(&q, f64::NEG_INFINITY, new_from, e.s, &e.steps, e.hk);
                } else if new_from + e.s + reach >= floor - 1e-9 {
                    ext.push((bi, c, new_from));
                }
            }
        }
        for (bi, c, new_from) in ext {
            let (prefix, e) = (&beams.keys[bi], &beams.vals[bi]);
            let lmp = lm.logp_key((e.hk << bits) | (c as u64 + 1));
            let mut q = prefix.clone();
            q.push(c);
            let mut steps = e.steps.clone();
            steps.push(t);
            let hk = ((e.hk << bits) | (c as u64 + 1)) & hmask;
            nxt.add(
                &q,
                f64::NEG_INFINITY,
                new_from,
                e.s + p.alpha * lmp + p.beta,
                &steps,
                hk,
            );
        }
        beams = nxt.top(p.beam);
    }
    let mut best = 0;
    for i in 1..beams.keys.len() {
        if Beams::total(&beams.vals[i]) > Beams::total(&beams.vals[best]) {
            best = i;
        }
    }
    (beams.keys[best].clone(), beams.vals[best].steps.clone())
}
