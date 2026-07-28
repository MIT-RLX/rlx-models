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

//! Load weights, compile the recognizer graph, run a forward pass, CTC-decode.

use crate::recognition::{NUM_CLASSES, build_recognition_graph};
use anyhow::{Result, anyhow};
use rlx_core::weight_map::WeightMap;
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Codemap entries `>= SENTINEL` are the CTC blank / non-printing markers.
const SENTINEL: u32 = 0xFFFE;

pub struct Recognizer {
    weights_path: PathBuf,
    codemap: Vec<u32>, // class index -> Unicode codepoint (blank/sentinel >= 0xFFFE)
    blank: usize,
    device: Device,
    cache: Mutex<HashMap<usize, CompiledGraph>>, // compiled graph per line-width
}

impl Recognizer {
    pub fn load(weights: &Path, codemap: &Path, device: Device) -> Result<Self> {
        let codemap: Vec<u32> = std::fs::read_to_string(codemap)?
            .split_whitespace()
            .map(|s| s.parse::<u32>())
            .collect::<std::result::Result<_, _>>()?;
        if codemap.len() != NUM_CLASSES {
            return Err(anyhow!(
                "codemap has {} entries, expected {NUM_CLASSES}",
                codemap.len()
            ));
        }
        let blank = codemap
            .iter()
            .position(|&v| v >= SENTINEL)
            .unwrap_or(codemap.len() - 1);
        Ok(Self {
            weights_path: weights.to_path_buf(),
            codemap,
            blank,
            device,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Recognize one line. `luma` is row-major `[32, width]` in `[0,1]` (background
    /// high, ink low); `width` must be a multiple of 4. Returns the raw CTC text
    /// (before any rescoring correction).
    pub fn recognize(&self, luma: &[f32], width: usize) -> Result<String> {
        let logits = self.forward_logits(luma, width)?;
        let seq = logits.len() / NUM_CLASSES;
        Ok(self.ctc_greedy(&logits, seq))
    }

    /// Raw `[seq * 439]` logits (row-major) — used by parity tests.
    pub fn forward_logits(&self, luma: &[f32], width: usize) -> Result<Vec<f32>> {
        if luma.len() != 32 * width {
            return Err(anyhow!("luma len {} != 32*{width}", luma.len()));
        }
        let mut cache = self.cache.lock().map_err(|_| anyhow!("lock poisoned"))?;
        if let std::collections::hash_map::Entry::Vacant(slot) = cache.entry(width) {
            let path_str = self
                .weights_path
                .to_str()
                .ok_or_else(|| anyhow!("weights path is not valid UTF-8"))?;
            let mut wm = WeightMap::from_file(path_str)?;
            let (graph, params) = build_recognition_graph(&mut wm, 1, width)?;
            slot.insert(crate::compile::compile_encoder(
                graph,
                params,
                self.device,
                false,
            ));
        }
        cache
            .get_mut(&width)
            .unwrap()
            .run(&[("image", luma)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("recognizer returned no output"))
    }

    /// Recognize with correction: CTC beam N-best, each rescored, best wins.
    pub fn recognize_with_rescorer(
        &self,
        luma: &[f32],
        width: usize,
        rescorer: &crate::rescore::Rescorer,
        beam: usize,
    ) -> anyhow::Result<String> {
        let mut logits = self.forward_logits(luma, width)?;
        let seq = logits.len() / NUM_CLASSES;
        for row in logits.chunks_mut(NUM_CLASSES) {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = row.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
            for x in row.iter_mut() {
                *x = (*x - m) - lse;
            }
        }
        let cands = crate::beam::ctc_beam_nbest(&logits, seq, NUM_CLASSES, self.blank, beam);
        let dbg = crate::env::rescore_debug();
        let mut best: Option<(f32, String)> = None;
        for (labels, rec_score) in cands.iter() {
            let s = self.labels_to_string(labels);
            let rescore_s = rescorer.score(&s);
            let total = rec_score + rescore_s;
            if dbg {
                eprintln!(
                    "  cand {s:?}  rec={rec_score:.3} rescore={rescore_s:.3} total={total:.3}"
                );
            }
            if best.as_ref().is_none_or(|(b, _)| total > *b) {
                best = Some((total, s));
            }
        }
        Ok(best.map(|(_, s)| s).unwrap_or_default())
    }

    /// Append the printable character for class `idx` (skips blank/sentinel classes).
    fn push_class(&self, out: &mut String, idx: usize) {
        if idx == self.blank {
            return;
        }
        let cp = self.codemap[idx];
        if cp < SENTINEL {
            if let Some(ch) = char::from_u32(cp) {
                out.push(ch);
            }
        }
    }

    fn labels_to_string(&self, labels: &[usize]) -> String {
        let mut out = String::new();
        for &l in labels {
            self.push_class(&mut out, l);
        }
        out
    }

    fn ctc_greedy(&self, logits: &[f32], seq: usize) -> String {
        let mut out = String::new();
        let mut prev = usize::MAX;
        for t in 0..seq {
            let row = &logits[t * NUM_CLASSES..(t + 1) * NUM_CLASSES];
            let mut best = 0usize;
            let mut best_v = row[0];
            for (c, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            if best != prev {
                self.push_class(&mut out, best);
            }
            prev = best;
        }
        out
    }
}
