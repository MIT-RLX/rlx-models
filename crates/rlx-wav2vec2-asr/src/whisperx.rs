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

//! WhisperX 3.8.6's `whisperx.align` (`alignment.py`) for space-separated
//! languages, step for step: character cleaning against the model
//! dictionary (unknown characters become a wildcard column), the CTC trellis
//! and backtrack in float32 (the trellis's blank `cumsum` accumulates in
//! double, as torch's CPU `cumsum` does), `merge_repeats`, character
//! timestamps with Python's correctly-rounded `round(x, 3)`, Punkt sentence
//! spans, word assembly with pandas' NaN-skipping `min`/`max`/`mean`
//! (numpy's pairwise sum; `round` on the mean is numpy's), `nearest`
//! interpolation of unaligned words, and the final `groupby(["start",
//! "end"])` that sorts and merges sentences.

use crate::punkt::Punkt;
use anyhow::Result;
use serde::Serialize;

pub const SAMPLE_RATE: usize = 16_000;

/// One ASR segment as `transcribe` returns it.
#[derive(Clone, Debug)]
pub struct AsrSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub avg_logprob: Option<f64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Word {
    pub word: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Segment {
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub text: String,
    pub words: Vec<Word>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprob: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Aligned {
    pub segments: Vec<Segment>,
    pub word_segments: Vec<Word>,
}

/// Python `round(x, 3)` on a float: the double nearest the correctly
/// rounded 3-decimal value.
pub fn py_round3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap()
}

/// numpy `round(x, 3)`: `rint(x * 1000) / 1000`.
pub fn np_round3(x: f64) -> f64 {
    (x * 1000.0).round_ties_even() / 1000.0
}

/// numpy's pairwise `add.reduce` over a contiguous float64 array.
pub fn np_pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut s = 0.0f64;
        for &v in a {
            s += v;
        }
        s
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let n2 = (n / 2) - ((n / 2) % 8);
        np_pairwise_sum(&a[..n2]) + np_pairwise_sum(&a[n2..])
    }
}

/// pandas `Series.mean()` (no bottleneck): NaNs filled with 0 and summed
/// pairwise, divided by the non-NaN count.
fn pd_mean(v: &[Option<f64>]) -> Option<f64> {
    let cnt = v.iter().filter(|x| x.is_some()).count();
    if cnt == 0 {
        return None;
    }
    let filled: Vec<f64> = v.iter().map(|x| x.unwrap_or(0.0)).collect();
    Some(np_pairwise_sum(&filled) / cnt as f64)
}

fn pd_min(v: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    v.flatten()
        .fold(None, |m, x| Some(m.map_or(x, |m: f64| m.min(x))))
}
fn pd_max(v: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    v.flatten()
        .fold(None, |m, x| Some(m.map_or(x, |m: f64| m.max(x))))
}

/// `interpolate_nans(x, "nearest")`: scipy-nearest (ties to the lower
/// index) between valid points when more than one is valid, then
/// `ffill().bfill()`.
pub fn interpolate_nearest(x: &[Option<f64>]) -> Vec<Option<f64>> {
    let valid: Vec<usize> = (0..x.len()).filter(|&i| x[i].is_some()).collect();
    let mut y = x.to_vec();
    if valid.len() > 1 {
        for i in valid[0]..=*valid.last().unwrap() {
            if y[i].is_none() {
                let p = *valid.iter().rev().find(|&&v| v < i).unwrap();
                let n = *valid.iter().find(|&&v| v > i).unwrap();
                y[i] = if i - p <= n - i { x[p] } else { x[n] };
            }
        }
    }
    // ffill then bfill
    let mut last = None;
    for v in y.iter_mut() {
        if v.is_some() {
            last = *v;
        } else {
            *v = last;
        }
    }
    let mut next = None;
    for v in y.iter_mut().rev() {
        if v.is_some() {
            next = *v;
        } else {
            *v = next;
        }
    }
    y
}

/// The CTC trellis (`get_trellis`), `[frames + 1][tokens + 1]`, float32.
fn trellis(em: &[f32], v: usize, frames: usize, tokens: &[usize], blank: usize) -> Vec<Vec<f32>> {
    let nt = tokens.len();
    let mut tr = vec![vec![0f32; nt + 1]; frames + 1];
    let mut acc = 0f64;
    for t in 0..frames {
        acc += em[t * v + blank] as f64;
        tr[t + 1][0] = acc as f32;
    }
    for j in 1..=nt {
        tr[0][j] = f32::NEG_INFINITY;
    }
    for t in frames + 1 - nt.min(frames + 1)..=frames {
        if nt > 0 {
            tr[t][0] = f32::INFINITY;
        }
    }
    for t in 0..frames {
        let b = em[t * v + blank];
        for j in 1..=nt {
            let stay = tr[t][j] + b;
            let change = tr[t][j - 1] + em[t * v + tokens[j - 1]];
            tr[t + 1][j] = stay.max(change);
        }
    }
    tr
}

struct Point {
    token: usize,
    time: usize,
    score: f64,
}

/// `backtrack`: `None` when the path never reaches the first token.
fn backtrack(
    tr: &[Vec<f32>],
    em: &[f32],
    v: usize,
    tokens: &[usize],
    blank: usize,
) -> Option<Vec<Point>> {
    let mut j = tokens.len();
    // torch.argmax: first maximal index (NaN-free here).
    let mut t_start = 0;
    for t in 1..tr.len() {
        if tr[t][j] > tr[t_start][j] {
            t_start = t;
        }
    }
    let mut path = Vec::new();
    let mut t = t_start;
    let mut done = false;
    while t > 0 {
        let stayed = tr[t - 1][j] + em[(t - 1) * v + blank];
        let changed = tr[t - 1][j - 1] + em[(t - 1) * v + tokens[j - 1]];
        let id = if changed > stayed {
            tokens[j - 1]
        } else {
            blank
        };
        path.push(Point {
            token: j - 1,
            time: t - 1,
            score: em[(t - 1) * v + id].exp() as f64,
        });
        if changed > stayed {
            j -= 1;
            if j == 0 {
                done = true;
                break;
            }
        }
        t -= 1;
    }
    if !done {
        return None;
    }
    path.reverse();
    Some(path)
}

struct CharSeg {
    start: usize,
    end: usize,
    score: f64,
}

fn merge_repeats(path: &[Point]) -> Vec<CharSeg> {
    let mut out = Vec::new();
    let (mut i1, mut i2) = (0, 0);
    while i1 < path.len() {
        while i2 < path.len() && path[i1].token == path[i2].token {
            i2 += 1;
        }
        let mut s = 0f64;
        for p in &path[i1..i2] {
            s += p.score;
        }
        out.push(CharSeg {
            start: path[i1].time,
            end: path[i2 - 1].time + 1,
            score: s / (i2 - i1) as f64,
        });
        i1 = i2;
    }
    out
}

/// Python `str.lower()` of one character as a string.
fn lower(c: char) -> String {
    c.to_lowercase().collect()
}

/// Align `segments` against `audio` (16 kHz mono, as `load_audio` returns
/// it). `labels` is the model dictionary (index = emission column, label 0
/// the blank); `emit(slice)` returns `(log_softmax emissions, frames)`.
pub fn align(
    segments: &[AsrSegment],
    audio: &[f32],
    labels: &[char],
    mut emit: impl FnMut(&[f32]) -> Result<(Vec<f32>, usize)>,
) -> Result<Aligned> {
    let punkt = Punkt::english();
    let dict: Vec<String> = labels.iter().map(|&c| lower(c)).collect();
    let code = |s: &str| dict.iter().position(|d| d == s);
    let blank = dict
        .iter()
        .position(|d| d == "[pad]" || d == "<pad>")
        .unwrap_or(0);
    let max_duration = audio.len() as f64 / SAMPLE_RATE as f64;
    let mut out_segments = Vec::new();

    for seg in segments {
        let text = &seg.text;
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let num_leading = n - text.trim_start().chars().count();
        let num_trailing = n - text.trim_end().chars().count();
        let mut clean_char: Vec<String> = Vec::new();
        let mut clean_cdx: Vec<usize> = Vec::new();
        for (cdx, &ch) in chars.iter().enumerate() {
            let c = lower(ch).replace(' ', "|");
            if cdx < num_leading || cdx + num_trailing > n - 1 {
                continue;
            }
            if code(&c).is_some() || (c != " " && c != "|") {
                clean_char.push(c);
                clean_cdx.push(cdx);
            }
        }
        let sentence_spans = punkt.span_tokenize(text);
        let unaligned = || Segment {
            start: Some(seg.start),
            end: Some(seg.end),
            text: text.clone(),
            words: Vec::new(),
            avg_logprob: seg.avg_logprob,
        };
        if clean_char.is_empty() || seg.start >= max_duration {
            out_segments.push(unaligned());
            continue;
        }
        let f1 = (seg.start * SAMPLE_RATE as f64) as usize;
        let f2 = ((seg.end * SAMPLE_RATE as f64) as usize).min(audio.len());
        let (mut em, frames) = emit(&audio[f1.min(f2)..f2])?;
        let mut v = labels.len();
        // Wildcard column for characters outside the dictionary.
        let has_wildcard = clean_char.iter().any(|c| code(c).is_none());
        if has_wildcard {
            let mut ext = Vec::with_capacity(frames * (v + 1));
            for t in 0..frames {
                let row = &em[t * v..(t + 1) * v];
                let m = row
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != blank)
                    .map(|(_, &x)| x)
                    .fold(f32::NEG_INFINITY, f32::max);
                ext.extend_from_slice(row);
                ext.push(m);
            }
            em = ext;
            v += 1;
        }
        let tokens: Vec<usize> = clean_char
            .iter()
            .map(|c| code(c).unwrap_or(v - 1))
            .collect();
        let tr = trellis(&em, v, frames, &tokens, blank);
        let Some(path) = backtrack(&tr, &em, v, &tokens, blank) else {
            out_segments.push(unaligned());
            continue;
        };
        let char_segs = merge_repeats(&path);
        let ratio = (seg.end - seg.start) / (tr.len() - 1) as f64;

        // Per-character rows (the DataFrame).
        struct Row {
            ch: char,
            start: Option<f64>,
            end: Option<f64>,
            score: Option<f64>,
            word: usize,
        }
        let mut rows = Vec::with_capacity(n);
        let mut word_idx = 0;
        for (cdx, &ch) in chars.iter().enumerate() {
            let (mut s, mut e, mut sc) = (None, None, None);
            if let Some(k) = clean_cdx.iter().position(|&c| c == cdx) {
                let cs = &char_segs[k];
                s = Some(py_round3(cs.start as f64 * ratio + seg.start));
                e = Some(py_round3(cs.end as f64 * ratio + seg.start));
                sc = Some(py_round3(cs.score));
            }
            rows.push(Row {
                ch,
                start: s,
                end: e,
                score: sc,
                word: word_idx,
            });
            if cdx == n - 1 || chars[cdx + 1] == ' ' {
                word_idx += 1;
            }
        }

        let mut subs: Vec<Segment> = Vec::new();
        for &(ss, se) in &sentence_spans {
            let curr: Vec<&Row> = rows
                .iter()
                .enumerate()
                .filter(|(i, _)| *i >= ss && *i <= se)
                .map(|(_, r)| r)
                .collect();
            let sentence_text: String = chars[ss.min(n)..se.min(n)].iter().collect();
            let s_start = pd_min(curr.iter().map(|r| r.start));
            let s_end = pd_max(curr.iter().filter(|r| r.ch != ' ').map(|r| r.end));
            let mut words = Vec::new();
            let mut seen = Vec::new();
            for r in &curr {
                if !seen.contains(&r.word) {
                    seen.push(r.word);
                }
            }
            for w in seen {
                let wc: Vec<&&Row> = curr.iter().filter(|r| r.word == w).collect();
                let wtext: String = wc
                    .iter()
                    .map(|r| r.ch)
                    .collect::<String>()
                    .trim()
                    .to_string();
                if wtext.is_empty() {
                    continue;
                }
                let wc: Vec<&&Row> = wc.into_iter().filter(|r| r.ch != ' ').collect();
                let start = pd_min(wc.iter().map(|r| r.start));
                let end = pd_max(wc.iter().map(|r| r.end));
                let scores: Vec<Option<f64>> = wc.iter().map(|r| r.score).collect();
                let score = pd_mean(&scores).map(np_round3);
                words.push(Word {
                    word: wtext,
                    start,
                    end,
                    score,
                });
            }
            if !words.is_empty() {
                let st: Vec<Option<f64>> = words.iter().map(|w| w.start).collect();
                if st.iter().any(Option::is_none) && st.iter().any(Option::is_some) {
                    let en: Vec<Option<f64>> = words.iter().map(|w| w.end).collect();
                    let (st, en) = (interpolate_nearest(&st), interpolate_nearest(&en));
                    for (i, w) in words.iter_mut().enumerate() {
                        if w.start.is_none() {
                            w.start = st[i];
                        }
                        if w.end.is_none() {
                            w.end = en[i];
                        }
                    }
                }
            }
            subs.push(Segment {
                start: s_start,
                end: s_end,
                text: sentence_text,
                words,
                avg_logprob: seg.avg_logprob,
            });
        }
        let st = interpolate_nearest(&subs.iter().map(|s| s.start).collect::<Vec<_>>());
        let en = interpolate_nearest(&subs.iter().map(|s| s.end).collect::<Vec<_>>());
        for (i, s) in subs.iter_mut().enumerate() {
            s.start = st[i];
            s.end = en[i];
        }
        // groupby(["start", "end"]): drop NaN keys, sort by key, join texts
        // with " ", concatenate words, keep the first avg_logprob.
        let mut keyed: Vec<(usize, Segment)> = subs
            .into_iter()
            .enumerate()
            .filter(|(_, s)| s.start.is_some() && s.end.is_some())
            .collect();
        keyed.sort_by(|a, b| {
            let ka = (a.1.start.unwrap(), a.1.end.unwrap());
            let kb = (b.1.start.unwrap(), b.1.end.unwrap());
            ka.partial_cmp(&kb).unwrap().then(a.0.cmp(&b.0))
        });
        let mut merged: Vec<Segment> = Vec::new();
        for (_, s) in keyed {
            match merged.last_mut() {
                Some(m) if m.start == s.start && m.end == s.end => {
                    m.text = format!("{} {}", m.text, s.text);
                    m.words.extend(s.words);
                }
                _ => merged.push(s),
            }
        }
        out_segments.extend(merged);
    }
    let word_segments = out_segments
        .iter()
        .flat_map(|s| s.words.iter().cloned())
        .collect();
    Ok(Aligned {
        segments: out_segments,
        word_segments,
    })
}
