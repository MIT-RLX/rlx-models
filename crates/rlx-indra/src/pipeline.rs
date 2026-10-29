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

//! The page pipeline: deskew and orientation, line detection, script routing,
//! CTC line recognition (with an n-gram beam search for unsure lines), line
//! splitting, reading order and layout markdown.

use crate::graph::{self, Compiled, Weights};
use crate::imgproc::{
    Border, Gray, Interp, connected_components, resize, rot90, rotate, rotation_matrix,
};
use crate::layout::{self, Block, MarkdownExtras, round_to, round1, union};
use crate::lines::{self, LineBox, PAD_X, WIDTH_MULTIPLE, py_round};
use crate::lm::{BeamParams, NgramLm, beam_search};
use crate::order::{self, GRID_LABELS, Rect, class_id, median};
use crate::text::{Vocab, flip, has_rtl, is_alpha, is_rtl, nfc, squash_spaces};
use anyhow::{Context, Result, bail};
use rlx_runtime::Device;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

/// Median line height (input px) under which detection runs again, larger.
const ADAPTIVE_MIN_H: f32 = 9.0;
const ADAPTIVE_TARGET_H: f64 = 12.0;
const ADAPTIVE_MAX_SIDE: f64 = 2048.0;
/// Gaps between CTC character centres, in line heights.
const SPLIT_GAP: f64 = 3.0;
const CELL_GAP: f64 = 1.8;
const GUTTER_GAP: f64 = 1.6;
const GUTTER_EDGES: usize = 3;
const ORIENT_SIDE: usize = 768;
/// A turn is applied only above this probability.
const ORIENT_MIN: f32 = 0.95;
/// Orientation class: the page is set in vertical columns.
const VERTICAL: usize = 4;
/// A recognized line under this confidence gets no text.
const MIN_CONF: f64 = 0.5;
const LM_GATE: f64 = 0.9;
const LM_ALPHA: f64 = 0.5;
const LM_BETA: f64 = 1.5;
const LM_BEAM: usize = 8;
const LM_CAND_FLOOR: f64 = 3e-3;
const DEFAULT_GROUPS: [&str; 4] = ["base", "cjk", "rtl", "other"];
/// A line goes to one recognizer when the router is surer than this.
const ROUTE_SURE: f32 = 0.95;
const PICTURE_MIN: f32 = 0.5;
const PICTURE_MIN_AREA: f32 = 0.004;
const DESKEW_MAX: f64 = 45.0;
const DESKEW_MIN: f64 = 0.3;
const PREVIEW_SIDE: usize = 1600;
const SKEW_SIDE: usize = 800;
/// Lines per recognizer batch.
const REC_BATCH: usize = 64;
const MARKER_GAP: f32 = 2.5;

/// Options of [`Indra`].
#[derive(Clone, Debug)]
pub struct Options {
    pub device: Device,
    /// Detect again at a larger scale when the lines come out tiny.
    pub adaptive: bool,
    /// Tables from the order model's row / column / cell heads.
    pub table_heads: bool,
    /// Keep page headers and footers in the markdown.
    pub furniture: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            adaptive: true,
            table_heads: false,
            furniture: false,
        }
    }
}

/// A recognized line in reading order.
#[derive(Clone, Debug, Serialize)]
pub struct Line {
    #[serde(rename = "box")]
    pub bbox: [f64; 4],
    pub text: String,
    pub conf: f64,
}

/// One page's result.
#[derive(Clone, Debug, Serialize)]
pub struct Page {
    pub lines: Vec<Line>,
    pub markdown: Option<String>,
    pub blocks: Vec<Block>,
    /// Degrees counter-clockwise the input was turned; boxes are upright.
    pub rotation: i32,
    /// Degrees of small tilt undone after the turn.
    pub skew: f64,
    /// Set in vertical columns: read turned a quarter turn, boxes on the page as it is.
    pub vertical: bool,
    pub note: String,
    pub timings: std::collections::BTreeMap<String, f64>,
}

impl Page {
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A recognizer token: a class id of the base vocabulary, or a character of
/// a group recognizer; with its page x.
#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Id(u32),
    Ch(String),
}

type Toks = Vec<(Tok, f64)>;

/// (text, confidence, tokens) of one line.
#[derive(Clone, Debug)]
pub struct Reading {
    pub text: String,
    pub conf: f64,
    pub toks: Toks,
}

struct Recognizer {
    net: Compiled,
    vocab: Vocab,
    height: usize,
    lm: Option<(NgramLm, BeamParams, f64)>,
}

/// The Indra page reader.
pub struct Indra {
    det: Compiled,
    base: Recognizer,
    groups: Vec<(String, Recognizer)>,
    script: Option<Compiled>,
    script_groups: Vec<String>,
    orient: Option<Compiled>,
    order: Option<Compiled>,
    long_side: usize,
    height: usize,
    window: usize,
    opts: Options,
}

/// Compile batch of `n` lines: powers of two up to 8, then multiples of 8.
fn batch_bucket(n: usize) -> usize {
    if n <= 8 {
        n.next_power_of_two()
    } else {
        n.div_ceil(8) * 8
    }
}

/// Compile width of a batch padded to `w` columns (multiples of 32 so every
/// network divides it; coarser as lines get longer).
fn width_bucket(w: usize) -> usize {
    let step = match w {
        0..=256 => 32,
        257..=1024 => 64,
        1025..=2048 => 128,
        _ => 256,
    };
    w.div_ceil(step) * step
}

/// Keys of `meta.json` in file order (serde_json maps may sort them).
fn meta_key_order(raw: &str, prefix: &str) -> Vec<String> {
    let pat = format!("\"{prefix}");
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = raw[from..].find(&pat) {
        let start = from + i + 1;
        let end = start + raw[start..].find('"').unwrap_or(0);
        let key = &raw[start..end];
        if raw[end + 1..].trim_start().starts_with(':') && !out.iter().any(|k: &String| k == key) {
            out.push(key.to_string());
        }
        from = end + 1;
    }
    out
}

impl Recognizer {
    fn load(
        dir: &Path,
        net: &str,
        vocab: Vocab,
        lm: Option<(&str, &Value)>,
        device: Device,
    ) -> Result<Self> {
        let w = Weights::load(&dir.join(format!("{net}.safetensors")))?;
        let spec = graph::rec::RecSpec::from_weights(&w)?;
        if spec.classes != vocab.classes() {
            bail!(
                "{net}: {} classes, vocabulary has {}",
                spec.classes,
                vocab.classes()
            );
        }
        let lm = match lm {
            Some((file, params)) if dir.join(file).exists() => {
                let get = |k: &str, d: f64| params.get(k).and_then(Value::as_f64).unwrap_or(d);
                Some((
                    NgramLm::load(&dir.join(file))?,
                    BeamParams {
                        alpha: get("alpha", LM_ALPHA),
                        beta: get("beta", LM_BETA),
                        beam: LM_BEAM,
                        blank_skip: 0.999,
                        cand_floor: LM_CAND_FLOOR,
                    },
                    get("gate", LM_GATE),
                ))
            }
            _ => None,
        };
        Ok(Self {
            net: Compiled::new(w, device, |w, k| graph::rec::build(w, k[0], k[1])),
            vocab,
            height: spec.height,
            lm,
        })
    }
}

/// `[B, 1, H, W]` batch of right-padded ink maps.
fn pad_batch(crops: &[&Crop], rows: usize, batch: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0f32; batch * rows * width];
    for (b, c) in crops.iter().enumerate() {
        for y in 0..rows {
            out[(b * rows + y) * width..(b * rows + y) * width + c.width]
                .copy_from_slice(&c.ink[y * c.width..(y + 1) * c.width]);
        }
    }
    out
}

/// A recognizer input: ink map `[rows, width]`.
#[derive(Clone)]
pub struct Crop {
    pub ink: Vec<f32>,
    pub rows: usize,
    pub width: usize,
}

fn ink_thumb(gray: &Gray, side: usize) -> Gray {
    let s = side as f64 / gray.width.max(gray.height) as f64;
    let w = (py_round(gray.width as f64 * s) as usize).max(1);
    let h = (py_round(gray.height as f64 * s) as usize).max(1);
    let mut small = resize(gray, w, h, Interp::Area);
    for v in &mut small.data {
        *v = 255 - *v;
    }
    small
}

/// How sharply the horizontal ink profile changes row to row, the page turned
/// by `a` degrees: highest when the text rows are level.
fn row_sharpness(ink: &Gray, a: f64) -> f64 {
    let m = rotation_matrix(ink.width as f32 / 2.0, ink.height as f32 / 2.0, a);
    let r = crate::imgproc::warp_affine(ink, &m, ink.width, ink.height, Border::Constant(0));
    let p: Vec<f64> = (0..r.height)
        .map(|y| r.row(y).iter().map(|&v| v as i64).sum::<i64>() as f64)
        .collect();
    p.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0])).sum()
}

fn score_all(ink: &Gray, angles: &[f64]) -> Vec<f64> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(angles.len().max(1));
    let chunk = angles.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = angles
            .chunks(chunk)
            .map(|c| s.spawn(move || c.iter().map(|&a| row_sharpness(ink, a)).collect::<Vec<_>>()))
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// `np.arange(lo, hi + 1e-6, step)`.
fn arange(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let n = ((hi + 1e-6 - lo) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| lo + i as f64 * step).collect()
}

/// Degrees (counter-clockwise) that straighten the text rows: a 2° search over
/// ±45° on a half-size thumbnail, then 0.5° and 0.1° refinements.
pub fn estimate_skew(gray: &Gray) -> f64 {
    let fine = ink_thumb(gray, SKEW_SIDE);
    if fine.mean() < 0.003 * 255.0 {
        return 0.0;
    }
    let coarse = ink_thumb(gray, SKEW_SIDE / 2);
    let grid = arange(-DESKEW_MAX, DESKEW_MAX, 2.0);
    let scores = score_all(&coarse, &grid);
    let mut idx: Vec<usize> = (0..grid.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut starts = vec![0.0];
    for &i in &idx[idx.len().saturating_sub(2)..] {
        if !starts.contains(&grid[i]) {
            starts.push(grid[i]);
        }
    }
    let near: Vec<f64> = starts
        .iter()
        .flat_map(|&a0| arange(a0 - 2.0, a0 + 2.0, 0.5))
        .collect();
    let mut uniq: Vec<f64> = Vec::new();
    for &a in &near {
        if !uniq.contains(&a) {
            uniq.push(a);
        }
    }
    let us = score_all(&fine, &uniq);
    let score_of = |a: f64| us[uniq.iter().position(|&u| u == a).unwrap()];
    let mut a1 = near[0];
    for &a in &near[1..] {
        if score_of(a) > score_of(a1) {
            a1 = a;
        }
    }
    let last = arange(a1 - 0.5, a1 + 0.5, 0.1);
    let vals = score_all(&fine, &last);
    let mut best = 0;
    for i in 1..last.len() {
        if vals[i] > vals[best] {
            best = i;
        }
    }
    last[best]
}

/// The page shrunk once to at most `side` pixels a side (area averaging).
pub fn preview(gray: &Gray, side: usize) -> Gray {
    let s = side as f64 / gray.width.max(gray.height) as f64;
    if s >= 1.0 {
        return gray.clone();
    }
    let w = (py_round(gray.width as f64 * s) as usize).max(1);
    let h = (py_round(gray.height as f64 * s) as usize).max(1);
    resize(gray, w, h, Interp::Area)
}

/// `[side, side]` ink map: long side scaled to `side`, padded.
fn orient_input(gray: &Gray, side: usize) -> Vec<f32> {
    let s = side as f64 / gray.width.max(gray.height) as f64;
    let w = (py_round(gray.width as f64 * s) as usize).max(1);
    let h = (py_round(gray.height as f64 * s) as usize).max(1);
    let small = resize(gray, w, h, Interp::Area);
    let mut out = vec![0f32; side * side];
    for y in 0..h.min(side) {
        for x in 0..w.min(side) {
            out[y * side + x] = 1.0 - small.at(x, y) as f32 / 255.0;
        }
    }
    out
}

pub fn rotate_page(gray: &Gray, degrees: f64) -> Gray {
    rotate(gray, degrees, Border::Replicate)
}

/// A box of the page `rotate_page(.., degrees)` made, mapped back.
fn unrotate_box(b: [f64; 4], degrees: f64, w: usize, h: usize) -> [f64; 4] {
    let m = rotation_matrix(w as f32 / 2.0, h as f32 / 2.0, -degrees);
    let xs = [b[0], b[2], b[0], b[2]];
    let ys = [b[1], b[1], b[3], b[3]];
    let px: Vec<f64> = (0..4).map(|k| m[0] * xs[k] + m[1] * ys[k] + m[2]).collect();
    let py: Vec<f64> = (0..4).map(|k| m[3] * xs[k] + m[4] * ys[k] + m[5]).collect();
    let (wf, hf) = (w as f64, h as f64);
    let mn = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    [
        round1(mn(&px).clamp(0.0, wf)),
        round1(mn(&py).clamp(0.0, hf)),
        round1(mx(&px).clamp(0.0, wf)),
        round1(mx(&py).clamp(0.0, hf)),
    ]
}

const LEADER_CHARS: &str = ".·…‥_-–—=";
const DASHES: &str = "-–—";
const BULLETS: &str = "•◦▪▫●○■□‣⁃∙*";

/// A numbered list marker: `1.`, `12)`, `a.`, `B)`, `(3)`, `(b)`, `(iv)`.
fn is_enumerator(t: &str) -> bool {
    let c: Vec<char> = t.chars().collect();
    let digits = |s: &[char]| {
        !s.is_empty()
            && s.len() <= 3
            && s.iter().all(|&ch| {
                crate::text::category(ch)
                    == unicode_general_category::GeneralCategory::DecimalNumber
            })
    };
    let letter = |s: &[char]| s.len() == 1 && s[0].is_ascii_alphabetic();
    let roman =
        |s: &[char]| !s.is_empty() && s.len() <= 5 && s.iter().all(|ch| "ivxlcIVXLC".contains(*ch));
    match c.as_slice() {
        [inner @ .., '.' | ')'] if c[0] != '(' && (digits(inner) || letter(inner)) => true,
        ['(', inner @ .., ')'] => digits(inner) || letter(inner) || roman(inner),
        _ => false,
    }
}

impl Indra {
    /// Load a model folder written by `rlx-indra-convert`.
    pub fn load(dir: impl AsRef<Path>, opts: Options) -> Result<Self> {
        let dir = dir.as_ref();
        let raw = std::fs::read_to_string(dir.join("meta.json"))
            .with_context(|| format!("{} lacks meta.json", dir.display()))?;
        let meta: Value = serde_json::from_str(&raw)?;
        let s = |k: &str| {
            meta.get(k)
                .and_then(Value::as_str)
                .with_context(|| format!("meta.json lacks {k}"))
        };
        let u = |k: &str| meta.get(k).and_then(Value::as_u64).map(|v| v as usize);
        let lm_params = meta.get("lm_params").cloned().unwrap_or(Value::Null);
        let empty = Value::Null;
        let dev = opts.device;
        let det = Compiled::new(Weights::load(&dir.join("det.safetensors"))?, dev, |w, k| {
            graph::det::build(w, k[0], k[1])
        });
        let base = Recognizer::load(
            dir,
            "rec",
            Vocab::from_json(s("vocab")?)?,
            Some(("lm.safetensors", lm_params.get("base").unwrap_or(&empty))),
            dev,
        )?;
        let mut groups = Vec::new();
        for key in meta_key_order(&raw, "vocab_") {
            let g = &key["vocab_".len()..];
            if !dir.join(format!("rec_{g}.safetensors")).exists() {
                continue;
            }
            let vocab = Vocab::from_json(s(&key)?)?;
            let lm_file = format!("lm_{g}.safetensors");
            let rec = Recognizer::load(
                dir,
                &format!("rec_{g}"),
                vocab,
                Some((lm_file.as_str(), lm_params.get(g).unwrap_or(&empty))),
                dev,
            )?;
            groups.push((g.to_string(), rec));
        }
        let opt_net =
            |name: &str,
             build: fn(
                &Weights,
                &[usize],
            )
                -> Result<(rlx_ir::Graph, std::collections::HashMap<String, Vec<f32>>)>|
             -> Result<Option<Compiled>> {
                let p = dir.join(format!("{name}.safetensors"));
                Ok(if p.exists() {
                    Some(Compiled::new(Weights::load(&p)?, dev, build))
                } else {
                    None
                })
            };
        let script = if groups.is_empty() {
            None
        } else {
            opt_net("script", |w, k| {
                graph::classifier::build_script(w, k[0], k[1])
            })?
        };
        let script_groups = meta
            .get("script_groups")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| DEFAULT_GROUPS.iter().map(|s| s.to_string()).collect());
        Ok(Self {
            det,
            base,
            groups,
            script,
            script_groups,
            orient: opt_net("orient", |w, k| graph::classifier::build_orient(w, 1, k[0]))?,
            order: opt_net("order", |w, k| graph::order::build(w, k[0]))?,
            long_side: u("det_long_side").context("meta.json lacks det_long_side")?,
            height: u("rec_height").unwrap_or(lines::LINE_HEIGHT),
            window: u("rec_window").unwrap_or(1),
            opts,
        })
    }

    // ── detection ───────────────────────────────────────────────────────────

    fn detect_at(
        &self,
        gray: &Gray,
        long_side: usize,
    ) -> Result<(Vec<LineBox>, f64, Option<(Vec<f32>, usize, usize)>)> {
        let (x, h, w, s) = lines::page_to_input(gray, long_side);
        let out = &self.det.run(&[h, w], &[("x", &x)])?[0];
        let (oh, ow) = (h / graph::det::OUT_STRIDE, w / graph::det::OUT_STRIDE);
        let channels = out.len() / (oh * ow);
        let pic = (channels > 3).then(|| (out[3 * oh * ow..4 * oh * ow].to_vec(), oh, ow));
        Ok((lines::decode(out, oh, ow), s, pic))
    }

    /// Line boxes in page pixels, and picture regions.
    pub fn detect(&self, gray: &Gray) -> Result<(Vec<LineBox>, Vec<Rect>)> {
        let (mut boxes, mut s, mut pic) = self.detect_at(gray, self.long_side)?;
        if self.opts.adaptive && boxes.len() >= 8 {
            let med_h = median(boxes.iter().map(|b| b.y1 - b.y0));
            if med_h < ADAPTIVE_MIN_H {
                let t = ADAPTIVE_MAX_SIDE
                    .min(self.long_side as f64 * ADAPTIVE_TARGET_H / (med_h as f64).max(1.0));
                let ls = (t as usize) / 32 * 32;
                if ls > self.long_side {
                    (boxes, s, pic) = self.detect_at(gray, ls)?;
                }
            }
        }
        let (w, h) = (gray.width as f32, gray.height as f32);
        let sf = s as f32;
        for b in &mut boxes {
            b.x0 = (b.x0 / sf).clamp(0.0, w);
            b.x1 = (b.x1 / sf).clamp(0.0, w);
            b.y0 = (b.y0 / sf).clamp(0.0, h);
            b.y1 = (b.y1 / sf).clamp(0.0, h);
        }
        let pictures = match pic {
            Some((logit, ph, pw)) => picture_boxes(&logit, ph, pw, s, gray.width, gray.height),
            None => Vec::new(),
        };
        Ok((boxes, pictures))
    }

    // ── recognition ─────────────────────────────────────────────────────────

    fn crops(&self, gray: &Gray, boxes: &[LineBox], window: usize, dark: &[bool]) -> Vec<Crop> {
        boxes
            .iter()
            .zip(dark)
            .map(|(b, &d)| {
                let (ink, rows, width) = lines::crop_line(gray, &b.rect(), self.height, window, d);
                Crop { ink, rows, width }
            })
            .collect()
    }

    /// Per line crop, the group whose recognizer reads it and whether the
    /// router is sure.
    fn line_groups(&self, crops: &[Crop]) -> Result<(Vec<String>, Vec<bool>)> {
        let mut names = vec!["base".to_string(); crops.len()];
        let mut sure = vec![true; crops.len()];
        let Some(script) = &self.script else {
            return Ok((names, sure));
        };
        let widths: Vec<usize> = crops.iter().map(|c| c.width).collect();
        for idx in lines::width_batches(&widths, REC_BATCH) {
            let wm = idx
                .iter()
                .map(|&j| widths[j])
                .max()
                .unwrap()
                .div_ceil(WIDTH_MULTIPLE)
                * WIDTH_MULTIPLE;
            if wm < graph::classifier::SCRIPT_STRIDE {
                // Upstream pools such a batch to nothing: a NaN reading, base, unsure.
                for &j in &idx {
                    sure[j] = false;
                }
                continue;
            }
            let (bb, wb) = (batch_bucket(idx.len()), width_bucket(wm));
            let sel: Vec<&Crop> = idx.iter().map(|&j| &crops[j]).collect();
            let x = pad_batch(&sel, self.height, bb, wb);
            let mut ws: Vec<usize> = idx.iter().map(|&j| widths[j]).collect();
            ws.resize(bb, WIDTH_MULTIPLE);
            let (mask, inv) =
                graph::classifier::script_masks(&ws, wb / graph::classifier::SCRIPT_STRIDE);
            let p = &script.run(
                &[bb, wb],
                &[("x", &x), ("mask", &mask), ("inv_count", &inv)],
            )?[0];
            let g = p.len() / bb;
            for (k, &j) in idx.iter().enumerate() {
                let row = &p[k * g..(k + 1) * g];
                let mut best = 0;
                for c in 1..g {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                let name = self.script_groups.get(best).cloned().unwrap_or_default();
                names[j] = if self.groups.iter().any(|(n, _)| *n == name) {
                    name
                } else {
                    "base".into()
                };
                sure[j] = row[best] > ROUTE_SURE;
            }
        }
        Ok((names, sure))
    }

    /// (text, confidence, tokens) per line; tokens carry the page x of their
    /// CTC peak. `vertical`: every line to the CJK recognizer.
    pub fn recognize(
        &self,
        gray: &Gray,
        boxes: &[LineBox],
        vertical: bool,
    ) -> Result<Vec<Reading>> {
        let n = boxes.len();
        let dark: Vec<bool> = boxes
            .iter()
            .map(|b| lines::is_dark_line(gray, &b.rect()))
            .collect();
        let crops = self.crops(gray, boxes, 1, &dark);
        let tall = if self.window > 1 {
            self.crops(gray, boxes, self.window, &dark)
        } else {
            crops.clone()
        };
        let mut out = vec![
            Reading {
                text: String::new(),
                conf: 0.0,
                toks: Vec::new()
            };
            n
        ];
        if n == 0 {
            return Ok(out);
        }
        // (recognizer, lines, tokens as characters, reads the tall window)
        let mut plan: Vec<(&Recognizer, Vec<usize>, bool, bool)> = Vec::new();
        let cjk = self.groups.iter().find(|(g, _)| g == "cjk");
        if let (true, Some((_, r))) = (vertical, cjk) {
            plan.push((r, (0..n).collect(), true, false));
        } else if self.script.is_some() {
            let (names, sure) = self.line_groups(&crops)?;
            plan.push((
                &self.base,
                (0..n).filter(|&j| names[j] == "base" || !sure[j]).collect(),
                false,
                true,
            ));
            for (g, r) in &self.groups {
                plan.push((r, (0..n).filter(|&j| names[j] == *g).collect(), true, false));
            }
        } else {
            plan.push((&self.base, (0..n).collect(), false, true));
        }
        let mut best_conf = vec![-1f64; n];
        for (rec, sel, as_chars, use_tall) in plan {
            let widths: Vec<usize> = sel.iter().map(|&j| crops[j].width).collect();
            for part in lines::width_batches(&widths, REC_BATCH) {
                let idx: Vec<usize> = part.iter().map(|&i| sel[i]).collect();
                let src: Vec<&Crop> = idx
                    .iter()
                    .map(|&j| if use_tall { &tall[j] } else { &crops[j] })
                    .collect();
                let wm = src
                    .iter()
                    .map(|c| c.width)
                    .max()
                    .unwrap()
                    .div_ceil(WIDTH_MULTIPLE)
                    * WIDTH_MULTIPLE;
                let (bb, wb) = (batch_bucket(idx.len()), width_bucket(wm));
                let rows = rec.height;
                let x = pad_batch(&src, rows, bb, wb);
                let t_all = wb / graph::rec::STRIDE;
                let mut ws: Vec<usize> = src.iter().map(|c| c.width).collect();
                ws.resize(bb, WIDTH_MULTIPLE);
                let (mask, inv) = graph::rec::line_masks(&ws, t_all);
                let outs = rec.net.run(
                    &[bb, wb],
                    &[("x", &x), ("mask", &mask), ("inv_count", &inv)],
                )?;
                let (best, lse, feats) = (&outs[0], &outs[1], &outs[2]);
                let head = graph::rec::Head::new(rec.net.weights())?;
                let c = head.channels();
                const K: usize = graph::rec::TOP_K;
                for (k, &j) in idx.iter().enumerate() {
                    let len = src[k].width.div_ceil(graph::rec::STRIDE);
                    let at = |t: usize| k * t_all + t;
                    let logp = |t: usize, class: usize| {
                        head.logp(&feats[at(t) * c..(at(t) + 1) * c], lse[at(t)], class)
                    };
                    let seq: Vec<u32> = (0..len).map(|t| best[at(t)] as u32).collect();
                    let pm: Vec<f64> = (0..len)
                        .map(|t| (logp(t, seq[t] as usize) as f64).exp())
                        .collect();
                    let keep: Vec<usize> = (0..len)
                        .filter(|&t| seq[t] != 0 && (t == 0 || seq[t] != seq[t - 1]))
                        .collect();
                    let b = boxes[j];
                    let (x0, x1) = (b.x0 as f64, b.x1 as f64);
                    let h = ((b.y1 - b.y0) as f64).max(1.0);
                    let ix0 = (x0 - PAD_X * h).floor();
                    let cw = (x1 + PAD_X * h).ceil() - ix0;
                    let f = cw / crops[j].width as f64;
                    let mut ids: Vec<u32> = keep.iter().map(|&t| seq[t]).collect();
                    let mut ts: Vec<usize> = keep.clone();
                    let conf = if keep.is_empty() {
                        0.0
                    } else {
                        keep.iter().map(|&t| pm[t]).sum::<f64>() / keep.len() as f64
                    };
                    if let Some((lm, params, gate)) = &rec.lm
                        && MIN_CONF <= conf
                        && conf < *gate
                    {
                        let (mut blank, mut topv, mut topc) = (
                            Vec::with_capacity(len),
                            Vec::with_capacity(len * K),
                            Vec::with_capacity(len * K),
                        );
                        for t in 0..len {
                            blank.push(logp(t, 0));
                            for (class, v) in
                                head.top_k(&feats[at(t) * c..(at(t) + 1) * c], lse[at(t)])
                            {
                                topv.push(v);
                                topc.push(class);
                            }
                        }
                        let topi = topc;
                        (ids, ts) = beam_search(&blank, &topv, &topi, K, lm, *params);
                    }
                    let stride = graph::rec::STRIDE as f64;
                    let xs: Vec<f64> = ts
                        .iter()
                        .map(|&t| ix0 + (t as f64 + 0.5) * stride * f)
                        .collect();
                    let toks: Toks = ids
                        .iter()
                        .zip(xs)
                        .map(|(&i, x)| {
                            let t = if as_chars {
                                Tok::Ch(rec.vocab.token(i).unwrap_or_default().to_string())
                            } else {
                                Tok::Id(i)
                            };
                            (t, x)
                        })
                        .collect();
                    if conf <= best_conf[j] {
                        continue;
                    }
                    best_conf[j] = conf;
                    out[j] = if conf >= MIN_CONF {
                        Reading {
                            text: rec.vocab.decode(&ids),
                            conf,
                            toks,
                        }
                    } else {
                        Reading {
                            text: String::new(),
                            conf,
                            toks: Vec::new(),
                        }
                    };
                }
            }
        }
        Ok(out)
    }

    // ── orientation ─────────────────────────────────────────────────────────

    /// Orientation class probabilities (`None` without the model).
    pub fn orientation_probs(&self, gray: &Gray) -> Result<Option<Vec<f32>>> {
        let Some(net) = &self.orient else {
            return Ok(None);
        };
        let x = orient_input(gray, ORIENT_SIDE);
        Ok(Some(net.run(&[ORIENT_SIDE], &[("x", &x)])?.swap_remove(0)))
    }

    fn orientation(&self, gray: &Gray) -> Result<usize> {
        let Some(p) = self.orientation_probs(gray)? else {
            return Ok(0);
        };
        let mut k = 0;
        for i in 1..p.len() {
            if p[i] > p[k] {
                k = i;
            }
        }
        Ok(if p[k] > ORIENT_MIN { k } else { 0 })
    }

    fn turn(&self, gray: &Gray, k: usize, mut a: f64, small: &Gray) -> (Gray, f64) {
        let q = if k == VERTICAL { 1 } else { -(k as i32) };
        let mut g = if k == VERTICAL {
            rot90(gray, 1)
        } else if k > 0 {
            rot90(gray, -(k as i32))
        } else {
            gray.clone()
        };
        if matches!(k, 1 | 3 | VERTICAL) {
            a = estimate_skew(&rot90(small, q));
        }
        if a.abs() >= DESKEW_MIN {
            g = rotate_page(&g, a);
        } else {
            a = 0.0;
        }
        (g, a)
    }

    // ── the page ────────────────────────────────────────────────────────────

    /// OCR of a page image. `turn = false`: the page is known to be upright.
    pub fn read(&self, gray: &Gray, turn: bool) -> Result<Page> {
        let t0 = Instant::now();
        let small = preview(gray, PREVIEW_SIDE);
        let a = estimate_skew(&small);
        let k = if turn {
            let s = if a.abs() >= DESKEW_MIN {
                rotate_page(&small, a)
            } else {
                small.clone()
            };
            self.orientation(&s)?
        } else {
            0
        };
        let orient_ms = ms(t0);
        let t0 = Instant::now();
        let (g, skew) = self.turn(gray, k, a, &small);
        let mut timings = std::collections::BTreeMap::new();
        timings.insert("orient_ms".to_string(), ms(t0) + orient_ms);
        let t0 = Instant::now();
        let (boxes, pictures) = self.detect(&g)?;
        timings.insert("det_ms".into(), ms(t0));
        let t0 = Instant::now();
        let vertical = k == VERTICAL;
        let rec = self.recognize(&g, &boxes, vertical)?;
        timings.insert("rec_ms".into(), ms(t0));
        let rects: Vec<Rect> = boxes.iter().map(LineBox::rect).collect();
        let mut page = self.assemble(
            (g.width, g.height),
            rects,
            rec,
            timings,
            vertical,
            if vertical { &[] } else { &pictures },
        )?;
        page.rotation = if vertical { 0 } else { 90 * k as i32 };
        page.skew = round_to(skew, 2);
        page.vertical = vertical;
        if skew != 0.0 {
            for l in &mut page.lines {
                l.bbox = unrotate_box(l.bbox, skew, g.width, g.height);
            }
            for b in &mut page.blocks {
                b.bbox = unrotate_box(b.bbox, skew, g.width, g.height);
            }
        }
        if vertical {
            let w = g.height as f64;
            let back = |b: [f64; 4]| [w - b[3], b[0], w - b[1], b[2]];
            for l in &mut page.lines {
                l.bbox = back(l.bbox);
            }
            for b in &mut page.blocks {
                b.bbox = back(b.bbox);
            }
        }
        Ok(page)
    }

    fn is_space(&self, t: &Tok) -> bool {
        match t {
            Tok::Ch(c) => c == " ",
            Tok::Id(i) => Some(*i) == self.base.vocab.space(),
        }
    }

    /// Text of tokens: base-vocabulary ids, or characters of a group
    /// recognizer (visual order: right-to-left text is flipped).
    fn decode_toks(&self, toks: &[(Tok, f64)], page_rtl: bool) -> String {
        if let Some((Tok::Ch(_), _)) = toks.first() {
            let s: String = toks
                .iter()
                .map(|(t, _)| match t {
                    Tok::Ch(c) => c.as_str(),
                    Tok::Id(_) => "",
                })
                .collect();
            let text = nfc(&squash_spaces(&s));
            return if has_rtl(&text) {
                flip(&text, page_rtl.then_some(true))
            } else {
                text
            };
        }
        let ids: Vec<u32> = toks
            .iter()
            .filter_map(|(t, _)| match t {
                Tok::Id(i) => Some(*i),
                Tok::Ch(_) => None,
            })
            .collect();
        self.base.vocab.decode(&ids)
    }

    /// Per line, the x positions of gaps that sit on a column gutter.
    fn gutter_cuts(&self, boxes: &[Rect], rec: &[Reading]) -> Vec<Vec<f64>> {
        boxes
            .iter()
            .zip(rec)
            .enumerate()
            .map(|(i, (b, r))| {
                let h = ((b[3] - b[1]) as f64).max(1.0);
                let xs: Vec<f64> = r
                    .toks
                    .iter()
                    .filter(|(t, _)| !self.is_space(t))
                    .map(|(_, x)| *x)
                    .collect();
                let mut mine = Vec::new();
                for w in xs.windows(2) {
                    let (a, c) = (w[0], w[1]);
                    if c - a <= GUTTER_GAP * h {
                        continue;
                    }
                    let (gl, gr) = (a + 0.35 * h, c - 0.35 * h);
                    let near = boxes
                        .iter()
                        .enumerate()
                        .filter(|&(j, o)| {
                            j != i
                                && (((o[2] as f64) - gl).abs() < 0.5 * h
                                    || ((o[0] as f64) - gr).abs() < 0.5 * h)
                        })
                        .count();
                    if near >= GUTTER_EDGES {
                        mine.push((a + c) / 2.0);
                    }
                }
                mine
            })
            .collect()
    }

    /// Cut a line where two neighbouring characters are more than `gap` line
    /// heights apart, or at a gutter in `at`.
    fn split_line(
        &self,
        b: &Rect,
        toks: &[(Tok, f64)],
        gap: f64,
        at: &[f64],
        page_rtl: bool,
    ) -> Vec<(Rect, String, Toks)> {
        let h = ((b[3] - b[1]) as f64).max(1.0);
        let mut at = at.to_vec();
        at.sort_by(f64::total_cmp);
        let mut segs: Vec<Toks> = vec![Vec::new()];
        let mut prev: Option<f64> = None;
        for tk in toks {
            let x = tk.1;
            if !self.is_space(&tk.0) {
                if let Some(p) = prev
                    && (x - p > gap * h || at.iter().any(|&c| p < c && c < x))
                {
                    segs.push(Vec::new());
                }
                prev = Some(x);
            }
            segs.last_mut().unwrap().push(tk.clone());
        }
        if segs.len() == 1 {
            return vec![(*b, self.decode_toks(toks, page_rtl), toks.to_vec())];
        }
        let mut out = Vec::new();
        for sg in segs {
            let text = self.decode_toks(&sg, page_rtl).trim().to_string();
            let xs: Vec<f64> = sg
                .iter()
                .filter(|(t, _)| !self.is_space(t))
                .map(|(_, x)| *x)
                .collect();
            if text.is_empty() || xs.is_empty() {
                continue;
            }
            let mn = xs.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let bx = [
                (b[0] as f64).max(mn - 0.35 * h) as f32,
                b[1],
                (b[2] as f64).min(mx + 0.35 * h) as f32,
                b[3],
            ];
            out.push((bx, text, sg));
        }
        out
    }

    /// Reading order, the layout heads `(cls, same, tab)` when the order model
    /// has them.
    #[allow(clippy::type_complexity)]
    fn reading_order(
        &self,
        boxes: &[Rect],
        texts: &[String],
        wh: (usize, usize),
    ) -> Result<(Vec<usize>, Option<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)>)> {
        let n = boxes.len();
        let Some(net) = self.order.as_ref().filter(|_| n >= 2) else {
            return Ok((
                if n > 0 {
                    order::order_xycut(boxes)
                } else {
                    Vec::new()
                },
                None,
            ));
        };
        let feats = order::line_features(boxes, wh.0 as f32, wh.1 as f32, texts);
        let np = n.div_ceil(16) * 16;
        let mut pf = feats;
        pf.resize(np * 16, 0.0);
        let mut bias = vec![0f32; np];
        bias[n..].fill(graph::order::PAD_BIAS);
        let outs = net.run(&[np], &[("feats", &pf), ("key_bias", &bias)])?;
        let crop = |m: &[f32], planes: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(planes * n * n);
            for p in 0..planes {
                for i in 0..n {
                    out.extend_from_slice(&m[(p * np + i) * np..(p * np + i) * np + n]);
                }
            }
            out
        };
        let classes = outs[1].len() / np;
        let a = crop(&outs[0], 1);
        let cls = outs[1][..n * classes].to_vec();
        let same = crop(&outs[2], 1);
        let tab = self.opts.table_heads.then(|| crop(&outs[3], 3));
        let order = order::decode_blocks_order(&a, &same, boxes, &cls, classes);
        Ok((order, Some((cls, same, tab))))
    }

    fn assemble(
        &self,
        wh: (usize, usize),
        boxes: Vec<Rect>,
        rec: Vec<Reading>,
        mut timings: std::collections::BTreeMap<String, f64>,
        vertical: bool,
        pictures: &[Rect],
    ) -> Result<Page> {
        let t0 = Instant::now();
        let letters: Vec<char> = rec
            .iter()
            .flat_map(|r| r.text.chars())
            .filter(|&c| is_alpha(c))
            .collect();
        let n_rtl = letters.iter().filter(|&&c| is_rtl(c)).count();
        let page_rtl = n_rtl > letters.len() - n_rtl;
        let width = wh.0 as f32;
        let geo = |b: &Rect| -> Rect {
            if page_rtl {
                [width - b[2], b[1], width - b[0], b[3]]
            } else {
                *b
            }
        };

        let cuts = self.gutter_cuts(&boxes, &rec);
        let (mut nb, mut nrec) = (Vec::new(), Vec::new());
        for ((b, r), at) in boxes.iter().zip(&rec).zip(&cuts) {
            for (bx, tx, tk) in self.split_line(b, &r.toks, SPLIT_GAP, at, page_rtl) {
                nb.push(bx);
                nrec.push(Reading {
                    text: tx,
                    conf: r.conf,
                    toks: tk,
                });
            }
        }
        let (mut boxes, mut rec) = tidy_lines(nb, nrec, page_rtl, self.base.vocab.space());
        let mut texts: Vec<String> = rec.iter().map(|r| r.text.clone()).collect();
        let gb: Vec<Rect> = boxes.iter().map(geo).collect();
        let (mut order, mut layout) = self.reading_order(&gb, &texts, wh)?;
        if vertical && let Some((cls, _, _)) = &layout {
            let blocks = order::xycut_blocks(&boxes);
            order = blocks.iter().flatten().copied().collect();
            let n = boxes.len();
            let mut same = vec![-1f32; n * n];
            for b in &blocks {
                for &i in b {
                    for &j in b {
                        same[i * n + j] = 1.0;
                    }
                }
            }
            let classes = cls.len() / n.max(1);
            let mut cls = cls.clone();
            for g in GRID_LABELS {
                let c = class_id(g);
                for i in 0..n {
                    cls[i * classes + c] = -1e4;
                }
            }
            layout = Some((cls, same, None));
        } else if let Some((cls, _, Some(_))) = &layout {
            let classes = cls.len() / boxes.len().max(1);
            let table = class_id("Table");
            let is_table: Vec<bool> = (0..boxes.len())
                .map(|i| order::block_class(cls, classes, &[i]) == table)
                .collect();
            if is_table.iter().any(|&t| t) {
                let (mut nb, mut nrec) = (Vec::new(), Vec::new());
                for (i, (b, r)) in boxes.iter().zip(&rec).enumerate() {
                    let parts = if is_table[i] {
                        self.split_line(b, &r.toks, CELL_GAP, &[], page_rtl)
                    } else {
                        vec![(*b, r.text.clone(), r.toks.clone())]
                    };
                    for (bx, tx, tk) in parts {
                        nb.push(bx);
                        nrec.push(Reading {
                            text: tx,
                            conf: r.conf,
                            toks: tk,
                        });
                    }
                }
                if nb.len() != boxes.len() {
                    boxes = nb;
                    rec = nrec;
                    texts = rec.iter().map(|r| r.text.clone()).collect();
                    let gb: Vec<Rect> = boxes.iter().map(geo).collect();
                    (order, layout) = self.reading_order(&gb, &texts, wh)?;
                }
            }
        }
        let note = if vertical {
            "order=xycut"
        } else {
            "order=model"
        }
        .to_string();
        timings.insert("order_ms".into(), ms(t0));
        let total: f64 = timings.values().sum();
        timings.insert("total_ms".into(), total);
        let lines_out: Vec<Line> = order
            .iter()
            .map(|&i| Line {
                bbox: boxes[i].map(|v| round1(v as f64)),
                text: rec[i].text.clone(),
                conf: round_to(rec[i].conf, 4),
            })
            .collect();
        let (mut markdown, mut blocks) = (None, Vec::new());
        if let Some((cls, same, tab)) = &layout {
            let classes = cls.len() / boxes.len().max(1);
            let gb: Vec<Rect> = boxes.iter().map(geo).collect();
            blocks = layout::page_blocks(&gb, &texts, &order, cls, classes, same);
            for b in &mut blocks {
                let r = [
                    b.bbox[0] as f32,
                    b.bbox[1] as f32,
                    b.bbox[2] as f32,
                    b.bbox[3] as f32,
                ];
                b.bbox = geo(&r).map(|v| round1(v as f64));
            }
            let cells: Vec<Vec<(Rect, String)>> = (0..boxes.len())
                .map(|i| {
                    self.split_line(&boxes[i], &rec[i].toks, CELL_GAP, &[], page_rtl)
                        .into_iter()
                        .map(|(bx, tx, _)| (geo(&bx), tx))
                        .collect()
                })
                .collect();
            let gp: Vec<Rect> = pictures.iter().map(geo).collect();
            markdown = Some(layout::page_markdown(
                &gb,
                &texts,
                &order,
                cls,
                classes,
                same,
                &MarkdownExtras {
                    keep_furniture: self.opts.furniture,
                    cells: Some(&cells),
                    tab: tab.as_deref(),
                    pictures: &gp,
                },
            ));
            for p in pictures {
                let cy = ((p[1] + p[3]) / 2.0) as f64;
                let at = blocks
                    .iter()
                    .position(|b| {
                        (b.bbox[1] + b.bbox[3]) / 2.0 > cy
                            && (p[2] as f64).min(b.bbox[2]) > (p[0] as f64).max(b.bbox[0])
                    })
                    .unwrap_or(blocks.len());
                blocks.insert(
                    at,
                    Block {
                        label: "Picture".into(),
                        bbox: p.map(|v| round1(v as f64)),
                        text: String::new(),
                    },
                );
            }
        } else if boxes.len() < 2 {
            markdown = Some(texts.join("\n"));
        }
        Ok(Page {
            lines: lines_out,
            markdown,
            blocks,
            rotation: 0,
            skew: 0.0,
            vertical: false,
            note,
            timings,
        })
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Picture regions from the detector's picture channel (at half the input
/// scale `s`): 8-connected regions above `PICTURE_MIN`, at least
/// `PICTURE_MIN_AREA` of the page.
fn picture_boxes(
    logit: &[f32],
    h: usize,
    w: usize,
    s: f64,
    page_w: usize,
    page_h: usize,
) -> Vec<Rect> {
    let thr = (PICTURE_MIN / (1.0 - PICTURE_MIN)).ln();
    let mask: Vec<bool> = logit.iter().map(|&v| v > thr).collect();
    let (_, comps) = connected_components(&mask, w, h, true);
    let f = 2.0 / s;
    let min_area = PICTURE_MIN_AREA as f64 * (page_w * page_h) as f64;
    comps
        .iter()
        .filter_map(|c| {
            let b = [
                c.left as f64 * f,
                c.top as f64 * f,
                ((c.left + c.width) as f64 * f).min(page_w as f64),
                ((c.top + c.height) as f64 * f).min(page_h as f64),
            ];
            ((b[2] - b[0]) * (b[3] - b[1]) >= min_area).then(|| b.map(|v| v as f32))
        })
        .collect()
}

/// Lines without text of their own: leader-only lines are dropped; a lone
/// list marker joins the line that starts just after it on the same row.
fn tidy_lines(
    mut boxes: Vec<Rect>,
    mut rec: Vec<Reading>,
    rtl: bool,
    space: Option<u32>,
) -> (Vec<Rect>, Vec<Reading>) {
    let n = rec.len();
    let mut keep = vec![true; n];
    let h: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let w: Vec<f32> = boxes.iter().map(|b| b[2] - b[0]).collect();
    for i in 0..n {
        let core: String = rec[i].text.chars().filter(|&c| c != ' ').collect();
        if !core.is_empty() && core.chars().all(|c| LEADER_CHARS.contains(c)) {
            let wide = w[i] >= 3.0 * h[i];
            if core.chars().count() >= 3 || wide || !core.chars().any(|c| DASHES.contains(c)) {
                keep[i] = false;
            }
        }
    }
    for i in 0..n {
        let t = rec[i].text.trim().to_string();
        let bullet = t.chars().count() == 1 && BULLETS.contains(t.as_str());
        if !keep[i] || !(bullet || is_enumerator(&t)) {
            continue;
        }
        let mut best: Option<(usize, f32)> = None;
        for j in 0..n {
            if j == i || !keep[j] || rec[j].text.trim().is_empty() {
                continue;
            }
            let ov = boxes[i][3].min(boxes[j][3]) - boxes[i][1].max(boxes[j][1]);
            if ov < 0.5 * h[i].min(h[j]) {
                continue;
            }
            let gap = if rtl {
                boxes[i][0] - boxes[j][2]
            } else {
                boxes[j][0] - boxes[i][2]
            };
            if -0.3 * h[i] <= gap
                && gap <= MARKER_GAP * h[i].max(h[j])
                && best.is_none_or(|(_, g)| gap < g)
            {
                best = Some((j, gap));
            }
        }
        let Some((bj, _)) = best else {
            continue;
        };
        let toks = rec[i].toks.clone();
        let mut kj = rec[bj].toks.clone();
        let chars = matches!(kj.first(), Some((Tok::Ch(_), _)));
        let sp = if chars {
            Some(Tok::Ch(" ".into()))
        } else {
            space.map(Tok::Id)
        };
        if !rtl
            && !toks.is_empty()
            && !kj.is_empty()
            && let Some(sp) = sp
            && matches!(toks[0].0, Tok::Ch(_)) == chars
        {
            let x = ((boxes[i][2] + boxes[bj][0]) / 2.0) as f64;
            let mut merged = toks;
            merged.push((sp, x));
            merged.extend(kj);
            kj = merged;
        }
        rec[bj] = Reading {
            text: format!("{t} {}", rec[bj].text.trim()),
            conf: rec[bj].conf,
            toks: kj,
        };
        boxes[bj] = union(&boxes, &[i, bj]);
        keep[i] = false;
    }
    let idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    (
        idx.iter().map(|&i| boxes[i]).collect(),
        idx.iter().map(|&i| rec[i].clone()).collect(),
    )
}

/// Grayscale pages of an image file (every frame of a multi-page TIFF is not
/// split: the first frame is read). Color is reduced with OpenCV's weights.
pub fn load_gray(path: &Path) -> Result<Gray> {
    let img = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("{}: not a readable image", path.display()))?;
    Ok(to_gray(&img))
}

pub fn to_gray(img: &image::DynamicImage) -> Gray {
    use image::DynamicImage as D;
    let (w, h) = (img.width() as usize, img.height() as usize);
    match img {
        D::ImageLuma8(g) => Gray::new(w, h, g.as_raw().clone()),
        _ => {
            let rgb = img.to_rgb8();
            let data = rgb
                .pixels()
                .map(|p| {
                    let [r, g, b] = p.0.map(|v| v as u32);
                    ((r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14) as u8
                })
                .collect();
            Gray::new(w, h, data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerators() {
        for t in ["1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)"] {
            assert!(is_enumerator(t), "{t}");
        }
        for t in ["1.5", "(1", "ab.", "1234.", "(.)"] {
            assert!(!is_enumerator(t), "{t}");
        }
    }

    #[test]
    fn arange_matches_numpy() {
        assert_eq!(arange(-45.0, 45.0, 2.0).len(), 46);
        assert_eq!(arange(-2.0, 2.0, 0.5).len(), 9);
        assert_eq!(arange(0.5, 1.5, 0.1).len(), 11);
    }

    #[test]
    fn meta_keys_in_file_order() {
        let raw =
            r#"{"vocab": "x", "vocab_world": "a", "n": 1, "vocab_rtl": "b", "vocab_cjk": "c"}"#;
        assert_eq!(
            meta_key_order(raw, "vocab_"),
            vec!["vocab_world", "vocab_rtl", "vocab_cjk"]
        );
    }
}
