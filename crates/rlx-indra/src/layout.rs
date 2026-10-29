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

//! Page markdown from the ordered lines and the layout heads. Walking the
//! reading order, a block starts where the same-block logit between
//! neighbours is negative; each block renders by its class (section header,
//! paragraph, list, HTML table; page headers and footers dropped unless kept).

use crate::order::{BLOCK_CLASSES, Rect, block_class, class_id, cluster, median};
use crate::text::{is_alpha, is_lower, is_space};
use serde::Serialize;

/// One layout block: its class, the union of its lines, its text.
#[derive(Clone, Debug, Serialize)]
pub struct Block {
    pub label: String,
    pub bbox: [f64; 4],
    pub text: String,
}

/// Python's `round(v, digits)`: the exact binary value rounded half to even.
pub fn round_to(v: f64, digits: usize) -> f64 {
    format!("{v:.digits$}").parse().unwrap_or(v)
}

pub fn round1(v: f64) -> f64 {
    round_to(v, 1)
}

fn is_cjk(c: char) -> bool {
    let o = c as u32;
    (0x3000..=0x30FF).contains(&o)
        || (0x3400..=0x9FFF).contains(&o)
        || (0xF900..=0xFAFF).contains(&o)
        || (0xFF00..=0xFFEF).contains(&o)
        || (0x20000..=0x2FA1F).contains(&o)
}

/// Join a paragraph's lines: a word hyphenated at a line end is rejoined, a
/// break between CJK characters takes no space.
pub fn join_lines<S: AsRef<str>>(texts: &[S]) -> String {
    let mut out: Vec<char> = Vec::new();
    for t in texts {
        let t: Vec<char> = t.as_ref().trim().chars().collect();
        if t.is_empty() {
            continue;
        }
        let n = out.len();
        if n > 1 && out[n - 1] == '-' && is_alpha(out[n - 2]) && is_alpha(t[0]) && is_lower(t[0]) {
            out.pop();
            out.extend(t);
        } else if n > 0 && is_cjk(out[n - 1]) && is_cjk(t[0]) {
            out.extend(t);
        } else {
            if n > 0 {
                out.push(' ');
            }
            out.extend(t);
        }
    }
    out.into_iter().collect()
}

/// Runs of the reading order with a positive same-block logit between neighbours.
pub fn split_blocks(order: &[usize], same: &[f32], n: usize) -> Vec<Vec<usize>> {
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for &i in order {
        match blocks.last_mut() {
            Some(b) if same[b[b.len() - 1] * n + i] > 0.0 => b.push(i),
            _ => blocks.push(vec![i]),
        }
    }
    blocks
}

fn escape(t: &str) -> String {
    t.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Cell lines grouped into rows by vertical centre, each row left to right.
fn rows_of(boxes: &[Rect]) -> Vec<Vec<usize>> {
    let cy: Vec<f32> = boxes.iter().map(|b| (b[1] + b[3]) / 2.0).collect();
    let h = median(boxes.iter().map(|b| b[3] - b[1]));
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| cy[a].total_cmp(&cy[b]));
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for i in order {
        if let Some(r) = rows.last_mut() {
            let mean = r.iter().map(|&k| cy[k]).sum::<f32>() / r.len() as f32;
            if (cy[i] - mean).abs() < 0.5 * h {
                r.push(i);
                continue;
            }
        }
        rows.push(vec![i]);
    }
    for r in &mut rows {
        r.sort_by(|&a, &b| boxes[a][0].total_cmp(&boxes[b][0]));
    }
    rows
}

/// Join the line rows of wrapped cells into one table row: a small gap
/// (under half the median row gap and 0.6 line heights) and fewer lines than
/// the row above.
fn merge_wrapped(rows: Vec<Vec<usize>>, boxes: &[Rect]) -> Vec<Vec<usize>> {
    if rows.len() < 3 {
        return rows;
    }
    let h = median(boxes.iter().map(|b| b[3] - b[1]));
    let gaps: Vec<f32> = rows
        .windows(2)
        .map(|w| {
            let top = w[1]
                .iter()
                .map(|&k| boxes[k][1])
                .fold(f32::INFINITY, f32::min);
            let bot = w[0]
                .iter()
                .map(|&k| boxes[k][3])
                .fold(f32::NEG_INFINITY, f32::max);
            top - bot
        })
        .collect();
    let med = median(gaps.iter().copied());
    let mut out = vec![rows[0].clone()];
    for (g, (prev, r)) in gaps.iter().zip(rows.iter().zip(rows.iter().skip(1))) {
        if *g < 0.5 * med && *g < 0.6 * h && r.len() < prev.len() {
            out.last_mut().unwrap().extend(r);
        } else {
            out.push(r.clone());
        }
    }
    out
}

fn tr(
    tag: &str,
    cells: &[(usize, String)],
    width: usize,
    at: impl Fn(usize) -> Option<usize>,
) -> String {
    let mut tds = String::new();
    let mut c = 0;
    while c < width {
        match at(c) {
            Some(k) => {
                let (n, text) = &cells[k];
                let attr = if *n > 1 {
                    format!(" colspan=\"{n}\"")
                } else {
                    String::new()
                };
                tds.push_str(&format!("<{tag}{attr}>{}</{tag}>", escape(text)));
                c += n;
            }
            None => {
                tds.push_str(&format!("<{tag}></{tag}>"));
                c += 1;
            }
        }
    }
    format!("<tr>{tds}</tr>")
}

/// An HTML table from cell lines: rows by vertical centre, columns anchored
/// on the fullest rows; a first row that spans most of the table is a title.
pub fn table_html(boxes: &[Rect], texts: &[String]) -> String {
    let rows = rows_of(boxes);
    let span_x = boxes.iter().map(|b| b[2]).fold(f32::NEG_INFINITY, f32::max)
        - boxes.iter().map(|b| b[0]).fold(f32::INFINITY, f32::min);
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if rows.len() > 2 && rows[0].len() == 1 && width > 1 {
        let t = rows[0][0];
        if boxes[t][2] - boxes[t][0] > 0.6 * span_x {
            let keep: Vec<usize> = (0..texts.len()).filter(|&k| k != t).collect();
            let kb: Vec<Rect> = keep.iter().map(|&k| boxes[k]).collect();
            let kt: Vec<String> = keep.iter().map(|&k| texts[k].clone()).collect();
            return format!("{}\n\n{}", texts[t], table_html(&kb, &kt));
        }
    }
    let full: Vec<&Vec<usize>> = rows.iter().filter(|r| r.len() == width).collect();
    let rows = merge_wrapped(rows.clone(), boxes);
    let bands: Vec<[f32; 2]> = (0..width)
        .map(|c| {
            [
                median(full.iter().map(|r| boxes[r[c]][0])),
                median(full.iter().map(|r| boxes[r[c]][2])),
            ]
        })
        .collect();
    let mut edges = vec![f32::NEG_INFINITY];
    for c in 0..width.saturating_sub(1) {
        edges.push((bands[c][1] + bands[c + 1][0]) / 2.0);
    }
    edges.push(f32::INFINITY);
    // np.searchsorted(edges, x, side="right") - 1
    let band_of = |x: f32| edges.iter().filter(|&&e| e <= x).count() as i64 - 1;
    let mut out = vec!["<table>".to_string()];
    for (ri, r) in rows.iter().enumerate() {
        let tag = if ri == 0 && rows.len() > 1 {
            "th"
        } else {
            "td"
        };
        let mut slots: Vec<Option<usize>> = vec![None; width];
        let mut cells: Vec<(usize, String)> = Vec::new();
        let mut sorted = r.clone();
        sorted.sort_by(|&a, &b| {
            boxes[a][1]
                .total_cmp(&boxes[b][1])
                .then(boxes[a][0].total_cmp(&boxes[b][0]))
        });
        for k in sorted {
            let (x0, x1) = (boxes[k][0], boxes[k][2]);
            let first = band_of(x0);
            let last = band_of(x1);
            let lo = first.max(0) as usize;
            let hi = last.min(width as i64 - 1);
            let mut span: Vec<usize> = (lo as i64..=hi)
                .map(|c| c as usize)
                .filter(|&c| {
                    x1.min(bands[c][1]) - x0.max(bands[c][0]) > 0.5 * (bands[c][1] - bands[c][0])
                })
                .collect();
            if span.is_empty() {
                span.push((first.max(0) as usize).min(width - 1));
            }
            let c0 = span[0];
            match slots[c0] {
                Some(i) => {
                    let (n, text) = &cells[i];
                    cells[i] = ((*n).max(span.len()), format!("{text} {}", texts[k]));
                }
                None => {
                    slots[c0] = Some(cells.len());
                    cells.push((span.len(), texts[k].clone()));
                }
            }
        }
        out.push(tr(tag, &cells, width, |c| slots[c]));
    }
    out.push("</table>".into());
    out.join("\n")
}

/// The cell head joins side-by-side lines in more than 10% of its cells:
/// it failed on this table, use geometry instead.
pub fn heads_plausible(boxes: &[Rect], cell_logits: &[f32]) -> bool {
    let n = boxes.len();
    let cells = cluster(cell_logits, n);
    let bad = cells
        .iter()
        .filter(|c| c.len() >= 2)
        .filter(|c| {
            c.iter().any(|&i| {
                c.iter()
                    .any(|&j| boxes[i][2].min(boxes[j][2]) - boxes[i][0].max(boxes[j][0]) < 0.0)
            })
        })
        .count();
    bad as f64 <= 0.1 * cells.len() as f64
}

/// An HTML table from the order model's table heads (`tab [3, N, N]`: same
/// row, column, cell) for one table block's lines.
pub fn table_html_heads(boxes: &[Rect], texts: &[String], tab: &[f32]) -> String {
    let n = boxes.len();
    let plane = |k: usize| &tab[k * n * n..(k + 1) * n * n];
    let cells = cluster(plane(2), n);
    let cb: Vec<Rect> = cells
        .iter()
        .map(|c| {
            [
                c.iter().map(|&k| boxes[k][0]).fold(f32::INFINITY, f32::min),
                c.iter().map(|&k| boxes[k][1]).fold(f32::INFINITY, f32::min),
                c.iter()
                    .map(|&k| boxes[k][2])
                    .fold(f32::NEG_INFINITY, f32::max),
                c.iter()
                    .map(|&k| boxes[k][3])
                    .fold(f32::NEG_INFINITY, f32::max),
            ]
        })
        .collect();
    let ctext: Vec<String> = cells
        .iter()
        .map(|c| {
            let mut c = c.clone();
            c.sort_by(|&a, &b| {
                boxes[a][1]
                    .total_cmp(&boxes[b][1])
                    .then(boxes[a][0].total_cmp(&boxes[b][0]))
            });
            c.iter()
                .map(|&k| texts[k].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let m = cells.len();
    let cell_logits = |k: usize| -> Vec<f32> {
        let p = plane(k);
        let mut out = vec![0f32; m * m];
        for a in 0..m {
            for b in a + 1..m {
                let s: f32 = cells[a]
                    .iter()
                    .flat_map(|&i| cells[b].iter().map(move |&j| p[i * n + j]))
                    .sum();
                let v = s / (cells[a].len() * cells[b].len()) as f32;
                out[a * m + b] = v;
                out[b * m + a] = v;
            }
        }
        out
    };
    let min_by =
        |g: &Vec<usize>, axis: usize| g.iter().map(|&c| cb[c][axis]).fold(f32::INFINITY, f32::min);
    let mut rows = cluster(&cell_logits(0), m);
    rows.sort_by(|a, b| min_by(a, 1).total_cmp(&min_by(b, 1)));
    let mut cols = cluster(&cell_logits(1), m);
    cols.sort_by(|a, b| min_by(a, 0).total_cmp(&min_by(b, 0)));
    let mut row_of = vec![0; m];
    let mut col_of = vec![0; m];
    for (r, g) in rows.iter().enumerate() {
        for &c in g {
            row_of[c] = r;
        }
    }
    for (k, g) in cols.iter().enumerate() {
        for &c in g {
            col_of[c] = k;
        }
    }
    let centers: Vec<f32> = cols
        .iter()
        .map(|g| median(g.iter().map(|&c| (cb[c][0] + cb[c][2]) / 2.0)))
        .collect();
    let h = median(boxes.iter().map(|b| b[3] - b[1]));
    let mut grid: std::collections::HashMap<(usize, usize), (usize, String)> = Default::default();
    for c in 0..m {
        let c0 = col_of[c];
        let mut span = 1;
        while c0 + span < cols.len()
            && cb[c][0] - h <= centers[c0 + span]
            && centers[c0 + span] <= cb[c][2] + h
        {
            span += 1;
        }
        let key = (row_of[c], c0);
        let entry = match grid.remove(&key) {
            Some((s, t)) => (s.max(span), format!("{t} {}", ctext[c])),
            None => (span, ctext[c].clone()),
        };
        grid.insert(key, entry);
    }
    let mut out = vec!["<table>".to_string()];
    for r in 0..rows.len() {
        let tag = if r == 0 && rows.len() > 1 { "th" } else { "td" };
        let cells: Vec<(usize, String)> = (0..cols.len())
            .map(|c| grid.get(&(r, c)).cloned().unwrap_or((0, String::new())))
            .collect();
        out.push(tr(tag, &cells, cols.len(), |c| {
            grid.contains_key(&(r, c)).then_some(c)
        }));
    }
    out.push("</table>".into());
    out.join("\n")
}

/// The layout blocks behind [`page_markdown`], in reading order.
pub fn page_blocks(
    boxes: &[Rect],
    texts: &[String],
    order: &[usize],
    cls: &[f32],
    classes: usize,
    same: &[f32],
) -> Vec<Block> {
    split_blocks(order, same, boxes.len())
        .into_iter()
        .map(|blk| Block {
            label: BLOCK_CLASSES[block_class(cls, classes, &blk)].to_string(),
            bbox: union(boxes, &blk).map(|v| round1(v as f64)),
            text: blk
                .iter()
                .map(|&i| texts[i].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        })
        .collect()
}

pub fn union(boxes: &[Rect], idx: &[usize]) -> Rect {
    [
        idx.iter()
            .map(|&i| boxes[i][0])
            .fold(f32::INFINITY, f32::min),
        idx.iter()
            .map(|&i| boxes[i][1])
            .fold(f32::INFINITY, f32::min),
        idx.iter()
            .map(|&i| boxes[i][2])
            .fold(f32::NEG_INFINITY, f32::max),
        idx.iter()
            .map(|&i| boxes[i][3])
            .fold(f32::NEG_INFINITY, f32::max),
    ]
}

/// A list marker followed by text or alone: `^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)`.
fn is_item(t: &str) -> bool {
    let c: Vec<char> = t.trim_start().chars().collect();
    let end_ok = |i: usize| c.get(i).is_none_or(|&ch| is_space(ch));
    if c.first().is_some_and(|ch| "•-–▪◦*·→".contains(*ch)) && end_ok(1) {
        return true;
    }
    let mut i = usize::from(c.first() == Some(&'('));
    let start = i;
    while i < c.len() && i - start < 3 && is_digit_d(c[i]) {
        i += 1;
    }
    if i > start && c.get(i).is_some_and(|ch| ".)".contains(*ch)) && end_ok(i + 1) {
        return true;
    }
    let i = start;
    c.get(i).is_some_and(char::is_ascii_alphabetic)
        && c.get(i + 1).is_some_and(|ch| ".)".contains(*ch))
        && end_ok(i + 2)
}

/// `\d` of Python's `re` on `str`: a decimal digit.
fn is_digit_d(c: char) -> bool {
    crate::text::category(c) == unicode_general_category::GeneralCategory::DecimalNumber
}

/// A line that is one formula: `^\$[^$]+\$$`.
fn whole_math(t: &str) -> bool {
    t.len() > 2 && t.starts_with('$') && t.ends_with('$') && !t[1..t.len() - 1].contains('$')
}

/// Optional inputs of [`page_markdown`].
#[derive(Default)]
pub struct MarkdownExtras<'a> {
    pub keep_furniture: bool,
    /// Per line, its table cells `(box, text)` when it holds several.
    pub cells: Option<&'a [Vec<(Rect, String)>]>,
    /// `[3, N, N]` same row / column / cell logits: tables from the heads.
    pub tab: Option<&'a [f32]>,
    /// Picture regions, written as `![](picture-<k>)`.
    pub pictures: &'a [Rect],
}

/// Markdown for one page.
pub fn page_markdown(
    boxes: &[Rect],
    texts: &[String],
    order: &[usize],
    cls: &[f32],
    classes: usize,
    same: &[f32],
    x: &MarkdownExtras<'_>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut part_boxes: Vec<Rect> = Vec::new();
    for blk in split_blocks(order, same, boxes.len()) {
        let before = parts.len();
        block_part(&mut parts, &blk, boxes, texts, cls, classes, x);
        if parts.len() > before {
            part_boxes.push(union(boxes, &blk));
        }
    }
    if !x.pictures.is_empty() {
        let mut slots: Vec<Vec<String>> = vec![Vec::new(); parts.len() + 1];
        for (k, p) in x.pictures.iter().enumerate() {
            let cy = (p[1] + p[3]) / 2.0;
            let at = part_boxes
                .iter()
                .position(|b| (b[1] + b[3]) / 2.0 > cy && p[2].min(b[2]) > p[0].max(b[0]))
                .unwrap_or(parts.len());
            slots[at].push(format!("![](picture-{k})"));
        }
        let mut merged = Vec::new();
        for (j, slot) in slots.into_iter().enumerate() {
            merged.extend(slot);
            if j < parts.len() {
                merged.push(std::mem::take(&mut parts[j]));
            }
        }
        parts = merged;
    }
    parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn block_part(
    parts: &mut Vec<String>,
    blk: &[usize],
    boxes: &[Rect],
    texts: &[String],
    cls: &[f32],
    classes: usize,
    x: &MarkdownExtras<'_>,
) {
    let c = block_class(cls, classes, blk);
    let tb: Vec<&str> = blk.iter().map(|&i| texts[i].as_str()).collect();
    if (c == class_id("Page-Header") || c == class_id("Page-Footer")) && !x.keep_furniture {
        return;
    }
    if tb.iter().all(|t| whole_math(t.trim())) {
        parts.push(
            tb.iter()
                .map(|t| {
                    let t = t.trim();
                    format!("$${}$$", &t[1..t.len() - 1])
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
        return;
    }
    let table = class_id("Table");
    let n = boxes.len();
    let bb: Vec<Rect> = blk.iter().map(|&i| boxes[i]).collect();
    let sub_tab = x.tab.map(|tab| {
        let m = blk.len();
        let mut out = vec![0f32; 3 * m * m];
        for k in 0..3 {
            for (a, &i) in blk.iter().enumerate() {
                for (b, &j) in blk.iter().enumerate() {
                    out[(k * m + a) * m + b] = tab[(k * n + i) * n + j];
                }
            }
        }
        out
    });
    if c == class_id("Section-Header") {
        parts.push(format!("## {}", join_lines(&tb)));
    } else if c == table
        && blk.len() > 1
        && let Some(st) = &sub_tab
        && heads_plausible(&bb, &st[2 * blk.len() * blk.len()..])
    {
        let tt: Vec<String> = tb.iter().map(|s| s.to_string()).collect();
        parts.push(table_html_heads(&bb, &tt, st));
    } else if c == table && blk.len() > 1 {
        let (mut cb, mut ct) = (Vec::new(), Vec::new());
        for &i in blk {
            match x
                .cells
                .and_then(|cells| cells.get(i))
                .filter(|v| !v.is_empty())
            {
                Some(v) => {
                    for (b, t) in v {
                        cb.push(*b);
                        ct.push(t.clone());
                    }
                }
                None => {
                    cb.push(boxes[i]);
                    ct.push(texts[i].clone());
                }
            }
        }
        parts.push(table_html(&cb, &ct));
    } else if c == class_id("List-Group") {
        let mut items: Vec<Vec<&str>> = Vec::new();
        for t in &tb {
            if items.is_empty() || is_item(t) {
                items.push(vec![t]);
            } else {
                items.last_mut().unwrap().push(t);
            }
        }
        parts.push(
            items
                .iter()
                .map(|it| join_lines(it))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    } else {
        parts.push(join_lines(&tb));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_rules() {
        assert_eq!(join_lines(&["tumori-", "genesis is"]), "tumorigenesis is");
        assert_eq!(join_lines(&["中文", "句子"]), "中文句子");
        assert_eq!(join_lines(&["a", " ", "b"]), "a b");
    }

    #[test]
    fn items() {
        for t in ["• one", "1. first", "(a) x", "12)", "-"] {
            assert!(is_item(t), "{t}");
        }
        for t in ["1.5 m", "ab. c", "Word"] {
            assert!(!is_item(t), "{t}");
        }
    }

    #[test]
    fn simple_table() {
        let boxes = [
            [0.0, 0.0, 40.0, 10.0],
            [100.0, 0.0, 140.0, 10.0],
            [0.0, 20.0, 40.0, 30.0],
            [100.0, 20.0, 140.0, 30.0],
        ];
        let texts: Vec<String> = ["A", "B", "1", "2 & 3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            table_html(&boxes, &texts),
            "<table>\n<tr><th>A</th><th>B</th></tr>\n<tr><td>1</td><td>2 &amp; 3</td></tr>\n</table>"
        );
    }
}
