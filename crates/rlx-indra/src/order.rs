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

//! Reading order: the order model's line features, decoding its pairwise
//! logits into an order, block clustering, and the XY-cut fallbacks.

use crate::text::{is_digit, is_upper, is_wide};

pub type Rect = [f32; 4];

/// Block classes of the layout head, in its output order.
pub const BLOCK_CLASSES: [&str; 10] = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
];
pub const GRID_LABELS: [&str; 3] = ["Table", "Form", "Table-Of-Contents"];

pub fn class_id(label: &str) -> usize {
    BLOCK_CLASSES.iter().position(|&c| c == label).unwrap_or(0)
}

const SENTENCE_END: &str = ".!?:;。！？：；．";

/// `np.median` of float32 values (mean of the middle two, in f32).
pub fn median(v: impl IntoIterator<Item = f32>) -> f32 {
    let mut v: Vec<f32> = v.into_iter().collect();
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Indices sorted by `key` (stable).
fn argsort(key: &[f32]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..key.len()).collect();
    o.sort_by(|&a, &b| key[a].total_cmp(&key[b]));
    o
}

fn display_len(t: &str) -> usize {
    t.chars().map(|c| if is_wide(c) { 2 } else { 1 }).sum()
}

/// `[N, 16]` order-model features: ten of geometry, six of text.
pub fn line_features(boxes: &[Rect], page_w: f32, page_h: f32, texts: &[String]) -> Vec<f32> {
    let n = boxes.len();
    let mut out = vec![0f32; n * 16];
    if n == 0 {
        return out;
    }
    let hs: Vec<f32> = boxes.iter().map(|b| (b[3] - b[1]).max(1e-3)).collect();
    let med_h = median(hs.iter().copied());
    for (i, b) in boxes.iter().enumerate() {
        let w = (b[2] - b[0]).max(1e-3);
        let h = hs[i];
        let f = &mut out[i * 16..(i + 1) * 16];
        f[0] = b[0] / page_w;
        f[1] = b[1] / page_h;
        f[2] = b[2] / page_w;
        f[3] = b[3] / page_h;
        f[4] = (b[0] + b[2]) / (2.0 * page_w);
        f[5] = (b[1] + b[3]) / (2.0 * page_h);
        f[6] = w / page_w;
        f[7] = (h / med_h).ln();
        f[8] = (w / h).ln() / 4.0;
        f[9] = (med_h as f64 / page_h as f64 * 20.0) as f32;
        let t = texts.get(i).map_or("", |s| s.trim());
        let chars: Vec<char> = t.chars().collect();
        if chars.is_empty() {
            continue;
        }
        f[10] = ((display_len(t) as f64).ln_1p() / 5.0) as f32;
        f[11] = (chars.iter().filter(|&&c| is_digit(c)).count() as f64 / chars.len() as f64) as f32;
        let head: String = chars.iter().take(3).collect();
        let head = head.trim_end_matches(['.', ')']);
        let numbered = !head.is_empty() && head.chars().all(is_digit) && chars.len() > 3;
        f[12] = ("•-–▪◦*·→".contains(chars[0]) || numbered) as u8 as f32;
        f[13] = (*chars.last().unwrap() == '-') as u8 as f32;
        f[14] = SENTENCE_END.contains(*chars.last().unwrap()) as u8 as f32;
        f[15] = is_upper(chars[0]) as u8 as f32;
    }
    out
}

/// Order from a pairwise logit matrix (`a[i·n + j] > 0`: i before j): the
/// row-sum ranking, then insertion moves that raise the summed log-probability
/// of the pairwise relations.
pub fn decode_order(a: &[f32], n: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }
    // log P(i before j) = -logaddexp(0, -a)
    let lp: Vec<f32> = a
        .iter()
        .map(|&v| {
            let x = -v;
            -(x.max(0.0) + (-(x.abs())).exp().ln_1p())
        })
        .collect();
    let key: Vec<f32> = (0..n)
        .map(|i| {
            let row: f32 = (0..n).map(|j| lp[i * n + j]).sum();
            let col: f32 = (0..n).map(|j| lp[j * n + i]).sum();
            -row + col
        })
        .collect();
    let mut order = argsort(&key);
    for _ in 0..4 {
        let mut improved = false;
        for x in order.clone() {
            let rest: Vec<usize> = order.iter().copied().filter(|&v| v != x).collect();
            let m = rest.len();
            let mut gains = vec![0f32; m + 1];
            let mut acc = 0f32;
            for k in 0..m {
                acc += lp[rest[k] * n + x];
                gains[k + 1] = acc;
            }
            let mut acc = 0f32;
            let mut tail = vec![0f32; m + 1];
            for k in (0..m).rev() {
                acc += lp[x * n + rest[k]];
                tail[k] = acc;
            }
            for k in 0..=m {
                gains[k] += tail[k];
            }
            let mut best = 0;
            for k in 1..=m {
                if gains[k] > gains[best] {
                    best = k;
                }
            }
            let cur = order.iter().position(|&v| v == x).unwrap();
            if gains[best] > gains[cur] + 1e-6 {
                let mut o = rest;
                o.insert(best, x);
                order = o;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    order
}

/// Rows top to bottom (centres within half a line height share a row), left
/// to right.
pub fn order_top_down(boxes: &[Rect]) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let h = median(boxes.iter().map(|b| b[3] - b[1]));
    let step = (h * 0.5).max(1e-3);
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    idx.sort_by(|&i, &j| {
        let ri = (((boxes[i][1] + boxes[i][3]) / 2.0) / step).round_ties_even();
        let rj = (((boxes[j][1] + boxes[j][3]) / 2.0) / step).round_ties_even();
        ri.total_cmp(&rj).then(boxes[i][0].total_cmp(&boxes[j][0]))
    });
    idx
}

/// Widest whitespace gap along `axis` (1 = rows, 0 = columns): (gap, split
/// position in the sorted order, the order).
fn best_cut(b: &[Rect], axis: usize) -> (f32, Option<usize>, Vec<usize>) {
    let (lo, hi) = if axis == 1 { (1, 3) } else { (0, 2) };
    let key: Vec<f32> = b.iter().map(|r| r[lo]).collect();
    let o = argsort(&key);
    let mut end = b[o[0]][hi];
    let (mut best_gap, mut best_k) = (0f32, None);
    for k in 1..o.len() {
        let gap = b[o[k]][lo] - end;
        if gap > best_gap {
            best_gap = gap;
            best_k = Some(k);
        }
        end = end.max(b[o[k]][hi]);
    }
    (best_gap, best_k, o)
}

/// Recursive XY-cut: split at the widest whitespace gap on either axis, upper
/// or left part first; without a gap, top-down order.
pub fn order_xycut(boxes: &[Rect]) -> Vec<usize> {
    fn rec(boxes: &[Rect], ids: Vec<usize>) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids;
        }
        let b: Vec<Rect> = ids.iter().map(|&i| boxes[i]).collect();
        let (gy, ky, oy) = best_cut(&b, 1);
        let (gx, kx, ox) = best_cut(&b, 0);
        if ky.is_none() && kx.is_none() {
            return order_top_down(&b).into_iter().map(|i| ids[i]).collect();
        }
        // The axis with the wider gap; a positive gap always has a cut.
        let (k, o) = if gy >= gx { (ky, oy) } else { (kx, ox) };
        let k = k.expect("the wider gap has a cut");
        let mut out = rec(boxes, o[..k].iter().map(|&i| ids[i]).collect());
        out.extend(rec(boxes, o[k..].iter().map(|&i| ids[i]).collect()));
        out
    }
    rec(boxes, (0..boxes.len()).collect())
}

/// Blocks in reading order by recursive XY-cut with a gap of at least one
/// median line height; lines inside a block by [`order_xycut`].
pub fn xycut_blocks(boxes: &[Rect]) -> Vec<Vec<usize>> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let h = median(boxes.iter().map(|b| b[3] - b[1]));
    fn rec(boxes: &[Rect], ids: Vec<usize>, h: f32, out: &mut Vec<Vec<usize>>) {
        let b: Vec<Rect> = ids.iter().map(|&i| boxes[i]).collect();
        if ids.len() > 1 {
            let (gy, ky, oy) = best_cut(&b, 1);
            let (gx, kx, ox) = best_cut(&b, 0);
            let (gap, k, o) = if gy >= gx { (gy, ky, oy) } else { (gx, kx, ox) };
            if let Some(k) = k
                && gap >= h
            {
                rec(boxes, o[..k].iter().map(|&i| ids[i]).collect(), h, out);
                rec(boxes, o[k..].iter().map(|&i| ids[i]).collect(), h, out);
                return;
            }
        }
        out.push(order_xycut(&b).into_iter().map(|i| ids[i]).collect());
    }
    let mut out = Vec::new();
    rec(boxes, (0..boxes.len()).collect(), h, &mut out);
    out
}

/// Average-linkage clustering of pairwise "same group" logits (`n × n`):
/// merge the two groups with the highest mean logit while it is positive.
pub fn cluster(logits: &[f32], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    let mut s: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    for i in 0..n {
        s[i * n + i] = 0.0;
    }
    let mut size = vec![1f64; n];
    let mut alive = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut m = s.clone();
    for i in 0..n {
        m[i * n + i] = f64::NEG_INFINITY;
    }
    let argmax = |row: &[f64]| -> usize {
        let mut b = 0;
        for j in 1..row.len() {
            if row[j] > row[b] {
                b = j;
            }
        }
        b
    };
    let mut best: Vec<usize> = (0..n).map(|i| argmax(&m[i * n..(i + 1) * n])).collect();
    let mut best_v: Vec<f64> = (0..n).map(|i| m[i * n + best[i]]).collect();
    for _ in 0..n - 1 {
        let a = argmax(&best_v);
        let b = best[a];
        if best_v[a] <= 0.0 {
            break;
        }
        let (a, b) = (a.min(b), a.max(b));
        for j in 0..n {
            s[a * n + j] += s[b * n + j];
        }
        for j in 0..n {
            s[j * n + a] += s[j * n + b];
        }
        s[a * n + a] = 0.0;
        size[a] += size[b];
        alive[b] = false;
        let mb = std::mem::take(&mut members[b]);
        members[a].extend(mb);
        let mut row: Vec<f64> = (0..n).map(|j| s[a * n + j] / (size[a] * size[j])).collect();
        for j in 0..n {
            if !alive[j] {
                row[j] = f64::NEG_INFINITY;
            }
        }
        row[a] = f64::NEG_INFINITY;
        for j in 0..n {
            m[a * n + j] = row[j];
            m[j * n + a] = row[j];
            m[b * n + j] = f64::NEG_INFINITY;
            m[j * n + b] = f64::NEG_INFINITY;
        }
        best_v[b] = f64::NEG_INFINITY;
        best[a] = argmax(&row);
        best_v[a] = row[best[a]];
        for j in 0..n {
            if j != a && alive[j] && (best[j] == a || best[j] == b) {
                best[j] = argmax(&m[j * n..(j + 1) * n]);
                best_v[j] = m[j * n + best[j]];
            }
        }
        for j in 0..n {
            if j != a && alive[j] && row[j] > best_v[j] {
                best[j] = a;
                best_v[j] = row[j];
            }
        }
    }
    (0..n)
        .filter(|&i| alive[i])
        .map(|i| {
            let mut g = members[i].clone();
            g.sort_unstable();
            g
        })
        .collect()
}

/// Class with the largest summed logit over `lines` (`cls` is `[N, C]`).
pub fn block_class(cls: &[f32], classes: usize, lines: &[usize]) -> usize {
    let mut sum = vec![0f32; classes];
    for &i in lines {
        for c in 0..classes {
            sum[c] += cls[i * classes + c];
        }
    }
    let mut best = 0;
    for c in 1..classes {
        if sum[c] > sum[best] {
            best = c;
        }
    }
    best
}

/// Reading order with the layout heads: blocks from the same-block logits,
/// lines in a block by XY-cut (row by row in a grid block), blocks by the mean
/// order logit between them.
pub fn decode_blocks_order(
    a: &[f32],
    same: &[f32],
    boxes: &[Rect],
    cls: &[f32],
    classes: usize,
) -> Vec<usize> {
    let n = boxes.len();
    if n <= 1 {
        return (0..n).collect();
    }
    let blocks = cluster(same, n);
    let grid: Vec<usize> = GRID_LABELS.iter().map(|g| class_id(g)).collect();
    let inner: Vec<Vec<usize>> = blocks
        .iter()
        .map(|b| {
            let bx: Vec<Rect> = b.iter().map(|&i| boxes[i]).collect();
            let local = if grid.contains(&block_class(cls, classes, b)) {
                order_top_down(&bx)
            } else {
                order_xycut(&bx)
            };
            local.into_iter().map(|k| b[k]).collect()
        })
        .collect();
    let m = blocks.len();
    let mut ab = vec![0f32; m * m];
    let mean = |p: &[usize], q: &[usize]| -> f64 {
        let s: f64 = p
            .iter()
            .flat_map(|&i| q.iter().map(move |&j| a[i * n + j] as f64))
            .sum();
        s / (p.len() * q.len()) as f64
    };
    for p in 0..m {
        for q in p + 1..m {
            let v = mean(&blocks[p], &blocks[q]);
            ab[p * m + q] = v as f32;
            ab[q * m + p] = -v as f32;
        }
    }
    decode_order(&ab, m)
        .into_iter()
        .flat_map(|k| inner[k].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_columns_read_column_by_column() {
        let boxes = [
            [10.0, 10.0, 90.0, 20.0],
            [110.0, 10.0, 190.0, 20.0],
            [10.0, 30.0, 90.0, 40.0],
            [110.0, 30.0, 190.0, 40.0],
        ];
        assert_eq!(order_xycut(&boxes), vec![0, 2, 1, 3]);
        assert_eq!(order_top_down(&boxes), vec![0, 1, 2, 3]);
    }

    #[test]
    fn cluster_joins_positive_pairs() {
        #[rustfmt::skip]
        let s = [
            0.0, 5.0, -5.0,
            5.0, 0.0, -5.0,
            -5.0, -5.0, 0.0,
        ];
        assert_eq!(cluster(&s, 3), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn decode_order_follows_logits() {
        // 2 before 0 before 1.
        #[rustfmt::skip]
        let a = [
            0.0, 4.0, -4.0,
            -4.0, 0.0, -4.0,
            4.0, 4.0, 0.0,
        ];
        assert_eq!(decode_order(&a, 3), vec![2, 0, 1]);
    }
}
