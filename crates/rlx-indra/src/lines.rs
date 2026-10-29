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

//! Detection input and decoding, and the recognizer's line crops.

use crate::graph::det::{INPUT_MULTIPLE, OUT_STRIDE};
use crate::imgproc::{Gray, Interp, connected_components, resize};

/// The line core is shorter than the line by this fraction of its height, at
/// each end (the decoder grows it back).
const CORE_X_SHRINK: f64 = 0.15;

/// A line box in pixels, with the detector's mean core probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl LineBox {
    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }
    pub fn rect(&self) -> [f32; 4] {
        [self.x0, self.y0, self.x1, self.y1]
    }
}

/// `round(v)` the Python way (half to even).
pub fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// The page scaled so its long side is `long_side` and padded with paper to
/// a multiple of 32: (ink map `[H, W]`, height, width, scale).
pub fn page_to_input(gray: &Gray, long_side: usize) -> (Vec<f32>, usize, usize, f64) {
    let (h, w) = (gray.height, gray.width);
    let scale = long_side as f64 / h.max(w) as f64;
    let nh = (py_round(h as f64 * scale) as usize).max(1);
    let nw = (py_round(w as f64 * scale) as usize).max(1);
    let small;
    let g = if (nh, nw) != (h, w) {
        let interp = if scale < 1.0 {
            Interp::Area
        } else {
            Interp::Linear
        };
        small = resize(gray, nw, nh, interp);
        &small
    } else {
        gray
    };
    let ph = nh.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let pw = nw.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let mut out = vec![0f32; ph * pw];
    for y in 0..nh {
        for x in 0..nw {
            out[y * pw + x] = 1.0 - g.at(x, y) as f32 / 255.0;
        }
    }
    (out, ph, pw, scale)
}

/// Line boxes (input pixels) from the detector's `[C, h, w]` output: 4-connected
/// regions of core probability above 0.5, tops and bottoms from the mean
/// predicted distances.
pub fn decode(out: &[f32], h: usize, w: usize) -> Vec<LineBox> {
    let plane = h * w;
    let prob: Vec<f32> = out[..plane]
        .iter()
        .map(|&l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    let mask: Vec<bool> = prob.iter().map(|&p| p > 0.5).collect();
    let (labels, comps) = connected_components(&mask, w, h, false);
    let n = comps.len() + 1;
    let s = OUT_STRIDE as f32;
    let (mut top, mut bot, mut sc, mut cnt) =
        (vec![0f64; n], vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let y = ((i / w) as f32 + 0.5) * s;
        let l = l as usize;
        top[l] += (y - out[plane + i]) as f64;
        bot[l] += (y + out[2 * plane + i]) as f64;
        sc[l] += prob[i] as f64;
        cnt[l] += 1.0;
    }
    let mut boxes = Vec::new();
    for (k, c) in comps.iter().enumerate() {
        let i = k + 1;
        if c.area < 2 {
            continue;
        }
        let d = cnt[i].max(1.0);
        let (y0, y1) = (top[i] / d, bot[i] / d);
        let lh = (y1 - y0).max(1.0);
        let ex = (CORE_X_SHRINK * lh).min(0.5 * lh);
        let s = s as f64;
        boxes.push(LineBox {
            x0: (c.left as f64 * s - ex) as f32,
            y0: y0 as f32,
            x1: ((c.left + c.width) as f64 * s + ex) as f32,
            y1: y1 as f32,
            score: (sc[i] / d) as f32,
        });
    }
    boxes
}

// ── line crops ──────────────────────────────────────────────────────────────

pub const LINE_HEIGHT: usize = 32;
pub const PAD_Y: f64 = 0.2;
pub const PAD_X: f64 = 0.3;
/// Line crops are padded to a multiple of this width (the timestep stride).
pub const WIDTH_MULTIPLE: usize = 8;
const MAX_WIDTH: usize = 4096;

/// `np.percentile(region, (2, 50, 98))` (linear) from the region's histogram.
fn percentiles(hist: &[u64; 256], n: u64) -> [f64; 3] {
    let mut cum = [0u64; 256];
    let mut acc = 0;
    for (i, &h) in hist.iter().enumerate() {
        acc += h;
        cum[i] = acc;
    }
    let search = |v: u64| cum.iter().position(|&c| c > v).unwrap_or(256) as f64;
    [0.02, 0.5, 0.98].map(|q| {
        let x = q * (n - 1) as f64;
        let i = x.floor();
        let g = x - i;
        let lo = search(i as u64);
        let hi = search((i as u64 + 1).min(n - 1));
        let d = hi - lo;
        if g >= 0.5 {
            hi - d * (1.0 - g)
        } else {
            lo + d * g
        }
    })
}

/// Light text on a dark ground: the ground (the median of the line box) lies
/// nearer the dark end than the text does. Read inverted.
pub fn is_dark_line(gray: &Gray, b: &[f32; 4]) -> bool {
    let (x0, y0) = (
        (b[0] as f64).max(0.0) as usize,
        (b[1] as f64).max(0.0) as usize,
    );
    let x1 = ((b[2] as f64) as i64 + 1).clamp(0, gray.width as i64) as usize;
    let y1 = ((b[3] as f64) as i64 + 1).clamp(0, gray.height as i64) as usize;
    if x1 <= x0 || y1 <= y0 {
        return false;
    }
    let mut hist = [0u64; 256];
    for y in y0..y1 {
        for &v in &gray.row(y)[x0..x1] {
            hist[v as usize] += 1;
        }
    }
    let [lo, med, hi] = percentiles(&hist, ((x1 - x0) * (y1 - y0)) as u64);
    hi - med > 2.0 * (med - lo) + 8.0
}

/// A line's recognizer input: the box grown by `PAD_Y`/`PAD_X` of its height,
/// cropped (paper outside the page), scaled to `height · window` rows; ink map
/// `[rows, W]` row-major. A tall window keeps the text scale and adds the page
/// above and below; its middle band is the `window = 1` crop.
pub fn crop_line(
    gray: &Gray,
    b: &[f32; 4],
    height: usize,
    window: usize,
    dark: bool,
) -> (Vec<f32>, usize, usize) {
    let pad_y = (window as f64 * (1.0 + 2.0 * PAD_Y) - 1.0) / 2.0;
    let rows = height * window;
    let [x0, y0, x1, y1] = b.map(|v| v as f64);
    let h = (y1 - y0).max(1.0);
    let (bx0, bx1) = (x0 - PAD_X * h, x1 + PAD_X * h);
    let (by0, by1) = (y0 - pad_y * h, y1 + pad_y * h);
    let (ix0, iy0) = (bx0.floor() as i64, by0.floor() as i64);
    let (ix1, iy1) = (bx1.ceil() as i64, by1.ceil() as i64);
    let cw = (ix1 - ix0).max(1) as usize;
    let ch = (iy1 - iy0).max(1) as usize;
    let (gw, gh) = (gray.width as i64, gray.height as i64);
    let (sx0, sy0, sx1, sy1) = (ix0.max(0), iy0.max(0), ix1.min(gw), iy1.min(gh));
    let mut crop = Gray::filled(cw, ch, if dark { 0 } else { 255 });
    if sx1 > sx0 && sy1 > sy0 {
        for y in sy0..sy1 {
            let src = &gray.row(y as usize)[sx0 as usize..sx1 as usize];
            let dy = (y - iy0) as usize;
            let dx = (sx0 - ix0) as usize;
            crop.data[dy * cw + dx..dy * cw + dx + src.len()].copy_from_slice(src);
        }
    }
    let out_w = (py_round(cw as f64 * rows as f64 / ch as f64) as usize)
        .clamp(WIDTH_MULTIPLE, MAX_WIDTH)
        .max(WIDTH_MULTIPLE);
    let interp = if ch > rows {
        Interp::Area
    } else {
        Interp::Linear
    };
    let r = resize(&crop, out_w, rows, interp);
    let ink = r
        .data
        .iter()
        .map(|&v| {
            let f = v as f32 / 255.0;
            if dark { f } else { 1.0 - f }
        })
        .collect();
    (ink, rows, out_w)
}

/// Line indices grouped into batches of similar width, narrowest first. A
/// batch closes at `max_batch` lines, at a line wider than 1.3× its first
/// (widths under 64 count as 64), or past 64 × 640 pixel columns.
pub fn width_batches(widths: &[usize], max_batch: usize) -> Vec<Vec<usize>> {
    const MAX_RATIO: f64 = 1.3;
    const MIN_WIDTH: usize = 64;
    const MAX_AREA: usize = 64 * 640;
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&i| widths[i]);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let wi = widths[i].max(MIN_WIDTH);
        if let Some(last) = batches.last_mut()
            && last.len() < max_batch
            && wi as f64 <= MAX_RATIO * widths[last[0]].max(MIN_WIDTH) as f64
            && (last.len() + 1) * wi <= MAX_AREA
        {
            last.push(i);
        } else {
            batches.push(vec![i]);
        }
    }
    batches
}
