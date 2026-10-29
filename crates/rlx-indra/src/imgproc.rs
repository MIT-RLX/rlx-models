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

//! 8-bit grayscale image operations the upstream pipeline takes from OpenCV,
//! written to OpenCV's arithmetic (fixed-point weights, rounding) so the
//! networks see the same pixels: `resize` (`INTER_AREA` / `INTER_LINEAR`),
//! `warpAffine` (bilinear, constant or replicated border), rotation matrices,
//! connected components, and quarter turns.

/// An 8-bit grayscale image, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Gray {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height, "gray buffer size");
        Self {
            width,
            height,
            data,
        }
    }

    pub fn filled(width: usize, height: usize, v: u8) -> Self {
        Self::new(width, height, vec![v; width * height])
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    /// The `[x0, x1) × [y0, y1)` region (clipped to the image).
    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Gray {
        let (x1, y1) = (x1.min(self.width), y1.min(self.height));
        let (x0, y0) = (x0.min(x1), y0.min(y1));
        let mut data = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            data.extend_from_slice(&self.row(y)[x0..x1]);
        }
        Gray::new(x1 - x0, y1 - y0, data)
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|&v| v as u64).sum::<u64>() as f64 / self.data.len() as f64
    }
}

/// OpenCV `saturate_cast<int>(double)` / `cvRound`: round half to even.
#[inline]
fn cv_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

#[inline]
fn sat_u8(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

// ── resize ──────────────────────────────────────────────────────────────────

/// Interpolation of [`resize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Area,
}

/// `cv2.resize(img, (width, height), interpolation=...)`.
pub fn resize(src: &Gray, width: usize, height: usize, interp: Interp) -> Gray {
    assert!(width > 0 && height > 0, "resize to an empty image");
    if (width, height) == (src.width, src.height) {
        return src.clone();
    }
    let inv_x = width as f64 / src.width as f64;
    let inv_y = height as f64 / src.height as f64;
    let (sx, sy) = (1.0 / inv_x, 1.0 / inv_y);
    let (isx, isy) = (sx.round_ties_even() as i64, sy.round_ties_even() as i64);
    let area_fast =
        (sx - isx as f64).abs() < f64::EPSILON && (sy - isy as f64).abs() < f64::EPSILON;
    let mut interp = interp;
    if interp == Interp::Linear && area_fast && isx == 2 && isy == 2 {
        interp = Interp::Area;
    }
    match interp {
        Interp::Area if inv_x >= 1.0 && inv_y >= 1.0 => {
            resize_linear(src, width, height, sx, sy, true)
        }
        Interp::Area if area_fast => {
            resize_area_fast(src, width, height, isx as usize, isy as usize)
        }
        Interp::Area => resize_area(src, width, height, sx, sy),
        Interp::Linear => resize_linear(src, width, height, sx, sy, false),
    }
}

const RESIZE_COEF_BITS: u32 = 11;
const RESIZE_COEF_SCALE: f32 = (1 << RESIZE_COEF_BITS) as f32;

/// Source index and 11-bit fixed-point weights per destination coordinate.
/// `area_up`: `INTER_AREA` enlarging, which replicates pixels and blends only
/// across source-pixel edges.
fn linear_taps(dsize: usize, ssize: usize, scale: f64, area_up: bool) -> Vec<(usize, i32, i32)> {
    let inv = 1.0 / scale;
    (0..dsize)
        .map(|d| {
            let (mut s, mut f) = if area_up {
                let s = (d as f64 * scale).floor() as i64;
                let f = ((d + 1) as f64 - (s + 1) as f64 * inv) as f32;
                (s, if f <= 0.0 { 0.0 } else { f - f.floor() })
            } else {
                let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
                let s = f.floor() as i64;
                (s, f - s as f32)
            };
            if s < 0 {
                f = 0.0;
                s = 0;
            }
            if s + 1 >= ssize as i64 {
                f = 0.0;
                s = ssize as i64 - 1;
            }
            let a0 = ((1.0 - f) * RESIZE_COEF_SCALE).round_ties_even() as i32;
            let a1 = (f * RESIZE_COEF_SCALE).round_ties_even() as i32;
            (s as usize, a0, a1)
        })
        .collect()
}

fn resize_linear(src: &Gray, width: usize, height: usize, sx: f64, sy: f64, area_up: bool) -> Gray {
    let xt = linear_taps(width, src.width, sx, area_up);
    let yt = linear_taps(height, src.height, sy, area_up);
    let hrow = |y: usize| -> Vec<i32> {
        let r = src.row(y);
        xt.iter()
            .map(|&(s, a0, a1)| r[s] as i32 * a0 + r[(s + 1).min(src.width - 1)] as i32 * a1)
            .collect()
    };
    let mut out = Vec::with_capacity(width * height);
    for &(s, b0, b1) in &yt {
        let r0 = hrow(s);
        let r1 = hrow((s + 1).min(src.height - 1));
        for x in 0..width {
            // VResizeLinear's 8-bit path: drop 4 bits, 16-bit high multiply, round.
            let t = (((r0[x] >> 4) * b0) >> 16) + (((r1[x] >> 4) * b1) >> 16);
            out.push(sat_u8(((t + 2) >> 2) as i64));
        }
    }
    Gray::new(width, height, out)
}

fn resize_area_fast(src: &Gray, width: usize, height: usize, kx: usize, ky: usize) -> Gray {
    let area = (kx * ky) as f32;
    let scale = 1.0 / area;
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u32;
            for dy in 0..ky {
                for dx in 0..kx {
                    sum += src.at(x * kx + dx, y * ky + dy) as u32;
                }
            }
            let v = if kx == 2 && ky == 2 {
                ((sum + 2) >> 2) as i64
            } else {
                cv_round((sum as f32 * scale) as f64)
            };
            out.push(sat_u8(v));
        }
    }
    Gray::new(width, height, out)
}

/// `computeResizeAreaTab`: (dst index, src index, weight) triples.
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<(usize, usize, f32)> {
    let mut tab = Vec::new();
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push((dx, (sx1 - 1) as usize, ((sx1 as f64 - fsx1) / cell) as f32));
        }
        for s in sx1..sx2 {
            tab.push((dx, s as usize, (1.0 / cell) as f32));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            let a = (fsx2 - sx2 as f64).min(1.0).min(cell) / cell;
            tab.push((dx, sx2 as usize, a as f32));
        }
    }
    tab
}

fn resize_area(src: &Gray, width: usize, height: usize, sx: f64, sy: f64) -> Gray {
    let xtab = area_tab(src.width, width, sx);
    let ytab = area_tab(src.height, height, sy);
    let mut out = vec![0u8; width * height];
    let mut buf = vec![0f32; width];
    let mut sum = vec![0f32; width];
    let mut prev = ytab.first().map_or(0, |t| t.0);
    for &(dy, sy, beta) in &ytab {
        buf.fill(0.0);
        let row = src.row(sy);
        for &(dx, s, alpha) in &xtab {
            buf[dx] += row[s] as f32 * alpha;
        }
        if dy != prev {
            let d = &mut out[prev * width..(prev + 1) * width];
            for x in 0..width {
                d[x] = sat_u8(cv_round(sum[x] as f64));
                sum[x] = beta * buf[x];
            }
            prev = dy;
        } else {
            for x in 0..width {
                sum[x] += beta * buf[x];
            }
        }
    }
    let d = &mut out[prev * width..(prev + 1) * width];
    for x in 0..width {
        d[x] = sat_u8(cv_round(sum[x] as f64));
    }
    Gray::new(width, height, out)
}

// ── affine warp ─────────────────────────────────────────────────────────────

/// `cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)` (center as `float`).
pub fn rotation_matrix(cx: f32, cy: f32, degrees: f64) -> [f64; 6] {
    let a = degrees * std::f64::consts::PI / 180.0;
    let (alpha, beta) = (a.cos(), a.sin());
    let (cx, cy) = (cx as f64, cy as f64);
    [
        alpha,
        beta,
        (1.0 - alpha) * cx - beta * cy,
        -beta,
        alpha,
        beta * cx + (1.0 - alpha) * cy,
    ]
}

/// Border handling of [`warp_affine`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    Constant(u8),
    Replicate,
}

const AB_BITS: i64 = 10;
const AB_SCALE: f64 = (1 << AB_BITS) as f64;
const INTER_BITS: i64 = 5;
const INTER_TAB: i64 = 1 << INTER_BITS;
const REMAP_BITS: i64 = 15;

/// `cv2.warpAffine(src, M, (width, height), flags=INTER_LINEAR, borderMode=...)`
/// with `M` mapping source to destination (OpenCV inverts it).
pub fn warp_affine(src: &Gray, m: &[f64; 6], width: usize, height: usize, border: Border) -> Gray {
    let mut m = *m;
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22) = (m[4] * d, m[0] * d);
    m[0] = a11;
    m[1] *= -d;
    m[3] *= -d;
    m[4] = a22;
    let b1 = -m[0] * m[2] - m[1] * m[5];
    let b2 = -m[3] * m[2] - m[4] * m[5];
    m[2] = b1;
    m[5] = b2;

    let adelta: Vec<i64> = (0..width)
        .map(|x| cv_round(m[0] * x as f64 * AB_SCALE))
        .collect();
    let bdelta: Vec<i64> = (0..width)
        .map(|x| cv_round(m[3] * x as f64 * AB_SCALE))
        .collect();
    let round_delta = (AB_SCALE as i64) / INTER_TAB / 2;
    let (sw, sh) = (src.width as i64, src.height as i64);
    let pix = |x: i64, y: i64| -> Option<u8> {
        match border {
            Border::Replicate => {
                Some(src.at(x.clamp(0, sw - 1) as usize, y.clamp(0, sh - 1) as usize))
            }
            Border::Constant(_) if x < 0 || y < 0 || x >= sw || y >= sh => None,
            Border::Constant(_) => Some(src.at(x as usize, y as usize)),
        }
    };
    let cval = match border {
        Border::Constant(v) => v as i64,
        Border::Replicate => 0,
    };
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let x0 = cv_round((m[1] * y as f64 + m[2]) * AB_SCALE) + round_delta;
        let y0 = cv_round((m[4] * y as f64 + m[5]) * AB_SCALE) + round_delta;
        for x in 0..width {
            let xx = (x0 + adelta[x]) >> (AB_BITS - INTER_BITS);
            let yy = (y0 + bdelta[x]) >> (AB_BITS - INTER_BITS);
            let (sx, sy) = (xx >> INTER_BITS, yy >> INTER_BITS);
            let (ax, ay) = (xx & (INTER_TAB - 1), yy & (INTER_TAB - 1));
            if matches!(border, Border::Constant(_))
                && (sx >= sw || sx + 1 < 0 || sy >= sh || sy + 1 < 0)
            {
                out.push(cval as u8);
                continue;
            }
            // Bilinear weights at 1/32 steps are exact in 15-bit fixed point.
            let w00 = (INTER_TAB - ax) * (INTER_TAB - ay);
            let w01 = ax * (INTER_TAB - ay);
            let w10 = (INTER_TAB - ax) * ay;
            let w11 = ax * ay;
            let v = |px: Option<u8>| px.map_or(cval, |p| p as i64);
            let acc = v(pix(sx, sy)) * w00
                + v(pix(sx + 1, sy)) * w01
                + v(pix(sx, sy + 1)) * w10
                + v(pix(sx + 1, sy + 1)) * w11;
            // Weights are in 1/1024 units: scale to the 15-bit table, then round.
            let acc = acc * ((1 << REMAP_BITS) / (INTER_TAB * INTER_TAB));
            out.push(sat_u8((acc + (1 << (REMAP_BITS - 1))) >> REMAP_BITS));
        }
    }
    Gray::new(width, height, out)
}

/// `cv2.warpAffine(gray, getRotationMatrix2D((w/2, h/2), degrees, 1), (w, h), ...)`.
pub fn rotate(src: &Gray, degrees: f64, border: Border) -> Gray {
    let m = rotation_matrix(src.width as f32 / 2.0, src.height as f32 / 2.0, degrees);
    warp_affine(src, &m, src.width, src.height, border)
}

/// `np.rot90(img, k)`: `k` quarter turns counter-clockwise (negative: clockwise).
pub fn rot90(src: &Gray, k: i32) -> Gray {
    let (w, h) = (src.width, src.height);
    match k.rem_euclid(4) {
        0 => src.clone(),
        1 => {
            let mut d = Vec::with_capacity(w * h);
            for i in 0..w {
                for j in 0..h {
                    d.push(src.at(w - 1 - i, j));
                }
            }
            Gray::new(h, w, d)
        }
        2 => {
            let mut d = src.data.clone();
            d.reverse();
            Gray::new(w, h, d)
        }
        _ => {
            let mut d = Vec::with_capacity(w * h);
            for i in 0..w {
                for j in 0..h {
                    d.push(src.at(i, h - 1 - j));
                }
            }
            Gray::new(h, w, d)
        }
    }
}

// ── connected components ────────────────────────────────────────────────────

/// One component of [`connected_components`] (OpenCV `CC_STAT_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Component {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
    pub area: usize,
}

/// `cv2.connectedComponentsWithStats(mask, connectivity)`: labels (0 =
/// background, components numbered in raster order of their first pixel)
/// and per-component stats (index `label - 1`).
pub fn connected_components(
    mask: &[bool],
    width: usize,
    height: usize,
    eight: bool,
) -> (Vec<u32>, Vec<Component>) {
    let mut labels = vec![0u32; width * height];
    let mut comps: Vec<Component> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..width * height {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        let id = comps.len() as u32 + 1;
        let (mut x0, mut y0, mut x1, mut y1, mut area) = (usize::MAX, usize::MAX, 0, 0, 0);
        labels[start] = id;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let (x, y) = (p % width, p / width);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            area += 1;
            let mut visit = |nx: isize, ny: isize| {
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    return;
                }
                let q = ny as usize * width + nx as usize;
                if mask[q] && labels[q] == 0 {
                    labels[q] = id;
                    stack.push(q);
                }
            };
            let (xi, yi) = (x as isize, y as isize);
            visit(xi - 1, yi);
            visit(xi + 1, yi);
            visit(xi, yi - 1);
            visit(xi, yi + 1);
            if eight {
                visit(xi - 1, yi - 1);
                visit(xi + 1, yi - 1);
                visit(xi - 1, yi + 1);
                visit(xi + 1, yi + 1);
            }
        }
        comps.push(Component {
            left: x0,
            top: y0,
            width: x1 - x0 + 1,
            height: y1 - y0 + 1,
            area,
        });
    }
    (labels, comps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rot90_matches_numpy() {
        // [[1, 2, 3], [4, 5, 6]]
        let g = Gray::new(3, 2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rot90(&g, 1).data, vec![3, 6, 2, 5, 1, 4]);
        assert_eq!(rot90(&g, -1).data, vec![4, 1, 5, 2, 6, 3]);
        assert_eq!(rot90(&g, 2).data, vec![6, 5, 4, 3, 2, 1]);
        assert_eq!(rot90(&rot90(&g, 1), -1), g);
    }

    #[test]
    fn components_raster_order() {
        #[rustfmt::skip]
        let m = [
            false, false, true,  true,
            true,  false, false, false,
            true,  false, true,  false,
        ];
        let (_, c) = connected_components(&m, 4, 3, false);
        assert_eq!(c.len(), 3);
        assert_eq!((c[0].left, c[0].top, c[0].width, c[0].area), (2, 0, 2, 2));
        assert_eq!((c[1].left, c[1].top, c[1].height), (0, 1, 2));
        let (_, c8) = connected_components(&m, 4, 3, true);
        assert_eq!(c8.len(), 3);
    }

    #[test]
    fn identity_warp_is_exact() {
        let g = Gray::new(5, 4, (0..20).map(|v| v * 12).collect());
        assert_eq!(rotate(&g, 0.0, Border::Replicate), g);
    }
}
