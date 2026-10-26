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

//! Turning a generated emoji into a sticker: super-resolution, background
//! cut-out, final resample.
//!
//! The LoRA was trained on macOS emoji composited onto **white** (image
//! models handle transparency poorly), so every output sits on white — the
//! same final step the vendor's Genmoji takes, visible as a faint white rim on dark
//! backgrounds. Because the background colour is *known*, the cut is solved as
//! a matting problem rather than thresholded: an edge pixel is
//! `C = α·F + (1-α)·white`, and with `F` taken from the nearest opaque
//! foreground the equation gives α directly, and `F` is the decontaminated
//! colour (no white halo).
//!
//! Order matters: matting runs **after** super-resolution, so the edge band is
//! a few pixels of a sharp 640-pixel image rather than one pixel of the 160-pixel
//! generation magnified five times.

use image::{RgbImage, Rgba, RgbaImage, imageops::FilterType};
use std::collections::VecDeque;

/// Matting settings, as "distance from white" `255 - min(r, g, b)`.
#[derive(Debug, Clone, Copy)]
pub struct Cutout {
    /// Background seeds: border-connected pixels at most this far from white.
    /// Low, so that near-white parts of the subject (wings, snow, teeth) are
    /// not eaten — the fill cannot cross them.
    pub flood: u8,
    /// Width of the soft edge band, as a fraction of the image side.
    pub band: f32,
    /// Enclosed pockets: how far from white a pixel outside the subject mask
    /// may be and still count as background (generations shade concave
    /// background pockets light grey).
    pub pocket: u8,
}

impl Default for Cutout {
    fn default() -> Self {
        Self {
            flood: 10,
            band: 0.006,
            pocket: 12,
        }
    }
}

fn dist_white(p: &[u8]) -> u8 {
    255 - p[0].min(p[1]).min(p[2])
}

/// Border-connected near-white region.
fn flood_background(img: &RgbImage, tol: u8) -> Vec<bool> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let dist: Vec<u8> = img.pixels().map(|p| dist_white(&p.0)).collect();
    let mut bg = vec![false; w * h];
    let mut q = VecDeque::new();
    let push = |i: usize, bg: &mut Vec<bool>, q: &mut VecDeque<usize>| {
        if !bg[i] && dist[i] <= tol {
            bg[i] = true;
            q.push_back(i);
        }
    };
    for x in 0..w {
        push(x, &mut bg, &mut q);
        push((h - 1) * w + x, &mut bg, &mut q);
    }
    for y in 0..h {
        push(y * w, &mut bg, &mut q);
        push(y * w + w - 1, &mut bg, &mut q);
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = (i % w, i / w);
        if x > 0 {
            push(i - 1, &mut bg, &mut q);
        }
        if x + 1 < w {
            push(i + 1, &mut bg, &mut q);
        }
        if y > 0 {
            push(i - w, &mut bg, &mut q);
        }
        if y + 1 < h {
            push(i + w, &mut bg, &mut q);
        }
    }
    bg
}

/// BFS distance (8-connected, in steps) from every pixel to the nearest `seed`
/// pixel, plus the index of that seed.
fn nearest(seed: &[bool], w: usize, h: usize) -> (Vec<u32>, Vec<u32>) {
    let mut d = vec![u32::MAX; w * h];
    let mut src = vec![u32::MAX; w * h];
    let mut q = VecDeque::new();
    for (i, &s) in seed.iter().enumerate() {
        if s {
            d[i] = 0;
            src[i] = i as u32;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let (nx, ny) = (x + dx, y + dy);
                if (dx, dy) == (0, 0) || nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if d[j] == u32::MAX {
                    d[j] = d[i] + 1;
                    src[j] = src[i];
                    q.push_back(j);
                }
            }
        }
    }
    (d, src)
}

/// Known-white-background matting: transparent background, soft decontaminated
/// edges, opaque subject (including enclosed or near-white parts the
/// background fill cannot reach).
pub fn remove_white_background(img: &RgbImage, c: Cutout) -> RgbaImage {
    remove_white_background_with(img, c, None)
}

/// [`remove_white_background`] with extra background pixels (`extra[i]`):
/// enclosed pockets a border fill cannot reach, e.g. from a segmentation of
/// the subject ([`background_pockets`]).
pub fn remove_white_background_with(
    img: &RgbImage,
    c: Cutout,
    extra: Option<&[bool]>,
) -> RgbaImage {
    matte(img, union(flood_background(img, c.flood), extra), c)
}

fn union(mut bg: Vec<bool>, extra: Option<&[bool]>) -> Vec<bool> {
    if let Some(e) = extra {
        bg.iter_mut().zip(e).for_each(|(b, &x)| *b |= x);
    }
    bg
}

/// Pixels the border fill does not reach (subject plus enclosed pockets).
pub fn not_background(img: &RgbImage, c: Cutout) -> Vec<bool> {
    flood_background(img, c.flood)
        .into_iter()
        .map(|b| !b)
        .collect()
}

/// Bounding box `(x0, y0, x1, y1)` of everything the border fill does not
/// reach — the subject, for a segmentation prompt.
pub fn subject_box(img: &RgbImage, c: Cutout) -> Option<(u32, u32, u32, u32)> {
    let w = img.width() as usize;
    let bg = flood_background(img, c.flood);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, &b) in bg.iter().enumerate() {
        if !b {
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
    }
    (x0 != usize::MAX).then_some((x0 as u32, y0 as u32, x1 as u32, y1 as u32))
}

/// Background pockets enclosed by the subject: near-white regions the border
/// fill cannot reach that a subject mask says are *not* subject.
///
/// `outside(x, y)` is the segmenter's verdict for pixel `(x, y)` of `img`
/// (true = confidently not the subject). The decision is per *region* — a
/// 4-connected component of near-white pixels — not per pixel: a region goes
/// only when nearly all of it (90 %) lies outside the mask and it is not tiny.
/// A segmentation that clips a pale part of the subject (a white wing, an eye
/// highlight) then cannot nibble it, and non-white pixels never go at all.
pub fn background_pockets(
    img: &RgbImage,
    c: Cutout,
    outside: impl Fn(u32, u32) -> bool,
) -> Vec<bool> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bg = flood_background(img, c.flood);
    let cand: Vec<bool> = img
        .pixels()
        .enumerate()
        .map(|(i, p)| !bg[i] && dist_white(&p.0) <= c.pocket)
        .collect();
    let min_area = ((w * h) as f32 * 0.0005).ceil() as usize;
    let mut out = vec![false; w * h];
    let mut seen = vec![false; w * h];
    for s in 0..w * h {
        if !cand[s] || seen[s] {
            continue;
        }
        let mut comp = vec![s];
        seen[s] = true;
        let mut k = 0;
        while k < comp.len() {
            let i = comp[k];
            k += 1;
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if cand[j] && !seen[j] {
                    seen[j] = true;
                    comp.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        let n_out = comp
            .iter()
            .filter(|&&i| outside((i % w) as u32, (i / w) as u32))
            .count();
        if comp.len() >= min_area && n_out * 10 >= comp.len() * 9 {
            comp.iter().for_each(|&i| out[i] = true);
        }
    }
    out
}

/// [`remove_white_background`] for a super-resolved image `hi` of the
/// generation `lo`.
///
/// *Where* the edge is comes from `lo`: its background is clean (almost all
/// within 2 of white) and an edge is about one pixel wide, so its matte is
/// reliable. A super-resolution network does neither — it leaves faint
/// off-white patches in the background and widens every anti-aliased edge into
/// a ring of light pixels, which a cut-out on `hi` alone keeps as haze or as
/// specks along the outline. So the low-resolution alpha, scaled up, marks what
/// is surely subject (alpha 1), surely background (alpha 0) and the band
/// between; *how much* of a band pixel is subject comes from `hi` by the
/// known-white projection against the nearest sure-subject colour. A white ring
/// pixel projects to ~0, a coloured edge pixel to its true coverage, at the
/// sharp edge's resolution.
pub fn remove_white_background_guided(
    hi: &RgbImage,
    lo: &RgbImage,
    c: Cutout,
    extra: Option<&[bool]>,
) -> RgbaImage {
    remove_white_background_hinted(hi, lo, c, extra, None)
}

/// [`remove_white_background_guided`] with a soft subject mask `soft` at
/// `hi`'s resolution (0..1, e.g. a segmenter's sigmoid). Where the subject's
/// own colour is near-white — a white wing against the white background — the
/// colour equation cannot place the edge and the generation barely shades it;
/// the segmenter's silhouette is the smooth, object-level answer there, and is
/// used for every pixel within reach of such a part (even ones the fill on
/// `lo` had already taken, like thin feather tips).
pub fn remove_white_background_hinted(
    hi: &RgbImage,
    lo: &RgbImage,
    c: Cutout,
    extra: Option<&[bool]>,
    soft: Option<&[f32]>,
) -> RgbaImage {
    let (hw, hh) = (hi.width() as usize, hi.height() as usize);
    let (lw, lh) = (lo.width() as usize, lo.height() as usize);
    let lo_matte = matte(
        lo,
        union(flood_background(lo, c.flood), extra),
        Cutout {
            band: 1.5 / lw.max(lh) as f32,
            ..c
        },
    );
    let a_lo: Vec<f32> = lo_matte.pixels().map(|p| p[3] as f32 / 255.0).collect();
    let (sx, sy) = (lw as f32 / hw as f32, lh as f32 / hh as f32);
    let p: Vec<f32> = (0..hw * hh)
        .map(|i| {
            let x = ((i % hw) as f32 + 0.5) * sx - 0.5;
            let y = ((i / hw) as f32 + 0.5) * sy - 0.5;
            let (x, y) = (x.clamp(0.0, lw as f32 - 1.0), y.clamp(0.0, lh as f32 - 1.0));
            let (x0, y0) = (x.floor() as usize, y.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(lw - 1), (y0 + 1).min(lh - 1));
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            let v = |x: usize, y: usize| a_lo[y * lw + x];
            (v(x0, y0) * (1.0 - fx) + v(x1, y0) * fx) * (1.0 - fy)
                + (v(x0, y1) * (1.0 - fx) + v(x1, y1) * fx) * fy
        })
        .collect();
    let sure_fg: Vec<bool> = p.iter().map(|&a| a >= 0.999).collect();
    let (from_fg, src) = nearest(&sure_fg, hw, hh);
    let raw = hi.as_raw();
    let reach = (3.0 * hw as f32 / lw as f32).ceil() as u32;
    let near_white_subject = |i: usize| -> bool {
        let s = src[i];
        s != u32::MAX && {
            let s = s as usize;
            let f = [
                raw[s * 3] as f32 - 255.0,
                raw[s * 3 + 1] as f32 - 255.0,
                raw[s * 3 + 2] as f32 - 255.0,
            ];
            f.iter().map(|v| v * v).sum::<f32>() < 30.0 * 30.0
        }
    };
    let mut out = RgbaImage::new(hw as u32, hh as u32);
    for i in 0..hw * hh {
        let cpx = [
            raw[i * 3] as f32,
            raw[i * 3 + 1] as f32,
            raw[i * 3 + 2] as f32,
        ];
        let a = if p[i] >= 0.999 {
            1.0
        } else if let Some(sm) = soft.filter(|_| from_fg[i] <= reach && near_white_subject(i)) {
            sm[i]
        } else if p[i] <= 0.001 || src[i] == u32::MAX {
            0.0
        } else {
            let s = src[i] as usize;
            let fw = [
                raw[s * 3] as f32 - 255.0,
                raw[s * 3 + 1] as f32 - 255.0,
                raw[s * 3 + 2] as f32 - 255.0,
            ];
            let ff: f32 = fw.iter().map(|v| v * v).sum();
            if ff < 30.0 * 30.0 {
                // A near-white subject (wings, fur, snow): the colour direction is
                // too short to project on; the same equation on distance-from-white
                // still separates it while the subject is at all off-white, and
                // the low-resolution alpha decides when it is not.
                let d_f = dist_white(&raw[s * 3..s * 3 + 3]) as f32;
                if d_f >= 6.0 {
                    (dist_white(&raw[i * 3..i * 3 + 3]) as f32 / d_f).clamp(0.0, 1.0)
                } else {
                    p[i]
                }
            } else {
                let cw = [cpx[0] - 255.0, cpx[1] - 255.0, cpx[2] - 255.0];
                (fw.iter().zip(&cw).map(|(a, b)| a * b).sum::<f32>() / ff).clamp(0.0, 1.0)
            }
        };
        let a = if a < 0.03 {
            0.0
        } else if a > 0.97 {
            1.0
        } else {
            a
        };
        let mut col = [raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]];
        if a > 0.0 && a < 1.0 {
            for k in 0..3 {
                col[k] = ((cpx[k] - (1.0 - a) * 255.0) / a).round().clamp(0.0, 255.0) as u8;
            }
        }
        out.as_mut()[i * 4..i * 4 + 4].copy_from_slice(&[
            col[0],
            col[1],
            col[2],
            (a * 255.0).round() as u8,
        ]);
    }
    out
}

/// Single-render alpha of `img` at its own resolution (border fill ∪ `extra`
/// as background), `[h·w]` in 0..1.
pub fn coarse_alpha(img: &RgbImage, c: Cutout, extra: Option<&[bool]>) -> Vec<f32> {
    let m = matte(
        img,
        union(flood_background(img, c.flood), extra),
        Cutout {
            band: 1.5 / img.width().max(img.height()) as f32,
            ..c
        },
    );
    m.pixels().map(|p| p[3] as f32 / 255.0).collect()
}

/// Nearest-neighbour resample of a mask from `w × h` to `ow × oh`.
pub fn resample_mask(m: &[bool], w: usize, h: usize, ow: usize, oh: usize) -> Vec<bool> {
    (0..ow * oh)
        .map(|i| {
            let (x, y) = (i % ow * w / ow, i / ow * h / oh);
            m[y * w + x]
        })
        .collect()
}

fn matte(img: &RgbImage, bg: Vec<bool>, c: Cutout) -> RgbaImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let px: Vec<[f32; 3]> = img
        .pixels()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let band = ((c.band * w.max(h) as f32).round() as u32).max(1);
    // Distance of every pixel from the background region.
    let (from_bg, _) = nearest(&bg, w, h);
    // Definite foreground: beyond the band. Edge pixels borrow the colour of
    // their nearest definite-foreground pixel as their `F`; background farther
    // than the band from any foreground is simply transparent.
    let fg: Vec<bool> = from_bg.iter().map(|&d| d > band).collect();
    let (from_fg, fg_src) = nearest(&fg, w, h);
    let mut out = RgbaImage::new(w as u32, h as u32);
    for i in 0..w * h {
        let cpx = px[i];
        let (alpha, color) = if fg[i] || fg_src[i] == u32::MAX {
            (if bg[i] { 0.0 } else { 1.0 }, cpx)
        } else if bg[i] && from_fg[i] > 2 * band {
            (0.0, cpx)
        } else {
            let f = px[fg_src[i] as usize];
            let fw = [f[0] - 255.0, f[1] - 255.0, f[2] - 255.0];
            let cw = [cpx[0] - 255.0, cpx[1] - 255.0, cpx[2] - 255.0];
            let ff: f32 = fw.iter().map(|v| v * v).sum();
            let a = if ff < 30.0 * 30.0 {
                // A near-white subject edge: the equation cannot separate it from
                // the background, so trust the region — inside it, a ramp on its
                // own whiteness; outside, opaque.
                if bg[i] {
                    (dist_white(&img.as_raw()[i * 3..i * 3 + 3]) as f32 / 30.0).min(1.0)
                } else {
                    1.0
                }
            } else {
                (fw.iter().zip(&cw).map(|(a, b)| a * b).sum::<f32>() / ff).clamp(0.0, 1.0)
            };
            let col = if a > 0.02 {
                let mut o = [0.0f32; 3];
                for k in 0..3 {
                    o[k] = ((cpx[k] - (1.0 - a) * 255.0) / a).clamp(0.0, 255.0);
                }
                o
            } else {
                f
            };
            (a, col)
        };
        out.as_mut()[i * 4..i * 4 + 4].copy_from_slice(&[
            color[0].round() as u8,
            color[1].round() as u8,
            color[2].round() as u8,
            (alpha * 255.0).round() as u8,
        ]);
    }
    out
}

/// Triangulation matting (Smith & Blinn 1996) from two renders of the same
/// subject, one over white (`on_white`) and one over black (`on_black`): per
/// pixel `C_w = α·F + (1-α)·255` and `C_b = α·F`, so
/// `α = 1 − (C_w − C_b)/255` and `F = C_b/α` exactly — no assumption about the
/// subject's colour, which is what a single render cannot give for a white wing
/// on white. `sure` marks pixels known to be solid subject (the eroded interior
/// of a single-render matte): they stay opaque with the white render's colour,
/// so small differences between the two renders cannot turn into translucency
/// inside the subject. `free` marks where the second render was free at all;
/// outside it the result is transparent.
pub fn triangulation_matte(
    on_white: &RgbImage,
    on_black: &RgbImage,
    sure: &[bool],
    free: &[bool],
) -> RgbaImage {
    let (w, h) = on_white.dimensions();
    let (cw, cb) = (on_white.as_raw(), on_black.as_raw());
    let mut out = RgbaImage::new(w, h);
    for i in 0..(w * h) as usize {
        let px = if sure[i] {
            [cw[i * 3], cw[i * 3 + 1], cw[i * 3 + 2], 255]
        } else if !free[i] {
            [0, 0, 0, 0]
        } else {
            let diff: f32 = (0..3)
                .map(|k| cw[i * 3 + k] as f32 - cb[i * 3 + k] as f32)
                .sum::<f32>()
                / 3.0;
            let a = (1.0 - diff / 255.0).clamp(0.0, 1.0);
            let a = if a < 0.03 {
                0.0
            } else if a > 0.97 {
                1.0
            } else {
                a
            };
            if a == 0.0 {
                [0, 0, 0, 0]
            } else {
                let f = |k: usize| (cb[i * 3 + k] as f32 / a).round().clamp(0.0, 255.0) as u8;
                [f(0), f(1), f(2), (a * 255.0).round() as u8]
            }
        };
        out.as_mut()[i * 4..i * 4 + 4].copy_from_slice(&px);
    }
    out
}

/// Erode a mask by `r` pixels (8-connected).
pub fn erode(mask: &[bool], w: usize, h: usize, r: u32) -> Vec<bool> {
    let not: Vec<bool> = mask.iter().map(|&m| !m).collect();
    let (d, _) = nearest(&not, w, h);
    d.iter().map(|&d| d > r).collect()
}

/// Dilate a mask by `r` pixels (8-connected).
pub fn dilate(mask: &[bool], w: usize, h: usize, r: u32) -> Vec<bool> {
    let (d, _) = nearest(mask, w, h);
    d.iter().map(|&d| d <= r).collect()
}

/// What a render on a white background says about its subject, as alpha:
/// clearly non-white (≥ `WHITE_PRIOR_LO` below white in some channel), in a
/// component covering at least `WHITE_PRIOR_MIN_AREA` of the image. Partial
/// pixels get `d / d_max` with `d_max` the strongest deficit within 3 px,
/// i.e. their coverage when blended against white. `None` when the image
/// border is not white, so the prior says nothing.
///
/// Saliency mattes (the vendor's subject lifting, BiRefNet) pick *the* salient
/// object: a fireplace render loses its wooden surround and keeps only the
/// flame (Vision itself does the same). Near-white pixels never count here,
/// so enclosed white background stays transparent and white parts of the
/// subject (wings) still come from the learned matte.
pub fn white_prior(img: &RgbImage) -> Option<Vec<f32>> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let d: Vec<u8> = img.pixels().map(|p| dist_white(&p.0)).collect();
    let border: Vec<usize> = (0..w)
        .flat_map(|x| [x, (h - 1) * w + x])
        .chain((0..h).flat_map(|y| [y * w, y * w + w - 1]))
        .collect();
    let white = border.iter().filter(|&&i| d[i] <= 16).count();
    if (white as f32) < 0.9 * border.len() as f32 {
        return None;
    }
    // Components of clearly non-white pixels; keep the large ones.
    let core: Vec<bool> = d.iter().map(|&v| v >= WHITE_PRIOR_LO).collect();
    let mut keep = vec![false; w * h];
    let mut seen = vec![false; w * h];
    let min_area = (WHITE_PRIOR_MIN_AREA * (w * h) as f32) as usize;
    for s in 0..w * h {
        if !core[s] || seen[s] {
            continue;
        }
        let mut comp = vec![s];
        seen[s] = true;
        let mut k = 0;
        while k < comp.len() {
            let i = comp[k];
            k += 1;
            let (x, y) = (i % w, i / w);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < h {
                    let j = ny * w + nx;
                    if core[j] && !seen[j] {
                        seen[j] = true;
                        comp.push(j);
                    }
                }
            }
        }
        if comp.len() >= min_area {
            for i in comp {
                keep[i] = true;
            }
        }
    }
    // Grow by the edge band so antialiased rims get their partial alpha.
    let near = dilate(&keep, w, h, 3);
    let r = 3isize;
    Some(
        (0..w * h)
            .map(|i| {
                if !near[i] || d[i] == 0 {
                    return 0.0;
                }
                let (x, y) = ((i % w) as isize, (i / w) as isize);
                let mut dmax = 0u8;
                for yy in (y - r).max(0)..=(y + r).min(h as isize - 1) {
                    for xx in (x - r).max(0)..=(x + r).min(w as isize - 1) {
                        dmax = dmax.max(d[yy as usize * w + xx as usize]);
                    }
                }
                (d[i] as f32 / dmax.max(1) as f32).clamp(0.0, 1.0)
            })
            .collect(),
    )
}

/// Channel deficit below white that [`white_prior`] counts as subject.
pub const WHITE_PRIOR_LO: u8 = 24;
/// Smallest component [`white_prior`] keeps, as a fraction of the image.
pub const WHITE_PRIOR_MIN_AREA: f32 = 0.002;

/// RGBA from an image on a **white** background and a learned alpha matte:
/// each partly transparent pixel is `C = α·F + (1-α)·255`, so its true colour
/// `F` is recovered rather than keeping the white it was blended with (no
/// halo). Alpha below 2 % is cut, above 98 % made solid.
pub fn apply_alpha_on_white(img: &RgbImage, alpha: &[f32]) -> RgbaImage {
    let raw = img.as_raw();
    RgbaImage::from_fn(img.width(), img.height(), |x, y| {
        let i = (y * img.width() + x) as usize;
        let a = alpha[i];
        let a = if a < 0.02 {
            0.0
        } else if a > 0.98 {
            1.0
        } else {
            a
        };
        let c = |k: usize| {
            let v = raw[i * 3 + k] as f32;
            if a > 0.0 && a < 1.0 {
                ((v - (1.0 - a) * 255.0) / a).round().clamp(0.0, 255.0) as u8
            } else {
                v as u8
            }
        };
        Rgba([c(0), c(1), c(2), (a * 255.0).round() as u8])
    })
}

/// Shortest opaque run, as a fraction of the subject's extent, that counts as
/// the frame cutting the subject off (see [`frame_square`]).
const CUT_RUN: f32 = 0.2;

/// Square framing like the vendor's `EmojiFramer`: crop to the subject (alpha >
/// 5 %), scale its long side to `fill` of a `size × size` canvas and centre it
/// — except that a side where the subject is cut off by the image edge (a
/// bust, a body cropped at the bottom) stays on that edge rather than floating
/// mid-canvas. Measured on macOS's own Genmoji: centred to within ~2 %, long
/// side 0.67–0.81 of the canvas (median 0.75), cut-off subjects anchored.
pub fn frame_square(img: &RgbaImage, size: u32, fill: f32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    for (x, y, p) in img.enumerate_pixels() {
        if p.0[3] > 12 {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return resize_rgba(img, size, size);
    }
    // Cut off = the subject reaches within 1 % of that image edge AND meets it
    // along a solid stretch: its extreme row/column there holds an opaque run
    // of ≥ 20 % of its extent. A bust cut by the frame qualifies; feet or a
    // tail tip that merely touch the edge do not, and stay centred (a
    // one-pixel difference in the render must not move the whole sticker).
    let edge = (w.max(h) / 100).max(1);
    let run = |line: &mut dyn Iterator<Item = u8>, extent: u32| -> bool {
        let (mut best, mut cur) = (0u32, 0u32);
        for a in line {
            cur = if a > 127 { cur + 1 } else { 0 };
            best = best.max(cur);
        }
        best as f32 >= CUT_RUN * extent as f32
    };
    let row = |y: u32| (x0..x1).map(move |x| img.get_pixel(x, y).0[3]);
    let col = |x: u32| (y0..y1).map(move |y| img.get_pixel(x, y).0[3]);
    let (bw0, bh0) = (x1 - x0, y1 - y0);
    let cut_l = x0 <= edge && run(&mut col(x0), bh0);
    let cut_r = x1 + edge >= w && run(&mut col(x1 - 1), bh0);
    let cut_t = y0 <= edge && run(&mut row(y0), bw0);
    let cut_b = y1 + edge >= h && run(&mut row(y1 - 1), bw0);
    let (bw, bh) = (x1 - x0, y1 - y0);
    let scale = fill * size as f32 / bw.max(bh) as f32;
    let (sw, sh) = (
        ((bw as f32 * scale).round() as u32).max(1),
        ((bh as f32 * scale).round() as u32).max(1),
    );
    let crop = image::imageops::crop_imm(img, x0, y0, bw, bh).to_image();
    let scaled = resize_rgba(&crop, sw, sh);
    let place = |len: u32, lo_cut: bool, hi_cut: bool| -> i64 {
        let free = size as i64 - len as i64;
        match (lo_cut, hi_cut) {
            (true, false) => 0,
            (false, true) => free,
            _ => free / 2,
        }
    };
    let (ox, oy) = (place(sw, cut_l, cut_r), place(sh, cut_t, cut_b));
    let mut out = RgbaImage::new(size, size);
    image::imageops::overlay(&mut out, &scaled, ox, oy);
    out
}

/// Lanczos resample to `w × h`, in premultiplied alpha so transparent pixels
/// do not bleed their (meaningless) colour into the edges.
pub fn resize_rgba(img: &RgbaImage, w: u32, h: u32) -> RgbaImage {
    if img.dimensions() == (w, h) {
        return img.clone();
    }
    let (iw, ih) = img.dimensions();
    // `image` treats f32 channels as [0, 1] and clamps to it.
    let pre = image::ImageBuffer::<Rgba<f32>, Vec<f32>>::from_fn(iw, ih, |x, y| {
        let p = img.get_pixel(x, y).0;
        let a = p[3] as f32 / 255.0;
        Rgba([
            p[0] as f32 / 255.0 * a,
            p[1] as f32 / 255.0 * a,
            p[2] as f32 / 255.0 * a,
            a,
        ])
    });
    let big = image::imageops::resize(&pre, w, h, FilterType::Lanczos3);
    RgbaImage::from_fn(w, h, |x, y| {
        let p = big.get_pixel(x, y).0;
        let a = p[3].clamp(0.0, 1.0);
        let un = |v: f32| {
            if a > 0.0 {
                (v / a * 255.0).round().clamp(0.0, 255.0) as u8
            } else {
                0
            }
        };
        Rgba([un(p[0]), un(p[1]), un(p[2]), (a * 255.0).round() as u8])
    })
}

/// Integer Lanczos upscale of an RGBA image (premultiplied).
pub fn upscale(img: &RgbaImage, factor: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    resize_rgba(img, w * factor.max(1), h * factor.max(1))
}

/// Lanczos resample of an opaque image.
pub fn resize_rgb(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    if img.dimensions() == (w, h) {
        return img.clone();
    }
    image::imageops::resize(img, w, h, FilterType::Lanczos3)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A red disc on white with a 1-px anti-aliased rim at 50 % coverage.
    fn disc() -> RgbImage {
        RgbImage::from_fn(41, 41, |x, y| {
            let r = ((x as f32 - 20.0).powi(2) + (y as f32 - 20.0).powi(2)).sqrt();
            if r < 12.0 {
                image::Rgb([200, 30, 40])
            } else if r < 13.0 {
                image::Rgb([228, 142, 148]) // half red, half white
            } else {
                image::Rgb([255, 255, 255])
            }
        })
    }

    #[test]
    fn matting_recovers_alpha_and_colour_on_the_rim() {
        let out = remove_white_background(
            &disc(),
            Cutout {
                flood: 10,
                band: 0.08,
                pocket: 24,
            },
        );
        assert_eq!(out.get_pixel(0, 0).0[3], 0, "background clear");
        assert_eq!(
            out.get_pixel(20, 20).0,
            [200, 30, 40, 255],
            "interior opaque"
        );
        // A rim pixel: x = 20 + 12.5 → (32/33, 20).
        let rim = (32..34)
            .map(|x| out.get_pixel(x, 20).0)
            .find(|p| p[3] > 0 && p[3] < 255)
            .expect("soft rim");
        assert!((rim[3] as i32 - 128).abs() < 20, "alpha ≈ 0.5: {rim:?}");
        assert!(
            (rim[0] as i32 - 200).abs() < 12 && rim[1] < 50,
            "colour decontaminated: {rim:?}"
        );
    }

    #[test]
    fn near_white_subject_parts_survive() {
        // A grey (235) wing touching a red body: not a background seed, so kept.
        let mut img = RgbImage::from_pixel(30, 30, image::Rgb([255, 255, 255]));
        for y in 10..20 {
            for x in 5..25 {
                let c = if x < 15 {
                    [200, 30, 40]
                } else {
                    [235, 235, 235]
                };
                img.put_pixel(x, y, image::Rgb(c));
            }
        }
        let out = remove_white_background(
            &img,
            Cutout {
                flood: 10,
                band: 0.03,
                pocket: 24,
            },
        );
        assert_eq!(
            out.get_pixel(20, 15).0[3],
            255,
            "near-white wing interior kept"
        );
        assert_eq!(out.get_pixel(1, 1).0[3], 0);
    }

    #[test]
    fn guided_matting_ignores_off_white_haze_in_the_upscaled_image() {
        let lo = disc();
        // "Upscaled" ×2 by nearest, plus an off-white patch (dist 20) in a corner
        // that a fill on the large image would not cross.
        let mut hi = image::imageops::resize(&lo, 82, 82, FilterType::Nearest);
        for y in 0..12 {
            for x in 0..12 {
                hi.put_pixel(x, y, image::Rgb([235, 238, 236]));
            }
        }
        let out = remove_white_background_guided(
            &hi,
            &lo,
            Cutout {
                flood: 10,
                band: 0.04,
                pocket: 24,
            },
            None,
        );
        assert_eq!(out.get_pixel(5, 5).0[3], 0, "haze cleared");
        assert_eq!(out.get_pixel(41, 41).0[3], 255);
    }

    #[test]
    fn enclosed_pocket_goes_clear_when_the_mask_says_so() {
        // A red ring enclosing a light-grey (dist 8) pocket the border fill cannot reach.
        let mut img = RgbImage::from_pixel(30, 30, image::Rgb([255, 255, 255]));
        for y in 5..25 {
            for x in 5..25 {
                let inner = (10..20).contains(&x) && (10..20).contains(&y);
                img.put_pixel(
                    x,
                    y,
                    image::Rgb(if inner {
                        [247, 247, 248]
                    } else {
                        [200, 30, 40]
                    }),
                );
            }
        }
        let c = Cutout {
            flood: 10,
            band: 0.03,
            pocket: 24,
        };
        let without = remove_white_background(&img, c);
        assert_eq!(
            without.get_pixel(15, 15).0[3],
            255,
            "fill alone keeps the pocket"
        );
        let pockets = background_pockets(&img, c, |x, y| {
            (10..20).contains(&x) && (10..20).contains(&y)
        });
        let with = remove_white_background_with(&img, c, Some(&pockets));
        assert_eq!(with.get_pixel(15, 15).0[3], 0, "pocket cleared");
        assert_eq!(with.get_pixel(7, 7).0[3], 255, "ring kept");
        // A pale subject part the mask wrongly excludes but which is not near-white stays.
        let none = background_pockets(&img, c, |_, _| true);
        assert!(!none[7 * 30 + 7], "red is never a pocket");
        // A mask that clips only part of the pocket region leaves it alone.
        let partial = background_pockets(&img, c, |x, _| x < 14);
        assert!(
            !partial[15 * 30 + 12],
            "a region mostly inside the mask is kept"
        );
    }

    #[test]
    fn triangulation_recovers_white_on_white() {
        // A white (250) subject with a half-covered rim: over white it vanishes,
        // over black it shows; triangulation recovers α and colour exactly.
        let (w, h) = (8u32, 1u32);
        let f = 250.0f32;
        let alphas = [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0];
        let on_white = RgbImage::from_fn(w, h, |x, _| {
            let a = alphas[x as usize];
            let v = (a * f + (1.0 - a) * 255.0).round() as u8;
            image::Rgb([v, v, v])
        });
        let on_black = RgbImage::from_fn(w, h, |x, _| {
            let v = (alphas[x as usize] * f).round() as u8;
            image::Rgb([v, v, v])
        });
        let out = triangulation_matte(&on_white, &on_black, &[false; 8], &[true; 8]);
        for (x, &a) in alphas.iter().enumerate() {
            let p = out.get_pixel(x as u32, 0).0;
            assert!((p[3] as f32 / 255.0 - a).abs() < 0.01, "x={x} alpha {p:?}");
            if a > 0.0 {
                assert!((p[0] as f32 - f).abs() <= 2.0, "x={x} colour {p:?}");
            }
        }
    }

    #[test]
    fn alpha_on_white_recovers_colour() {
        // 50 % red over white = (227.5, 142.5, 147.5).
        let img = RgbImage::from_pixel(1, 1, image::Rgb([228, 142, 148]));
        let out = apply_alpha_on_white(&img, &[0.5]);
        let p = out.get_pixel(0, 0).0;
        assert!(
            (p[0] as i32 - 200).abs() <= 2 && (p[1] as i32 - 30).abs() <= 2 && p[3] == 128,
            "{p:?}"
        );
    }

    #[test]
    fn framing_centres_and_anchors_cut_off_subjects() {
        // A 20×10 opaque block in a 100×100 canvas, top-left.
        let mut img = RgbaImage::new(100, 100);
        for y in 10..20 {
            for x in 10..30 {
                img.put_pixel(x, y, Rgba([200, 30, 40, 255]));
            }
        }
        let out = frame_square(&img, 200, 0.75);
        let bbox = |o: &RgbaImage| {
            let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
            for (x, y, p) in o.enumerate_pixels() {
                if p.0[3] > 128 {
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                }
            }
            (x0, y0, x1, y1)
        };
        let (x0, y0, x1, y1) = bbox(&out);
        assert!(
            (x1 - x0) as i32 - 150 <= 2 && ((x0 + x1) as i32 - 200).abs() <= 2,
            "centred, 75 % wide: {x0}..{x1}"
        );
        assert!(
            ((y0 + y1) as i32 - 200).abs() <= 2,
            "vertically centred: {y0}..{y1}"
        );
        // Same block cut off by the bottom edge: anchored to the canvas bottom.
        let mut cut = RgbaImage::new(100, 100);
        for y in 60..100 {
            for x in 30..70 {
                cut.put_pixel(x, y, Rgba([200, 30, 40, 255]));
            }
        }
        let (_, _, _, y1) = bbox(&frame_square(&cut, 200, 0.75));
        assert!(y1 >= 198, "bottom-anchored, got {y1}");
    }

    #[test]
    fn resize_keeps_opaque_colour_and_size() {
        let img = RgbaImage::from_pixel(4, 3, Rgba([10, 200, 30, 255]));
        let big = upscale(&img, 5);
        assert_eq!(big.dimensions(), (20, 15));
        assert_eq!(big.get_pixel(10, 7).0, [10, 200, 30, 255]);
    }

    /// A brown frame (with a white pocket inside) and a tiny speck on white.
    #[test]
    fn white_prior_keeps_large_non_white_regions_only() {
        let img = RgbImage::from_fn(100, 100, |x, y| {
            let frame = (20..80).contains(&x) && (20..80).contains(&y);
            let pocket = (35..65).contains(&x) && (35..65).contains(&y);
            if frame && !pocket || (x == 5 && y == 5) {
                image::Rgb([120, 70, 30])
            } else {
                image::Rgb([255, 255, 255])
            }
        });
        let a = white_prior(&img).expect("white border");
        assert_eq!(a[25 * 100 + 25], 1.0);
        assert_eq!(a[50 * 100 + 50], 0.0, "enclosed white stays background");
        assert_eq!(a[5 * 100 + 5], 0.0, "speck dropped");
        let scene = RgbImage::from_pixel(50, 50, image::Rgb([20, 30, 60]));
        assert!(white_prior(&scene).is_none());
    }

    /// Feet touching the bottom edge keep the subject centred; a body cut by
    /// the edge anchors it there.
    #[test]
    fn frame_anchors_only_cut_subjects() {
        let mk = |cut: bool| {
            RgbaImage::from_fn(100, 100, |x, y| {
                let body = (30..70).contains(&x) && (20..80).contains(&y);
                let feet = !cut && y >= 80 && (x == 40 || x == 60);
                let bust = cut && (30..70).contains(&x) && y >= 20;
                if body || feet || bust {
                    Rgba([200, 100, 50, 255])
                } else {
                    Rgba([0, 0, 0, 0])
                }
            })
        };
        let bottom_alpha = |img: &RgbaImage| img.get_pixel(50, img.height() - 1).0[3];
        assert_eq!(bottom_alpha(&frame_square(&mk(false), 100, 0.75)), 0);
        assert_eq!(bottom_alpha(&frame_square(&mk(true), 100, 0.75)), 255);
    }
}
