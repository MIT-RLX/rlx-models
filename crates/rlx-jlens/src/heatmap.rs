//! Rendering a patch-grid magnitude map over the image it came from.
//!
//! Three decisions worth stating, because each is the kind that quietly makes a
//! figure lie:
//!
//! * **Sequential encoding, one hue, light→dark.** Magnitude is a magnitude:
//!   it gets a single-hue ramp where lightness carries the value. A rainbow
//!   would invent category boundaries at every hue change — the eye reads
//!   green-to-yellow as a step even where the data is smooth.
//! * **The photo is desaturated first.** A saliency map laid over a
//!   full-colour photograph puts two colour signals in one frame, and a reader
//!   cannot tell which one carries the number. Greyscale underneath leaves the
//!   ramp as the only hue in the image, so colour means magnitude and nothing
//!   else. The greys are also compressed into a mid band so the dark end of the
//!   ramp still has somewhere to go over a dark photo.
//! * **Resampling is a choice, not a detail.** [`Upsample::Bilinear`] between
//!   patch *centres* is the default: it renders the measured field as the
//!   continuous quantity it samples, which is what a reader expects and what
//!   makes an overlay legible against a photograph. It adds no information —
//!   every patch centre still carries exactly its measured value — but it does
//!   hide where the samples are, so state the grid size alongside the figure.
//!   [`Upsample::Nearest`] keeps the lattice visible and is the honest choice
//!   when the question is "which patch"; it reads as a mosaic.
//!
//! The ramp is the house sequential blue, 100→700, used unchanged.

/// House sequential ramp: blue, steps 100 → 700, light to dark.
pub const SEQUENTIAL_BLUE: [[u8; 3]; 13] = [
    [0xcd, 0xe2, 0xfb], // 100
    [0xb7, 0xd3, 0xf6], // 150
    [0x9e, 0xc5, 0xf4], // 200
    [0x86, 0xb6, 0xef], // 250
    [0x6d, 0xa7, 0xec], // 300
    [0x55, 0x98, 0xe7], // 350
    [0x39, 0x87, 0xe5], // 400
    [0x2a, 0x78, 0xd6], // 450
    [0x25, 0x6a, 0xbf], // 500
    [0x1c, 0x5c, 0xab], // 550
    [0x18, 0x4f, 0x95], // 600
    [0x10, 0x42, 0x81], // 650
    [0x0d, 0x36, 0x6b], // 700
];

/// Chart surface, for the gaps between tiles and around the colour bar.
pub const SURFACE: [u8; 3] = [0xfc, 0xfc, 0xfb];

/// The ramp at `t ∈ [0, 1]`, linearly interpolated between steps.
pub fn sample(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let last = SEQUENTIAL_BLUE.len() - 1;
    let x = t * last as f32;
    let i = (x.floor() as usize).min(last);
    let j = (i + 1).min(last);
    let f = x - i as f32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        let a = SEQUENTIAL_BLUE[i][c] as f32;
        let b = SEQUENTIAL_BLUE[j][c] as f32;
        out[c] = (a + (b - a) * f).round() as u8;
    }
    out
}

/// Perceptual luminance, 0..1.
#[cfg(any(feature = "render", test))]
fn luma(p: [u8; 3]) -> f32 {
    (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0
}

/// How a patch grid is resampled to pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Upsample {
    /// Bilinear between patch **centres**, edge-clamped. The default: it renders
    /// the measured field as the continuous thing it is sampling, which is what
    /// a reader of a saliency map expects. It does not add information — every
    /// patch centre still carries exactly its measured value — but it does hide
    /// where the samples are, so pair it with the grid size in the caption.
    #[default]
    Bilinear,
    /// One flat block per patch. Honest about the sampling lattice and useful
    /// when the question is "which patch", but it reads as a mosaic.
    Nearest,
}

/// Sample the `[gx, gy]` grid at normalised image coordinates `fx, fy ∈ [0, 1]`.
///
/// Patch centres sit at `((cx + 0.5) / gx, (cy + 0.5) / gy)`, not at cell
/// corners, so the grid-to-pixel map has a half-cell offset. Interpolating on
/// raw cell indices instead shifts the whole field by half a patch — half a
/// patch is 14 px here, which is enough to move a peak off the object it
/// belongs to. Outside the centre lattice the value is clamped rather than
/// extrapolated, so edge patches stay flat instead of ramping to nothing.
pub fn sample_grid(mask: &[f32], gx: usize, gy: usize, fx: f32, fy: f32, up: Upsample) -> f32 {
    debug_assert_eq!(mask.len(), gx * gy);
    let at = |cx: usize, cy: usize| mask[cy.min(gy - 1) * gx + cx.min(gx - 1)];
    match up {
        Upsample::Nearest => {
            let cx = ((fx * gx as f32) as usize).min(gx - 1);
            let cy = ((fy * gy as f32) as usize).min(gy - 1);
            at(cx, cy)
        }
        Upsample::Bilinear => {
            // Position in "centre lattice" units: cell centre c sits at c.
            let gxf = (fx * gx as f32 - 0.5).clamp(0.0, (gx - 1) as f32);
            let gyf = (fy * gy as f32 - 0.5).clamp(0.0, (gy - 1) as f32);
            let (x0, y0) = (gxf.floor() as usize, gyf.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(gx - 1), (y0 + 1).min(gy - 1));
            let (tx, ty) = (gxf - x0 as f32, gyf - y0 as f32);
            let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
            let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
            top * (1.0 - ty) + bot * ty
        }
    }
}

#[cfg(feature = "render")]
mod render {
    use super::{SURFACE, Upsample, luma, sample, sample_grid};
    use image::RgbImage;

    /// Height of the colour bar drawn under each overlay, in pixels.
    const BAR_H: u32 = 14;
    /// Surface gap between a fill and its neighbour.
    const GAP: u32 = 2;

    /// `mask` over `photo`, as a `[gx, gy]` grid normalised to its own maximum.
    ///
    /// Returns the overlay with a ramp legend along the bottom, so a reader can
    /// tell light-means-low without being told. The scale is per-image — each
    /// call normalises independently — so magnitudes are comparable *within* an
    /// image and not across a set; print the absolute peak alongside.
    pub fn overlay(photo: &RgbImage, mask: &[f32], gx: usize, gy: usize) -> RgbImage {
        let hi = mask.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lo = mask.iter().copied().fold(f32::INFINITY, f32::min);
        overlay_scaled(photo, mask, gx, gy, lo, hi)
    }

    /// [`overlay_scaled`] with an explicit resampling mode.
    pub fn overlay_with(
        photo: &RgbImage,
        mask: &[f32],
        gx: usize,
        gy: usize,
        lo: f32,
        hi: f32,
        up: Upsample,
    ) -> RgbImage {
        render_overlay(photo, mask, gx, gy, lo, hi, up)
    }

    /// As [`overlay`], with the colour scale pinned to an explicit `lo..hi`.
    ///
    /// Use this whenever several overlays will be looked at together. Scaling
    /// each one to its own extremes is the right default for a single figure and
    /// actively misleading for a set: a layer holding 2% of the attention and one
    /// holding 22% both saturate the ramp, so the strongest visual signal in the
    /// contact sheet — "how dark is it" — carries no information, and a viewer
    /// reads sharp early-layer focus that is not there. A shared scale makes
    /// darkness mean the same thing in every tile.
    pub fn overlay_scaled(
        photo: &RgbImage,
        mask: &[f32],
        gx: usize,
        gy: usize,
        lo: f32,
        hi: f32,
    ) -> RgbImage {
        render_overlay(photo, mask, gx, gy, lo, hi, Upsample::default())
    }

    fn render_overlay(
        photo: &RgbImage,
        mask: &[f32],
        gx: usize,
        gy: usize,
        lo: f32,
        hi: f32,
        up: Upsample,
    ) -> RgbImage {
        assert_eq!(mask.len(), gx * gy, "mask is not {gx}x{gy}");
        let (w, h) = photo.dimensions();
        let mut out = RgbImage::new(w, h + GAP + BAR_H);
        let span = (hi - lo).max(1e-30);

        for y in 0..h {
            for x in 0..w {
                let src = photo.get_pixel(x, y).0;
                // Compress the photo into a mid grey band: pure greyscale would
                // leave the dark end of the ramp invisible over dark pixels.
                let g = 0.45 + 0.50 * luma(src);
                let gi = (g * 255.0).clamp(0.0, 255.0);

                // Sample at the pixel *centre*, so the field is not biased half
                // a pixel toward the origin.
                let fx = (x as f32 + 0.5) / w as f32;
                let fy = (y as f32 + 0.5) / h as f32;
                let t = (sample_grid(mask, gx, gy, fx, fy, up) - lo) / span;

                let c = sample(t);
                // Near-zero recedes to the grey photo; the peak keeps a little
                // texture rather than going flat.
                let a = (0.85 * t).clamp(0.0, 0.85);
                let px = [
                    (gi * (1.0 - a) + c[0] as f32 * a) as u8,
                    (gi * (1.0 - a) + c[1] as f32 * a) as u8,
                    (gi * (1.0 - a) + c[2] as f32 * a) as u8,
                ];
                out.put_pixel(x, y, image::Rgb(px));
            }
        }
        for y in h..h + GAP {
            for x in 0..w {
                out.put_pixel(x, y, image::Rgb(SURFACE));
            }
        }
        for y in h + GAP..h + GAP + BAR_H {
            for x in 0..w {
                out.put_pixel(x, y, image::Rgb(sample(x as f32 / (w - 1).max(1) as f32)));
            }
        }
        out
    }

    /// Tile equal-sized images into a contact sheet, `cols` across.
    ///
    /// Tiles are separated by a surface gap rather than butted together: two
    /// adjacent fills with no gap read as one region, which is exactly the
    /// misreading a per-layer sheet invites.
    pub fn contact_sheet(tiles: &[RgbImage], cols: usize) -> Option<RgbImage> {
        let first = tiles.first()?;
        let (tw, th) = first.dimensions();
        let cols = cols.max(1) as u32;
        let rows = (tiles.len() as u32).div_ceil(cols);
        let mut out = RgbImage::from_pixel(
            cols * tw + (cols + 1) * GAP,
            rows * th + (rows + 1) * GAP,
            image::Rgb(SURFACE),
        );
        for (i, tile) in tiles.iter().enumerate() {
            let (cx, cy) = (i as u32 % cols, i as u32 / cols);
            let (ox, oy) = (GAP + cx * (tw + GAP), GAP + cy * (th + GAP));
            for y in 0..th.min(tile.height()) {
                for x in 0..tw.min(tile.width()) {
                    out.put_pixel(ox + x, oy + y, *tile.get_pixel(x, y));
                }
            }
        }
        Some(out)
    }
}

#[cfg(feature = "render")]
pub use render::{contact_sheet, overlay, overlay_scaled, overlay_with};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_ends_are_the_documented_steps() {
        assert_eq!(sample(0.0), SEQUENTIAL_BLUE[0]);
        assert_eq!(sample(1.0), SEQUENTIAL_BLUE[SEQUENTIAL_BLUE.len() - 1]);
        // Out of range is clamped, not wrapped or panicking: masks arrive
        // normalised by a caller, and one bad value should not abort a render.
        assert_eq!(sample(-5.0), SEQUENTIAL_BLUE[0]);
        assert_eq!(sample(5.0), SEQUENTIAL_BLUE[SEQUENTIAL_BLUE.len() - 1]);
    }

    /// The whole point of a sequential ramp: lightness has to fall as the value
    /// rises, monotonically, or the encoding is not readable as a magnitude.
    #[test]
    fn lightness_falls_monotonically() {
        let mut prev = f32::INFINITY;
        for i in 0..=100 {
            let l = luma(sample(i as f32 / 100.0));
            assert!(
                l <= prev + 1e-3,
                "lightness rose at t = {}",
                i as f32 / 100.0
            );
            prev = l;
        }
        assert!(
            luma(sample(0.0)) - luma(sample(1.0)) > 0.5,
            "the ramp needs real lightness range end to end"
        );
    }

    /// A patch centre must come back with exactly its own value: bilinear that
    /// is off by half a cell shifts the whole field by 14 px here, enough to
    /// move a peak onto the wrong object.
    #[test]
    fn bilinear_reproduces_patch_centres() {
        let (gx, gy) = (4usize, 3usize);
        let mask: Vec<f32> = (0..gx * gy).map(|i| i as f32).collect();
        for cy in 0..gy {
            for cx in 0..gx {
                let fx = (cx as f32 + 0.5) / gx as f32;
                let fy = (cy as f32 + 0.5) / gy as f32;
                let got = sample_grid(&mask, gx, gy, fx, fy, Upsample::Bilinear);
                assert!(
                    (got - mask[cy * gx + cx]).abs() < 1e-4,
                    "centre ({cx},{cy}) gave {got}, want {}",
                    mask[cy * gx + cx]
                );
            }
        }
    }

    #[test]
    fn bilinear_is_monotone_between_two_patches() {
        let mask = [0.0f32, 1.0];
        let mut prev = f32::NEG_INFINITY;
        for i in 0..=50 {
            let v = sample_grid(&mask, 2, 1, i as f32 / 50.0, 0.5, Upsample::Bilinear);
            assert!(v >= prev - 1e-6, "not monotone at {i}");
            prev = v;
        }
        // Clamped, not extrapolated, outside the centre lattice.
        assert!((sample_grid(&mask, 2, 1, 0.0, 0.5, Upsample::Bilinear) - 0.0).abs() < 1e-6);
        assert!((sample_grid(&mask, 2, 1, 1.0, 0.5, Upsample::Bilinear) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nearest_returns_the_containing_cell() {
        let mask: Vec<f32> = (0..6).map(|i| i as f32).collect();
        assert_eq!(sample_grid(&mask, 3, 2, 0.1, 0.1, Upsample::Nearest), 0.0);
        assert_eq!(sample_grid(&mask, 3, 2, 0.9, 0.9, Upsample::Nearest), 5.0);
    }

    #[test]
    fn interpolation_is_continuous() {
        let a = sample(0.5);
        let b = sample(0.5 + 1e-4);
        for c in 0..3 {
            assert!(
                (a[c] as i32 - b[c] as i32).abs() <= 1,
                "ramp jumps mid-range"
            );
        }
    }
}
