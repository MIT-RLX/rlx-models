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

//! Images: the checkpoint's processor (`Qwen2VLImageProcessor` on the
//! torchvision backend) and the positions Qwen3.5 gives image tokens.
//!
//! The processor, reproduced step for step:
//!
//! 1. decode to RGB8 (`PIL.Image.convert("RGB")`);
//! 2. [`smart_resize`]: both sides a multiple of `patch * merge` (32), the
//!    area within `[min_pixels, max_pixels]`, aspect kept — Python `round`
//!    (half to even) and `math.floor`/`ceil` on `f64`;
//! 3. `torchvision...resize(uint8, BICUBIC, antialias=True)`, i.e. torch's
//!    int16 fixed-point separable path ([`rlx_core::image_preprocess::torch_resize_rgb8_aa`]);
//! 4. `(x - 127.5) / 127.5` in `f32` (rescale and normalize fused, as the
//!    backend does).
//!
//! JPEG is the one input that cannot match exactly: the `image` crate and
//! libjpeg-turbo (which PIL uses) disagree on a small fraction of pixels.
//! PNG and other lossless formats decode identically.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use rlx_core::image_preprocess::{Filter, torch_resize_rgb8_aa};

/// Spatial patch size and merge factor of the Qwen3.5 vision tower.
pub const PATCH: usize = 16;
pub const MERGE: usize = 2;

/// An RGB8 image, row-major HWC.
#[derive(Debug, Clone, PartialEq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl RgbImage {
    /// Decode encoded image bytes (PNG, JPEG, …).
    ///
    /// With the `libjpeg-turbo` feature, JPEGs go through libjpeg-turbo
    /// (accurate integer IDCT, fancy upsampling — PIL's defaults), which
    /// reproduces the reference's pixels exactly; anything it cannot turn
    /// into RGB (CMYK, say) falls back to the `image` crate.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        #[cfg(feature = "libjpeg-turbo")]
        if is_jpeg(bytes)
            && let Ok(img) = decode_jpeg_turbo(bytes)
        {
            return Ok(img);
        }
        let img = image::load_from_memory(bytes).context("decoding image")?;
        let rgb = img.to_rgb8();
        Ok(Self {
            width: rgb.width() as usize,
            height: rgb.height() as usize,
            pixels: rgb.into_raw(),
        })
    }

    /// An `images` entry: a `data:image/...;base64,` URL or a file path
    /// (relative paths resolve against `base`).
    pub fn open(entry: &str, base: Option<&Path>) -> Result<Self> {
        if let Some(rest) = entry.strip_prefix("data:image/") {
            let (_, payload) = rest
                .split_once(',')
                .ok_or_else(|| anyhow!("data URL without a ',' separator"))?;
            return Self::decode(&base64_decode(payload)?);
        }
        let path = PathBuf::from(entry);
        let path = match base {
            Some(b) if path.is_relative() => b.join(path),
            _ => path,
        };
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        Self::decode(&bytes).with_context(|| format!("decoding {}", path.display()))
    }
}

/// Whether `bytes` start like a JPEG (SOI marker).
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8, 0xFF])
}

/// Whether JPEGs decode exactly as the reference does (the `libjpeg-turbo`
/// feature); otherwise they are close but not bit-identical.
pub const JPEG_EXACT: bool = cfg!(feature = "libjpeg-turbo");

#[cfg(feature = "libjpeg-turbo")]
fn decode_jpeg_turbo(bytes: &[u8]) -> Result<RgbImage> {
    let img: turbojpeg::Image<Vec<u8>> =
        turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB).context("libjpeg-turbo")?;
    let row = img.width * 3;
    let pixels = if img.pitch == row {
        img.pixels
    } else {
        img.pixels
            .chunks(img.pitch)
            .take(img.height)
            .flat_map(|r| r[..row].iter().copied())
            .collect()
    };
    Ok(RgbImage {
        width: img.width,
        height: img.height,
        pixels,
    })
}

/// Strict standard base64 (`base64.b64decode(..., validate=True)`): the
/// standard alphabet, `=` padding only at the end, length a multiple of 4.
pub fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let b = s.as_bytes();
    if !b.len().is_multiple_of(4) {
        bail!("base64 length {} is not a multiple of 4", b.len());
    }
    let val = |c: u8| -> Result<u32> {
        Ok(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => bail!("invalid base64 character {:?}", c as char),
        } as u32)
    };
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    let chunks = b.len() / 4;
    for (i, q) in b.as_chunks::<4>().0.iter().enumerate() {
        let pad = q.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 || (pad > 0 && i + 1 != chunks) || q[..4 - pad].contains(&b'=') {
            bail!("misplaced base64 padding");
        }
        let mut n = 0u32;
        for &c in &q[..4 - pad] {
            n = (n << 6) | val(c)?;
        }
        n <<= 6 * pad as u32;
        let bytes = n.to_be_bytes();
        out.extend_from_slice(&bytes[1..4 - pad]);
    }
    Ok(out)
}

/// The processor's `smart_resize`, returning `(height, width)`.
///
/// Errors on an aspect ratio above 200, as the reference does.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        bail!("empty image");
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        bail!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        );
    }
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// One image ready for the tower.
#[derive(Debug, Clone, PartialEq)]
pub struct Preprocessed {
    /// Resized RGB8, row-major HWC (what the processor normalizes).
    pub resized: RgbImage,
    /// `[3, height, width]` normalized pixels.
    pub nchw: Vec<f32>,
    /// Merged-token grid: `height / 32` rows by `width / 32` columns.
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Preprocessed {
    /// Image tokens this image occupies in the prompt.
    pub fn n_tokens(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// The processor pipeline for one image.
pub fn preprocess(img: &RgbImage, min_pixels: usize, max_pixels: usize) -> Result<Preprocessed> {
    let factor = PATCH * MERGE;
    let (th, tw) = smart_resize(img.height, img.width, factor, min_pixels, max_pixels)?;
    let pixels = torch_resize_rgb8_aa(&img.pixels, img.width, img.height, tw, th, Filter::Bicubic);
    let plane = tw * th;
    let mut nchw = vec![0f32; 3 * plane];
    for (i, px) in pixels.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            nchw[c * plane + i] = (f32::from(px[c]) - 127.5) / 127.5;
        }
    }
    Ok(Preprocessed {
        resized: RgbImage {
            width: tw,
            height: th,
            pixels,
        },
        nchw,
        grid_h: th / factor,
        grid_w: tw / factor,
    })
}

/// M-RoPE positions `[t, h, w, 0]` for every token (`get_rope_index`).
///
/// Text advances all three axes together. A run of image tokens starts at
/// the current position `p`: token `(r, c)` of a `gh x gw` merged grid gets
/// `[p, p + r, p + c]`, and text resumes at `p + max(gh, gw)`. `images` gives
/// each image run's grid in order; `is_image` marks the image tokens.
pub fn mrope_positions(is_image: &[bool], images: &[(usize, usize)]) -> Result<Vec<[usize; 4]>> {
    let mut out = Vec::with_capacity(is_image.len());
    let mut grids = images.iter();
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < is_image.len() {
        if !is_image[i] {
            out.push([pos, pos, pos, 0]);
            pos += 1;
            i += 1;
            continue;
        }
        let &(gh, gw) = grids
            .next()
            .ok_or_else(|| anyhow!("more image runs than images"))?;
        let n = gh * gw;
        if i + n > is_image.len() || !is_image[i..i + n].iter().all(|&b| b) {
            bail!("image run at {i} is not {n} tokens long");
        }
        for r in 0..gh {
            for c in 0..gw {
                out.push([pos, pos + r, pos + c, 0]);
            }
        }
        pos += gh.max(gw);
        i += n;
    }
    if grids.next().is_some() {
        bail!("fewer image runs than images");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_the_reference_cases() {
        // Grids the reference processor produced for the fixture images.
        assert_eq!(
            smart_resize(900, 1400, 32, 65_536, 1_638_400).unwrap(),
            (896, 1408)
        );
        assert_eq!(
            smart_resize(120, 200, 32, 65_536, 1_638_400).unwrap(),
            (224, 352)
        );
        assert_eq!(
            smart_resize(1754, 1240, 32, 65_536, 1_638_400).unwrap(),
            (1504, 1056)
        );
        // Python rounds half to even: 48/32 = 1.5 -> 2, 80/32 = 2.5 -> 2.
        assert_eq!(smart_resize(48, 80, 32, 1, 1 << 30).unwrap(), (64, 64));
        assert!(smart_resize(1, 201, 32, 1, 1 << 30).is_err());
    }

    #[test]
    fn base64_is_strict() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGk=").unwrap(), b"hi");
        assert_eq!(base64_decode("aGVsbG8hIQ==").unwrap(), b"hello!!");
        assert!(base64_decode("aGVsbG8").is_err());
        assert!(base64_decode("aG=sbG8=").is_err());
        assert!(base64_decode("aGVs bG8=").is_err());
    }

    #[test]
    fn positions_follow_get_rope_index() {
        // 2 text, one 2x3 image, 1 text.
        let mut is_image = vec![false, false];
        is_image.extend([true; 6]);
        is_image.push(false);
        let p = mrope_positions(&is_image, &[(2, 3)]).unwrap();
        assert_eq!(p[0], [0, 0, 0, 0]);
        assert_eq!(p[1], [1, 1, 1, 0]);
        assert_eq!(p[2], [2, 2, 2, 0]); // (r, c) = (0, 0)
        assert_eq!(p[4], [2, 2, 4, 0]); // (0, 2)
        assert_eq!(p[5], [2, 3, 2, 0]); // (1, 0)
        assert_eq!(p[7], [2, 3, 4, 0]); // (1, 2)
        assert_eq!(p[8], [5, 5, 5, 0]); // 2 + max(2, 3)
        assert!(mrope_positions(&is_image, &[]).is_err());
        assert!(mrope_positions(&is_image, &[(2, 3), (1, 1)]).is_err());
    }
}
