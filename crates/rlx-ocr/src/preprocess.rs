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

//! Greyscale image preprocessing for OCR models.

use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use std::fmt;

/// Normalized greyscale background value used by ocrs models (matches `ocrs` 0.12.x).
pub const BLACK_VALUE: f32 = -0.5;

/// Errors when constructing an [`ImageSource`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSourceError {
    UnsupportedChannelCount,
    InvalidDataLength,
}

impl fmt::Display for ImageSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedChannelCount => f.write_str("channel count is not 1, 3 or 4"),
            Self::InvalidDataLength => {
                f.write_str("data length is not a multiple of width * height")
            }
        }
    }
}

impl std::error::Error for ImageSourceError {}

/// Pixel layout for image tensors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DimOrder {
    Hwc,
    Chw,
}

enum ImagePixels<'a> {
    #[allow(dead_code)]
    Floats(NdTensorView<'a, f32, 3>),
    Bytes(NdTensorView<'a, u8, 3>),
    FloatsOwned(NdTensor<f32, 3>),
}

/// Input image for [`crate::OcrEngine::prepare_input`].
pub struct ImageSource<'a> {
    data: ImagePixels<'a>,
    order: DimOrder,
}

impl<'a> ImageSource<'a> {
    /// RGB/RGBA/greyscale bytes in HWC order (`image` crate layout).
    pub fn from_bytes(bytes: &'a [u8], dimensions: (u32, u32)) -> Result<Self, ImageSourceError> {
        let (width, height) = dimensions;
        let channel_len = (width as usize).saturating_mul(height as usize);
        if channel_len == 0 {
            return Err(ImageSourceError::UnsupportedChannelCount);
        }
        if !bytes.len().is_multiple_of(channel_len) {
            return Err(ImageSourceError::InvalidDataLength);
        }
        let chans = bytes.len() / channel_len;
        if !matches!(chans, 1 | 3 | 4) {
            return Err(ImageSourceError::UnsupportedChannelCount);
        }
        let view = NdTensorView::from_data([height as usize, width as usize, chans], bytes);
        Ok(Self {
            data: ImagePixels::Bytes(view),
            order: DimOrder::Hwc,
        })
    }

    /// Existing CHW or HWC float tensor (copied).
    pub fn from_tensor(
        tensor: NdTensorView<'_, f32, 3>,
        order: DimOrder,
    ) -> Result<ImageSource<'static>, ImageSourceError> {
        let chans = match order {
            DimOrder::Hwc => tensor.size(2),
            DimOrder::Chw => tensor.size(0),
        };
        if chans == 0 || !matches!(chans, 1 | 3 | 4) {
            return Err(ImageSourceError::UnsupportedChannelCount);
        }
        let owned = NdTensor::from_data(tensor.shape(), tensor.to_vec());
        Ok(ImageSource {
            data: ImagePixels::FloatsOwned(owned),
            order,
        })
    }
}

/// Convert an image to a normalized greyscale CHW tensor `[1, H, W]`.
pub fn prepare_image(img: ImageSource<'_>) -> NdTensor<f32, 3> {
    match (&img.data, img.order) {
        (ImagePixels::Floats(f), DimOrder::Hwc) => prepare_floats::<true>(f.view()),
        (ImagePixels::Floats(f), DimOrder::Chw) => prepare_floats::<false>(f.view()),
        (ImagePixels::FloatsOwned(f), DimOrder::Hwc) => prepare_floats::<true>(f.view()),
        (ImagePixels::FloatsOwned(f), DimOrder::Chw) => prepare_floats::<false>(f.view()),
        (ImagePixels::Bytes(b), DimOrder::Hwc) => prepare_bytes::<true>(b.view()),
        (ImagePixels::Bytes(b), DimOrder::Chw) => prepare_bytes::<false>(b.view()),
    }
}

fn prepare_floats<const CHANS_LAST: bool>(floats: NdTensorView<'_, f32, 3>) -> NdTensor<f32, 3> {
    const ITU: [f32; 3] = [0.299, 0.587, 0.114];
    let n = if CHANS_LAST {
        floats.shape()[2]
    } else {
        floats.shape()[0]
    };
    match n {
        1 => convert_pixels::<f32, 1, 1, CHANS_LAST>(floats, [1.]),
        3 => convert_pixels::<f32, 3, 3, CHANS_LAST>(floats, ITU),
        4 => convert_pixels::<f32, 4, 3, CHANS_LAST>(floats, ITU),
        _ => panic!("expected greyscale, RGB or RGBA input image"),
    }
}

fn prepare_bytes<const CHANS_LAST: bool>(bytes: NdTensorView<'_, u8, 3>) -> NdTensor<f32, 3> {
    const ITU: [f32; 3] = [0.299, 0.587, 0.114];
    let weights = ITU.map(|w| w / 255.0);
    let n = if CHANS_LAST {
        bytes.shape()[2]
    } else {
        bytes.shape()[0]
    };
    match n {
        1 => convert_pixels::<u8, 1, 1, CHANS_LAST>(bytes, [1. / 255.0]),
        3 => convert_pixels::<u8, 3, 3, CHANS_LAST>(bytes, weights),
        4 => convert_pixels::<u8, 4, 3, CHANS_LAST>(bytes, weights),
        _ => panic!("expected greyscale, RGB or RGBA input image"),
    }
}

fn convert_pixels<
    T: Copy + Into<f32>,
    const PIXEL_STRIDE: usize,
    const CHANS: usize,
    const CHANS_LAST: bool,
>(
    src: NdTensorView<'_, T, 3>,
    chan_weights: [f32; CHANS],
) -> NdTensor<f32, 3> {
    let [height, width, chans] = if CHANS_LAST {
        src.shape()
    } else {
        let [c, h, w] = src.shape();
        [h, w, c]
    };
    assert_eq!(chans, PIXEL_STRIDE);

    let mut out_pixels = Vec::with_capacity(height * width);
    if CHANS_LAST {
        let src = src.to_contiguous();
        let mut iter = src.data().chunks_exact(PIXEL_STRIDE);
        debug_assert!(iter.remainder().is_empty());
        for in_pixel in iter.by_ref() {
            let mut pixel = BLACK_VALUE;
            for (c, &w) in chan_weights.iter().enumerate() {
                pixel += in_pixel[c].into() * w;
            }
            out_pixels.push(pixel);
        }
    } else {
        for y in 0..height {
            out_pixels.extend((0..width).map(|x| {
                let mut pixel = BLACK_VALUE;
                for (c, &w) in chan_weights.iter().enumerate() {
                    pixel += src[[c, y, x]].into() * w;
                }
                pixel
            }));
        }
    }
    NdTensor::from_data([1, height, width], out_pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_greyscale_bytes() {
        let out = prepare_image(ImageSource::from_bytes(&[0, 128, 255, 64], (2, 2)).unwrap());
        assert_eq!(out.shape(), [1, 2, 2]);
        assert!((out[[0, 0, 0]] - BLACK_VALUE).abs() < 1e-5);
        assert!((out[[0, 0, 1]] - (BLACK_VALUE + 128.0 / 255.0)).abs() < 1e-5);
    }
}
