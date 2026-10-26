// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! A super-resolution checkpoint exported as a package upscales what
//! `Upscaler` does, byte for byte: the real Compact net in `weights/image/
//! upscale` (or `RLX_UPSCALE_EXPORT_MODEL`; skipped where absent), an RGB
//! image that fits one tile, one that takes several, and an RGBA one.

#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_runtime::Device;
use rlx_runtime::model::Tensor;
use rlx_upscale::{UpscaleOptions, Upscaler};
use serde_json::json;

fn model() -> Option<PathBuf> {
    let p = rlx_ir::env::var("RLX_UPSCALE_EXPORT_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../weights/image/upscale/realesr-animevideov3.pth")
        });
    p.is_file().then_some(p)
}

fn image(w: usize, h: usize, c: usize) -> Vec<u8> {
    (0..w * h * c)
        .map(|i| {
            let (p, ch) = (i / c, i % c);
            let (y, x) = (p / w, p % w);
            if ch == 3 {
                // An alpha that is not all opaque.
                return ((x * 255) / w.max(1)) as u8;
            }
            ((x * 3 + y * 5 + ch * 70 + (x * y) % 17) % 256) as u8
        })
        .collect()
}

#[test]
fn an_exported_upscaler_gives_the_runners_pixels() {
    let Some(path) = model() else {
        eprintln!("skipped: no upscale checkpoint");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("upscale.rlxp");
    let report = rlx_upscale::export::export_rlxp(
        &path,
        &out,
        &rlx_upscale::export::ExportOptions::default(),
    )
    .unwrap();
    eprintln!("{report}");
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();
    let mut up = Upscaler::open(&path, Device::Cpu, UpscaleOptions::default()).unwrap();
    // The runner's default tile is its memory budget's; the package's is
    // the default at export. Pin both to the same small tile so the larger
    // image takes several tiles.
    let mut tiled = Upscaler::open(
        &path,
        Device::Cpu,
        UpscaleOptions {
            tile: Some(64),
            ..Default::default()
        },
    )
    .unwrap();
    for (w, h, c, tile) in [(37, 29, 3, None), (150, 90, 3, Some(64)), (40, 24, 4, None)] {
        let px = image(w, h, c);
        let runner = if tile.is_some() { &mut tiled } else { &mut up };
        let (want, ow, oh) = if c == 4 {
            runner.upscale_rgba8(&px, w, h).unwrap()
        } else {
            runner.upscale_rgb8(&px, w, h).unwrap()
        };
        let input = Tensor::new("image", rlx_ir::DType::U8, vec![h, w, c], px);
        let options = match tile {
            Some(t) => json!({"tile": t}),
            None => json!({}),
        };
        let outs = m.run(&[input], &options).unwrap();
        let got = &outs[0];
        assert_eq!(got.shape, vec![oh, ow, c], "{w}x{h}x{c}");
        let off = got.data.iter().zip(&want).filter(|(a, b)| a != b).count();
        let worst = got
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);

        eprintln!(
            "{w}x{h}x{c}: {off} of {} values differ, by at most {worst}",
            want.len()
        );
        assert_eq!(off, 0, "{w}x{h}x{c}: off by up to {worst}");
    }
}
