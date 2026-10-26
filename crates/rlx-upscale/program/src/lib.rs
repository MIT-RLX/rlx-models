// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only

//! The upscaler's host logic as an RLX program, answering `run`: what
//! `Upscaler::upscale_rgb8` / `upscale_rgba8` do around the network — the
//! tile plan (the crate's own `tile.rs`), each tile through the package's
//! network graph, the stitch, the 8-bit conversion and the alpha channel.
//!
//! In: `image`, `u8 [height, width, channels]` (RGB, or RGBA). Out: `image`,
//! the upscaled `u8 [height·s, width·s, channels]`. Options: `alpha`
//! (`"upscale"`, `"resize"` is not offered — the network's own pass —,
//! `"discard"`), `tile`, `overlap`.

#[allow(dead_code)]
#[path = "../../src/tile.rs"]
mod tile;

use rlx_program::{Message, Program, Tensor, run};
use serde_json::{Value, json};

struct Upscale {
    scale: usize,
    in_ch: usize,
    out_ch: usize,
    multiple: usize,
    overlap: usize,
    tile: usize,
    /// The network: one function symbolic in its tile, or one per tile side.
    symbolic: Option<String>,
    sized: Vec<(usize, String)>,
}

impl Upscale {
    /// The runner's plan: the tile on the window grid, shrunk toward the
    /// image, never below the halo's floor.
    fn plan(
        &self,
        w: usize,
        h: usize,
        tile: usize,
        overlap: usize,
    ) -> Result<tile::TilePlan, String> {
        let m = self.multiple;
        let round_up = |v: usize| v.div_ceil(m) * m;
        let floor = round_up(2 * overlap + m);
        let want = round_up(tile).max(floor);
        let fits = round_up(w.max(h)).max(floor);
        let tile = want.min(fits);
        tile::TilePlan::new(w, h, tile, overlap, self.scale, m).map_err(|e| e.to_string())
    }

    fn function(&self, tile: usize) -> Result<String, String> {
        if let Some(f) = &self.symbolic {
            return Ok(f.clone());
        }
        self.sized
            .iter()
            .find(|(t, _)| *t == tile)
            .map(|(_, f)| f.clone())
            .ok_or_else(|| format!("the package has no network for a {tile}px tile"))
    }

    /// `Upscaler::upscale_planar`: planar CHW in `[0, 1]` → planar CHW.
    fn planar(
        &self,
        src: &[f32],
        w: usize,
        h: usize,
        tile: usize,
        overlap: usize,
    ) -> Result<Vec<f32>, String> {
        let c = self.in_ch;
        let plan = self.plan(w, h, tile, overlap)?;
        let function = self.function(plan.tile)?;
        let (ow, oh) = (plan.out_w(), plan.out_h());
        let oc = self.out_ch;
        let mut out = vec![0f32; oc * ow * oh];
        let t = plan.tile;
        let tile_out = t * plan.scale;
        for tl in &plan.tiles {
            let patch = tile::extract(src, c, w, h, tl.src);
            let y = run(
                &function,
                vec![Tensor::f32("image", vec![1, c, t, t], &patch)],
            )?;
            let y = y.first().ok_or("the network returned nothing")?.to_f32();
            if y.len() != oc * tile_out * tile_out {
                return Err(format!(
                    "a tile's output is {} samples, not {}",
                    y.len(),
                    oc * tile_out * tile_out
                ));
            }
            tile::blit(
                &mut out, ow, oh, &y, tile_out, tile_out, oc, tl.crop, tl.dst,
            );
        }
        Ok(out)
    }

    /// `Upscaler::upscale_rgb8`: interleaved 8-bit, `c` channels.
    fn rgb8(
        &self,
        rgb: &[u8],
        w: usize,
        h: usize,
        tile: usize,
        overlap: usize,
    ) -> Result<Vec<u8>, String> {
        let c = self.in_ch;
        let mut planar = vec![0f32; c * w * h];
        for i in 0..w * h {
            for ci in 0..c {
                planar[ci * w * h + i] = rgb[i * c + ci] as f32 / 255.0;
            }
        }
        let out = self.planar(&planar, w, h, tile, overlap)?;
        let (ow, oh) = (w * self.scale, h * self.scale);
        let oc = self.out_ch;
        let mut pixels = vec![0u8; oc * ow * oh];
        for i in 0..ow * oh {
            for ci in 0..oc {
                pixels[i * oc + ci] =
                    (out[ci * ow * oh + i] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(pixels)
    }
}

impl Program for Upscale {
    fn init(config: Value) -> Result<Self, String> {
        let n = |k: &str| {
            config[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or(format!("config has no {k}"))
        };
        let sized = config["functions"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|f| {
                        Some((
                            f["tile"].as_u64()? as usize,
                            f["function"].as_str()?.to_string(),
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Upscale {
            scale: n("scale")?,
            in_ch: n("in_channels")?,
            out_ch: n("out_channels")?,
            multiple: n("size_multiple")?,
            overlap: n("overlap")?,
            tile: n("tile")?,
            symbolic: config["function"].as_str().map(String::from),
            sized,
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        if task != "run" {
            return Err(format!("the upscaler answers run, not {task}"));
        }
        let image = req.tensor("image").ok_or("run takes an `image` tensor")?;
        if image.dtype != "u8" || image.shape.len() != 3 {
            return Err("`image` is u8 [height, width, channels]".into());
        }
        let (h, w, c) = (image.shape[0], image.shape[1], image.shape[2]);
        let o = &req.body["options"];
        let tile = o["tile"].as_u64().map_or(self.tile, |v| v as usize);
        let overlap = o["overlap"].as_u64().map_or(self.overlap, |v| v as usize);
        let alpha = o["alpha"].as_str().unwrap_or("upscale");
        let (ow, oh) = (w * self.scale, h * self.scale);
        let px = &image.data;
        let (pixels, oc) = if c == self.in_ch {
            (self.rgb8(px, w, h, tile, overlap)?, self.out_ch)
        } else if c == 4 && self.in_ch == 3 {
            // `Upscaler::upscale_rgba8`: the colour, then the mask through the
            // same network (or dropped).
            let mut rgb = vec![0u8; 3 * w * h];
            let mut a = vec![0u8; w * h];
            for i in 0..w * h {
                rgb[i * 3..i * 3 + 3].copy_from_slice(&px[i * 4..i * 4 + 3]);
                a[i] = px[i * 4 + 3];
            }
            let colour = self.rgb8(&rgb, w, h, tile, overlap)?;
            match alpha {
                "discard" => (colour, self.out_ch),
                "upscale" => {
                    let up = if a.iter().all(|&v| v == 255) {
                        vec![255u8; ow * oh]
                    } else {
                        let mut grey = vec![0u8; 3 * w * h];
                        for i in 0..w * h {
                            grey[i * 3..i * 3 + 3].fill(a[i]);
                        }
                        let wide = self.rgb8(&grey, w, h, tile, overlap)?;
                        (0..ow * oh).map(|i| wide[i * self.out_ch]).collect()
                    };
                    let mut out = vec![0u8; 4 * ow * oh];
                    for i in 0..ow * oh {
                        out[i * 4..i * 4 + 3]
                            .copy_from_slice(&colour[i * self.out_ch..i * self.out_ch + 3]);
                        out[i * 4 + 3] = up[i];
                    }
                    (out, 4)
                }
                other => return Err(format!("alpha {other:?}: upscale or discard")),
            }
        } else {
            return Err(format!(
                "a {c}-channel image for a {}-channel network",
                self.in_ch
            ));
        };
        Ok(Message::new(
            json!({"width": ow, "height": oh}),
            vec![Tensor::new("image", "u8", vec![oh, ow, oc], pixels)],
        ))
    }
}

rlx_program::program!(Upscale);
