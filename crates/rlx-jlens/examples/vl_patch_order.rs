//! Does vision token `p` correspond to grid cell `(p % gx, p / gx)`?
//!
//! Every spatial claim made from a patch-grid heatmap rests on this mapping,
//! and it is not free: the Qwen2.5-VL tower runs **window attention**, which
//! permutes the token sequence and permutes it back, and the merger folds each
//! 2x2 patch block into one token. A map drawn on the wrong assumption is not
//! obviously wrong — it is a plausible-looking picture of nothing.
//!
//! So test it rather than read it, and test it by *sensitivity* rather than by
//! outlier-hunting. Encode a uniform grey field once as a baseline, then encode
//! it again with one cell brightened, and take the per-token change
//! `‖emb' - emb‖`. The token whose receptive field contains that cell is the one
//! that moves most. Sweeping every cell turns the mapping into a permutation
//! check.
//!
//! Two weaker probes were tried first and both are misleading, which is worth
//! recording because each *looks* like evidence against the ordering:
//!
//! * **Furthest from the batch mean.** A vision transformer mixes patches, so a
//!   bright cell perturbs the whole sequence and the most unusual token need not
//!   be the one that saw it. Scores far above chance, is not a permutation.
//! * **Raw argmax of the per-token delta.** The tower has its own sink tokens
//!   that absorb every perturbation, so one or two indices win almost regardless
//!   of where the light was.
//!
//! The fix is to score each token against *its own* behaviour: build the full
//! `[cell, token]` delta matrix and z-score down each token column, so a token
//! that always moves a lot cannot win on offset alone. The headline number is
//! then not the argmax at all but the **mean z at the expected index** against
//! the mean elsewhere — an argmax can be noisy while the mapping is still
//! plainly right.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen25-vl,metal --release --example vl_patch_order -- \
//!     --device metal --mmproj .../mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
//! ```

#![cfg(feature = "qwen25-vl")]

use anyhow::{Context, Result, bail};

/// Side of one merged token in pixels: `patch_size * spatial_merge_size`.
const CELL: u32 = 28;

fn main() -> Result<()> {
    let mut dir = "/Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct".to_string();
    let mut device = rlx_runtime::Device::Cpu;
    let mut mmproj: Option<String> = None;
    // Grid to test, in merged tokens.
    let (mut gx, mut gy) = (16u32, 9u32);

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .with_context(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--weights" => {
                dir = need(i)?;
                i += 2;
            }
            "--device" => {
                device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            "--mmproj" => {
                mmproj = Some(need(i)?);
                i += 2;
            }
            "--grid" => {
                let v = need(i)?;
                let (a, b) = v.split_once('x').context("--grid wants WxH")?;
                gx = a.parse()?;
                gy = b.parse()?;
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }

    let mmproj_path = match mmproj {
        Some(p) => std::path::PathBuf::from(p),
        None => std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf"))
            })
            .context("no mmproj-*.gguf beside the weights; pass --mmproj")?,
    };
    let mut runner = rlx_qwen25_vl::runner::Qwen25VlRunner::builder()
        .weights(&dir)
        .hf_config(std::path::Path::new(&dir).join("config.json"))
        .mmproj(&mmproj_path)
        .device(device)
        .build()?;

    let (w, h) = (gx * CELL, gy * CELL);
    println!("probing a {gx}x{gy} grid on a {w}x{h} field\n");

    // The unperturbed field every probe is differenced against.
    let flat = vec![128u8; (w * h * 3) as usize];
    let base_out = runner.encode_image(&flat, w as usize, h as usize)?;
    if base_out.grid_x as u32 != gx || base_out.grid_y as u32 != gy {
        bail!(
            "asked for {gx}x{gy}, encoder produced {}x{} — preprocessing resized the field",
            base_out.grid_x,
            base_out.grid_y
        );
    }
    let base_emb = base_out.embeddings.clone();
    let total = (gx * gy) as usize;

    // ── stripe test ──
    //
    // A single cell is a weak stimulus against a tower that mixes patches. A
    // whole row or column is 9-16x the signal and its expected footprint is a
    // whole band of token indices, so local blur cannot move the answer far. If
    // token order is raster, brightening row `r` must light up tokens
    // `r*gx .. (r+1)*gx`, and brightening column `c` must light up
    // `{c, c+gx, c+2gx, …}`. These are different index patterns, so passing both
    // pins the mapping in both axes.
    let stripe = |runner: &mut rlx_qwen25_vl::runner::Qwen25VlRunner,
                  horizontal: bool,
                  idx: u32|
     -> Result<Vec<f32>> {
        let mut rgb = vec![128u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let on = if horizontal {
                    y / CELL == idx
                } else {
                    x / CELL == idx
                };
                if on {
                    let o = ((y * w + x) * 3) as usize;
                    rgb[o] = 255;
                    rgb[o + 1] = 255;
                    rgb[o + 2] = 255;
                }
            }
        }
        let out = runner.encode_image(&rgb, w as usize, h as usize)?;
        let n = out.n_tokens;
        let d = out.embeddings.len() / n;
        Ok((0..n)
            .map(|p| {
                (0..d)
                    .map(|j| (out.embeddings[p * d + j] - base_emb[p * d + j]).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .collect())
    };

    for (label, horizontal, count) in [("row", true, gy), ("column", false, gx)] {
        let mut raw = Vec::with_capacity(count as usize);
        for i in 0..count {
            raw.push(stripe(&mut runner, horizontal, i)?);
        }
        // Z-score down each token column across the stripe probes, same reason
        // as before: the tower's sink tokens move for every stimulus.
        let n = raw[0].len();
        let mut z = vec![vec![0f32; n]; count as usize];
        for p in 0..n {
            let col: Vec<f32> = raw.iter().map(|r| r[p]).collect();
            let mean = col.iter().sum::<f32>() / count as f32;
            let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / count as f32)
                .sqrt()
                .max(1e-12);
            for (c, zc) in z.iter_mut().enumerate() {
                zc[p] = (raw[c][p] - mean) / sd;
            }
        }
        let width = if horizontal { gx as usize } else { gy as usize };
        let mut inside = 0f64;
        let mut hit = 0usize;
        for i in 0..count as usize {
            let belongs = |p: usize| {
                if horizontal {
                    p / gx as usize == i
                } else {
                    p % gx as usize == i
                }
            };
            // Of the `width` strongest responders, how many are in the band the
            // raster hypothesis predicts? Chance is `width / total`.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| z[i][b].partial_cmp(&z[i][a]).unwrap());
            let got = order[..width].iter().filter(|&&p| belongs(p)).count();
            hit += got;
            inside += (0..n)
                .filter(|&p| belongs(p))
                .map(|p| z[i][p] as f64)
                .sum::<f64>()
                / width as f64;
        }
        let inside = inside / count as f64;
        let chance = 100.0 * width as f64 / n as f64;
        println!(
            "{label} stripes: {hit}/{} of the top-{width} responders fall in the predicted band \
             ({:.0}% vs {chance:.0}% chance), mean z inside the band {inside:+.2}",
            count as usize * width,
            100.0 * hit as f64 / (count as usize * width) as f64,
        );
    }
    println!();

    // delta[cell][token]
    let mut delta = vec![vec![0f32; total]; total];
    for cy in 0..gy {
        for cx in 0..gx {
            // Mid grey everywhere, one white cell. Grey rather than black so the
            // bright cell is the only thing unusual in the frame.
            let mut rgb = vec![128u8; (w * h * 3) as usize];
            for y in cy * CELL..(cy + 1) * CELL {
                for x in cx * CELL..(cx + 1) * CELL {
                    let o = ((y * w + x) * 3) as usize;
                    rgb[o] = 255;
                    rgb[o + 1] = 255;
                    rgb[o + 2] = 255;
                }
            }
            let out = runner.encode_image(&rgb, w as usize, h as usize)?;
            if out.grid_x as u32 != gx || out.grid_y as u32 != gy {
                bail!("encoder produced {}x{}", out.grid_x, out.grid_y);
            }
            let n = out.n_tokens;
            let d = out.embeddings.len() / n;
            let cell = (cy * gx + cx) as usize;
            for p in 0..n {
                delta[cell][p] = (0..d)
                    .map(|j| (out.embeddings[p * d + j] - base_emb[p * d + j]).powi(2))
                    .sum::<f32>()
                    .sqrt();
            }
        }
    }

    // Z-score down each token column: how unusual is this token's response to
    // this cell, relative to how it responds to every other cell.
    let mut z = vec![vec![0f32; total]; total];
    for p in 0..total {
        let col: Vec<f32> = (0..total).map(|c| delta[c][p]).collect();
        let mean = col.iter().sum::<f32>() / total as f32;
        let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / total as f32)
            .sqrt()
            .max(1e-12);
        for c in 0..total {
            z[c][p] = (delta[c][p] - mean) / sd;
        }
    }

    let mut hits = 0usize;
    let mut z_expected = 0f64;
    let mut z_other = 0f64;
    let mut rank_sum = 0usize;
    let mut worst: Vec<(usize, usize, usize)> = Vec::new();
    for cell in 0..total {
        let best = (0..total)
            .max_by(|&a, &b| z[cell][a].partial_cmp(&z[cell][b]).unwrap())
            .unwrap_or(0);
        if best == cell {
            hits += 1;
        } else {
            worst.push((cell, best, 0));
        }
        z_expected += z[cell][cell] as f64;
        for p in 0..total {
            if p != cell {
                z_other += z[cell][p] as f64 / (total - 1) as f64;
            }
        }
        let rank = (0..total).filter(|&p| z[cell][p] > z[cell][cell]).count();
        rank_sum += rank;
    }
    let z_expected = z_expected / total as f64;
    let z_other = z_other / total as f64;
    let mean_rank = rank_sum as f64 / total as f64;

    println!("argmax lands on the expected token for {hits} of {total} cells");
    println!("mean z at the EXPECTED token index : {z_expected:+.2}");
    println!("mean z at every other token index  : {z_other:+.2}");
    println!("mean rank of the expected token    : {mean_rank:.2} of {total} (0 = best)");
    println!();
    if z_expected > 3.0 && mean_rank < (total as f64 * 0.05) {
        println!(
            "token p IS grid cell (p % gx, p / gx). The expected token is the one that \n\
             responds, by a wide margin, at essentially every cell — patch-grid heatmaps \n\
             are spatially sound."
        );
    } else if z_expected > 1.0 {
        println!(
            "the expected token responds more than chance but not decisively \n\
             (z {z_expected:+.2}, mean rank {mean_rank:.1}) — the mapping is probably right \n\
             but this probe cannot carry a strong spatial claim on its own."
        );
    } else {
        println!(
            "the expected token does NOT respond preferentially (z {z_expected:+.2}) — \n\
             token order is not raster order, and every spatial reading of a patch-grid \n\
             heatmap is suspect until the true permutation is recovered."
        );
        for (cell, best, _) in worst.iter().take(12) {
            println!(
                "  cell ({},{}) -> token ({},{})",
                cell % total.min(gx as usize),
                cell / gx as usize,
                best % gx as usize,
                best / gx as usize
            );
        }
    }
    Ok(())
}
