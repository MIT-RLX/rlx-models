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

//! Measure the memory/speed trade-off a tile size buys, for one checkpoint on
//! one machine.
//!
//! The crate picks a tile from an *estimate* of peak working set — a planning
//! aid with a factor-of-two error bar, since how well the arena reuses a
//! transient belongs to the backend's memory planner. This runs the real thing
//! and prints estimate beside measurement, so the guess can be checked and a
//! tile chosen deliberately.
//!
//! Two costs pull against each other:
//!
//! * **Memory** is quadratic in the tile. Halving the tile is roughly a quarter
//!   of the working set.
//! * **Work** is *not* proportional to output. Only the `(tile − 2·overlap)²`
//!   interior survives the crop, so the halo is computed and thrown away. At
//!   tile 176 with a 16px halo that is 1.5× redundancy; at tile 80 it is 2.8×.
//!
//! Which dominates depends on the architecture. For a convolutional net a
//! smaller tile is often *faster* — the working set starts fitting in cache.
//! For a deep transformer it is markedly slower, because the redundancy
//! multiplies a much more expensive per-pixel cost.
//!
//! Peak RSS is process-wide and monotonic (`getrusage` high-water mark), so
//! each tile is measured in a **fresh process**: the harness re-executes itself
//! once per tile. Measuring them in one process would report the largest tile's
//! peak for every row.
//!
//! Run:
//!   cargo run --release -p rlx-upscale --example tile_sweep -- \
//!     --model model.pth --image photo.png
//!
//!   cargo run --release -p rlx-upscale --features metal --example tile_sweep -- \
//!     --model model.pth --image photo.png --device metal --tiles 64,96,128,192

use anyhow::{Context, Result, bail};
use rlx_cli::req;
use rlx_runtime::Device;
use rlx_upscale::{UpscaleOptions, Upscaler, parse_upscale_device};
use std::path::PathBuf;
use std::time::Instant;

/// Set on the re-executed child so it runs exactly one tile and reports it.
const CHILD_TILE: &str = "RLX_UPSCALE_SWEEP_TILE";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut device_name = "cpu".to_string();
    let mut tiles: Option<Vec<usize>> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model = Some(PathBuf::from(req(&args, &mut i)?)),
            "--image" => image = Some(PathBuf::from(req(&args, &mut i)?)),
            "--device" => {
                device_name = req(&args, &mut i)?;
                device = parse_upscale_device(&device_name)?;
            }
            "--tiles" => {
                tiles = Some(
                    req(&args, &mut i)?
                        .split(',')
                        .map(|t| t.trim().parse::<usize>().context("--tiles"))
                        .collect::<Result<_>>()?,
                )
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    let model = model.context("--model is required")?;
    let image = image.context("--image is required")?;

    // A child measures one tile and prints a single row.
    if let Ok(t) = rlx_ir::env::read(CHILD_TILE) {
        return measure(&model, &image, device, t.parse()?);
    }

    let probe = Upscaler::open(&model, device, UpscaleOptions::default())?;
    let cfg = probe.config().clone();
    drop(probe);
    let m = cfg.size_multiple().max(1);
    let chosen = rlx_upscale::runner::budget_tile(&cfg);

    // Default sweep: powers-of-two-ish steps around the chosen tile, snapped to
    // the window grid so every candidate is legal for this architecture.
    let tiles = tiles.unwrap_or_else(|| {
        let mut v: Vec<usize> = [0.5f32, 0.75, 1.0, 1.5, 2.0]
            .iter()
            .map(|f| ((chosen as f32 * f) as usize / m).max(2) * m)
            .collect();
        v.dedup();
        v
    });

    println!("{}", cfg.summary());
    println!("  device {device_name}, window grid {m}, crate would choose tile {chosen}\n");
    println!(
        "{:>6} {:>10} {:>10} {:>8} {:>9} {:>7}",
        "tile", "est peak", "peak RSS", "est/meas", "time", "halo"
    );

    let exe = std::env::current_exe().context("locating this example's binary")?;
    let mut seen: Vec<usize> = Vec::new();
    for tile in tiles {
        let est = cfg.peak_working_set_bytes(tile) >> 20;
        let out = std::process::Command::new(&exe)
            .env(CHILD_TILE, tile.to_string())
            .args([
                "--model",
                &model.to_string_lossy(),
                "--image",
                &image.to_string_lossy(),
                "--device",
                &device_name,
            ])
            .output()
            .context("re-executing for a fresh RSS high-water mark")?;
        let text = String::from_utf8_lossy(&out.stdout);
        let Some(row) = text.lines().find(|l| l.starts_with("ROW ")) else {
            println!(
                "{tile:>6} {est:>7} MB   (failed: {})",
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .next_back()
                    .unwrap_or("no output")
            );
            continue;
        };
        let f: Vec<&str> = row.split_whitespace().collect();
        let (actual, rss, secs, halo) = (
            f[1].parse::<usize>().unwrap_or(tile),
            f[2].parse::<u64>().unwrap_or(0),
            f[3].parse::<f64>().unwrap_or(0.0),
            f[4].parse::<f64>().unwrap_or(1.0),
        );
        if actual != tile {
            // The image is smaller than the request; anything above this is the
            // same run, so stop rather than print duplicate rows.
            println!("{tile:>6}   (clamped to {actual} — image is smaller)");
            if seen.contains(&actual) {
                continue;
            }
        }
        seen.push(actual);
        let est = cfg.peak_working_set_bytes(actual) >> 20;
        let ratio = if rss > 0 {
            format!("{:.2}×", est as f64 / rss as f64)
        } else {
            "—".into()
        };
        println!("{actual:>6} {est:>7} MB {rss:>7} MB {ratio:>8} {secs:>8.2}s {halo:>6.1}×");
    }
    println!(
        "\nest/meas near 1 means the estimate is calibrated for this model. Above 1\n\
         is conservative — usual for shallow conv nets, whose live-buffer count was\n\
         fitted to deeper ones. Below 1 is optimistic and worth a smaller --tile."
    );
    Ok(())
}

/// One tile, in this process, reported as a parsable row.
fn measure(model: &PathBuf, image: &PathBuf, device: Device, tile: usize) -> Result<()> {
    let img = image::open(image)
        .with_context(|| format!("opening {}", image.display()))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);

    let mut up = Upscaler::open(
        model,
        device,
        UpscaleOptions {
            tile: Some(tile),
            ..Default::default()
        },
    )?;
    let t0 = Instant::now();
    let _ = up.upscale_rgb8(img.as_raw(), w, h)?;
    let secs = t0.elapsed().as_secs_f64();
    // Report the tile that was actually *compiled*: a request larger than the
    // padded image is clamped down to fit, and reporting the request would
    // invent rows that never ran.
    println!(
        "ROW {} {} {secs:.3} {:.3}",
        up.compiled_tile().unwrap_or(tile),
        rlx_core::asr_bench::peak_rss_mb(),
        up.last_redundancy().unwrap_or(1.0),
    );
    Ok(())
}
