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

//! Figure 4 — inverse rendering, adaptive voxel grids against fixed ones.
//!
//! ```text
//! # synthetic scene, no download needed
//! cargo run -p rlx-fgd --release --example radiance -- --steps 60 --fixed
//!
//! # the paper's setting on NeRF-Synthetic (download the dataset first)
//! cargo run -p rlx-fgd --release --example radiance -- \
//!     --scene /path/to/nerf_synthetic/ficus --steps 200 --max-level 3 --out /tmp/ficus
//! ```
//!
//! With `--out`, writes PPM renders of held-out views beside the truth.

use rlx_fgd::algorithm::{FgdConfig, Refinement, Refinement::Fixed, Report, run};
use rlx_fgd::rep::VoxelGrid;
use rlx_fgd::scene::{
    Image, LoadOptions, RenderOptions, Scene, Split, load_nerf_synthetic, render_image,
};
use rlx_fgd::tasks::radiance::{Occupancy, RadianceField};
use std::time::Instant;

fn flag(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    flag(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn present(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// A two-blob object, for when no dataset is on hand.
fn synthetic_truth(resolution: usize, half_extent: f64, sh_degree: u32) -> VoxelGrid {
    let mut grid = VoxelGrid::uniform(resolution, half_extent, sh_degree, 0.0, [0.0; 3]);
    let coefficients = grid.coefficients();
    for voxel in 0..grid.voxels() {
        let c = grid.voxel_center(voxel);
        let sphere = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let blob = ((c[0] - 0.5).powi(2) + (c[1] + 0.3).powi(2) + c[2].powi(2)).sqrt();
        if sphere >= 0.6 && blob >= 0.35 {
            continue;
        }
        grid.density_mut()[voxel] = 4.0;
        let color = [0.5 + 0.5 * c[0], 0.5 + 0.5 * c[1], 0.5 - 0.4 * c[2]];
        for channel in 0..3 {
            grid.color_mut()[(voxel * 3 + channel) * coefficients] =
                rlx_fgd::rep::harmonics::constant_coefficient(color[channel]);
        }
    }
    grid
}

fn write_ppm(path: &str, image: &Image) {
    let mut out = format!("P3\n{} {}\n255\n", image.width(), image.height());
    for value in image.pixels() {
        out.push_str(&format!(
            "{} ",
            (value.clamp(0.0, 1.0) * 255.0).round() as u8
        ));
    }
    std::fs::write(path, out).expect("could not write a render");
}

fn main() -> anyhow::Result<()> {
    let steps: usize = arg("--steps", 60);
    let learning_rate: f64 = arg("--lr", 1.0);
    let tolerance: f64 = arg("--eps", 0.4);
    let initial_resolution: usize = arg("--init-res", 8);
    let max_level: u32 = arg("--max-level", 2);
    let sh_degree: u32 = arg("--sh-deg", 1);
    let half_extent: f64 = arg("--radius", 1.3);
    let samples_per_voxel: usize = arg("--samples-per-voxel", 8);
    let rays_per_pixel: usize = arg("--rays-per-pixel", 4);
    let resolution: usize = arg("--res", 32);
    let views: usize = arg("--views", 12);
    let with_fixed = present("--fixed");

    let render = RenderOptions {
        t_near: 2.0,
        t_far: 6.0,
        samples_per_pixel: rays_per_pixel,
        seed: 0,
    };

    // The held-out views: either the dataset's test split, or fresh viewpoints
    // on the same sphere rendered from the truth. Either way they are not
    // trained on, and `reference` is what the reconstruction is scored against.
    let (train, novel, reference) = match flag("--scene") {
        Some(directory) => {
            let path = std::path::Path::new(&directory);
            let options = LoadOptions {
                downsample: arg("--downsample", 5),
                limit: Some(views),
                skip: 1,
            };
            let train = load_nerf_synthetic(path, Split::Train, [1.0; 3], &options)?;
            let test = load_nerf_synthetic(
                path,
                Split::Test,
                [1.0; 3],
                &LoadOptions {
                    limit: None,
                    skip: 8,
                    ..options
                },
            )?;
            println!(
                "{}: {} train / {} test views at {}×{}",
                path.display(),
                train.len(),
                test.len(),
                train.views[0].image.height(),
                train.views[0].image.width()
            );
            let cameras: Vec<_> = test.views.iter().map(|v| v.camera.clone()).collect();
            let images: Vec<Image> = test.views.into_iter().map(|v| v.image).collect();
            (train, cameras, images)
        }
        None => {
            let focal = 0.5 * resolution as f64 / (0.5 * 0.6911f64).tan();
            let truth = synthetic_truth(4 * resolution.max(8), half_extent, sh_degree);
            let train = Scene::synthesize(
                &truth,
                Scene::cameras_on_sphere(views, 4.0, focal, 0.8),
                resolution,
                resolution,
                &render,
                [1.0; 3],
            );
            let novel = Scene::cameras_on_sphere(5, 4.0, focal, 0.4);
            println!(
                "synthetic scene: {views} train views at {resolution}×{resolution}, 5 novel views"
            );
            let reference: Vec<Image> = novel
                .iter()
                .map(|camera| {
                    render_image(&truth, camera, resolution, resolution, &render, [1.0; 3])
                })
                .collect();
            (train, novel, reference)
        }
    };

    let problem = RadianceField::new(train)
        .with_samples_per_voxel(samples_per_voxel)
        .with_render(render)
        .with_occupancy(if present("--no-hull") {
            Occupancy::All
        } else {
            Occupancy::VisualHull { tolerance: 1e-6 }
        });

    let (height, width) = (
        problem.scene.views[0].image.height(),
        problem.scene.views[0].image.width(),
    );
    // Standard novel-view PSNR: render each held-out camera and score it
    // against what that camera actually recorded.
    let novel_psnr = |grid: &VoxelGrid| -> Option<f64> {
        Some(
            novel
                .iter()
                .zip(&reference)
                .map(|(camera, truth)| {
                    render_image(grid, camera, height, width, &render, [1.0; 3]).psnr(truth)
                })
                .sum::<f64>()
                / novel.len() as f64,
        )
    };

    println!(
        "\n{:<24} {:>10} {:>12} {:>11} {:>13} {:>8}",
        "method", "dofs", "train loss", "novel PSNR", "work", "time"
    );
    println!("{}", "-".repeat(84));
    let row = |name: String, report: &Report<VoxelGrid>, psnr: Option<f64>, secs: f64| {
        println!(
            "{name:<24} {:>10} {:>12.4e} {:>11} {:>13} {:>7.1}s",
            report.history.last().map_or(0, |s| s.dofs),
            report.final_loss().unwrap_or(f64::NAN),
            psnr.map_or("—".to_string(), |v| format!("{v:.2} dB")),
            report.total_work(),
            secs
        );
    };

    if with_fixed {
        for level in 0..=max_level {
            let resolution = initial_resolution << level;
            let started = Instant::now();
            let report = run(
                &problem,
                problem.initial_grid(resolution, half_extent, sh_degree),
                FgdConfig {
                    learning_rate,
                    tolerance,
                    steps,
                    refinement: Fixed,
                    seed: 0,
                },
            );
            let psnr = novel_psnr(&report.solution);
            row(
                format!("fixed {resolution}³"),
                &report,
                psnr,
                started.elapsed().as_secs_f64(),
            );
        }
    }

    let started = Instant::now();
    let report = run(
        &problem,
        problem.initial_grid(initial_resolution, half_extent, sh_degree),
        FgdConfig {
            learning_rate,
            tolerance,
            steps,
            refinement: Refinement::Adaptive { max_level },
            seed: 0,
        },
    );
    let psnr = novel_psnr(&report.solution);
    row(
        "adaptive (ours)".to_string(),
        &report,
        psnr,
        started.elapsed().as_secs_f64(),
    );

    println!("\nrefinement schedule:");
    let mut last = u32::MAX;
    for record in &report.history {
        if record.level != last {
            let voxels = initial_resolution << record.level;
            println!(
                "  step {:>4}: {voxels}³ voxels, ‖g‖ = {:.3e}, U = {:.3e}, L = {:.4e}",
                record.step,
                record.gradient_norm,
                record.error_bound,
                record.loss.unwrap_or(f64::NAN),
            );
            last = record.level;
        }
    }
    println!(
        "\nworst RelErr {:.3}, Theorem 3.10 precondition held: {}",
        report.worst_relative_error(),
        report.guarantees_hold()
    );

    if let Some(out) = flag("--out") {
        std::fs::create_dir_all(&out)?;
        for (index, camera) in novel.iter().enumerate() {
            let rendered = render_image(&report.solution, camera, height, width, &render, [1.0; 3]);
            write_ppm(&format!("{out}/novel{index:02}.ppm"), &rendered);
            write_ppm(&format!("{out}/truth{index:02}.ppm"), &reference[index]);
        }
        println!("wrote {} novel views to {out}", novel.len());
    }
    Ok(())
}
