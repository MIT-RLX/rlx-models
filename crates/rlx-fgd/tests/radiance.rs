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

//! Section 4.3: the renderer, the camera pair, Proposition A.3's gradient, and
//! a reconstruction from posed images.

use rlx_fgd::algorithm::{FgdConfig, Functional, Refinement, Representation, Rng, run};
use rlx_fgd::render::{Ray, march, render_ray};
use rlx_fgd::rep::VoxelGrid;
use rlx_fgd::rep::harmonics::{self, coefficient_count};
use rlx_fgd::scene::{Camera, RenderOptions, Scene, render_image};
use rlx_fgd::tasks::radiance::{Occupancy, RadianceField};

const HALF: f64 = 1.3;
const DEGREE: u32 = 1;

/// A two-blob object with position-dependent colour.
fn truth(resolution: usize) -> VoxelGrid {
    let mut grid = VoxelGrid::uniform(resolution, HALF, DEGREE, 0.0, [0.0; 3]);
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
                harmonics::constant_coefficient(color[channel]);
        }
    }
    grid
}

fn focal_for(width: usize) -> f64 {
    // NeRF-Synthetic's field of view.
    0.5 * width as f64 / (0.5 * 0.6911f64).tan()
}

// ── Spherical harmonics ─────────────────────────────────────────────────────

/// The colour norm of a voxel is the plain sum of its squared coefficients,
/// which is only true because the basis is orthonormal on the sphere. If it
/// were not, `‖g‖` would be measured in one metric and the error in another,
/// and the ratio Algorithm 1 tests would mean nothing.
#[test]
fn spherical_harmonics_are_orthonormal_on_the_sphere() {
    let degree = 3u32;
    let n = coefficient_count(degree);
    let samples = 120_000;
    let mut rng = Rng::new(4);
    let mut gram = vec![0.0f64; n * n];
    for _ in 0..samples {
        // Uniform on S²: z uniform, azimuth uniform.
        let z = 2.0 * rng.uniform01() - 1.0;
        let phi = std::f64::consts::TAU * rng.uniform01();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let basis = harmonics::basis(degree, [r * phi.cos(), r * phi.sin(), z]);
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] += basis[i] * basis[j];
            }
        }
    }
    let area = 4.0 * std::f64::consts::PI;
    for i in 0..n {
        for j in 0..n {
            let entry = gram[i * n + j] * area / samples as f64;
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (entry - expected).abs() < 0.03,
                "⟨Y_{i}, Y_{j}⟩ = {entry:.4}, expected {expected}"
            );
        }
    }
}

/// The kernel of the degree-`d` RKHS is `Σ_l (2l+1)/(4π) P_l(ω·ω')` by the
/// addition theorem — an identity the basis has to satisfy and a check on its
/// normalisation that no amount of self-consistency would catch.
#[test]
fn the_rkhs_kernel_matches_the_addition_theorem() {
    let legendre = |l: u32, x: f64| -> f64 {
        match l {
            0 => 1.0,
            1 => x,
            2 => 0.5 * (3.0 * x * x - 1.0),
            3 => 0.5 * (5.0 * x * x * x - 3.0 * x),
            _ => unreachable!(),
        }
    };
    let mut rng = Rng::new(9);
    for degree in 0..=3u32 {
        for _ in 0..200 {
            let a = harmonics::normalize([
                rng.uniform01() - 0.5,
                rng.uniform01() - 0.5,
                rng.uniform01() - 0.5,
            ]);
            let b = harmonics::normalize([
                rng.uniform01() - 0.5,
                rng.uniform01() - 0.5,
                rng.uniform01() - 0.5,
            ]);
            let dot = (0..3).map(|i| a[i] * b[i]).sum::<f64>();
            let expected: f64 = (0..=degree)
                .map(|l| (2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) * legendre(l, dot))
                .sum();
            let got = harmonics::kernel(degree, a, b);
            assert!(
                (got - expected).abs() < 1e-12,
                "degree {degree}: K = {got}, addition theorem says {expected}"
            );
        }
    }
}

// ── Renderer ────────────────────────────────────────────────────────────────

/// A uniform slab has a transmittance you can write down: `exp(−σ·L)` over the
/// path length inside the box. The DDA has to reproduce it however many voxels
/// it chops the path into, or the "exact integration" claim is empty.
#[test]
fn transmittance_through_a_uniform_medium_is_exact() {
    for resolution in [1usize, 2, 7, 16] {
        let grid = VoxelGrid::uniform(resolution, 1.0, 0, 0.75, [0.0; 3]);
        // Straight down the x axis, so the path length inside is exactly 2.
        let ray = Ray {
            origin: [-5.0, 0.13, -0.27],
            direction: [1.0, 0.0, 0.0],
        };
        let walk = march(&grid, &ray, 0.0, 20.0, None);
        let expected = (-0.75 * 2.0f64).exp();
        assert!(
            (walk.transmittance - expected).abs() < 1e-12,
            "resolution {resolution}: T = {}, expected {expected}",
            walk.transmittance
        );
    }
}

/// The rendering equation over a uniform, uniformly coloured medium collapses
/// to `(1 − e^{−σL})·c + e^{−σL}·b`, whatever the grid.
#[test]
fn a_uniform_medium_renders_to_its_closed_form() {
    let grid = VoxelGrid::uniform(9, 1.0, 1, 0.9, [0.3, 0.6, 0.2]);
    let ray = Ray {
        origin: [-4.0, 0.0, 0.05],
        direction: [1.0, 0.0, 0.0],
    };
    let background = [1.0, 0.25, 0.5];
    let rendered = render_ray(&grid, &ray, 0.0, 20.0, background);
    let transmittance = (-0.9 * 2.0f64).exp();
    for channel in 0..3 {
        let expected =
            (1.0 - transmittance) * [0.3, 0.6, 0.2][channel] + transmittance * background[channel];
        assert!(
            (rendered[channel] - expected).abs() < 1e-12,
            "channel {channel}: {} vs {expected}",
            rendered[channel]
        );
    }
}

/// Stopping mid-march has to give exactly what a separate march to that point
/// would. Proposition A.3's density gradient subtracts one from the other, so
/// any drift between them lands directly in the gradient.
#[test]
fn the_recorded_stop_matches_a_separate_march() {
    let mut rng = Rng::new(21);
    let mut grid = VoxelGrid::uniform(8, HALF, 1, 0.0, [0.0; 3]);
    for value in grid.density_mut() {
        *value = 2.0 * rng.uniform01();
    }
    for value in grid.color_mut() {
        *value = rng.uniform01() - 0.5;
    }
    let ray = Ray {
        origin: [-3.1, 0.4, 1.2],
        direction: [1.0, -0.13, -0.37],
    };
    for stop in [0.5, 1.9, 2.4, 3.0, 3.61, 4.2, 5.5, 9.0] {
        let combined = march(&grid, &ray, 0.5, 6.0, Some(stop));
        let separate = march(&grid, &ray, 0.5, stop.clamp(0.5, 6.0), None);
        // Past the far plane the stop is the full march, which is what
        // Proposition A.3's indicator makes irrelevant anyway.
        let expected = if stop > 6.0 {
            (combined.color, combined.transmittance)
        } else {
            (separate.color, separate.transmittance)
        };
        for channel in 0..3 {
            assert!(
                (combined.color_at_stop[channel] - expected.0[channel]).abs() < 1e-12,
                "stop {stop}: colour {} vs {}",
                combined.color_at_stop[channel],
                expected.0[channel]
            );
        }
        assert!(
            (combined.transmittance_at_stop - expected.1).abs() < 1e-12,
            "stop {stop}: transmittance {} vs {}",
            combined.transmittance_at_stop,
            expected.1
        );
    }
}

// ── Cameras ─────────────────────────────────────────────────────────────────

/// `project` has to be the exact inverse of `ray_through`. A mismatch would
/// not fail anywhere — it would quietly charge each point's error to the wrong
/// pixel, and the reconstruction would just come out wrong.
#[test]
fn projection_inverts_ray_generation() {
    let mut rng = Rng::new(77);
    for camera in Scene::cameras_on_sphere(7, 4.0, focal_for(64), 0.85) {
        for _ in 0..500 {
            let uv = [
                60.0 * (rng.uniform01() - 0.5),
                60.0 * (rng.uniform01() - 0.5),
            ];
            let t = 2.0 + 4.0 * rng.uniform01();
            let x = camera.ray_through(uv).at(t);
            let (recovered_uv, recovered_t) = camera.project(x);
            assert!((recovered_t - t).abs() < 1e-9, "depth {recovered_t} vs {t}");
            assert!(
                (recovered_uv[0] - uv[0]).abs() < 1e-9 && (recovered_uv[1] - uv[1]).abs() < 1e-9,
                "uv {recovered_uv:?} vs {uv:?}"
            );
        }
    }
}

#[test]
fn pixel_and_image_plane_coordinates_round_trip() {
    let camera = Camera::look_at([0.0, 0.0, 4.0], [0.0; 3], [0.0, 1.0, 0.0], focal_for(40));
    for row in 0..40 {
        for column in 0..24 {
            let uv = camera.pixel_to_uv([row as f64, column as f64], 40, 24);
            let back = camera.uv_to_pixel(uv, 40, 24);
            assert!((back[0] - row as f64).abs() < 1e-12);
            assert!((back[1] - column as f64).abs() < 1e-12);
        }
    }
}

// ── Proposition A.3 ─────────────────────────────────────────────────────────

/// The decisive check on the gradient: compare it against finite differences
/// of the loss it claims to be the gradient of.
///
/// Both sides are Monte-Carlo — the render averages jittered rays, the gradient
/// averages points in the voxel — so the comparison is a convergence claim, not
/// a single number. At the coarse setting the two disagree by tens of percent
/// and it says nothing; the content is that refining both drives the gap down.
#[test]
fn the_functional_gradient_matches_finite_differences() {
    let (height, width) = (8usize, 8usize);
    let cameras = Scene::cameras_on_sphere(2, 4.0, focal_for(width), 0.5);
    let reference = truth(8);

    let mut f = VoxelGrid::uniform(4, HALF, DEGREE, 0.6, [0.4, 0.5, 0.6]);
    f.density_clamp = None;
    let stride = 3 * f.coefficients();
    let volume = f.voxel_volume();
    let probes = [5usize, 21, 37, 42];
    let delta = 1e-4;

    let mut previous = f64::INFINITY;
    for &(rays, samples) in &[(256usize, 4096usize), (2048, 32768)] {
        let render = RenderOptions {
            t_near: 2.0,
            t_far: 6.0,
            samples_per_pixel: rays,
            seed: 3,
        };
        let scene = Scene::synthesize(
            &reference,
            cameras.clone(),
            height,
            width,
            &render,
            [1.0; 3],
        );
        // The visual hull would zero the gradient outside the object, and the
        // point here is the gradient itself.
        let problem = RadianceField::new(scene)
            .with_render(render)
            .with_occupancy(Occupancy::All)
            .with_samples_per_voxel(samples);

        let state = problem.begin_step(&f);
        let gradient = problem.approximate_gradient(&f, &state, 11);

        let mut worst = 0.0f64;
        for &voxel in &probes {
            for (label, slot) in [("density", usize::MAX), ("colour", voxel * stride + 1)] {
                let mut up = f.clone();
                let mut down = f.clone();
                let analytic = if slot == usize::MAX {
                    up.density_mut()[voxel] += delta;
                    down.density_mut()[voxel] -= delta;
                    volume * gradient.gradient.density()[voxel]
                } else {
                    up.color_mut()[slot] += delta;
                    down.color_mut()[slot] -= delta;
                    volume * gradient.gradient.color()[slot]
                };
                let numeric = (problem.photometric_loss(&problem.render_views(&up))
                    - problem.photometric_loss(&problem.render_views(&down)))
                    / (2.0 * delta);
                assert!(
                    numeric.abs() > 1e-9,
                    "voxel {voxel} {label}: the probe barely moves the loss"
                );
                let relative = (numeric - analytic).abs() / numeric.abs();
                worst = worst.max(relative);
            }
        }
        assert!(
            worst < previous,
            "refining the sampling did not tighten the agreement: {previous:e} → {worst:e}"
        );
        previous = worst;
    }
    assert!(
        previous < 0.05,
        "at the finest sampling the gradient still disagreed by {previous:e}"
    );
}

// ── Representation ──────────────────────────────────────────────────────────

#[test]
fn voxel_refinement_preserves_the_encoded_scene() {
    let mut rng = Rng::new(5);
    let mut grid = VoxelGrid::uniform(5, HALF, 2, 0.0, [0.0; 3]);
    for value in grid.density_mut() {
        *value = rng.uniform01();
    }
    for value in grid.color_mut() {
        *value = rng.uniform01() - 0.5;
    }
    let fine = grid.refine();
    assert_eq!(fine.resolution(), 10);
    assert_eq!(fine.dofs(), 8 * grid.dofs());
    for _ in 0..3000 {
        let x = [
            2.0 * HALF * (rng.uniform01() - 0.5),
            2.0 * HALF * (rng.uniform01() - 0.5),
            2.0 * HALF * (rng.uniform01() - 0.5),
        ];
        let direction = harmonics::normalize([
            rng.uniform01() - 0.5,
            rng.uniform01() - 0.5,
            rng.uniform01() - 0.5,
        ]);
        assert_eq!(grid.eval_density(x), fine.eval_density(x));
        assert_eq!(grid.eval_color(x, direction), fine.eval_color(x, direction));
    }
    // The norms are measure-weighted, so they survive too.
    assert!(
        (grid.density_norm_squared() - fine.density_norm_squared()).abs()
            < 1e-12 * grid.density_norm_squared()
    );
    assert!(
        (grid.color_norm_squared() - fine.color_norm_squared()).abs()
            < 1e-12 * grid.color_norm_squared()
    );
}

#[test]
fn separate_step_sizes_and_the_density_clamp_are_applied() {
    let mut grid = VoxelGrid::uniform(2, 1.0, 0, 1.0, [1.0; 3]);
    grid.density_rate = 20.0;
    grid.color_rate = 8.0;
    grid.density_clamp = Some((0.0, 10.0));
    let mut gradient = grid.clone();
    gradient.density_mut().fill(0.01);
    gradient.color_mut().fill(0.1);
    let before = grid.color()[0];
    grid.descend_by(1.0, &gradient);
    assert!((grid.density()[0] - (1.0 - 20.0 * 0.01)).abs() < 1e-12);
    assert!((grid.color()[0] - (before - 8.0 * 0.1)).abs() < 1e-12);

    // A step big enough to drive the density negative is projected back.
    gradient.density_mut().fill(10.0);
    grid.descend_by(1.0, &gradient);
    assert!(grid.density().iter().all(|d| *d >= 0.0));
}

// ── Reconstruction ──────────────────────────────────────────────────────────

fn reconstruction_problem(views: usize, pixels: usize, occupancy: Occupancy) -> RadianceField {
    let render = RenderOptions {
        t_near: 2.0,
        t_far: 6.0,
        samples_per_pixel: 4,
        seed: 0,
    };
    let scene = Scene::synthesize(
        &truth(32),
        Scene::cameras_on_sphere(views, 4.0, focal_for(pixels), 0.8),
        pixels,
        pixels,
        &render,
        [1.0; 3],
    );
    RadianceField::new(scene)
        .with_render(render)
        .with_samples_per_voxel(8)
        .with_occupancy(occupancy)
}

/// The visual hull is a hard constraint the data implies: a voxel that any
/// camera sees as background provably holds nothing.
#[test]
fn the_visual_hull_carves_the_empty_space_away() {
    let problem = reconstruction_problem(12, 32, Occupancy::VisualHull { tolerance: 1e-6 });
    let grid = truth(32);
    let kept = (0..grid.voxels())
        .filter(|&voxel| problem.is_occupied(grid.voxel_center(voxel)))
        .count();
    assert!(
        kept < grid.voxels() / 4,
        "the hull kept {kept} of {} voxels, which carves nothing",
        grid.voxels()
    );
    // The object's interior has to survive it. Only the interior: a voxel on
    // the silhouette projects into a pixel the object barely covers, which
    // reads as background, so a visual hull shaves the outermost shell at any
    // finite image resolution. That is the hull being conservative, not wrong
    // — but it is why the scene box has room to spare around the object.
    let n = grid.resolution();
    let solid = |i: usize, j: usize, k: usize| grid.density()[(i * n + j) * n + k] > 0.0;
    let mut interior = 0usize;
    let mut carved = 0usize;
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            for k in 1..n - 1 {
                let surrounded = solid(i, j, k)
                    && solid(i - 1, j, k)
                    && solid(i + 1, j, k)
                    && solid(i, j - 1, k)
                    && solid(i, j + 1, k)
                    && solid(i, j, k - 1)
                    && solid(i, j, k + 1);
                if !surrounded {
                    continue;
                }
                interior += 1;
                if !problem.is_occupied(grid.voxel_center((i * n + j) * n + k)) {
                    carved += 1;
                }
            }
        }
    }
    assert!(
        interior > 100,
        "the probe found only {interior} interior voxels"
    );
    assert_eq!(
        carved, 0,
        "the hull carved away {carved} of {interior} interior voxels"
    );

    // And it keeps the bulk of the object, not a sliver of it.
    let occupied = (0..grid.voxels())
        .filter(|&voxel| grid.density()[voxel] > 0.0)
        .count();
    let retained = (0..grid.voxels())
        .filter(|&voxel| {
            grid.density()[voxel] > 0.0 && problem.is_occupied(grid.voxel_center(voxel))
        })
        .count();
    assert!(
        retained as f64 > 0.8 * occupied as f64,
        "the hull kept only {retained} of the object's {occupied} voxels"
    );
}

/// Refinement inherits the parent's density into children that the hull
/// excludes, where the gradient is identically zero — so without
/// `after_refine` that density is permanent. Measured, it costs an order of
/// magnitude in the final loss.
#[test]
fn refinement_reimposes_the_visual_hull() {
    let problem = reconstruction_problem(8, 24, Occupancy::VisualHull { tolerance: 1e-6 });
    let mut grid = problem.initial_grid(8, HALF, DEGREE);
    grid.density_mut().fill(3.0);
    problem.apply_occupancy(&mut grid);
    let before = grid.density().iter().filter(|d| **d > 0.0).count();

    let mut refined = grid.refine();
    let inherited = refined.density().iter().filter(|d| **d > 0.0).count();
    assert_eq!(inherited, 8 * before, "refinement should copy the parents");

    problem.after_refine(&mut refined);
    let kept = refined.density().iter().filter(|d| **d > 0.0).count();
    assert!(
        kept < inherited,
        "re-imposing the hull removed nothing: {kept} of {inherited}"
    );
}

/// End to end: recover a scene from posed images alone, and check the result
/// on viewpoints that were never trained on.
#[test]
fn adaptive_fgd_reconstructs_a_scene_from_posed_images() {
    let problem = reconstruction_problem(12, 32, Occupancy::VisualHull { tolerance: 1e-6 });
    let reference = truth(32);
    let initial = problem.initial_grid(8, HALF, DEGREE);
    let before = problem.loss(&initial).expect("the loss is reported");

    let report = run(
        &problem,
        initial,
        FgdConfig {
            learning_rate: 1.0,
            tolerance: 0.4,
            steps: 40,
            refinement: Refinement::Adaptive { max_level: 2 },
            seed: 0,
        },
    );

    let after = report.final_loss().expect("the loss is reported");
    assert!(
        after < before / 20.0,
        "photometric loss only went {before:e} → {after:e}"
    );
    assert!(report.final_level() > 0, "nothing was ever refined");

    // Novel views: cameras at elevations the training set does not cover.
    let render = problem.render;
    let novel = Scene::cameras_on_sphere(5, 4.0, focal_for(32), 0.35);
    let psnr: f64 = novel
        .iter()
        .map(|camera| {
            let recovered = render_image(&report.solution, camera, 32, 32, &render, [1.0; 3]);
            let expected = render_image(&reference, camera, 32, 32, &render, [1.0; 3]);
            recovered.psnr(&expected)
        })
        .sum::<f64>()
        / novel.len() as f64;
    assert!(
        psnr > 28.0,
        "novel-view PSNR was only {psnr:.2} dB, so the geometry is wrong"
    );
}
