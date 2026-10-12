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

//! Learning radiance fields — Section 4.3 and Appendix A.1.4.
//!
//! Reconstruct a scene from posed images by minimising the photometric loss
//!
//! ```text
//! L_CV(σ, c) = (1/N) Σ_i (1/P) Σ_j ½‖R̃_{σ,c}(p_ij) − Y_ij‖²
//! ```
//!
//! over `σ ∈ L²(Ω)` and `c ∈ L²(Ω → [H^S]³)`. Highly nonconvex, and yet it has
//! functional gradients in closed form (Proposition A.3): for a point `x`, each
//! training camera contributes the error of the pixel that saw `x`, weighted by
//! how much `x` could have changed that pixel.
//!
//! ```text
//! ∇_σ L(x)   = (1/N) Σ_i a_i · D̃_i · ( T_i(x)·c(x, ω_i) − (R_i − R_i^{≤x}) )
//! ∇_{c_k}L(x,ω) = (1/N) Σ_i a_i · [D̃_i]_k · T_i(x) · σ(x) · K(ω_i, ω)
//! a_i = f_i² · 1[t_i(x) ∈ [t_near, t_far]] / (P · t_i(x)² · |det A_i|)
//! ```
//!
//! The two terms of `∇_σ` are the two ways a bit of density at `x` changes what
//! the camera sees: it adds its own colour, and it occludes everything behind
//! it. `a_i` is the Jacobian of the change of variables from the camera's
//! `(t, u, v)` to world space — density far from a camera covers more pixels
//! per unit volume, and this is what accounts for it.
//!
//! # The colour gradient is exactly representable
//!
//! `∇_{c_k}L(x, ·)` is a finite sum of kernels `K(ω_i, ·)`, and `K` is the
//! reproducing kernel of the span of the harmonics up to degree `d`. Expanding
//! `K(ω_i, ω) = Σ_j Y_j(ω_i) Y_j(ω)` gives the coefficients directly:
//!
//! ```text
//! [∇_{c_k}L(x, ·)]_j = (1/N) Σ_i a_i [D̃_i]_k T_i(x) σ(x) Y_j(ω_i).
//! ```
//!
//! So the representation can hold the colour gradient *exactly* in the
//! direction variable, and the only error left in `c` is spatial — the same
//! kind the density has. The authors instead fit the harmonics by least
//! squares over a sampled set of directions, which recovers the same
//! coefficients when that system has full rank and silently loses components
//! when it does not (fewer directions than harmonics). Computing the
//! coefficients directly removes the sampling, the pseudo-inverse, and that
//! failure mode at once.
//!
//! With that, `σ` and every colour coefficient are fitted the same way: the
//! voxel's value is the sample mean over the voxel, and the error is the
//! sample variance — [`CellFit`], exactly as in [`crate::tasks::fit`].

use crate::algorithm::{Functional, GradientApprox, Rng};
use crate::render::march;
use crate::rep::harmonics::{self, coefficient_count};
use crate::rep::{CellFit, VoxelGrid};
use crate::scene::{Image, RenderOptions, Scene, render_image};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// The largest colour-coefficient block a voxel can hold, `3 × 25`.
const MAX_COLOR_COEFFICIENTS: usize = 3 * 25;

/// Which voxels the optimisation is allowed to touch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Occupancy {
    /// Every voxel.
    All,
    /// The visual hull: a voxel is dropped as soon as *any* training view sees
    /// background where it projects, or does not see it at all.
    ///
    /// This is the authors' `ForegroundPreconditioner` with a zero background
    /// multiplier. Formally it is a change of measure on `L²(Ω)` — a diagonal
    /// preconditioner — and it is what keeps the empty space around the object
    /// from soaking up gradient and, after refinement, memory.
    VisualHull {
        /// How close to the background colour counts as background. The
        /// authors test exact equality against white.
        tolerance: f64,
    },
}

/// The photometric loss over voxel grids.
pub struct RadianceField {
    /// Posed training images.
    pub scene: Scene,
    /// Near plane.
    pub t_near: f64,
    /// Far plane.
    pub t_far: f64,
    /// Monte-Carlo samples per voxel.
    pub samples_per_voxel: usize,
    /// How training images are re-rendered each step.
    pub render: RenderOptions,
    /// Which voxels may be updated.
    pub occupancy: Occupancy,
    /// Renders of the current scene, keyed by the grid that produced them.
    ///
    /// Algorithm 1 asks for the loss at `f_{t+1}` and then, next iteration,
    /// re-renders the same `f_{t+1}` to build the step state. Rendering every
    /// training view is the second most expensive thing here, so the exact
    /// grid is kept alongside its renders and compared; a hash would be
    /// cheaper to store and wrong on a collision.
    cache: Mutex<Option<(VoxelGrid, Arc<Vec<Image>>)>>,
}

impl RadianceField {
    /// A reconstruction problem over `scene`, with NeRF's `[2, 6]` depth range.
    pub fn new(scene: Scene) -> Self {
        Self {
            scene,
            t_near: 2.0,
            t_far: 6.0,
            samples_per_voxel: 8,
            render: RenderOptions::default(),
            occupancy: Occupancy::VisualHull { tolerance: 1e-6 },
            cache: Mutex::new(None),
        }
    }

    /// Set the depth range rays are integrated over.
    pub fn with_depth_range(mut self, t_near: f64, t_far: f64) -> Self {
        assert!(t_far > t_near, "the depth range must be non-degenerate");
        self.t_near = t_near;
        self.t_far = t_far;
        self.render.t_near = t_near;
        self.render.t_far = t_far;
        self
    }

    /// Set the Monte-Carlo samples drawn per voxel.
    pub fn with_samples_per_voxel(mut self, samples: usize) -> Self {
        self.samples_per_voxel = samples;
        self
    }

    /// Set how training views are re-rendered each step.
    ///
    /// The depth range is taken from [`RadianceField::with_depth_range`], not
    /// from here — rendering with one near plane and differentiating against
    /// another is a mismatch nothing would report.
    pub fn with_render(mut self, render: RenderOptions) -> Self {
        self.render = RenderOptions {
            t_near: self.t_near,
            t_far: self.t_far,
            ..render
        };
        self
    }

    /// Set which voxels the optimisation may touch.
    pub fn with_occupancy(mut self, occupancy: Occupancy) -> Self {
        self.occupancy = occupancy;
        self
    }

    /// An initial grid: empty space, mid-grey, at `resolution³` voxels.
    ///
    /// Density starts at zero so nothing is asserted about the scene before
    /// any image has been looked at. The step sizes are Section 4.3's, `20`
    /// for density against `8` for colour, and the density is projected into
    /// `[0, 10]`.
    pub fn initial_grid(&self, resolution: usize, half_extent: f64, sh_degree: u32) -> VoxelGrid {
        let mut grid = VoxelGrid::uniform(resolution, half_extent, sh_degree, 0.0, [0.5; 3]);
        grid.density_rate = 20.0;
        grid.color_rate = 8.0;
        grid.density_clamp = Some((0.0, 10.0));
        grid
    }

    /// Render every training view of `f`.
    pub fn render_views(&self, f: &VoxelGrid) -> Vec<Image> {
        self.scene
            .views
            .par_iter()
            .map(|view| {
                render_image(
                    f,
                    &view.camera,
                    view.image.height(),
                    view.image.width(),
                    &self.render,
                    self.scene.background,
                )
            })
            .collect()
    }

    /// [`RadianceField::render_views`], reusing the last result when `f` is
    /// unchanged.
    pub fn cached_views(&self, f: &VoxelGrid) -> Arc<Vec<Image>> {
        let cache = self.cache.lock().expect("render cache poisoned");
        if let Some((grid, images)) = cache.as_ref()
            && grid == f
        {
            return Arc::clone(images);
        }
        drop(cache);
        let images = Arc::new(self.render_views(f));
        let mut cache = self.cache.lock().expect("render cache poisoned");
        *cache = Some((f.clone(), Arc::clone(&images)));
        images
    }

    /// `L_CV` given renders of the training views.
    pub fn photometric_loss(&self, images: &[Image]) -> f64 {
        let mut total = 0.0;
        for (view, rendered) in self.scene.views.iter().zip(images) {
            let truth = view.image.pixels();
            let pixels = rendered.pixels();
            let mut view_loss = 0.0;
            for (a, b) in pixels.iter().zip(truth) {
                view_loss += 0.5 * (a - b) * (a - b);
            }
            // The 1/P of Equation 11 counts pixels, not channels.
            total += view_loss / (rendered.height() * rendered.width()) as f64;
        }
        total / self.scene.len() as f64
    }

    /// Whether the optimisation may put matter at `x`.
    pub fn is_occupied(&self, x: [f64; 3]) -> bool {
        let tolerance = match self.occupancy {
            Occupancy::All => return true,
            Occupancy::VisualHull { tolerance } => tolerance,
        };
        for view in &self.scene.views {
            let (uv, t) = view.camera.project(x);
            // Behind the camera: the projection formula divides by `t` and
            // would otherwise place the point in front of it.
            if t.is_nan() || t <= 0.0 {
                return false;
            }
            let pixel = view
                .camera
                .uv_to_pixel(uv, view.image.height(), view.image.width());
            let (row, column) = (pixel[0].round() as isize, pixel[1].round() as isize);
            if !view.image.contains(row, column) {
                return false;
            }
            let observed = view.image.at(row, column);
            let is_background =
                (0..3).all(|c| (observed[c] - self.scene.background[c]).abs() <= tolerance);
            if is_background {
                return false;
            }
        }
        true
    }

    /// Zero the density of every voxel outside the visual hull.
    pub fn apply_occupancy(&self, f: &mut VoxelGrid) {
        if matches!(self.occupancy, Occupancy::All) {
            return;
        }
        let keep: Vec<bool> = (0..f.voxels())
            .into_par_iter()
            .map(|voxel| self.is_occupied(f.voxel_center(voxel)))
            .collect();
        f.mask_density(|voxel| keep[voxel]);
    }

    /// Proposition A.3 at one point: the density gradient, and the colour
    /// gradient already expressed in spherical-harmonic coefficients.
    ///
    /// `color` is written as `[channel][coefficient]`.
    fn functional_gradient(
        &self,
        f: &VoxelGrid,
        images: &[Image],
        x: [f64; 3],
        color: &mut [f64],
    ) -> f64 {
        let coefficients = coefficient_count(f.sh_degree());
        color[..3 * coefficients].fill(0.0);
        let mut density = 0.0;

        let density_at_x = f.eval_density(x);
        let mut basis = [0.0f64; 25];

        for (view, rendered) in self.scene.views.iter().zip(images) {
            let (height, width) = (view.image.height(), view.image.width());
            let (uv, t) = view.camera.project(x);
            // Behind the camera, or outside the depth range Equation 11
            // covers — the `1[t(x;i) ∈ F]` of Proposition A.3's `a_i`.
            if t.is_nan() || t <= 0.0 || t < self.t_near || t > self.t_far {
                continue;
            }
            let pixel = view.camera.uv_to_pixel(uv, height, width);
            let (row, column) = (pixel[0].round() as isize, pixel[1].round() as isize);
            if !view.image.contains(row, column) {
                continue;
            }

            // The antialiasing kernel is a box over the pixel, whose support is
            // the single pixel the point projects into; the weight there is 1.
            let observed = view.image.at(row, column);
            let predicted = rendered.at(row, column);
            let difference = [
                predicted[0] - observed[0],
                predicted[1] - observed[1],
                predicted[2] - observed[2],
            ];

            let ray = view.camera.ray_through(uv);
            let walk = march(f, &ray, self.t_near, self.t_far, Some(t));
            let transmittance = walk.transmittance_at_stop;
            let color_at_x = f.eval_color(x, ray.direction);

            let jacobian = view.camera.focal * view.camera.focal
                / ((height * width) as f64 * view.camera.rotation_determinant() * t * t);

            // ∇_σ: the colour this point adds, less everything it occludes.
            // The background belongs to the full render and not to the partial
            // one, so it stays inside the difference.
            let mut density_term = 0.0;
            for channel in 0..3 {
                let behind = walk.color[channel]
                    + walk.transmittance * self.scene.background[channel]
                    - walk.color_at_stop[channel];
                density_term +=
                    difference[channel] * (transmittance * color_at_x[channel] - behind);
            }
            density += jacobian * density_term;

            // ∇_c: a kernel centred on this view's direction, which is a fixed
            // combination of harmonics — so accumulate the coefficients.
            let weight = jacobian * transmittance * density_at_x;
            if weight != 0.0 {
                harmonics::basis_into(
                    f.sh_degree(),
                    harmonics::normalize(ray.direction),
                    &mut basis[..coefficients],
                );
                for channel in 0..3 {
                    let scale = weight * difference[channel];
                    let row = &mut color[channel * coefficients..(channel + 1) * coefficients];
                    for (slot, b) in row.iter_mut().zip(&basis[..coefficients]) {
                        *slot += scale * b;
                    }
                }
            }
        }

        let inverse_views = 1.0 / self.scene.len() as f64;
        for slot in color[..3 * coefficients].iter_mut() {
            *slot *= inverse_views;
        }
        density * inverse_views
    }
}

/// Renders of the training views at the current iterate.
///
/// This is the per-step work Algorithm 1's inner loop must not repeat:
/// refining the grid does not change the scene it encodes, so the renders and
/// the loss are the same for every refinement attempt within a step.
pub struct RenderedScene {
    /// One render per training view.
    pub images: Arc<Vec<Image>>,
    /// `L_CV` at the iterate these came from.
    pub loss: f64,
}

impl Functional for RadianceField {
    type Rep = VoxelGrid;
    type StepState = RenderedScene;

    fn begin_step(&self, f: &Self::Rep) -> Self::StepState {
        let images = self.cached_views(f);
        let loss = self.photometric_loss(&images);
        RenderedScene { images, loss }
    }

    fn approximate_gradient(
        &self,
        f: &Self::Rep,
        state: &Self::StepState,
        seed: u64,
    ) -> GradientApprox<Self::Rep> {
        let samples = self.samples_per_voxel.max(2);
        let coefficients = coefficient_count(f.sh_degree());
        let size = f.voxel_size();

        // One set of offsets shared by every voxel and by every refinement
        // attempt within the step, so refinement competes against a fixed draw
        // rather than against resampling noise.
        let mut rng = Rng::new(seed);
        let offsets: Vec<[f64; 3]> = (0..samples)
            .map(|_| {
                [
                    rng.uniform01() * size,
                    rng.uniform01() * size,
                    rng.uniform01() * size,
                ]
            })
            .collect();

        let images = &state.images;
        let stride = 3 * coefficients;
        let mut gradient = f.clone();
        // The gradient of a step is never clamped or masked — only the iterate is.
        gradient.density_clamp = None;

        // Written in place rather than collected: at the paper's 128³ a
        // per-voxel result struct wide enough for degree-4 harmonics would be
        // 1.3 GB of buffer for values that are about to be copied out anyway.
        let (density_out, color_out) = gradient.parts_mut();
        let error_squared: f64 = density_out
            .par_iter_mut()
            .zip(color_out.par_chunks_mut(stride))
            .enumerate()
            .map(|(voxel, (density_slot, color_slot))| {
                if !self.is_occupied(f.voxel_center(voxel)) {
                    // Outside the visual hull the gradient is defined to be
                    // zero, so it is represented exactly and contributes no
                    // approximation error.
                    *density_slot = 0.0;
                    color_slot.fill(0.0);
                    return 0.0;
                }
                let corner = f.voxel_corner(voxel);
                let mut density_sum = 0.0;
                let mut density_squares = 0.0;
                let mut color_sum = [0.0f64; MAX_COLOR_COEFFICIENTS];
                let mut color_squares = [0.0f64; MAX_COLOR_COEFFICIENTS];
                let mut color = [0.0f64; MAX_COLOR_COEFFICIENTS];

                for offset in &offsets {
                    let x = [
                        corner[0] + offset[0],
                        corner[1] + offset[1],
                        corner[2] + offset[2],
                    ];
                    let density = self.functional_gradient(f, images, x, &mut color);
                    density_sum += density;
                    density_squares += density * density;
                    for slot in 0..stride {
                        color_sum[slot] += color[slot];
                        color_squares[slot] += color[slot] * color[slot];
                    }
                }

                let fit = CellFit::from_moments(density_sum, density_squares, samples);
                *density_slot = fit.mean;
                let mut error = fit.error_density;
                for slot in 0..stride {
                    let cell = CellFit::from_moments(color_sum[slot], color_squares[slot], samples);
                    color_slot[slot] = cell.mean;
                    error += cell.error_density;
                }
                error
            })
            .sum::<f64>()
            * f.voxel_volume();

        GradientApprox {
            norm: gradient.norm(),
            gradient,
            error_bound: error_squared.sqrt(),
        }
    }

    /// Re-impose the visual hull at the new resolution.
    ///
    /// A voxel whose centre is inside the hull can have children whose centres
    /// are not, and those children inherit its density. The gradient is zero
    /// outside the hull — that is what the preconditioner means — so nothing
    /// would ever take that density away again. Left in, it shows up as a
    /// blocky shell around the object that the loss plateaus on: measured on
    /// the synthetic scene of `tests/radiance.rs`, the adaptive run stalls
    /// thirty times above where the same grid reaches when the hull is
    /// re-applied.
    fn after_refine(&self, f: &mut Self::Rep) {
        self.apply_occupancy(f);
    }

    fn loss(&self, f: &Self::Rep) -> Option<f64> {
        Some(self.photometric_loss(&self.cached_views(f)))
    }
}
