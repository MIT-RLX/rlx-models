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

//! Voxel grids with spherical-harmonic colour — the representation of Section 4.3.
//!
//! The scene is a pair of functions: a density `σ : Ω → R` and a
//! view-dependent colour `c : Ω × S² → R³`. Both are held on the same uniform
//! grid over a cube, with `σ` piecewise constant per voxel and `c` stored as
//! per-voxel spherical-harmonic coefficients (three channels × `(deg+1)²`).
//! Refinement splits every voxel eight ways and copies the parent into each
//! child, so the encoded scene is unchanged — the property Algorithm 1 needs.
//!
//! Two departures from the other representations in this crate, both forced by
//! the problem rather than chosen:
//!
//! - **Density and colour carry separate step sizes.** Section 4.3's reported
//!   run uses `20` for `σ` and `8` for `c`. That is a diagonal preconditioner —
//!   a reweighting of the `L²` measure, as the authors' `preconditioners.py`
//!   puts it — so the iteration is still gradient descent, in a rescaled inner
//!   product. Set both to `1` to recover ordinary FGD.
//! - **Density is clamped after each step.** A negative density has no physical
//!   meaning and makes the transmittance grow along a ray; the authors clamp to
//!   `[0, 10]`. This is a projection the paper's analysis does not cover, so it
//!   is optional and off unless asked for.

use crate::rep::harmonics::{self, coefficient_count};

/// A uniform voxel grid over the cube `[-half_extent, half_extent]³` holding a
/// density and a spherical-harmonic colour per voxel.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelGrid {
    resolution: usize,
    half_extent: f64,
    sh_degree: u32,
    level: u32,
    density: Vec<f64>,
    /// `[voxel][channel][coefficient]`, flattened.
    color: Vec<f64>,
    /// Relative step size for `σ`.
    pub density_rate: f64,
    /// Relative step size for `c`.
    pub color_rate: f64,
    /// Bounds the density is projected back into after each step.
    pub density_clamp: Option<(f64, f64)>,
}

impl VoxelGrid {
    /// A grid of `resolution³` voxels holding a constant density and colour.
    pub fn uniform(
        resolution: usize,
        half_extent: f64,
        sh_degree: u32,
        density: f64,
        color: [f64; 3],
    ) -> Self {
        assert!(resolution > 0, "resolution must be positive");
        assert!(half_extent > 0.0, "the scene box must be non-degenerate");
        let voxels = resolution * resolution * resolution;
        let coefficients = coefficient_count(sh_degree);
        let mut grid = Self {
            resolution,
            half_extent,
            sh_degree,
            level: 0,
            density: vec![density; voxels],
            color: vec![0.0; voxels * 3 * coefficients],
            density_rate: 1.0,
            color_rate: 1.0,
            density_clamp: None,
        };
        // A constant colour is the degree-zero coefficient alone.
        for voxel in 0..voxels {
            for channel in 0..3 {
                grid.color[(voxel * 3 + channel) * coefficients] =
                    harmonics::constant_coefficient(color[channel]);
            }
        }
        grid
    }

    /// Voxels per axis.
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Half the side length of the scene cube.
    pub fn half_extent(&self) -> f64 {
        self.half_extent
    }

    /// Spherical-harmonic degree.
    pub fn sh_degree(&self) -> u32 {
        self.sh_degree
    }

    /// Coefficients per colour channel, `(deg+1)²`.
    pub fn coefficients(&self) -> usize {
        coefficient_count(self.sh_degree)
    }

    /// Number of voxels.
    pub fn voxels(&self) -> usize {
        self.resolution * self.resolution * self.resolution
    }

    /// The side length of one voxel.
    pub fn voxel_size(&self) -> f64 {
        2.0 * self.half_extent / self.resolution as f64
    }

    /// The volume of one voxel.
    pub fn voxel_volume(&self) -> f64 {
        self.voxel_size().powi(3)
    }

    /// Density values, one per voxel.
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// Mutable density values.
    pub fn density_mut(&mut self) -> &mut [f64] {
        &mut self.density
    }

    /// Colour coefficients, `[voxel][channel][coefficient]`.
    pub fn color(&self) -> &[f64] {
        &self.color
    }

    /// Mutable colour coefficients.
    pub fn color_mut(&mut self) -> &mut [f64] {
        &mut self.color
    }

    /// Density and colour together, both mutable.
    ///
    /// Filling a gradient in parallel needs to write both per voxel, and two
    /// separate `_mut` accessors cannot be held at once.
    pub fn parts_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (&mut self.density, &mut self.color)
    }

    /// The slice of colour coefficients for one voxel, all three channels.
    pub fn voxel_color(&self, voxel: usize) -> &[f64] {
        let stride = 3 * self.coefficients();
        &self.color[voxel * stride..(voxel + 1) * stride]
    }

    /// The `(i, j, k)` index of a flat voxel index.
    pub fn voxel_index(&self, voxel: usize) -> [usize; 3] {
        let n = self.resolution;
        [voxel / (n * n), (voxel / n) % n, voxel % n]
    }

    /// The flat index of `(i, j, k)`.
    pub fn flat_index(&self, index: [usize; 3]) -> usize {
        let n = self.resolution;
        (index[0] * n + index[1]) * n + index[2]
    }

    /// The lower corner of a voxel.
    pub fn voxel_corner(&self, voxel: usize) -> [f64; 3] {
        let index = self.voxel_index(voxel);
        let size = self.voxel_size();
        [
            -self.half_extent + index[0] as f64 * size,
            -self.half_extent + index[1] as f64 * size,
            -self.half_extent + index[2] as f64 * size,
        ]
    }

    /// The centre of a voxel.
    pub fn voxel_center(&self, voxel: usize) -> [f64; 3] {
        let corner = self.voxel_corner(voxel);
        let half = 0.5 * self.voxel_size();
        [corner[0] + half, corner[1] + half, corner[2] + half]
    }

    /// The voxel containing `x`, or `None` when `x` is outside the scene cube.
    pub fn locate(&self, x: [f64; 3]) -> Option<usize> {
        let n = self.resolution;
        let mut index = [0usize; 3];
        for axis in 0..3 {
            let t = (x[axis] + self.half_extent) / (2.0 * self.half_extent);
            if t.is_nan() || t < 0.0 || t > 1.0 {
                return None;
            }
            index[axis] = ((t * n as f64).floor() as usize).min(n - 1);
        }
        Some(self.flat_index(index))
    }

    /// `σ(x)`, and zero outside the scene cube — outside it there is no scene,
    /// and extrapolating the boundary voxel would put matter behind the camera
    /// that nothing ever observed.
    pub fn eval_density(&self, x: [f64; 3]) -> f64 {
        self.locate(x).map_or(0.0, |voxel| self.density[voxel])
    }

    /// `c(x, ω)`, and black outside the scene cube.
    pub fn eval_color(&self, x: [f64; 3], direction: [f64; 3]) -> [f64; 3] {
        match self.locate(x) {
            Some(voxel) => self.eval_voxel_color(voxel, direction),
            None => [0.0; 3],
        }
    }

    /// `c` of one voxel in a given direction.
    pub fn eval_voxel_color(&self, voxel: usize, direction: [f64; 3]) -> [f64; 3] {
        let coefficients = self.coefficients();
        let mut basis = [0.0f64; 25];
        harmonics::basis_into(
            self.sh_degree,
            harmonics::normalize(direction),
            &mut basis[..coefficients],
        );
        let base = voxel * 3 * coefficients;
        let mut out = [0.0; 3];
        for (channel, slot) in out.iter_mut().enumerate() {
            let row =
                &self.color[base + channel * coefficients..base + (channel + 1) * coefficients];
            *slot = row
                .iter()
                .zip(&basis[..coefficients])
                .map(|(c, b)| c * b)
                .sum();
        }
        out
    }

    /// `‖σ‖²_{L²(Ω)}`.
    pub fn density_norm_squared(&self) -> f64 {
        self.voxel_volume() * self.density.iter().map(|d| d * d).sum::<f64>()
    }

    /// `‖c‖²_{L²(Ω → [H^S]³)}`.
    ///
    /// The harmonics are orthonormal on `S²`, so a voxel's colour norm is the
    /// sum of its squared coefficients — no quadrature over directions needed.
    pub fn color_norm_squared(&self) -> f64 {
        self.voxel_volume() * self.color.iter().map(|c| c * c).sum::<f64>()
    }

    /// `‖(σ, c)‖` in the product space.
    pub fn norm(&self) -> f64 {
        (self.density_norm_squared() + self.color_norm_squared()).sqrt()
    }

    /// Double the resolution, copying each parent voxel into its eight
    /// children. The encoded scene is unchanged.
    pub fn refined(&self) -> Self {
        let coarse = self.resolution;
        let fine = coarse * 2;
        let coefficients = self.coefficients();
        let stride = 3 * coefficients;
        let mut density = vec![0.0; fine * fine * fine];
        let mut color = vec![0.0; fine * fine * fine * stride];
        for voxel in 0..density.len() {
            let i = voxel / (fine * fine);
            let j = (voxel / fine) % fine;
            let k = voxel % fine;
            let parent = ((i / 2) * coarse + (j / 2)) * coarse + (k / 2);
            density[voxel] = self.density[parent];
            color[voxel * stride..(voxel + 1) * stride]
                .copy_from_slice(&self.color[parent * stride..(parent + 1) * stride]);
        }
        Self {
            resolution: fine,
            level: self.level + 1,
            density,
            color,
            ..self.clone()
        }
    }

    /// `self ← self − η·gradient`, with the per-field rates applied and the
    /// density projected back into [`VoxelGrid::density_clamp`].
    pub fn descend_by(&mut self, learning_rate: f64, gradient: &Self) {
        assert_eq!(
            self.density.len(),
            gradient.density.len(),
            "gradient must be at the iterate's refinement level"
        );
        assert_eq!(self.color.len(), gradient.color.len());
        let density_step = learning_rate * self.density_rate;
        let color_step = learning_rate * self.color_rate;
        for (value, g) in self.density.iter_mut().zip(&gradient.density) {
            *value -= density_step * g;
        }
        for (value, g) in self.color.iter_mut().zip(&gradient.color) {
            *value -= color_step * g;
        }
        if let Some((low, high)) = self.density_clamp {
            for value in self.density.iter_mut() {
                *value = value.clamp(low, high);
            }
        }
    }

    /// Zero the density wherever `keep` is false.
    ///
    /// Used by the visual-hull preconditioner of Section 4.3: a voxel that
    /// projects onto background in some training image cannot hold matter, and
    /// refinement would otherwise hand its children a density the gradient no
    /// longer touches.
    pub fn mask_density(&mut self, keep: impl Fn(usize) -> bool) {
        for voxel in 0..self.density.len() {
            if !keep(voxel) {
                self.density[voxel] = 0.0;
            }
        }
    }
}

impl crate::algorithm::Representation for VoxelGrid {
    fn dofs(&self) -> usize {
        self.density.len() + self.color.len()
    }

    fn level(&self) -> u32 {
        self.level
    }

    fn refine(&self) -> Self {
        self.refined()
    }

    fn descend(&mut self, learning_rate: f64, gradient: &Self) {
        self.descend_by(learning_rate, gradient);
    }
}
