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

//! The volumetric renderer of Section 4.3.
//!
//! ```text
//! R(γ) = ∫ T(t)·σ(γ(t))·c(γ(t), γ_ω) dt + T(t_far)·b,    T(t) = exp(−∫σ ds)
//! ```
//!
//! On a [`VoxelGrid`] this integral is not approximated: `σ` is constant inside
//! a voxel, so over a segment of length `Δt` the contribution is exactly
//! `T·(1 − e^{−σΔt})·c` and the transmittance is exactly multiplied by
//! `e^{−σΔt}`. A DDA walk visits each voxel the ray crosses once, which is why
//! the authors call this mode *exact integration* — the only error left is the
//! representation's own, which is what Algorithm 1 is measuring.
//!
//! [`march`] also records the state at one interior `t` while continuing to the
//! far plane. Proposition A.3's density gradient needs both the full render and
//! the render up to the point `x`, and the authors' code obtains them with two
//! separate traversals of the same ray; splitting one segment instead halves
//! the cost of the term that dominates the whole experiment.

use crate::rep::VoxelGrid;

/// A ray `γ(t) = origin + t·direction`.
///
/// `direction` is deliberately not normalised: the camera parametrisation of
/// Section 4.3 fixes its component along the optical axis at `−1`, so `t` is
/// depth along that axis and `[t_near, t_far] = [2, 6]` means what NeRF means
/// by it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// `γ(0)`.
    pub origin: [f64; 3],
    /// `dγ/dt`.
    pub direction: [f64; 3],
}

impl Ray {
    /// `γ(t)`.
    pub fn at(&self, t: f64) -> [f64; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }

    /// The `t` at which the ray passes closest to `x` — exact when `x` is on
    /// the ray, which is how Proposition A.3 uses it.
    pub fn parameter_at(&self, x: [f64; 3]) -> f64 {
        let d = self.direction;
        let numerator: f64 = (0..3).map(|a| (x[a] - self.origin[a]) * d[a]).sum();
        let denominator: f64 = (0..3).map(|a| d[a] * d[a]).sum();
        numerator / denominator
    }
}

/// What one traversal produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct March {
    /// `∫ T σ c dt` over the whole interval, without the background term.
    pub color: [f64; 3],
    /// `T(t_end)`.
    pub transmittance: f64,
    /// `∫ T σ c dt` up to the requested stop, or the full value if none.
    pub color_at_stop: [f64; 3],
    /// `T(stop)`, or `T(t_end)` if no stop was requested.
    pub transmittance_at_stop: f64,
}

/// Entry parameter of the ray into the grid's cube, at or after `t_start`.
///
/// `None` when the ray misses the box.
pub fn box_entry(grid: &VoxelGrid, ray: &Ray, t_start: f64) -> Option<f64> {
    let half = grid.half_extent();
    let mut t_min = t_start;
    let mut t_max = f64::INFINITY;
    for axis in 0..3 {
        let d = ray.direction[axis];
        let o = ray.origin[axis];
        let (lo, hi) = (-half, half);
        if d == 0.0 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let t1 = (lo - o) / d;
        let t2 = (hi - o) / d;
        let (near, far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        t_min = t_min.max(near);
        t_max = t_max.min(far);
    }
    if t_max >= t_min { Some(t_min) } else { None }
}

/// March a ray through `grid` over `[t_start, t_end]`.
///
/// When `stop_at` falls inside the interval, the accumulated colour and
/// transmittance at that point are recorded in the `*_at_stop` fields and the
/// walk continues; the segment containing it is split so the result matches a
/// separate march ending there exactly.
pub fn march(grid: &VoxelGrid, ray: &Ray, t_start: f64, t_end: f64, stop_at: Option<f64>) -> March {
    let mut color = [0.0f64; 3];
    let mut optical_depth = 0.0f64;
    // Set once the walk passes `stop_at`. Staying `None` means the stop was
    // never reached, and the full-interval values stand in for it — which is
    // the right answer for an `x` beyond the far plane, whose contribution
    // Proposition A.3's indicator zeroes anyway.
    let mut recorded: Option<([f64; 3], f64)> = None;

    let entry = match box_entry(grid, ray, t_start) {
        Some(entry) if entry < t_end => entry,
        // The ray misses the grid: nothing is accumulated anywhere along it.
        _ => {
            return March {
                color,
                transmittance: 1.0,
                color_at_stop: color,
                transmittance_at_stop: 1.0,
            };
        }
    };

    let n = grid.resolution();
    let half = grid.half_extent();
    let size = grid.voxel_size();

    // Locate the entry voxel, clamping the entry point onto the box face so a
    // grazing ray cannot start one voxel outside it.
    let entry_point = ray.at(entry);
    let mut index = [0usize; 3];
    for axis in 0..3 {
        let clamped = entry_point[axis].clamp(-half, half);
        let t = (clamped + half) / (2.0 * half);
        index[axis] = ((t * n as f64).floor() as isize).clamp(0, n as isize - 1) as usize;
    }

    // Standard DDA setup: `t_next[a]` is where the ray crosses the next voxel
    // boundary on axis `a`, `delta[a]` how much `t` advances per voxel there.
    let mut t_next = [f64::INFINITY; 3];
    let mut delta = [f64::INFINITY; 3];
    let mut step = [0isize; 3];
    for axis in 0..3 {
        let d = ray.direction[axis];
        if d == 0.0 {
            continue;
        }
        step[axis] = if d > 0.0 { 1 } else { -1 };
        delta[axis] = (size / d).abs();
        let boundary = if d > 0.0 {
            -half + (index[axis] + 1) as f64 * size
        } else {
            -half + index[axis] as f64 * size
        };
        t_next[axis] = (boundary - ray.origin[axis]) / d;
    }

    let mut current = entry;
    // A ray crosses at most `3n − 2` voxels of an `n³` grid.
    for _ in 0..(3 * n).saturating_sub(2) {
        if current >= t_end {
            break;
        }
        let axis = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
            0
        } else if t_next[1] <= t_next[2] {
            1
        } else {
            2
        };
        let boundary = t_next[axis];
        let segment_end = boundary.min(t_end);

        let voxel = grid.flat_index(index);
        let density = grid.density()[voxel];
        let voxel_color = grid.eval_voxel_color(voxel, ray.direction);

        // Split the segment if the requested stop falls inside it, so the
        // recorded state matches a separate march ending exactly there.
        let mut sub_start = current;
        if let Some(stop) = stop_at
            && recorded.is_none()
            && stop < segment_end
        {
            let split = stop.max(current);
            accumulate(
                &mut color,
                &mut optical_depth,
                density,
                voxel_color,
                split - sub_start,
            );
            recorded = Some((color, (-optical_depth).exp()));
            sub_start = split;
        }
        accumulate(
            &mut color,
            &mut optical_depth,
            density,
            voxel_color,
            segment_end - sub_start,
        );
        if let Some(stop) = stop_at
            && recorded.is_none()
            && stop <= segment_end
        {
            recorded = Some((color, (-optical_depth).exp()));
        }

        index[axis] = match index[axis].checked_add_signed(step[axis]) {
            Some(next) if next < n => next,
            _ => break,
        };
        current = boundary;
        t_next[axis] += delta[axis];
    }

    let transmittance = (-optical_depth).exp();
    let stop = recorded.unwrap_or((color, transmittance));
    March {
        color,
        transmittance,
        color_at_stop: stop.0,
        transmittance_at_stop: stop.1,
    }
}

/// One voxel segment: exact for a constant `σ` and `c`.
fn accumulate(
    color: &mut [f64; 3],
    optical_depth: &mut f64,
    density: f64,
    voxel_color: [f64; 3],
    length: f64,
) {
    if length <= 0.0 {
        return;
    }
    let step = density * length;
    // `−expm1(−step)` is `1 − e^{−step}` without the cancellation at small σΔt.
    let weight = (-*optical_depth).exp() * -(-step).exp_m1();
    for channel in 0..3 {
        color[channel] += weight * voxel_color[channel];
    }
    *optical_depth += step;
}

/// `R(γ)` including the background term — the rendering equation in full.
pub fn render_ray(
    grid: &VoxelGrid,
    ray: &Ray,
    t_near: f64,
    t_far: f64,
    background: [f64; 3],
) -> [f64; 3] {
    let result = march(grid, ray, t_near, t_far, None);
    let mut out = result.color;
    for channel in 0..3 {
        out[channel] += result.transmittance * background[channel];
    }
    out
}
