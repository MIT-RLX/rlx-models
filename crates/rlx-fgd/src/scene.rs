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

//! Cameras, images and posed-image datasets for Section 4.3.
//!
//! The camera convention is NeRF's, and matching it exactly matters more than
//! it looks: Proposition A.3's gradient is assembled by walking *backwards*
//! from a world point to the pixel that saw it, so [`Camera::project`] has to
//! be the exact inverse of [`Camera::ray_through`]. A convention mismatch
//! between them does not fail — it silently attributes each point's error to
//! the wrong pixel, and the reconstruction merely comes out wrong.
//!
//! `uv` here is in `(row, column)` order, as in the authors' code, and the
//! optical axis is `−z` in camera space with `t` measured along it. That is
//! what makes `[t_near, t_far] = [2, 6]` mean what NeRF means by it.

use crate::algorithm::Rng;
use crate::render::{Ray, render_ray};
use crate::rep::VoxelGrid;

/// A pinhole camera with a `4 × 4` camera-to-world pose.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// Camera-to-world transform, row-major.
    pose: [f64; 16],
    /// Inverse of the rotation block, cached for [`Camera::project`].
    inverse_rotation: [f64; 9],
    /// `det` of the rotation block — the `|det A_i|` of Proposition A.3.
    rotation_determinant: f64,
    /// Focal length in pixels.
    pub focal: f64,
    /// Whether pixel `(i, j)` samples its centre rather than its corner.
    pub rays_at_centers: bool,
}

impl Camera {
    /// Build from a camera-to-world pose (row-major `4 × 4`) and focal length.
    pub fn new(pose: [f64; 16], focal: f64) -> Self {
        let r = [
            pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[8], pose[9], pose[10],
        ];
        let determinant = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]);
        assert!(
            determinant.abs() > 1e-12,
            "camera pose has a singular rotation block"
        );
        let inv = 1.0 / determinant;
        let inverse_rotation = [
            (r[4] * r[8] - r[5] * r[7]) * inv,
            (r[2] * r[7] - r[1] * r[8]) * inv,
            (r[1] * r[5] - r[2] * r[4]) * inv,
            (r[5] * r[6] - r[3] * r[8]) * inv,
            (r[0] * r[8] - r[2] * r[6]) * inv,
            (r[2] * r[3] - r[0] * r[5]) * inv,
            (r[3] * r[7] - r[4] * r[6]) * inv,
            (r[1] * r[6] - r[0] * r[7]) * inv,
            (r[0] * r[4] - r[1] * r[3]) * inv,
        ];
        Self {
            pose,
            inverse_rotation,
            rotation_determinant: determinant,
            focal,
            rays_at_centers: true,
        }
    }

    /// A camera at `eye` looking at `target`, with the given focal length.
    pub fn look_at(eye: [f64; 3], target: [f64; 3], up: [f64; 3], focal: f64) -> Self {
        // NeRF's convention: the camera looks down −z, so +z is the backward axis.
        let backward = normalize(sub(eye, target));
        let right = normalize(cross(up, backward));
        let true_up = cross(backward, right);
        let pose = [
            right[0],
            true_up[0],
            backward[0],
            eye[0],
            right[1],
            true_up[1],
            backward[1],
            eye[1],
            right[2],
            true_up[2],
            backward[2],
            eye[2],
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        Self::new(pose, focal)
    }

    /// The camera-to-world pose, row-major.
    pub fn pose(&self) -> &[f64; 16] {
        &self.pose
    }

    /// `|det A_i|` of Proposition A.3's `a_i`.
    pub fn rotation_determinant(&self) -> f64 {
        self.rotation_determinant.abs()
    }

    /// The camera's position in world space.
    pub fn position(&self) -> [f64; 3] {
        [self.pose[3], self.pose[7], self.pose[11]]
    }

    /// The ray through image-plane coordinate `uv`, in `(row, column)` order.
    pub fn ray_through(&self, uv: [f64; 2]) -> Ray {
        let (u, v) = (uv[1], uv[0]);
        let camera_direction = [u / self.focal, -v / self.focal, -1.0];
        let r = &self.pose;
        let direction = [
            r[0] * camera_direction[0] + r[1] * camera_direction[1] + r[2] * camera_direction[2],
            r[4] * camera_direction[0] + r[5] * camera_direction[1] + r[6] * camera_direction[2],
            r[8] * camera_direction[0] + r[9] * camera_direction[1] + r[10] * camera_direction[2],
        ];
        Ray {
            origin: self.position(),
            direction,
        }
    }

    /// The exact inverse of [`Camera::ray_through`]: the image-plane coordinate
    /// a world point projects to, and its depth `t` along the optical axis.
    ///
    /// A point behind the camera comes back with `t ≤ 0`; callers must reject
    /// it, because the projection formula divides by `t` and would otherwise
    /// place it in front.
    pub fn project(&self, x: [f64; 3]) -> ([f64; 2], f64) {
        let o = self.position();
        let d = [x[0] - o[0], x[1] - o[1], x[2] - o[2]];
        let m = &self.inverse_rotation;
        let local = [
            m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
            m[3] * d[0] + m[4] * d[1] + m[5] * d[2],
            m[6] * d[0] + m[7] * d[1] + m[8] * d[2],
        ];
        let t = -local[2];
        let u = local[0] * self.focal / t;
        let v = -local[1] * self.focal / t;
        ([v, u], t)
    }

    /// Image-plane coordinate of pixel `(row, column)`.
    pub fn pixel_to_uv(&self, pixel: [f64; 2], height: usize, width: usize) -> [f64; 2] {
        let shift = if self.rays_at_centers { 0.5 } else { 0.0 };
        [
            pixel[0] - 0.5 * height as f64 + shift,
            pixel[1] - 0.5 * width as f64 + shift,
        ]
    }

    /// Fractional pixel coordinate of an image-plane point.
    pub fn uv_to_pixel(&self, uv: [f64; 2], height: usize, width: usize) -> [f64; 2] {
        let shift = if self.rays_at_centers { 0.5 } else { 0.0 };
        [
            uv[0] + 0.5 * height as f64 - shift,
            uv[1] + 0.5 * width as f64 - shift,
        ]
    }
}

/// An RGB image with values in `[0, 1]`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    height: usize,
    width: usize,
    pixels: Vec<f64>,
}

impl Image {
    /// Build from row-major RGB values.
    pub fn new(height: usize, width: usize, pixels: Vec<f64>) -> Self {
        assert_eq!(pixels.len(), height * width * 3, "pixel count mismatch");
        Self {
            height,
            width,
            pixels,
        }
    }

    /// A constant image.
    pub fn filled(height: usize, width: usize, color: [f64; 3]) -> Self {
        let mut pixels = Vec::with_capacity(height * width * 3);
        for _ in 0..height * width {
            pixels.extend_from_slice(&color);
        }
        Self::new(height, width, pixels)
    }

    /// Rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Columns.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Row-major RGB values.
    pub fn pixels(&self) -> &[f64] {
        &self.pixels
    }

    /// Mutable RGB values.
    pub fn pixels_mut(&mut self) -> &mut [f64] {
        &mut self.pixels
    }

    /// The colour at `(row, column)`, clamped to the image.
    pub fn at(&self, row: isize, column: isize) -> [f64; 3] {
        let r = row.clamp(0, self.height as isize - 1) as usize;
        let c = column.clamp(0, self.width as isize - 1) as usize;
        let base = (r * self.width + c) * 3;
        [
            self.pixels[base],
            self.pixels[base + 1],
            self.pixels[base + 2],
        ]
    }

    /// Whether `(row, column)` is inside the image.
    pub fn contains(&self, row: isize, column: isize) -> bool {
        row >= 0 && column >= 0 && (row as usize) < self.height && (column as usize) < self.width
    }

    /// Mean squared error against another image of the same size.
    pub fn mean_squared_error(&self, other: &Image) -> f64 {
        assert_eq!(self.pixels.len(), other.pixels.len(), "size mismatch");
        self.pixels
            .iter()
            .zip(&other.pixels)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / self.pixels.len() as f64
    }

    /// Peak signal-to-noise ratio in dB against another image.
    pub fn psnr(&self, other: &Image) -> f64 {
        let mse = self.mean_squared_error(other);
        if mse <= 0.0 {
            f64::INFINITY
        } else {
            -10.0 * mse.log10()
        }
    }

    /// Box-downsample by an integer factor — what the authors' `INTER_AREA`
    /// resize reduces to at their `0.2` (`800 → 160`).
    pub fn downsample(&self, factor: usize) -> Image {
        assert!(factor >= 1, "factor must be positive");
        if factor == 1 {
            return self.clone();
        }
        let height = self.height / factor;
        let width = self.width / factor;
        let mut pixels = vec![0.0; height * width * 3];
        let scale = 1.0 / (factor * factor) as f64;
        for r in 0..height {
            for c in 0..width {
                for channel in 0..3 {
                    let mut sum = 0.0;
                    for dr in 0..factor {
                        for dc in 0..factor {
                            let base = ((r * factor + dr) * self.width + c * factor + dc) * 3;
                            sum += self.pixels[base + channel];
                        }
                    }
                    pixels[(r * width + c) * 3 + channel] = sum * scale;
                }
            }
        }
        Image::new(height, width, pixels)
    }
}

/// A posed image.
#[derive(Debug, Clone, PartialEq)]
pub struct View {
    /// The camera that took it.
    pub camera: Camera,
    /// What it saw.
    pub image: Image,
}

/// A set of posed images of one scene.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// The views.
    pub views: Vec<View>,
    /// Colour behind the object; NeRF-Synthetic composites onto white.
    pub background: [f64; 3],
}

impl Scene {
    /// Build from views.
    pub fn new(views: Vec<View>, background: [f64; 3]) -> Self {
        assert!(!views.is_empty(), "a scene needs at least one view");
        Self { views, background }
    }

    /// Number of views.
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether there are no views.
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Cameras placed on a sphere around the origin, on a Fibonacci spiral so
    /// they cover it evenly at any count.
    pub fn cameras_on_sphere(
        count: usize,
        radius: f64,
        focal: f64,
        elevation_limit: f64,
    ) -> Vec<Camera> {
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        (0..count)
            .map(|i| {
                let fraction = (i as f64 + 0.5) / count as f64;
                // Restrict to a band of elevations, as NeRF-Synthetic's
                // hemispherical capture does.
                let z = elevation_limit * (2.0 * fraction - 1.0);
                let r = (1.0 - z * z).sqrt();
                let theta = golden * i as f64;
                let eye = [
                    radius * r * theta.cos(),
                    radius * r * theta.sin(),
                    radius * z,
                ];
                Camera::look_at(eye, [0.0; 3], [0.0, 0.0, 1.0], focal)
            })
            .collect()
    }

    /// Render `grid` from every camera, producing a synthetic training set.
    pub fn synthesize(
        grid: &VoxelGrid,
        cameras: Vec<Camera>,
        height: usize,
        width: usize,
        options: &RenderOptions,
        background: [f64; 3],
    ) -> Scene {
        let views = cameras
            .into_iter()
            .map(|camera| {
                let image = render_image(grid, &camera, height, width, options, background);
                View { camera, image }
            })
            .collect();
        Scene::new(views, background)
    }
}

/// How an image is rendered from a grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderOptions {
    /// Near plane.
    pub t_near: f64,
    /// Far plane.
    pub t_far: f64,
    /// Jittered sub-rays per pixel.
    ///
    /// Section 4.3's antialiasing kernel is a box over the pixel, so averaging
    /// uniformly jittered rays *is* the convolution — this is the continuous
    /// supersampling the appendix describes. The authors use 2048 on a GPU;
    /// far fewer is visually indistinguishable and this renderer is on a CPU.
    pub samples_per_pixel: usize,
    /// Seed for the jitter.
    pub seed: u64,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            t_near: 2.0,
            t_far: 6.0,
            samples_per_pixel: 4,
            seed: 0,
        }
    }
}

/// Render one image from a grid.
pub fn render_image(
    grid: &VoxelGrid,
    camera: &Camera,
    height: usize,
    width: usize,
    options: &RenderOptions,
    background: [f64; 3],
) -> Image {
    let samples = options.samples_per_pixel.max(1);
    let mut pixels = vec![0.0; height * width * 3];
    for row in 0..height {
        for column in 0..width {
            // One stream per pixel keeps the image independent of how the work
            // is split across threads.
            let mut rng = Rng::new(
                options
                    .seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((row * width + column) as u64),
            );
            let mut accumulated = [0.0f64; 3];
            for _ in 0..samples {
                let jitter = if samples == 1 {
                    [0.0, 0.0]
                } else {
                    [rng.uniform01() - 0.5, rng.uniform01() - 0.5]
                };
                let uv = camera.pixel_to_uv(
                    [row as f64 + jitter[0], column as f64 + jitter[1]],
                    height,
                    width,
                );
                let ray = camera.ray_through(uv);
                let color = render_ray(grid, &ray, options.t_near, options.t_far, background);
                for channel in 0..3 {
                    accumulated[channel] += color[channel];
                }
            }
            let base = (row * width + column) * 3;
            for channel in 0..3 {
                pixels[base + channel] = accumulated[channel] / samples as f64;
            }
        }
    }
    Image::new(height, width, pixels)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    crate::rep::harmonics::normalize(v)
}

/// Which split of a NeRF-Synthetic scene to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// `transforms_train.json`.
    Train,
    /// `transforms_val.json`.
    Val,
    /// `transforms_test.json`.
    Test,
}

impl Split {
    fn file_name(self) -> &'static str {
        match self {
            Split::Train => "transforms_train.json",
            Split::Val => "transforms_val.json",
            Split::Test => "transforms_test.json",
        }
    }
}

/// How much of a split to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    /// Integer box-downsampling factor. Section 4.3 uses `5`, taking the
    /// dataset's `800 × 800` images to `160 × 160`.
    pub downsample: usize,
    /// Keep only the first `limit` frames, as Section 4.3 does with `24`.
    pub limit: Option<usize>,
    /// Keep every `skip`-th frame. The authors evaluate with `8`.
    pub skip: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            downsample: 5,
            limit: Some(24),
            skip: 1,
        }
    }
}

/// Load a NeRF-Synthetic scene (`lego`, `ficus`, …) from its directory.
///
/// Expects the dataset's own layout: `transforms_{split}.json` beside the
/// `{train,val,test}/` image folders. The images are RGBA and are composited
/// onto `background` — matching the dataset's convention, and the reason
/// `background` also has to be handed to the renderer, which adds it behind
/// whatever transmittance is left.
pub fn load_nerf_synthetic(
    directory: &std::path::Path,
    split: Split,
    background: [f64; 3],
    options: &LoadOptions,
) -> anyhow::Result<Scene> {
    use anyhow::Context;

    let manifest = directory.join(split.file_name());
    let text = std::fs::read_to_string(&manifest)
        .with_context(|| format!("reading {}", manifest.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&text)?;
    let angle_x = parsed["camera_angle_x"]
        .as_f64()
        .context("transforms JSON has no camera_angle_x")?;
    let frames = parsed["frames"]
        .as_array()
        .context("transforms JSON has no frames array")?;

    let selected: Vec<&serde_json::Value> = frames
        .iter()
        .take(options.limit.unwrap_or(usize::MAX))
        .step_by(options.skip.max(1))
        .collect();
    anyhow::ensure!(!selected.is_empty(), "no frames selected");

    let mut views = Vec::with_capacity(selected.len());
    for frame in selected {
        let relative = frame["file_path"]
            .as_str()
            .context("a frame has no file_path")?;
        let mut path = directory.join(relative);
        if path.extension().is_none() {
            path.set_extension("png");
        }
        let decoded = image::open(&path)
            .with_context(|| format!("reading {}", path.display()))?
            .to_rgba8();
        let (width, height) = (decoded.width() as usize, decoded.height() as usize);
        let mut pixels = Vec::with_capacity(height * width * 3);
        for pixel in decoded.pixels() {
            let alpha = pixel[3] as f64 / 255.0;
            for channel in 0..3 {
                let value = pixel[channel] as f64 / 255.0;
                pixels.push(value * alpha + background[channel] * (1.0 - alpha));
            }
        }
        let image = Image::new(height, width, pixels).downsample(options.downsample.max(1));

        let matrix = frame["transform_matrix"]
            .as_array()
            .context("a frame has no transform_matrix")?;
        let mut pose = [0.0f64; 16];
        for (r, row) in matrix.iter().enumerate() {
            let row = row.as_array().context("malformed transform_matrix row")?;
            for (c, value) in row.iter().enumerate() {
                pose[r * 4 + c] = value.as_f64().context("non-numeric pose entry")?;
            }
        }
        // The focal length follows the *rendered* width, so downsampling scales
        // it with the image rather than needing a separate correction.
        let focal = 0.5 * image.width() as f64 / (0.5 * angle_x).tan();
        views.push(View {
            camera: Camera::new(pose, focal),
            image,
        });
    }
    Ok(Scene::new(views, background))
}
