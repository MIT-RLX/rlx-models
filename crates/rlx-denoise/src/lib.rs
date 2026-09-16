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

//! Monte-Carlo render denoiser: a guided U-Net, trainable end to end in RLX.
//!
//! A path tracer converges as `1/sqrt(n)`, so the last halving of its noise
//! costs three quarters of the render. Every production renderer therefore
//! stops early and reconstructs, and the two that matter both do it with a
//! trained network: Cycles calls OpenImageDenoise, and OptiX has one inside the
//! driver.
//!
//! Neither is portable. OptiX's is reached only through `optixDenoiserInvoke`
//! — the architecture and its weights live in `libnvoptix`, and there is no
//! intermediate representation to read. OpenImageDenoise is Apache-2.0 down to
//! its weights, so it *can* be read, but it is an Intel-shaped U-Net in a
//! format of its own and tens of megabytes of parameters.
//!
//! This is the same idea built natively: the architecture in [`model`], the
//! training in [`train`], and both expressed in ops that already have autodiff
//! rules and backend coverage — so the network trains on CUDA and runs
//! anywhere RLX runs.
//!
//! # What it takes
//!
//! Nine channels in — the render's colour plus the albedo and normal a path
//! tracer produces at its first useful hit — and three out. The guides are what
//! make the difference between a denoiser and a blur: they carry no
//! Monte-Carlo error, so where they disagree there is real detail and where
//! they agree there is only noise.
//!
//! Guides taken at the *first* hit describe a mirror rather than the image in
//! it, so a renderer feeding this should follow the path through specular and
//! transmissive surfaces before recording them, as Cycles does.
//!
//! # Training data
//!
//! Pairs cost nothing but time: render a scene at a low sample count and again
//! at a high one. No corpus needs collecting and no licence attaches to the
//! result.
//!
//! ```no_run
//! use rlx_denoise::{Batch, DenoiseNet, TrainConfig, Trainer, Widths};
//! use rlx_runtime::Device;
//!
//! let net = DenoiseNet::new(Widths::default());
//! let mut trainer = Trainer::new(net, 8, 128, 128, Device::Cpu, TrainConfig::default())?;
//! # let (input, target) = (vec![0.0; 8 * 9 * 128 * 128], vec![0.0; 8 * 3 * 128 * 128]);
//! let loss = trainer.step(&Batch { n: 8, h: 128, w: 128, input: &input, target: &target })?;
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod checkpoint;
pub mod dataset;
pub mod infer;
pub mod model;
pub mod train;

pub use dataset::Dataset;
pub use infer::Denoiser;
pub use model::{DenoiseNet, IN_CHANNELS, OUT_CHANNELS, ParamSpec, Widths, init_params};
pub use train::{Batch, LOSS_EPSILON, TRAIN_EPSILON, TrainConfig, Trainer};
