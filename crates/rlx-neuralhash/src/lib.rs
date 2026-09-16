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

//! Apple NeuralHash perceptual image hashing, natively on RLX.
//!
//! A native port of [AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX)'s
//! `nnhash.py`. The pipeline is four stages:
//!
//! ```text
//!   image ──resize 360×360 (PIL bicubic), /255·2−1, NCHW──▶ [1, 3, 360, 360]
//!         ──descriptor CNN (native rlx-ir graph)──────────▶ 128 f32
//!         ──seed matrix dot product ([96, 128])───────────▶ 96 f32
//!         ──binary step (score ≥ 0)──────────────────────▶ 96-bit hash
//! ```
//!
//! # Native, not an ONNX shim
//!
//! The descriptor network is built as an rlx-ir graph, so it runs on every rlx
//! backend (CPU / Metal / MLX / CUDA / ROCm / wgpu / Vulkan). The architecture
//! is read straight from the vendor's own model container:
//!
//! ```text
//!   NeuralHashv3b_fp16-current.espresso.net      (JSON layer list) ─┐
//!   NeuralHashv3b_fp16-current.espresso.shape    (blob geometry)    ├─▶ NeuralHashSpec ─▶ rlx-ir
//!   NeuralHashv3b_fp16-current.espresso.weights  (f16/f32 blobs)   ─┘
//! ```
//!
//! [`espresso`] parses the container, [`spec`] normalizes it into a
//! serializable architecture description, [`weights`] extracts the tensors and
//! [`flow`] emits the graph. Nothing on the inference path touches ONNX; the
//! optional `onnx-parity` feature exists only to diff the native graph against
//! the reference `model.onnx`.
//!
//! # Inputs
//!
//! Two files, both installed by the operating system (this crate ships and
//! redistributes neither):
//!
//! * `NeuralHashv3b_fp16-current.espresso.{net,shape,weights}` — the descriptor network.
//! * `neuralhash_128x96_seed1.dat` — the `[96, 128]` output projection.
//!
//! On macOS they live in `Vision.framework`; see the README for the path.
//!
//! # Example
//!
//! ```no_run
//! use rlx_neuralhash::{NeuralHash, NeuralHasher};
//! use rlx_runtime::Device;
//!
//! let r = "/System/Library/Frameworks/Vision.framework/Versions/A/Resources";
//! let mut hasher = NeuralHasher::open_espresso(
//!     format!("{r}/NeuralHashv3b_fp16-current.espresso.net"),
//!     format!("{r}/neuralhash_128x96_seed1.dat"),
//!     Device::Cpu,
//! )?;
//! let h: NeuralHash = hasher.hash_image("photo.jpg")?;
//! println!("{h}");                       // e.g. ab14febaa837b6c1484c35e6
//!
//! let other = hasher.hash_image("photo_resaved.jpg")?;
//! println!("{} bits differ", h.hamming(&other));
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Reproducibility
//!
//! NeuralHash is not bit-stable across implementations — the reference
//! implementation notes the same. Descriptor floats land arbitrarily close to
//! zero, so any change in
//! floating-point association flips that bit. The same caveat applies across
//! rlx backends: CPU and Metal will agree on the overwhelming majority of
//! bits, not necessarily all 96. Compare with [`NeuralHash::hamming`], never
//! with equality, unless you produced both on the same backend.
//!
//! # Scope
//!
//! This is a perceptual hash — a similarity descriptor over image content. It
//! tells you whether two images look alike; it is not a cryptographic hash and
//! carries no integrity or authenticity guarantee.

pub mod cli;
pub mod espresso;
pub mod flow;
pub mod hash;
pub mod model;
pub mod preprocess;
pub mod seed;
pub mod spec;
pub mod synth;
pub mod weights;

/// Validation-only cross-check against the reference `model.onnx`.
/// Requires the `onnx-parity` feature; not on the inference path.
#[cfg(feature = "onnx-parity")]
pub mod parity;

pub use espresso::EspressoNet;
pub use hash::{HASH_BITS, HASH_BYTES, NeuralHash};
pub use model::NeuralHashModel;
pub use preprocess::{INPUT_ELEMS, INPUT_SIZE};
pub use seed::{EMBED_DIM, SEED_FILE_BYTES, SEED_HEADER_BYTES, SeedMatrix};
pub use spec::NeuralHashSpec;

use anyhow::Result;
use rlx_runtime::Device;
use std::path::Path;

/// The full image → 96-bit hash pipeline: descriptor model + seed projection.
pub struct NeuralHasher {
    model: NeuralHashModel,
    seed: SeedMatrix,
}

impl NeuralHasher {
    /// Load the Espresso network and the `[96, 128]` seed matrix.
    pub fn open_espresso(
        net: impl AsRef<Path>,
        seed: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        // Parse the seed first: it is cheap and catches the most common setup
        // mistake before the graph build.
        let seed = SeedMatrix::open(seed)?;
        let model = NeuralHashModel::open_espresso(net, device)?;
        Ok(Self { model, seed })
    }

    /// Assemble from an already-built model and parsed seed matrix.
    pub fn from_parts(model: NeuralHashModel, seed: SeedMatrix) -> Self {
        Self { model, seed }
    }

    /// The backend the descriptor network runs on.
    pub fn device(&self) -> Device {
        self.model.device()
    }

    /// The seed projection matrix.
    pub fn seed(&self) -> &SeedMatrix {
        &self.seed
    }

    /// Hash an image file.
    pub fn hash_image(&mut self, path: impl AsRef<Path>) -> Result<NeuralHash> {
        let input = preprocess::load_image(path)?;
        self.hash_tensor(&input)
    }

    /// Hash an in-memory HWC RGB8 buffer (`rgb.len() == w * h * 3`).
    pub fn hash_rgb8(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<NeuralHash> {
        let input = preprocess::from_rgb8(rgb, w, h)?;
        self.hash_tensor(&input)
    }

    /// Hash an already-preprocessed `[3, 360, 360]` NCHW tensor.
    pub fn hash_tensor(&mut self, input: &[f32]) -> Result<NeuralHash> {
        let embedding = self.model.embed(input)?;
        self.hash_embedding(&embedding)
    }

    /// Project + threshold an existing 128-float descriptor.
    pub fn hash_embedding(&self, embedding: &[f32]) -> Result<NeuralHash> {
        let scores = self.seed.project(embedding)?;
        NeuralHash::from_scores(&scores)
    }

    /// The raw 128-float descriptor for an image file (pre-projection).
    pub fn embed_image(&mut self, path: impl AsRef<Path>) -> Result<Vec<f32>> {
        let input = preprocess::load_image(path)?;
        self.model.embed(&input)
    }

    /// Mutable access to the underlying descriptor network.
    pub fn model_mut(&mut self) -> &mut NeuralHashModel {
        &mut self.model
    }
}
