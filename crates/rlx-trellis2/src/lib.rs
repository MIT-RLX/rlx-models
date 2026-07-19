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

//! # rlx-trellis2 — Microsoft TRELLIS.2-4B image-to-3D, native for RLX
//!
//! TRELLIS.2 turns a single image into a textured 3D mesh through a cascade of
//! flow-matching **DiTs** and sparse-3D-conv **VAEs**:
//!
//! ```text
//!   image ─▶ [rembg] ─▶ DINOv3 features ─┐
//!                                         ├─▶ sparse-structure DiT ─▶ dense conv3d decoder ─▶ active voxels
//!                                         ├─▶ shape-SLat DiT (512→1024 cascade) ─▶ FlexiDualGrid VAE ─▶ dual grid ─▶ mesh
//!                                         └─▶ texture-SLat DiT ─────────────────▶ SparseUnet VAE ─▶ PBR voxels
//!                                                                                                     └─▶ GLB/OBJ
//! ```
//!
//! Module map (built bottom-up, each parity-checked against the upstream
//! PyTorch reference where fixtures exist):
//!   - [`config`]      — checkpoint JSON (`pipeline.json` + per-model `ckpts/*.json`)
//!   - [`rope`]        — 3-D axial rotary embedding
//!   - [`dit_host`]    — host DiT forward (structure + SLat)
//!   - [`dit_flow`]    — compiled DiT (AdaLN + SDPA + GatedResidual)
//!   - [`sampler`]     — flow-Euler + CFG guidance interval
//!   - [`conv3d`] / [`ss_decoder`] — dense structure VAE
//!   - [`sparse`] / [`shape_decoder`] — sparse shape/texture VAEs
//!   - [`mesh`]        — flexible dual-grid → triangles (+ PBR voxel attrs)
//!   - [`preprocess`]  — alpha crop / black composite (BiRefNet not bundled)
//!   - [`weights`]     — checkpoint resolution + WeightMap load
//!   - [`pipeline`]    — end-to-end [`Trellis2Runner`]
//!
//! The image conditioner reuses [`rlx_dinov3`] (DINOv3 ViT-L/16). The core RLX
//! compiler/runtime lives in the sibling `../rlx` workspace.

pub mod cli;
pub mod config;
pub mod conv3d;
pub mod dit_flow;
pub mod dit_host;
pub mod mesh;
pub mod pipeline;
pub mod preprocess;
pub mod rng;
pub mod rope;
pub mod sampler;
pub mod shape_decoder;
pub mod sparse;
pub mod ss_decoder;
pub mod weights;

pub use config::{
    DitConfig, DitKind, Normalization, PipelineConfig, PipelineType, SamplerParams,
    SparseStructureVaeArgs, SparseVaeConfig, SparseVaeKind,
};
pub use mesh::{Mesh, MeshWithPbr};
pub use pipeline::{
    Trellis2Input, Trellis2Output, Trellis2Runner, Trellis2RunnerBuilder, Trellis2ShapeOutput,
};
pub use preprocess::{PreprocessOptions, PreprocessedImage};
