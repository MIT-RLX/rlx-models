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

//! [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (Zheng et al., MIT
//! license) — bilateral-reference dichotomous image segmentation, the open
//! high-resolution background-removal model — for RLX.
//!
//! Reads the published `ZhengPeng7/BiRefNet*` `model.safetensors` directly
//! (`BiRefNet_lite`: Swin-T; `BiRefNet`: Swin-L). See [`graph`] for the
//! architecture and how deformable convolution is expressed; parity against
//! the reference code is checked by `examples/parity.rs` with
//! `scripts/birefnet_reference.py` dumps.

#[cfg(feature = "export")]
pub mod export;
pub mod graph;
pub mod model;
pub mod weights;

pub use model::BiRefNet;
pub use weights::Weights;
