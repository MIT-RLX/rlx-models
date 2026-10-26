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

//! The vendor's subject lifting — the network behind Vision's
//! `VNGenerateForegroundInstanceMaskRequest` ("lift subject" in Photos, and
//! the background removal Genmoji's pipeline calls through MediaAnalysis) —
//! run natively by RLX.
//!
//! The model is read **in place** from the copy macOS ships
//! ([`SYSTEM_MODEL`], a vendor `int8` network); this crate contains no vendor
//! weights and is useful only on a Mac that has them.
//!
//! What it is (from the model file): a 512×512 RGB input; an EfficientNet-style
//! backbone (SiLU as `x·sigmoid(x)`, squeeze-excite via global average pools,
//! depthwise convs, ReLU6) with a ×2-upsample/concat decoder; two heads —
//! `saliency` (soft foreground probability, 512²) and `gating_confidence`
//! (whether there is a liftable subject at all). Vision then upsamples the
//! low-resolution soft mask to image resolution against the image
//! (`-[VNInstanceMaskObservation _upsampleLowResMask:reference:…]`).
//!
//! The settings the file does not record (pixel encoding, upsample filter) are
//! fixed by parity against the Vision framework itself — see
//! `examples/parity.rs`.

#[cfg(feature = "export")]
pub mod export;
pub mod graph;
pub mod lift;

pub use graph::Upsample;
pub use lift::{FEATHER, InputNorm, LiftOptions, Lifted, SYSTEM_MODEL, SubjectLift};
