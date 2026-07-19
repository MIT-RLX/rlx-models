// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! **SnapViT** elastic structured pruning + **GLARE** continual SSL
//! pre-training for Vision Transformers, on the RLX backends.
//!
//! - [`vit`] — a generic, differentiable ViT forward (built at the
//!   `rlx_ir::Graph` level so it composes with autodiff, masks, and adapters)
//!   plus the DINO ViT-B/16 and UNI2-h backbones.
//! - `dino` — shared DINO self-supervised machinery (projection head, DINO
//!   loss, multi-crop augmentation, EMA teacher).
//! - `snapvit` — the SnapViT prunability score (local squared-gradient Hessian
//!   diagonal + xNES global correlation) and elastic structured pruning.
//! - `glare` — the UniAdapter, the three consistency losses, and the
//!   adapter-only continual pre-training loop.
//!
//! Papers: SnapViT (arXiv:2510.17700), GLARE (arXiv:2509.17816).

pub mod data;
pub mod dino;
pub mod glare;
pub mod snapvit;
pub mod vit;

pub use vit::{FfnKind, VitConfig};
