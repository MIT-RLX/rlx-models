// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! GLARE: continual self-supervised pre-training that adapts a frozen SSL ViT
//! to a new domain by training only a UniAdapter (+ cross-attention + head)
//! with a student–teacher (EMA) DINO objective at three levels — global
//! ([CLS]), regional (cross-attention), and local (patch strong-blur).

pub mod adapter;
pub mod cross_attn;
pub mod losses;
pub mod patch_aug;
pub mod regions;
pub mod train;

pub use adapter::{AdapterConfig, init_adapter_params};
pub use cross_attn::{build_cross_attention, init_cross_attention_params};
pub use losses::{GlareCore, GlareStudent, GlareWeights, build_glare_core, build_glare_student};
pub use patch_aug::strong_blur_patches;
pub use regions::RegionLayout;
pub use train::{GlareConfig, GlareTrainer};
