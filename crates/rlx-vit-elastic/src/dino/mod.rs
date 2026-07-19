// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Shared DINO self-supervised machinery: multi-crop augmentation, the
//! projection head, the cross-view consistency loss, and the EMA teacher.

pub mod crops;
pub mod head;
pub mod loss;
pub mod teacher;

pub use crops::{CropConfig, Rng, multi_crop, stack_crops};
pub use head::{DinoHeadConfig, build_dino_head, init_head_params};
pub use loss::{build_dino_loss, dino_ce_aligned, l2_normalize, pair_mask};
pub use teacher::{Center, ema_update, teacher_targets};
