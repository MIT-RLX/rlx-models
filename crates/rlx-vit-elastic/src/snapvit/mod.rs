// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! SnapViT: retraining-free, label-free structured pruning.
//!
//! - [`loss`] — the SnapViT SSL objective graph (DINO cross-view loss).
//! - [`local`] — local Hessian-diagonal per-structure scores (Eq. 2/3).
//! - [`fitness`] — label-free PCA-cosine fitness (Eq. 6), forward-only.
//! - [`xnes`] — the global block-scaling search (§3.3).
//! - [`prune`] — prunability score + elastic structured pruning (Eq. 8/9).
//! - [`run`] — end-to-end: local → xNES → elastic sub-networks.

pub mod fitness;
pub mod local;
pub mod loss;
pub mod mask;
pub mod prune;
pub mod run;
pub mod xnes;

pub use fitness::Fitness;
pub use local::{CalibImage, LocalScores, SnapVitConfig, compute_local_scores};
pub use loss::{SnapVitLoss, build_snapvit_loss};
pub use mask::{
    masks_from_kept, num_ffn_structures, num_head_structures, ones_ffn_mask, ones_head_mask,
};
pub use prune::{Prunability, PruneResult, coeffs_len, prunability, prune_at};
pub use run::{ElasticEntry, SnapVitParams, SnapVitResult, run};
pub use xnes::{XnesConfig, XnesResult, optimize};
