// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! `no_std` + `alloc` mel frontend and WakeCnn for embedded / WASM wakeword.
//!
//! Weight layout matches `rlx-wake`. Ternary path: [`WakeCnnWeights::ternarize`] + fused
//! add/sub kernels when tensors are exact `{−1,0,+1}`. Pack header: [`PackHeader`] (`RLXW`).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod cnn;
pub mod mel;
pub mod ops;
pub mod pack;
pub mod ternary;

pub use cnn::{WakeCnn, WakeCnnConfig, WakeCnnWeights};
pub use mel::{MelConfig, MelFrontend, SAMPLE_RATE_16K};
pub use pack::{PACK_MAGIC, PACK_VERSION, PackHeader, QuantScales};
pub use ternary::{
    TernaryOpts, TernaryStats, is_ternary_f32, pack_trits, ternarize, ternarize_inplace,
    unpack_trits,
};
