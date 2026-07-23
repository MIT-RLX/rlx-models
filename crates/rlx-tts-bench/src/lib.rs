//! Unified TTS bench harness (binary: `rlx-tts-bench`).
//!
//! Library entry: [`prelude`] (`use rlx_tts_bench::prelude::*`).

pub mod adapter;
pub mod adapters;
pub mod cli;
pub mod corpus;
pub mod devices;
pub mod isolate;
pub mod metrics;
pub mod phrases;
pub mod prelude;
pub mod report;
pub mod stress;
pub mod suite;
pub mod wav;
