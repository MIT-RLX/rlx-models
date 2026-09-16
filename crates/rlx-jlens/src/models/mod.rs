//! [`crate::model::LensModel`] implementations.
//!
//! Each model lives behind its own feature so the core crate depends on none of
//! them. Adding a model is a new module plus a feature — no change to the
//! estimator, the fitting loop, or the readout.

#[cfg(feature = "dinov3")]
pub mod dinov3;
#[cfg(feature = "qwen25-vl")]
pub mod qwen25_vl;
#[cfg(feature = "qwen3")]
pub mod qwen3;
#[cfg(feature = "qwen35")]
pub mod qwen35;
