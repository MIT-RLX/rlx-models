//! PP-OCRv6 tiny / small — native RLX detection + recognition.
//!
//! Inference loads **safetensors** and builds offline-decomposed HIR
//! ([`native`]). Host DB post-process + CTC decode complete the pipeline.
//! No ONNX Runtime and no runtime `rlx-onnx-import`.
//!
//! # Quick start
//!
//! ```ignore
//! use rlx_ppocrv6::{PpOcrV6Runner, Tier};
//! use rlx_runtime::Device;
//!
//! let runner = PpOcrV6Runner::builder()
//!     .tier(Tier::Tiny)
//!     .model_dir(".cache/ppocrv6/tiny")
//!     .device(Device::Cpu)
//!     .build()?;
//! let out = runner.predict_path("page.png")?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! See the crate [README](https://github.com/MIT-RLX/rlx-models/blob/main/crates/rlx-ppocrv6/README.md)
//! for fetch recipes, model layout, and backend notes.

pub mod backbone;
pub mod capabilities;
pub mod cli;
pub mod config;
pub mod detection;
pub mod engine;
pub mod model;
pub mod native;
pub mod preprocess;
pub mod recognition;
pub mod rlx;
pub mod runner;
pub mod weights;

pub use capabilities::{STANDARD_DEVICE_NAMES, STANDARD_DEVICES, validate_device};
pub use config::{DetectionParams, RecognitionParams, Tier};
pub use detection::DetBox;
pub use engine::{OcrLine, OcrResult, PpOcrV6Engine};
pub use runner::{PpOcrV6Runner, PpOcrV6RunnerBuilder};
