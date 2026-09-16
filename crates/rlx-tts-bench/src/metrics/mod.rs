//! Shared model-agnostic metrics.

mod noise;
mod rss;
mod spectral;
mod whisper;

pub use noise::{NoiseMetrics, noise_metrics};
pub use rss::{RssMetrics, RssTracker, peak_rss_mb};
pub use spectral::{SpectralMetrics, spectral_vs_ref};
pub use whisper::{WhisperMetrics, WhisperState, try_load_whisper, whisper_coverage};
