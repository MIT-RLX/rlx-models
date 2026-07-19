//! Convenience re-exports for `rlx-tts-bench` library users.
//!
//! ```rust,ignore
//! use rlx_tts_bench::prelude::*;
//!
//! let models = select_models("chatterbox,supertonic");
//! let devices = filter_available(&parse_device_list("auto")?);
//! let rows = run_suite(&RunConfig { /* … */ })?;
//! write_html(Path::new("report.html"), &rows, &write_summary_json(Path::new("summary.json"), &rows)?)?;
//! ```

// ── Adapter trait + catalog ──────────────────────────────────────
pub use crate::adapter::{
    AdapterFactory, AdapterMeta, CloneRequest, SynthRequest, SynthResult, TtsAdapter, WeightHints,
};
pub use crate::adapters::{all_model_ids, catalog, factory_for, make_adapter};

// ── Suite planner ────────────────────────────────────────────────
pub use crate::report::{
    BenchRow, Summary, append_results_jsonl, read_results_jsonl, write_html, write_results_jsonl,
    write_summary_json,
};
pub use crate::suite::{
    RunConfig, failed_row, gate_failed, list_adapters, run_suite, scenarios_for_flags,
    select_models,
};

// ── Devices ──────────────────────────────────────────────────────
pub use crate::devices::{auto_devices, device_label, filter_available, parse_device_list};
pub use rlx_runtime::{Device, is_available};

// ── Phrases ──────────────────────────────────────────────────────
pub use crate::phrases::{DEFAULT_LONG, DEFAULT_SHORT, FOX_WORDS, content_words};

// ── Metrics ──────────────────────────────────────────────────────
pub use crate::metrics::{
    NoiseMetrics, SpectralMetrics, WhisperMetrics, WhisperState, noise_metrics, spectral_vs_ref,
    try_load_whisper, whisper_coverage,
};

// ── WAV / DSP helpers ────────────────────────────────────────────
pub use crate::wav::{
    add_gaussian_noise, cosine, median, peak_normalize, read_wav_mono, resample_linear,
    write_wav_mono,
};
