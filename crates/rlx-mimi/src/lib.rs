// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Kyutai Mimi neural audio codec — native Rust encode/decode for
// https://huggingface.co/kyutai/mimi

pub mod audio;
pub mod codec;
pub mod codes;
pub mod config;
pub mod conv;
pub mod device;
pub mod download;
pub mod graph;
pub mod layout;
pub mod rvq;
pub mod seanet;
pub mod transformer;

#[cfg(feature = "gpu-codec")]
pub mod gpu;

#[cfg(feature = "cli")]
pub mod cli;

pub use codec::{FRAME_RATE, MimiCodec, RoundtripStats, SAMPLE_RATE};
pub use codes::MimiCodes;
pub use config::MimiConfig;
pub use device::{
    candle_codec_available, device_ready, parse_mimi_device, resolve_codec_device, test_devices,
};
pub use download::{
    HF_MIMI_REPO, MIMI_CANDLE_SIDECAR, default_mimi_dir, ensure_weights, fetch_mimi,
    resolve_candle_weights, resolve_model_dir,
};
pub use rlx_core::audio_codec::{AudioCodec, ChunkStreamer, CodecInfo, RvqCodes};

#[cfg(feature = "cli")]
pub use cli::run;
