// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Descript Audio Codec (DAC) — native Rust encode/decode for
// https://github.com/descriptinc/descript-audio-codec

pub mod audio;
pub mod build;
pub mod codec;
pub mod codes;
pub mod config;
pub mod download;
pub mod graph;
pub mod layers;
pub mod ops;
pub mod quantize;
pub mod weights;

#[cfg(feature = "cli")]
pub mod cli;

pub use codec::{
    DacCodec, RoundtripStats, SAMPLE_RATE_16KHZ, SAMPLE_RATE_24KHZ, SAMPLE_RATE_44KHZ,
};
pub use codes::DacCodes;
pub use config::DacConfig;
pub use download::{
    HF_DAC_16KHZ, HF_DAC_24KHZ, HF_DAC_44KHZ, default_dac_dir, ensure_weights, fetch_dac,
    resolve_model_dir,
};
pub use rlx_core::audio_codec::{AudioCodec, ChunkStreamer, CodecInfo, RvqCodes};

#[cfg(feature = "cli")]
pub use cli::run;
