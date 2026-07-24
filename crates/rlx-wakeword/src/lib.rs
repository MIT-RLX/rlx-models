// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! First-party wakeword product: streaming session, multi-phrase train/pack, optional VAD / speaker-id.
//!
//! - [`WakewordSession`] — 16 kHz PCM → [`WakeEvent`]
//! - [`TrainBuilder`] — RLX CNN SGD → [`WakewordBundle`]
//! - Ternary: [`TernaryOpts`] after train (bake TQ2 / fused kernels in core)
//! - WASM / Web Worker: crate `rlx-wakeword-wasm` (not published)

pub mod bundle;
pub mod cascade;
pub mod cli;
pub mod config;
pub mod session;
pub mod train;

pub use bundle::{WakewordBundle, save_bundle, stub_bundle, stub_bundle_n};
pub use cascade::{AsrConfirm, SpeakerGate, Unsupported};
pub use config::{PhraseConfig, WakewordConfig, hop_ms_to_samples, samples_to_hop_ms};
pub use session::{WakeEvent, WakewordSession};
pub use train::{
    PhraseTrainSpec, TrainBuilder, TrainOpts, merge_phrase_into_dir, parse_phrase_arg,
    specs_from_phrases_dir, train_one_phrase, train_phrases, train_synth_n, validate_hop_ms,
};

pub use rlx_wake::{TernaryOpts, TernaryStats, is_ternary_f32};

#[cfg(feature = "speaker-id")]
pub use cascade::speaker::{EnrolledSpeaker, SpeakerIdConfig, SpeakerIdGate};

pub use rlx_wake::{
    SAMPLE_RATE_16K, bind_streaming_device, load_wav_mono_f32, parse_device_list, resolve_device,
    resample_linear,
};
pub use rlx_wakeword_core::{PACK_MAGIC, PackHeader};
