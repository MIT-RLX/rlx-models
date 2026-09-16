// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Speaker diarization: sliding-window embeddings + centroid clustering.
//!
//! - Default backend: mel-stat embeddings (no neural weights).
//! - Feature `wespeaker`: WeSpeaker ResNet34-LM on **native RLX** (256-d) when
//!   [`DiarizeConfig::wespeaker_dir`] / `RLX_WESPEAKER_DIR` resolves (ONNX under
//!   `onnx/` is import/reference only).

pub mod cluster;
pub mod embed;
pub mod session;
pub mod sortformer;

pub use session::{DiarizeConfig, DiarizeSession, SpeakerTurn, best_speaker};
pub use sortformer::{SortformerConfig, activity_to_turns, sort_speakers_by_arrival};
