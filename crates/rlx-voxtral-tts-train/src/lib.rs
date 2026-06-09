// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Native RLX training for Voxtral voice cloning.
//!
//! Phase 1: codec encoder (reconstruction + VQ auxiliary losses).
//! Phase 2: LoRA adapters on the 4B LM (embedding distillation).
//! Export/inject weights into `consolidated.safetensors` for inference.

pub mod adam;
pub mod asr_loss;
pub mod audio_metrics;
pub mod backward_prep;
pub mod checkpoint;
pub mod codec_graph;
pub mod compile;
pub mod config;
pub mod dataset;
pub mod device;
pub mod discriminator;
pub mod distill_dataset;
pub mod distill_text;
pub mod early_stop;
pub mod encoder_loss;
pub mod encoder_report;
pub mod encoder_train;
pub mod lm_lora_graph;
pub mod lora_train;
pub mod teacher;
pub mod train_pipeline;
pub mod weights;

pub use backward_prep::{needs_portable_backward_prep, prepare_backward_for_device};
pub use checkpoint::{
    export_encoder_weights, export_lora_weights, inject_weights, load_encoder_weights,
    load_lora_weights,
};
pub use compile::{TrainSession, backward_cpu_only_from_env, compile_train_session};
pub use config::{EncoderTrainConfig, LoraTrainConfig, TrainProfile};
pub use device::{pick_auto_device, resolve_train_device};
pub use encoder_train::{EncoderTrainResult, train_encoder};
pub use lora_train::train_lora;
pub use train_pipeline::{TrainAllConfig, TrainAllResult, default_train_all, train_all};
pub use weights::{codec_has_encoder, merge_codec_encoder_overlay};
