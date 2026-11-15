// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Custom wake-word training **entirely in RLX** (`rlx-cpu` BLAS + host backprop).
//!
//! No PyTorch / openWakeWord / nanowakeword training loops required.

pub mod cnn;
pub mod dataset;
pub mod graph_cnn;
pub mod mlp;
pub mod multi;
pub mod report;
pub mod sgd;

pub use cnn::{CnnTrainConfig, train_new_lite_cnn, train_wake_cnn};
pub use graph_cnn::{GraphTrainConfig, train_wake_cnn_device};
pub use multi::{MultiClip, MultiReport, MultiTrainConfig, init_multi, load_mels_cached, train_multi};
pub use dataset::{LabeledClip, load_pos_neg_dirs, synth_pos_neg_dataset, write_synth_corpus};
pub use mlp::{MlpConfig, MlpWeights, clips_to_mel_features, mel_mean_feature, train_mlp};
pub use report::TrainReport;
pub use sgd::SgdConfig;
