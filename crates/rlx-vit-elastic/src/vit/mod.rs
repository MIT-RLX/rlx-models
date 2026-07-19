// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Generic differentiable Vision Transformer forward + backbones.

pub mod backbones;
pub mod config;
pub mod forward;
pub mod preprocess;
pub mod runner;
pub mod weights;

pub use backbones::{config_for, load_backbone};

pub use config::{FfnKind, IMAGENET_MEAN, IMAGENET_STD, VitConfig};
pub use forward::{
    AdapterOpts, ParamSpec, VitGraph, build_vit_graph, build_vit_graph_with, extract_cls,
};
pub use preprocess::{
    PreprocessWeights, assemble_hidden, extract_preprocess_weights, rgb_u8_to_imagenet_nchw,
};
pub use runner::VitRunner;
pub use weights::{LoadedVit, load_vit, prepare_from_weightmap, synthetic_checkpoint};
