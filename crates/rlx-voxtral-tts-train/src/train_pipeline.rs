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

//! Full voice-clone training pipeline (encoder → LoRA → inject).

use anyhow::Result;
use std::path::Path;

use crate::checkpoint::inject_weights;
use crate::config::{EncoderTrainConfig, LoraTrainConfig};
use crate::encoder_train::train_encoder;
use crate::lora_train::train_lora;

pub struct TrainAllConfig {
    pub encoder: EncoderTrainConfig,
    pub lora: LoraTrainConfig,
    pub inject: bool,
}

pub struct TrainAllResult {
    pub encoder_loss: f64,
    pub lora_loss: f64,
    pub consolidated: Option<std::path::PathBuf>,
}

pub fn train_all(cfg: &TrainAllConfig) -> Result<TrainAllResult> {
    let enc = train_encoder(&cfg.encoder)?;
    let mut lora_cfg = cfg.lora.clone();
    lora_cfg.encoder_weights = Some(cfg.encoder.out_dir.join("best_encoder.safetensors"));
    let lora = train_lora(&lora_cfg)?;
    let consolidated = if cfg.inject {
        Some(inject_weights(
            &cfg.encoder.model_dir,
            Some(&cfg.encoder.out_dir.join("best_encoder.safetensors")),
            Some(&cfg.lora.out_dir.join("lora_adapters.safetensors")),
        )?)
    } else {
        None
    };
    Ok(TrainAllResult {
        encoder_loss: enc.best_recon_l1,
        lora_loss: lora.best_loss,
        consolidated,
    })
}

pub fn default_train_all(
    model_dir: &Path,
    wav_dir: &Path,
    out_root: &Path,
    manifest: Option<std::path::PathBuf>,
) -> TrainAllConfig {
    let mut encoder = EncoderTrainConfig::from_cli(
        model_dir.to_path_buf(),
        wav_dir.to_path_buf(),
        out_root.join("encoder"),
    );
    encoder.manifest = manifest.clone();
    let mut lora = LoraTrainConfig::from_cli(
        model_dir.to_path_buf(),
        wav_dir.to_path_buf(),
        out_root.join("lora"),
    );
    lora.manifest = manifest;
    TrainAllConfig {
        encoder,
        lora,
        inject: true,
    }
}
