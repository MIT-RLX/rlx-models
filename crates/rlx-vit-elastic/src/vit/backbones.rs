// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Named backbones + loading.

use std::path::Path;

use anyhow::{Result, bail};

use super::config::VitConfig;
use super::weights::{LoadedVit, load_vit, prepare_from_weightmap, synthetic_checkpoint};

/// Resolve a backbone name to a [`VitConfig`].
///
/// - `dino-vitb16` — `facebook/dino-vitb16` (DINO ViT-B/16, the SSL backbone).
/// - `uni2h` — `MahmoodLab/UNI2-h` (DINOv2-family ViT-H/14, packed SwiGLU).
/// - `synthetic` / `synthetic-uni2` — tiny built-in topologies (no weights).
pub fn config_for(name: &str) -> Result<VitConfig> {
    Ok(match name {
        "dino-vitb16" | "dino_vitb16" | "dino" => VitConfig::dino_vitb16(),
        "uni2h" | "uni2-h" | "uni2" => VitConfig::uni2_h(224),
        "synthetic" => VitConfig::synthetic(),
        "synthetic-uni2" => VitConfig::synthetic_uni2(),
        other => bail!(
            "unknown backbone '{other}' (want dino-vitb16 | uni2h | synthetic | synthetic-uni2)"
        ),
    })
}

/// Load a backbone: real weights from `weights` if given, else a deterministic
/// synthetic checkpoint (for the `synthetic*` configs or quick demos).
pub fn load_backbone(name: &str, weights: Option<&Path>) -> Result<(VitConfig, LoadedVit)> {
    let cfg = config_for(name)?;
    let loaded = match weights {
        Some(path) => load_vit(path, &cfg)?,
        None => prepare_from_weightmap(synthetic_checkpoint(&cfg, 0x5EED), &cfg)?,
    };
    Ok((cfg, loaded))
}
