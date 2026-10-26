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

//! Export a DINOv2 checkpoint (timm names) as an `.rlxp` package rlx runs
//! alone, through the one exporter ([`rlx_core::export::vision`]): an RGB
//! image of any size in — resized and normalized as the runner does, its
//! patches, class and register tokens and position embeddings assembled as
//! the runner assembles them on the host — and the encoder's tokens and
//! class embedding (or, with a classifier head, its logits) out.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::vision::{PatchStem, Positions, Vit, export_vit};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_core::weight_map::WeightMap;

use crate::config::{DinoV2Config, IMAGENET_MEAN, IMAGENET_STD};
use crate::flow::build_dinov2_built;

/// The config a checkpoint's tensors describe: width and patch size from
/// the patch projection, the image size from the position table, depth from
/// the blocks, the MLP ratio from `fc1`, heads at 64 wide (every DINOv2), a
/// classifier from `head.weight`. A `config.json` of DINOv2 fields wins.
pub fn config_of(ck: &Checkpoint, tensors: &WeightMap) -> Result<DinoV2Config> {
    if let Ok(cfg) = serde_json::from_value::<DinoV2Config>(ck.config.clone()) {
        return Ok(cfg);
    }
    let shape = |k: &str| {
        tensors
            .get(k)
            .map(|(_, s)| s.to_vec())
            .with_context(|| format!("the checkpoint has no '{k}'"))
    };
    let proj = shape("patch_embed.proj.weight")?;
    let (width, patch) = (proj[0], proj[2]);
    let registers = shape("register_tokens").map_or(0, |s| s[1]);
    let rows = shape("pos_embed")?[1];
    let side = ((rows - 1) as f64).sqrt().round() as usize;
    let depth = (0..)
        .take_while(|i| tensors.get(&format!("blocks.{i}.norm1.weight")).is_some())
        .count();
    let fc1 = shape("blocks.0.mlp.fc1.weight")?[0];
    let mut cfg = DinoV2Config::new(side * patch, depth, width, (width / 64).max(1), registers);
    cfg.patch_size = patch;
    cfg.mlp_ratio = fc1 as f64 / width as f64;
    cfg.num_classes = shape("head.weight").map_or(0, |s| s[0]);
    Ok(cfg)
}

/// Export the DINOv2 checkpoint in `dir` (timm `model.safetensors`) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let mut loader = ck.loader()?;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for k in loader.remaining_keys() {
        let v = loader.take(&k)?;
        t.insert(k, v);
    }
    drop(loader);
    let mut wm = WeightMap::from_tensors(t);
    let cfg = config_of(&ck, &wm)?;
    let built = build_dinov2_built(&cfg, &mut wm, 1)?;
    let (encoder, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    let stem = PatchStem {
        image_size: cfg.img_size,
        patch_size: cfg.patch_size,
        channels: 3,
        width: cfg.hidden_size,
        weight: "patch_embed.proj.weight".into(),
        bias: "patch_embed.proj.bias".into(),
        class_token: "cls_token".into(),
        registers: (cfg.num_register_tokens > 0)
            .then(|| ("register_tokens".to_string(), cfg.num_register_tokens)),
        positions: Positions::AllRows("pos_embed".into()),
    };
    let vit = Vit {
        family: "dinov2".into(),
        exported_by: "rlx-dinov2".into(),
        stem,
        encoder,
        encoder_input: "hidden".into(),
        params,
        output: if cfg.num_classes > 0 {
            "logits"
        } else {
            "tokens"
        }
        .into(),
        mean: IMAGENET_MEAN,
        std: IMAGENET_STD,
    };
    export_vit(&ck, vit, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("dinov2", "image-feature-extraction", export_rlxp).claims(&["dinov2"]),
    );
}
