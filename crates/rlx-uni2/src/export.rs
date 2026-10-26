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

//! Export a UNI2 checkpoint (timm names) as an `.rlxp` package rlx runs
//! alone, through the one exporter ([`rlx_core::export::vision`]): an RGB
//! image of any size in, the encoder's tokens and the `[CLS]` embedding
//! out, its patches assembled as the runner assembles them on the host
//! (position embeddings on the patches only, `no_embed_class`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::vision::{PatchStem, Positions, Vit, export_vit};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_core::weight_map::WeightMap;

use crate::config::{IMAGENET_MEAN, IMAGENET_STD, Uni2Config};
use crate::flow::build_uni2_built;

/// The config a checkpoint's tensors describe (the timm `config.json` is a
/// nested `pretrained_cfg` that says none of it): width and patch size from
/// the patch projection, the image size from the patch position table,
/// registers from `reg_token`, depth from the blocks, the packed SwiGLU
/// width from `fc1`, heads at 64 wide. A `config.json` of UNI2 fields wins.
pub fn config_of(ck: &Checkpoint, tensors: &WeightMap) -> Result<Uni2Config> {
    if let Ok(cfg) = serde_json::from_value::<Uni2Config>(ck.config.clone()) {
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
    let side = (shape("pos_embed")?[1] as f64).sqrt().round() as usize;
    let registers = shape("reg_token")
        .or_else(|_| shape("register_tokens"))
        .map_or(0, |s| s[1]);
    let depth = (0..)
        .take_while(|i| tensors.get(&format!("blocks.{i}.norm1.weight")).is_some())
        .count();
    Ok(Uni2Config {
        hidden_size: width,
        num_hidden_layers: depth,
        num_attention_heads: (width / 64).max(1),
        img_size: side * patch,
        patch_size: patch,
        mlp_hidden_dim: shape("blocks.0.mlp.fc1.weight")?[0],
        layer_norm_eps: 1e-6,
        num_register_tokens: registers,
    })
}

/// Export the UNI2 checkpoint in `dir` (timm `model.safetensors`) as the
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
    let registers_name = if t.contains_key("reg_token") {
        "reg_token"
    } else {
        "register_tokens"
    };
    let mut wm = WeightMap::from_tensors(t);
    let cfg = config_of(&ck, &wm)?;
    let built = build_uni2_built(&cfg, &mut wm, 1)?;
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
            .then(|| (registers_name.to_string(), cfg.num_register_tokens)),
        positions: Positions::Patches("pos_embed".into()),
    };
    let vit = Vit {
        family: "uni2".into(),
        exported_by: "rlx-uni2".into(),
        stem,
        encoder,
        encoder_input: "hidden".into(),
        params,
        output: "tokens".into(),
        mean: IMAGENET_MEAN,
        std: IMAGENET_STD,
    };
    export_vit(&ck, vit, out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("uni2", "image-feature-extraction", export_rlxp).claims(&["uni2"]),
    );
}
