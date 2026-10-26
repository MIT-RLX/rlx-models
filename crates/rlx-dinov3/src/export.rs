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

//! Export a Hugging Face DINOv3 ViT checkpoint as an `.rlxp` package rlx
//! runs alone, through the one exporter ([`rlx_core::export::vision`]): an
//! RGB image of any size in, the encoder's tokens and the `[CLS]` embedding
//! out. The patches, class and register tokens are assembled as the runner
//! assembles them on the host; there are no position embeddings (the
//! encoder rotates its queries and keys, its tables held by the package).

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rlx_core::export::vision::{PatchStem, Positions, Vit, export_vit};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_core::weight_map::WeightMap;

use crate::config::{DinoV3Config, IMAGENET_MEAN, IMAGENET_STD};
use crate::flow::build_dinov3_built;

/// Export the DINOv3 checkpoint in `dir` (`config.json`,
/// `model.safetensors`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let cfg: DinoV3Config = serde_json::from_value(ck.config.clone())?;
    let mut loader = ck.loader()?;
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for k in loader.remaining_keys() {
        let v = loader.take(&k)?;
        t.insert(k, v);
    }
    drop(loader);
    let mut wm = WeightMap::from_tensors(t);
    let built = build_dinov3_built(&cfg, &mut wm, 1)?;
    let (encoder, params) = rlx_core::flow_util::graph_from_built(built.model)?;
    let stem = PatchStem {
        image_size: cfg.image_size,
        patch_size: cfg.patch_size,
        channels: cfg.num_channels,
        width: cfg.hidden_size,
        weight: "embeddings.patch_embeddings.weight".into(),
        bias: "embeddings.patch_embeddings.bias".into(),
        class_token: "embeddings.cls_token".into(),
        registers: (cfg.num_register_tokens > 0).then(|| {
            (
                "embeddings.register_tokens".to_string(),
                cfg.num_register_tokens,
            )
        }),
        positions: Positions::None,
    };
    let vit = Vit {
        family: "dinov3".into(),
        exported_by: "rlx-dinov3".into(),
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
        Exporter::new("dinov3", "image-feature-extraction", export_rlxp)
            .claims(&["dinov3_vit", "dinov3"]),
    );
}
