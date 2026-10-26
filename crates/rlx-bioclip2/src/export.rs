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

//! Export BioCLIP-2 as an `.rlxp` package rlx runs alone, through the one
//! exporter ([`rlx_core::export`]): the entry `image`, a pipeline (an RGB
//! image of any size in, its embedding out), and `text`, an `embed` entry.
//! Opened as one model, `run` takes an image and `embed` texts.
//!
//! What [`crate::BioClip2Runner`] does on the host is in the graphs: the
//! image's OpenCLIP preprocessing ([`vision::clip_resize_crop`]), the
//! patch stem ([`crate::assemble_vision_hidden`]; OpenCLIP's convolution
//! has no bias, so its bias is zeros), the token and position embeddings
//! ([`crate::assemble_text_hidden`]), and the end-of-text row through the
//! text projection. Both embeddings are as the runner returns them; the
//! `embed` entry normalizes the text's.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow};
use rlx_core::export::graph::{Pipeline, Stage};
use rlx_core::export::vision::{
    self, CLIP_IOTA, HEIGHT, PatchStem, Positions, WIDTH, clip_iota, clip_resize_crop, patch_tokens,
};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport};
use rlx_core::flow_util::graph_from_built;
use rlx_core::weight_map::WeightMap;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::config::{BioClip2Config, CLIP_MEAN, CLIP_STD};
use crate::flow::{build_text_flow, build_vision_flow};

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// The zero bias the patch projection is given.
const STEM_BIAS: &str = "visual.conv1.bias";

/// `input_ids` and `attention_mask` `[1, ctx]` → `embedding` `[1, 1,
/// embed]`: the tokens' and positions' embeddings through the text tower,
/// the last real token's row (the end of text) through the projection.
fn encode_text(
    cfg: &BioClip2Config,
    wm: &mut WeightMap,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (ctx, width, embed) = (cfg.text.context_length, cfg.text.width, cfg.embed_dim);
    let (tower, mut params) = graph_from_built(build_text_flow(cfg, wm, 1)?)?;
    let (table, shape) = wm.take("token_embedding.weight")?;
    let vocab = shape[0];
    params.insert("token_embedding.weight".into(), table);
    params.insert(
        "positional_embedding".into(),
        wm.take("positional_embedding")?.0,
    );
    params.insert("text_projection".into(), wm.take("text_projection")?.0);

    let mut g = Graph::new(format!("encode_text_{ctx}"));
    let ids = g.input("input_ids", Shape::new(&[1, ctx], DType::I32));
    let mask = g.input("attention_mask", f32s(&[1, ctx]));
    let table = g.param("token_embedding.weight", f32s(&[vocab, width]));
    let flat = g.reshape(ids, vec![ctx as i64], Shape::new(&[ctx], DType::I32));
    let rows = g.gather(table, flat, 0, f32s(&[ctx, width]));
    let pos = g.param("positional_embedding", f32s(&[ctx, width]));
    let hidden = g.add(rows, pos);
    let hidden = g.reshape(
        hidden,
        vec![1, ctx as i64, width as i64],
        f32s(&[1, ctx, width]),
    );
    let spliced = g
        .splice(&tower, &rlx_ir::Splice::new().bind("hidden", hidden))
        .map_err(|e| anyhow!("splicing the text tower: {e}"))?;
    // The end of text, the last real token: row `Σ mask − 1`. The tower is
    // causal, so the padding after it does not reach it.
    let one = g.add_node(
        Op::Constant {
            data: 1f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let count = g.sum(mask, vec![1], false);
    let last = g.binary(BinaryOp::Sub, count, one, f32s(&[1]));
    let last = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![last],
        Shape::new(&[1], DType::I32),
    );
    let states = g.reshape(
        spliced.outputs[0],
        vec![ctx as i64, width as i64],
        f32s(&[ctx, width]),
    );
    let row = g.gather(states, last, 0, f32s(&[1, width]));
    let proj = g.param("text_projection", f32s(&[width, embed]));
    let out = g.mm(row, proj);
    let out = g.reshape(out, vec![1, 1, embed as i64], f32s(&[1, 1, embed]));
    g.node_mut(out).name = Some("embedding".into());
    g.set_outputs(vec![out]);
    Ok((g, params))
}

/// Export the BioCLIP-2 checkpoint in `dir` (OpenCLIP's layout:
/// `open_clip_config.json`, `open_clip_model.safetensors`,
/// `tokenizer.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let cfg_path = dir.join("open_clip_config.json");
    let cfg = if cfg_path.is_file() {
        BioClip2Config::from_open_clip_json(&cfg_path)?
    } else {
        BioClip2Config::vit_l_14()
    };
    let weights = dir.join("open_clip_model.safetensors");
    let mut wm = WeightMap::from_file(
        weights
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
    )?;
    let v = cfg.vision;
    let stem = PatchStem {
        image_size: v.image_size,
        patch_size: v.patch_size,
        channels: 3,
        width: v.width,
        weight: "visual.conv1.weight".into(),
        bias: STEM_BIAS.into(),
        class_token: "visual.class_embedding".into(),
        registers: None,
        positions: Positions::AllRows("visual.positional_embedding".into()),
    };
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    for name in [
        "visual.conv1.weight",
        "visual.class_embedding",
        "visual.positional_embedding",
    ] {
        params.insert(name.into(), wm.take(name)?.0);
    }
    params.insert(STEM_BIAS.into(), vec![0.0; v.width]);
    params.insert(CLIP_IOTA.into(), clip_iota());
    let (mut tower, tower_params) = graph_from_built(build_vision_flow(&cfg, &mut wm, 1)?)?;
    tower.name = "vision".into();
    params.extend(tower_params);
    let (text, text_params) = encode_text(&cfg, &mut wm)?;
    params.extend(text_params);

    let stages = vec![
        Stage::new(clip_resize_crop(v.image_size, CLIP_MEAN, CLIP_STD)?)
            .input("image", "image")
            .input("dims", "image#shape")
            .outputs(&["pixels"])
            .on("cpu"),
        Stage::new(patch_tokens(&stem)).outputs(&["hidden"]),
        Stage::new(tower)
            .input("hidden", "hidden")
            .outputs(&["image_embedding"]),
    ];
    let mut p = Pipeline::new("bioclip2", "image-feature-extraction", stages);
    p.inputs = vec![vision::image_contract()];
    p.dims = vec![("height".into(), HEIGHT), ("width".into(), WIDTH)];
    p.metadata = vec![
        ("image_size".into(), json!(v.image_size)),
        ("embed_dim".into(), json!(cfg.embed_dim)),
        (
            "similarity".into(),
            json!("cosine of the image and text embeddings, each normalized"),
        ),
    ];
    let functions = vec![text.name.clone()];
    p.functions = vec![text];
    p.exported_by = "rlx-bioclip2".into();
    let mut pkg = p.package(params, Some(&ck), out, opts)?;
    let tokenizer = pkg.tokenizer(&ck, None)?;
    pkg.rename_entry("pipeline", "image")?;
    pkg.entry(
        "text",
        json!({
            "functions": functions,
            "tokenizer": tokenizer,
            "inputs": {"input_ids": "token_ids", "attention_mask": "attention_mask"},
            // The graph takes the end of text: its one row.
            "pooling": "cls",
            "normalize": true,
            "kind": "embed",
        }),
    );
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("bioclip2", "image-feature-extraction", export_rlxp).claims(&["bioclip2"]),
    );
}
