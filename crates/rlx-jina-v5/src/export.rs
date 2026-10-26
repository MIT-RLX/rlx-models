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

//! Export jina-embeddings-v5 with one task adapter as an `.rlxp` package
//! with an `embed` entry, through the one exporter ([`rlx_core::export`]).
//!
//! What [`crate::JinaV5`] does on the host is in the graph: the token
//! embeddings gathered, the key-padding bias from the attention mask, and —
//! after the runner's encoder, its adapter merged — the last real token's
//! row. The entry normalizes it. The task is the export's `variant`
//! (`retrieval` by default); the merged projections are the package's own,
//! the rest referenced from the checkpoint.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlx_core::SafetensorsSource;
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use rlx_llama32::Llama32Config;
use serde_json::json;

use crate::{MASKED, TASKS, build_encoder};

/// Lengths the encoder is built at when the options list none.
const DEFAULT_SEQS: [usize; 5] = [64, 128, 256, 512, 2048];

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, values: &[f32]) -> NodeId {
    let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], f32s(&[values.len()]))
}

/// The embedding of up to `seq` tokens: `input_ids` and `attention_mask`
/// `[1, seq]` in, the last real token's state `[1, 1, hidden]` out.
fn encode(
    cfg: &Llama32Config,
    dir: &Path,
    adapter_dir: &Path,
    adapter_cfg: &serde_json::Value,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (h, nh) = (cfg.hidden_size, cfg.num_attention_heads);
    let base = SafetensorsSource::File(dir.join("model.safetensors")).loader()?;
    let adapter =
        SafetensorsSource::File(adapter_dir.join("adapter_model.safetensors")).loader()?;
    let (encoder, params) = rlx_core::flow_util::graph_from_built(build_encoder(
        cfg,
        base,
        adapter,
        adapter_cfg,
        seq,
    )?)?;
    let mut g = Graph::new(format!("encode_{seq}"));
    let ids = g.input("input_ids", Shape::new(&[1, seq], DType::I32));
    let mask = g.input("attention_mask", f32s(&[1, seq]));
    let table = g.param("embed_tokens.weight", f32s(&[cfg.vocab_size, h]));
    let flat = g.reshape(ids, vec![seq as i64], Shape::new(&[seq], DType::I32));
    let rows = g.gather(table, flat, 0, f32s(&[seq, h]));
    let embeds = g.reshape(rows, vec![1, seq as i64, h as i64], f32s(&[1, seq, h]));
    // `(mask − 1) · 1e9`: 0 for a real key, −1e9 for padding, every query.
    let one = constant(&mut g, &[1.0]);
    let big = constant(&mut g, &[-MASKED]);
    let off = g.binary(BinaryOp::Sub, mask, one, f32s(&[1, seq]));
    let keys = g.binary(BinaryOp::Mul, off, big, f32s(&[1, seq]));
    let keys = g.reshape(keys, vec![1, 1, 1, seq as i64], f32s(&[1, 1, 1, seq]));
    let bias = g.add_node(
        Op::Expand {
            target_shape: vec![1, nh as i64, seq as i64, seq as i64],
        },
        vec![keys],
        f32s(&[1, nh, seq, seq]),
    );
    let spliced = g
        .splice(
            &encoder,
            &rlx_ir::Splice::new()
                .bind("inputs_embeds", embeds)
                .bind("attn_bias", bias),
        )
        .map_err(|e| anyhow::anyhow!("splicing the encoder: {e}"))?;
    // The last real token: row `Σ mask − 1`.
    let count = g.sum(mask, vec![1], false);
    let last = g.binary(BinaryOp::Sub, count, one, f32s(&[1]));
    let last = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![last],
        Shape::new(&[1], DType::I32),
    );
    let states = g.reshape(
        spliced.outputs[0],
        vec![seq as i64, h as i64],
        f32s(&[seq, h]),
    );
    let row = g.gather(states, last, 0, f32s(&[1, h]));
    let out = g.reshape(row, vec![1, 1, h as i64], f32s(&[1, 1, h]));
    g.node_mut(out).name = Some("embedding".into());
    g.set_outputs(vec![out]);
    Ok((g, params))
}

/// Export the jina-embeddings-v5 checkpoint in `dir` with the task adapter
/// `opts.variant` (`retrieval` by default) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let task = opts.variant.as_deref().unwrap_or("retrieval");
    if !TASKS.contains(&task) {
        bail!("jina-v5: unknown task {task:?}; expected one of {TASKS:?}");
    }
    let ck = Checkpoint::open(dir)?;
    let cfg: Llama32Config = serde_json::from_value(ck.config.clone())?;
    let adapter_dir = dir.join("adapters").join(task);
    let adapter_cfg = ck.json(&format!("adapters/{task}/adapter_config.json"));
    if adapter_cfg.is_null() {
        bail!("{}: no adapter_config.json", adapter_dir.display());
    }
    let table = ck.loader()?.take("embed_tokens.weight")?.0;
    let mut refr = ck.referencer(opts.embed_weights)?;
    refr.take(HashMap::from([("embed_tokens.weight".to_string(), table)]))?;
    let family = "jina-v5";
    let mut pkg = Package::for_checkpoint(&ck, opts, family, "embedding", "rlx-jina-v5");
    let mut functions = Vec::new();
    for seq in opts
        .seqs_or(&DEFAULT_SEQS)
        .into_iter()
        .filter(|&s| s <= cfg.max_position_embeddings)
    {
        let (g, params) = encode(&cfg, dir, &adapter_dir, &adapter_cfg, seq)
            .with_context(|| format!("the encoder at {seq} tokens"))?;
        refr.take(params)?;
        functions.push(g.name.clone());
        pkg.function(g);
    }
    pkg.reference(refr, out)?;
    let tokenizer = pkg.tokenizer(&ck, None)?;
    pkg.entry(
        "embed",
        json!({
            "functions": functions,
            "tokenizer": tokenizer,
            "inputs": {"input_ids": "token_ids", "attention_mask": "attention_mask"},
            // The graph takes the last real token: its one row.
            "pooling": "cls",
            "normalize": true,
        }),
    );
    pkg.metadata("task", json!(task));
    pkg.metadata("hidden_size", json!(cfg.hidden_size));
    let prompts = ck.json("config_sentence_transformers.json")["prompts"].clone();
    if !prompts.is_null() {
        pkg.metadata("prompts", prompts);
    }
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("jina-v5", "embedding", export_rlxp)
            .claims(&["jina_embeddings_v5"])
            .architectures(&["JinaEmbeddingsV5Model"]),
    );
}
