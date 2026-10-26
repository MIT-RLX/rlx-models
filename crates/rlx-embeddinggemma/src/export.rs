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

//! Export EmbeddingGemma as an `.rlxp` package with an `embed` entry, so
//! `Model.open("embeddinggemma.rlxp").embed(texts)` works in every
//! language with no rlx-models code, through the one exporter
//! ([`rlx_core::export`]).
//!
//! What [`crate::EmbeddingGemma`] does on the host is in the graph: the
//! token embeddings gathered and scaled by √hidden, the full and the
//! bidirectional sliding-window attention biases from the attention mask
//! (padding keys at −1e9), then — after the runner's own encoder graph —
//! the mean over the real tokens and the composed dense head. The graph
//! returns that one row; the entry normalizes it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rlx_core::AssetSource;
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package};
use rlx_gemma::GemmaConfig;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::{MASKED, bidirectional_window, build_encoder, compose_dense_head};

/// Lengths the encoder is built at when the options list none.
const DEFAULT_SEQS: [usize; 5] = [64, 128, 256, 512, 2048];

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, values: &[f32], dims: &[usize]) -> NodeId {
    let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], f32s(dims))
}

/// The embedding of up to `seq` tokens: `input_ids` and `attention_mask`
/// `[1, seq]` in, `embedding` `[1, 1, dim]` (not yet normalized) out.
fn encode(
    cfg: &GemmaConfig,
    ck: &Checkpoint,
    vocab: usize,
    head: &[f32],
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let (h, nh) = (cfg.hidden_size, cfg.num_attention_heads);
    let dim = head.len() / h;
    let (encoder, mut params) =
        rlx_core::flow_util::graph_from_built(build_encoder(cfg, ck.loader()?, seq)?)?;
    let mut g = Graph::new(format!("encode_{seq}"));
    let ids = g.input("input_ids", Shape::new(&[1, seq], DType::I32));
    let mask = g.input("attention_mask", f32s(&[1, seq]));

    // The token embeddings, scaled as the runner scales them.
    let table = g.param("embed_tokens.weight", f32s(&[vocab, h]));
    let flat = g.reshape(ids, vec![seq as i64], Shape::new(&[seq], DType::I32));
    let rows = g.gather(table, flat, 0, f32s(&[seq, h]));
    let scale = constant(&mut g, &[(h as f64).sqrt() as f32], &[1]);
    let rows = g.binary(BinaryOp::Mul, rows, scale, f32s(&[seq, h]));
    let hidden_in = g.reshape(rows, vec![1, seq as i64, h as i64], f32s(&[1, seq, h]));

    // A padding key masked for every query: `(mask − 1) · 1e9` is 0 or −1e9.
    let one = constant(&mut g, &[1.0], &[1]);
    let big = constant(&mut g, &[-MASKED], &[1]);
    let off = g.binary(BinaryOp::Sub, mask, one, f32s(&[1, seq]));
    let keys = g.binary(BinaryOp::Mul, off, big, f32s(&[1, seq]));
    let keys = g.reshape(keys, vec![1, 1, 1, seq as i64], f32s(&[1, 1, 1, seq]));
    let full_shape = f32s(&[1, nh, seq, seq]);
    let expand = |g: &mut Graph, x: NodeId| {
        g.add_node(
            Op::Expand {
                target_shape: vec![1, nh as i64, seq as i64, seq as i64],
            },
            vec![x],
            full_shape.clone(),
        )
    };
    let full = expand(&mut g, keys);
    // Sliding layers also mask keys `window` or more away.
    let window = cfg.sliding_window.map_or(usize::MAX, bidirectional_window);
    let band: Vec<f32> = (0..seq * seq)
        .map(|i| {
            if (i / seq).abs_diff(i % seq) < window {
                0.0
            } else {
                MASKED
            }
        })
        .collect();
    let band = constant(&mut g, &band, &[1, 1, seq, seq]);
    let sliding = g.binary(BinaryOp::Min, keys, band, f32s(&[1, 1, seq, seq]));
    let sliding = expand(&mut g, sliding);

    let spliced = g
        .splice(
            &encoder,
            &rlx_ir::Splice::new()
                .bind("prefill_hidden", hidden_in)
                .bind("attn_bias_full", full)
                .bind("attn_bias_sliding", sliding),
        )
        .map_err(|e| anyhow::anyhow!("splicing the encoder: {e}"))?;
    let hidden = spliced.outputs[0];

    // The mean over the real tokens, then the dense head.
    let m3 = g.reshape(mask, vec![1, seq as i64, 1], f32s(&[1, seq, 1]));
    let kept = g.binary(BinaryOp::Mul, hidden, m3, f32s(&[1, seq, h]));
    let sum = g.sum(kept, vec![1], false);
    let count = g.sum(mask, vec![1], true);
    let inv = g.recip(count);
    let pooled = g.binary(BinaryOp::Mul, sum, inv, f32s(&[1, h]));
    let proj = g.param("embeddinggemma.head", f32s(&[dim, h]));
    let proj_t = g.transpose_(proj, vec![1, 0]);
    let out = g.mm(pooled, proj_t);
    let out = g.reshape(out, vec![1, 1, dim as i64], f32s(&[1, 1, dim]));
    g.node_mut(out).name = Some("embedding".into());
    g.set_outputs(vec![out]);
    params.insert("embeddinggemma.head".into(), head.to_vec());
    Ok((g, params))
}

/// Export the EmbeddingGemma checkpoint in `dir` (sentence-transformers
/// layout: `config.json`, `model.safetensors`, `2_Dense`, `3_Dense`,
/// `tokenizer.json`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let raw = std::fs::read_to_string(dir.join("config.json"))?;
    let cfg = GemmaConfig::from_hf_value(&ck.config, &raw)?;
    let src = AssetSource::dir(dir.to_path_buf());
    let head = compose_dense_head(&src, cfg.hidden_size)?;
    let max_seq = ck.json("sentence_bert_config.json")["max_seq_length"]
        .as_u64()
        .map_or(cfg.max_position_embeddings, |v| v as usize);
    let vocab = cfg.vocab_size;

    // The token table, read as stored: the referencer finds it in the
    // checkpoint.
    let table = ck.loader()?.take("embed_tokens.weight")?.0;
    let mut refr = ck.referencer(opts.embed_weights)?;
    refr.take(HashMap::from([("embed_tokens.weight".to_string(), table)]))?;
    let mut pkg = Package::for_checkpoint(
        &ck,
        opts,
        "embeddinggemma",
        "embedding",
        "rlx-embeddinggemma",
    );
    let mut functions = Vec::new();
    for seq in opts
        .seqs_or(&DEFAULT_SEQS)
        .into_iter()
        .filter(|&s| s <= max_seq)
    {
        let (g, params) = encode(&cfg, &ck, vocab, &head, seq)?;
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
            // The graph pools and projects: its one row is the embedding.
            "pooling": "cls",
            "normalize": true,
        }),
    );
    pkg.metadata("hidden_size", json!(cfg.hidden_size));
    // The task prompts a text is embedded with (`"task: search result |
    // query: "` …), prepended by the caller.
    let prompts = ck.json("config_sentence_transformers.json")["prompts"].clone();
    if !prompts.is_null() {
        pkg.metadata("prompts", prompts);
    }
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// for a bare `Gemma3TextModel` (an embedder, where a causal Gemma is
/// `Gemma3ForCausalLM`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("embeddinggemma", "embedding", export_rlxp)
            .architectures(&["Gemma3TextModel"]),
    );
}
