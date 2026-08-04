// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A **full BERT encoder written in the `rlx!` DSL, run with real pretrained
//! weights**, verified against this crate's `rlx-flow` reference builder.
//!
//! It downloads `google/bert_uncased_L-2_H-128_A-2` (a standard 2-layer HF BERT)
//! from the Hugging Face hub at runtime, then builds the forward pass two ways:
//! the crate's production path `rlx_bert::build_bert_graph_sized` (rlx-flow), and
//! a hand-written `rlx!` graph in `build_bert_dsl` below. It binds the *same*
//! real weights into both, runs them on CPU over a real tokenized sentence, and
//! asserts the hidden states agree. The `rlx!` version is the whole model:
//! embedding gather + LayerNorm, then the encoder layers (a reusable `fn` block —
//! Q/K/V projections, fused attention, output dense, residual+LayerNorm, GELU
//! feed-forward, residual+LayerNorm).
//!
//! ```text
//! cargo run -p rlx-bert --example bert_rlx_dsl_real_weights --release
//! ```

use std::collections::HashMap;
use std::path::Path;

use hf_hub::api::sync::Api;
use rlx::rlx;
use rlx::runtime::{Device, Session};
use rlx_core::config::BertConfig;
use rlx_core::weight_map::WeightMap;
use rlx_ir::Graph;

const REPO: &str = "google/bert_uncased_L-2_H-128_A-2";

fn main() {
    // ── 1. Fetch the real checkpoint from the HF hub (cached on disk) ──
    let api = Api::new().expect("hf-hub api");
    let repo = api.model(REPO.to_string());
    let cfg_path = repo.get("config.json").expect("download config.json");
    let w_path = repo
        .get("model.safetensors")
        .expect("download model.safetensors");
    println!("checkpoint: {REPO}");

    let cfg = BertConfig::from_file(&cfg_path).expect("parse config");
    let tensors = load_safetensors(&w_path); // keys keep the `bert.` prefix
    println!(
        "config: {} layers, hidden {}, heads {}, intermediate {}, vocab {}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.intermediate_size,
        cfg.vocab_size
    );

    // ── 2. A real tokenized sentence: "the cat sat on the mat" ──
    let seq = 8usize;
    let ids: Vec<f32> = vec![101., 1996., 4937., 2938., 2006., 1996., 13523., 102.];
    let mask = vec![1.0f32; seq]; // no padding → attention is unmasked
    let types = vec![0.0f32; seq];
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();

    // ── 3. Reference (rlx-flow) with the real weights — ground truth ──
    let mut wm = WeightMap::from_tensors(tensors.clone());
    let (ref_graph, ref_params) =
        rlx_bert::build_bert_graph_sized(&cfg, &mut wm, 1, seq).expect("reference graph");
    let ref_out = run(
        ref_graph,
        &ref_params,
        &[
            ("input_ids", &ids),
            ("attention_mask", &mask),
            ("token_type_ids", &types),
            ("position_ids", &pos),
        ],
    );

    // ── 4. The full BERT written in rlx!, bound to the same real weights ──
    let (dsl_graph, dsl_params) = build_bert_dsl(&cfg, &tensors);
    let dsl_out = run(
        dsl_graph,
        &dsl_params,
        // The DSL graph uses `MaskKind::None` (this sentence has no padding), so
        // it needs only the three inputs it reads.
        &[
            ("input_ids", &ids),
            ("token_type_ids", &types),
            ("position_ids", &pos),
        ],
    );

    // ── 5. Compare hidden states ──
    assert_eq!(ref_out.len(), dsl_out.len());
    let maxdiff = ref_out
        .iter()
        .zip(&dsl_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine(&ref_out, &dsl_out);

    println!("\nhidden_states  [1, {seq}, {}]", cfg.hidden_size);
    println!("  max |rlx-flow − rlx!| = {maxdiff:.3e}");
    println!("  cosine similarity     = {cos:.6}");
    println!(
        "  rlx! [CLS] embedding (first 8 dims) = {:?}",
        &dsl_out[..8]
    );

    assert!(
        maxdiff < 1e-3,
        "rlx! BERT diverged from the rlx-flow reference (max diff {maxdiff:.3e})"
    );
    println!("\n✔ full BERT in rlx! matches the reference on real weights.");
}

/// Compile `graph`, bind `params` by name, run over `inputs`, return output[0].
fn run(graph: Graph, params: &HashMap<String, Vec<f32>>, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(graph);
    for (name, data) in params {
        c.set_param(name, data);
    }
    c.run(inputs).into_iter().next().expect("one output")
}

/// The complete BERT encoder as an `rlx!` graph, plus its `param → weights` map.
/// Every `param … @ "key"` names itself by its HF checkpoint key (with `{i}`
/// filled per family element), so weight binding is a one-liner: iterate the
/// graph's params and pull each by name. `linear(x, w, b)` = `x·Wᵀ+b` fused;
/// `gelu(linear(..))` folds the activation. Written for a 2-layer checkpoint.
fn build_bert_dsl(
    cfg: &BertConfig,
    tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> (Graph, HashMap<String, Vec<f32>>) {
    assert_eq!(
        cfg.num_hidden_layers, 2,
        "this example is written for 2 layers"
    );
    let eps = cfg.layer_norm_eps as f32;

    let graph = rlx! {
        graph "bert_dsl";
        input input_ids: [1, 8];
        input token_type_ids: [1, 8];
        input position_ids: [1, 8];

        param word_emb @ "bert.embeddings.word_embeddings.weight"        : [30522, 128];
        param pos_emb  @ "bert.embeddings.position_embeddings.weight"    : [512, 128];
        param type_emb @ "bert.embeddings.token_type_embeddings.weight"  : [2, 128];
        param emb_ln_g @ "bert.embeddings.LayerNorm.weight"             : [128];
        param emb_ln_b @ "bert.embeddings.LayerNorm.bias"               : [128];

        // Per-layer weight families — one declaration each covers all layers.
        param qw[2]   @ "bert.encoder.layer.{i}.attention.self.query.weight"        : [128, 128];
        param qb[2]   @ "bert.encoder.layer.{i}.attention.self.query.bias"          : [128];
        param kw[2]   @ "bert.encoder.layer.{i}.attention.self.key.weight"          : [128, 128];
        param kb[2]   @ "bert.encoder.layer.{i}.attention.self.key.bias"            : [128];
        param vw[2]   @ "bert.encoder.layer.{i}.attention.self.value.weight"        : [128, 128];
        param vb[2]   @ "bert.encoder.layer.{i}.attention.self.value.bias"          : [128];
        param ow[2]   @ "bert.encoder.layer.{i}.attention.output.dense.weight"      : [128, 128];
        param ob[2]   @ "bert.encoder.layer.{i}.attention.output.dense.bias"        : [128];
        param alng[2] @ "bert.encoder.layer.{i}.attention.output.LayerNorm.weight"  : [128];
        param alnb[2] @ "bert.encoder.layer.{i}.attention.output.LayerNorm.bias"    : [128];
        param iw[2]   @ "bert.encoder.layer.{i}.intermediate.dense.weight"          : [512, 128];
        param ib[2]   @ "bert.encoder.layer.{i}.intermediate.dense.bias"            : [512];
        param dw[2]   @ "bert.encoder.layer.{i}.output.dense.weight"                : [128, 512];
        param db[2]   @ "bert.encoder.layer.{i}.output.dense.bias"                  : [128];
        param olng[2] @ "bert.encoder.layer.{i}.output.LayerNorm.weight"            : [128];
        param olnb[2] @ "bert.encoder.layer.{i}.output.LayerNorm.bias"              : [128];

        // One HF BERT encoder layer (post-LayerNorm). `linear` = HF `x·Wᵀ+b`.
        fn layer(x, qw, qb, kw, kb, vw, vb, ow, ob, alng, alnb, iw, ib, dw, db, olng, olnb) {
            let q = linear(x, qw, qb);
            let k = linear(x, kw, kb);
            let v = linear(x, vw, vb);
            let ctx = q.attention(k, v, 2, 64, MaskKind::None);
            let attn = (linear(ctx, ow, ob) + x).layer_norm(alng, alnb, (eps));
            let inter = gelu(linear(attn, iw, ib));
            let y = (linear(inter, dw, db) + attn).layer_norm(olng, olnb, (eps));
        }

        // Embeddings: word + position + token-type lookups (`embed` sugar =
        // gather), summed, then LayerNorm.
        let we = embed(word_emb, input_ids);
        let pe = embed(pos_emb, position_ids);
        let te = embed(type_emb, token_type_ids);
        let emb = (we + pe + te).layer_norm(emb_ln_g, emb_ln_b, (eps));

        // Two distinct-weight encoder layers.
        repeat i in 0..2 {
            let emb = layer(emb,
                qw[i], qb[i], kw[i], kb[i], vw[i], vb[i], ow[i], ob[i],
                alng[i], alnb[i], iw[i], ib[i], dw[i], db[i], olng[i], olnb[i]);
        }
        out emb;
    };

    // Every param already carries its HF key as its name → bind by iterating.
    let params = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Param { name } => Some((
                name.clone(),
                tensors
                    .get(name)
                    .unwrap_or_else(|| panic!("missing weight {name}"))
                    .0
                    .clone(),
            )),
            _ => None,
        })
        .collect();
    (graph, params)
}

/// Load an all-F32 safetensors file into `name → (row-major f32, shape)`.
fn load_safetensors(path: &Path) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let bytes = std::fs::read(path).expect("read weights");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let mut out = HashMap::new();
    for name in st.names() {
        let t = st.tensor(name).expect("tensor");
        assert_eq!(
            t.dtype(),
            safetensors::Dtype::F32,
            "{name} is not F32 ({:?})",
            t.dtype()
        );
        let data: Vec<f32> = t
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(name.to_string(), (data, t.shape().to_vec()));
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-20)
}
