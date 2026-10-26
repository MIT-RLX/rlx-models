// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Export Zonos v0.1 as an `.rlxp` package rlx runs alone: text in,
//! 44.1 kHz speech out, through the one exporter ([`rlx_core::export`]).
//!
//! Its graphs, as [`generate_codes_compiled`](crate::generate) computes:
//! `prefix` (the phonemes' rows and the six conditioning rows through the
//! prefix conditioner's projection and LayerNorm; symbolic in the
//! phonemes), the backbone (the CFG pair over the whole context, its last
//! hidden states; symbolic in the context, its RoPE tables as long as the
//! package reaches), `embed_codes` (a frame's nine codebook rows summed),
//! `heads` (the nine codebooks' logits for both halves of the pair), and
//! DAC's `lookup` and `decode` ([`rlx_dac::export::dac_functions`]). The
//! conditioning's small tensors (the unconditional rows, the Fourier
//! embedders, the speaker projection, the language table) are files the
//! program reads. The program (`program/`) phonemizes with espeak-ng, builds
//! the six conditioning rows, runs the delay-pattern loop with classifier-
//! free guidance and the runner's sampler, and decodes the codes.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{broadcast_params, symbolic_in};
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use serde_json::json;

use crate::conditioner::{DEFAULT_EMOTION, LANGUAGE_CODES};
use crate::config::{
    CODEBOOK_SIZE, DEFAULT_DAC_DIR, EOS_TOKEN_ID, MASKED_TOKEN_ID, N_CODEBOOKS, SAMPLE_RATE,
    ZonosFileConfig,
};
use crate::flow::{ZonosDims, build_prefill_graph, graph_params, rope_tables};
use crate::weights::WeightMap;

/// The context's, the phonemes', DAC's frames' and codebooks' symbols.
const CONTEXT: u32 = 0;
const PHONEMES: u32 = 1;
const DAC_FRAMES: u32 = 2;
const DAC_BOOKS: u32 = 3;
/// The longest context the backbone's RoPE tables reach.
const MAX_CONTEXT: usize = 4096;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// The backbone at `seq`: the CFG pair `[2, seq, d]` → its final-normed
/// hidden states, its RoPE tables params (`rope.cos`, `rope.sin`).
fn backbone(dims: &ZonosDims, seq: usize) -> Result<Graph> {
    let g = build_prefill_graph(dims, seq, false);
    let mut g = rlx_ir::stateful::rebuild(&g, |out, _, node, _| match &node.op {
        Op::Input { name } if name == "rope_cos" || name == "rope_sin" => {
            Ok(Some(out.param(name.replace('_', "."), node.shape.clone())))
        }
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    let hidden = g.outputs[0];
    g.node_mut(hidden).name = Some("hidden".into());
    g.set_outputs(vec![hidden]);
    g.name = "backbone".into();
    Ok(g)
}

/// `phonemes` `[n]` i32 and `conditions` `[6, d]` (the speaker, emotion,
/// fmax, pitch, speaking rate and language rows) → the prefix `[n + 6, d]`:
/// the phonemes' rows and the conditions, through the projection and the
/// LayerNorm.
fn prefix(d: usize, vocab: usize, eps: f32, n: usize) -> Graph {
    let mut g = Graph::new("prefix");
    let ids = g.input("phonemes", Shape::new(&[n], DType::I32));
    let cond = g.input("conditions", f32s(&[6, d]));
    let table = g.param(
        "prefix_conditioner.conditioners.0.phoneme_embedder.weight",
        f32s(&[vocab, d]),
    );
    let rows = g.gather(table, ids, 0, f32s(&[n, d]));
    // The phonemes first, then the speaker; the four Fourier rows; the
    // language — the conditioners' order.
    let all = g.concat_(vec![rows, cond], 0);
    let w = g.param("prefix_conditioner.project.weight", f32s(&[d, d]));
    let b = g.param("prefix_conditioner.project.bias", f32s(&[d]));
    let wt = g.transpose_(w, vec![1, 0]);
    let y = g.mm(all, wt);
    let y = g.add(y, b);
    let gamma = g.param("prefix_conditioner.norm.weight", f32s(&[d]));
    let beta = g.param("prefix_conditioner.norm.bias", f32s(&[d]));
    let y = g.ln(y, gamma, beta, eps);
    g.node_mut(y).name = Some("prefix".into());
    g.set_outputs(vec![y]);
    g
}

/// `codes` `[9]` i32 → the frame's row `[1, d]`: each codebook's embedding
/// row, summed.
fn embed_codes(d: usize, vocab: usize) -> Graph {
    let mut g = Graph::new("embed_codes");
    let codes = g.input("codes", Shape::new(&[N_CODEBOOKS], DType::I32));
    let mut sum = None;
    for q in 0..N_CODEBOOKS {
        let id = g.narrow_(codes, 0, q, 1);
        let table = g.param(format!("embeddings.{q}.weight"), f32s(&[vocab, d]));
        let row = g.gather(table, id, 0, f32s(&[1, d]));
        sum = Some(match sum {
            Some(s) => g.add(s, row),
            None => row,
        });
    }
    let row = sum.expect("nine codebooks");
    g.set_outputs(vec![row]);
    g
}

/// `h` `[2, d]` (the CFG pair's last states) → each codebook's logits for
/// both, `[2, 9 · (codebook + 1)]`.
fn heads(d: usize) -> Graph {
    let v = CODEBOOK_SIZE + 1;
    let mut g = Graph::new("heads");
    let h = g.input("h", f32s(&[2, d]));
    let w = g.param("heads.stacked", f32s(&[N_CODEBOOKS * v, d]));
    let wt = g.transpose_(w, vec![1, 0]);
    let y = g.mm(h, wt);
    g.set_outputs(vec![y]);
    g
}

/// Export the Zonos directory `dir` (`config.json`, `model.safetensors`;
/// DAC from `weights/tts/parler-dac` beside it, or `ZONOS_DAC_DIR`) as the
/// package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let cfg = ZonosFileConfig::load(dir.join("config.json"))?;
    let w = WeightMap::load(dir.join("model.safetensors"))?;
    let dims = ZonosDims::from_cfg(&cfg);
    let d = dims.d_model;
    let mut pkg = Package::for_checkpoint(&ck, opts, "zonos", "text-to-speech", "rlx-zonos");

    // The backbone, symbolic in the context.
    let g = symbolic_in(CONTEXT, (5, 7, 3), |n| backbone(&dims, n))?;
    let at = backbone(&dims, 3)?;
    let mut params = graph_params(&dims, &w)?;
    let half = dims.head_dim / 2;
    let (c3, s3) = rope_tables(&dims, 3);
    params.insert("rope.cos".into(), c3);
    params.insert("rope.sin".into(), s3);
    let (cos, sin) = rope_tables(&dims, MAX_CONTEXT);
    let g = broadcast_params(
        &g,
        &at,
        &mut params,
        &[("rope.cos", &cos, half), ("rope.sin", &sin, half)],
    )?;
    pkg.function(g);

    // The prefix, the frames' rows, the heads.
    let ph_vocab = w.shape("prefix_conditioner.conditioners.0.phoneme_embedder.weight")?[0];
    let eps = dims.eps;
    // `n + 6` rows: a constant past the phonemes, so checked at a third size.
    let p = symbolic_in(PHONEMES, (37, 53, 20), |n| Ok(prefix(d, ph_vocab, eps, n)))?;
    pkg.function(p);
    let emb_vocab = w.shape("embeddings.0.weight")?[0];
    pkg.function(embed_codes(d, emb_vocab));
    pkg.function(heads(d));
    for name in [
        "prefix_conditioner.conditioners.0.phoneme_embedder.weight",
        "prefix_conditioner.project.weight",
        "prefix_conditioner.project.bias",
        "prefix_conditioner.norm.weight",
        "prefix_conditioner.norm.bias",
    ] {
        params.insert(name.into(), w.get(name)?.to_vec());
    }
    for q in 0..N_CODEBOOKS {
        let n = format!("embeddings.{q}.weight");
        params.insert(n.clone(), w.get(&n)?.to_vec());
    }
    let mut stacked = Vec::new();
    for q in 0..N_CODEBOOKS {
        stacked.extend_from_slice(w.get(&format!("heads.{q}.weight"))?);
    }
    params.insert("heads.stacked".into(), stacked);

    // DAC.
    let dac_dir = std::env::var("ZONOS_DAC_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dir.parent()
                .map_or_else(|| DEFAULT_DAC_DIR.into(), |p| p.join("parler-dac"))
        });
    let dac =
        Checkpoint::open(&dac_dir).with_context(|| format!("DAC from {}", dac_dir.display()))?;
    let (dac_graphs, dac_params, _) = rlx_dac::export::dac_functions(&dac, DAC_FRAMES, DAC_BOOKS)?;
    for mut g in dac_graphs {
        g.name = format!("dac_{}", g.name);
        pkg.function(g);
    }
    params.extend(dac_params);
    pkg.dims = vec![
        ("context".into(), CONTEXT),
        ("phonemes".into(), PHONEMES),
        ("dac_frames".into(), DAC_FRAMES),
        ("codebooks".into(), DAC_BOOKS),
    ];
    let dims_of = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let dd = dims_of
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), dd, &params[n]);
    }

    // The conditioning's small tensors, files the program reads.
    let c = |i: usize, t: &str| format!("prefix_conditioner.conditioners.{i}.{t}");
    let mut files = vec![
        ("speaker_weight", c(1, "project.weight")),
        ("speaker_bias", c(1, "project.bias")),
        ("language", c(6, "int_embedder.weight")),
    ];
    let uncond: Vec<(String, String)> = (1..=6)
        .map(|i| (format!("uncond_{i}"), c(i, "uncond_vector")))
        .collect();
    let fourier: Vec<(String, String)> = (2..=5)
        .map(|i| (format!("fourier_{i}"), c(i, "weight")))
        .collect();
    let mut owned: Vec<(String, String)> =
        files.drain(..).map(|(a, b)| (a.to_string(), b)).collect();
    owned.extend(uncond);
    owned.extend(fourier);
    for (file, tensor) in owned {
        pkg.sidecar(
            &format!("cond/{file}.f32"),
            "application/octet-stream",
            w.get(&tensor)?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        );
    }
    let espeak = tempfile::tempdir()?;
    espeak_ng::install_bundled_language(espeak.path(), "en")
        .map_err(|e| anyhow::anyhow!("espeak-ng's English data: {e}"))?;
    let mut data = Vec::new();
    files_under(espeak.path(), espeak.path(), &mut data)?;
    for (rel, p) in data {
        pkg.sidecar(
            &format!("espeak/{rel}"),
            "application/octet-stream",
            std::fs::read(p)?,
        );
    }

    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "espeak": "/package/espeak",
            "cond": "/package/cond",
            "d_model": d,
            "n_codebooks": N_CODEBOOKS,
            "codebook_size": CODEBOOK_SIZE,
            "eos": EOS_TOKEN_ID,
            "masked": MASKED_TOKEN_ID,
            "languages": LANGUAGE_CODES,
            "emotion": DEFAULT_EMOTION,
            "cfg_scale": 2.0,
            "min_p": 0.1,
            "temperature": 1.0,
            "repetition_penalty": 3.0,
            "seed": 1337,
            "speaking_rate": 15.0,
            "fmax": 22_050.0,
            "pitch_std": 20.0,
            "language": "en-us",
            "sample_rate": SAMPLE_RATE,
        }),
    );
    pkg.metadata("sample_rate", json!(SAMPLE_RATE));
    pkg.write(out, opts)
}

fn files_under(dir: &Path, base: &Path, out: &mut Vec<(String, std::path::PathBuf)>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            files_under(&p, base, out)?;
        } else {
            let rel = p.strip_prefix(base)?.to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
    Ok(())
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("zonos", "text-to-speech", export_rlxp));
}
