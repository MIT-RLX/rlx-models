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

//! Export an openWakeWord directory (`embedding.safetensors`,
//! `phrase.safetensors`) as an `.rlxp` package rlx runs alone: 16 kHz audio
//! in, a score per 80 ms push and the detections out.
//!
//! Its graphs are the engine's: `mel` (`rlx_wake`'s frontend, symbolic in
//! the frames), `embed` (the embedding net over `MEL_WINDOW` frames: three
//! 3×3 convolutions, the spatial mean, the linear) and `phrase` (the phrase
//! head over the last `EMBED_HISTORY` embeddings, its sigmoid). The program
//! is the wake words' (`rlx_wake::export::detect_entry`), embedding every
//! `MEL_STEP` frames as the engine's queue does.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted_many;
use rlx_core::export::{ExportOptions, ExportReport, Package};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_wake::{MelConfig, MelFrontend};
use serde_json::json;

use crate::embedding::{EMBED_DIM, EmbeddingWeights, MEL_STEP, MEL_WINDOW};
use crate::phrase::{EMBED_HISTORY, PhraseWeights};

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `x · Wᵀ + b` for `x` `[1, i]`, the params `{name}.weight` `[o, i]` and
/// `{name}.bias`.
fn linear(g: &mut Graph, x: NodeId, name: &str, i: usize, o: usize) -> NodeId {
    let w = g.param(format!("{name}.weight"), f32s(&[o, i]));
    let w = g.transpose_(w, vec![1, 0]);
    let y = g.mm(x, w);
    let b = g.param(format!("{name}.bias"), f32s(&[o]));
    let b = g.reshape_(b, vec![1, o as i64]);
    g.binary(BinaryOp::Add, y, b, f32s(&[1, o]))
}

/// `embed`: `mel` `[MEL_WINDOW, n_mels]` → `embedding` `[1, EMBED_DIM]`.
fn embed(w: &EmbeddingWeights) -> Graph {
    let mut g = Graph::new("embed");
    let x = g.input("mel", f32s(&[MEL_WINDOW, w.n_mels]));
    let mut x = g.reshape_(x, vec![1, 1, MEL_WINDOW as i64, w.n_mels as i64]);
    for (name, c_in, c_out, stride) in [
        ("embed.conv1", 1, w.c1, 1),
        ("embed.conv2", w.c1, w.c2, 2),
        ("embed.conv3", w.c2, w.c3, 2),
    ] {
        let k = g.param(format!("{name}.weight"), f32s(&[c_out, c_in, 3, 3]));
        let y = g.conv2d(x, k, [3, 3], [stride, stride], [1, 1], [1, 1], 1);
        let b = g.param(format!("{name}.bias"), f32s(&[c_out]));
        let b = g.reshape_(b, vec![1, c_out as i64, 1, 1]);
        let y = g.add(y, b);
        x = g.relu(y);
    }
    let pooled = g.mean(x, vec![2, 3], false);
    let pooled = g.reshape_(pooled, vec![1, w.c3 as i64]);
    let out = linear(&mut g, pooled, "embed.fc", w.c3, EMBED_DIM);
    g.node_mut(out).name = Some("embedding".into());
    g.set_outputs(vec![out]);
    g
}

/// `phrase`: `embeddings` `[EMBED_HISTORY, EMBED_DIM]` → `score` `[1, 1]`.
fn phrase(w: &PhraseWeights) -> Graph {
    let mut g = Graph::new("phrase");
    let x = g.input("embeddings", f32s(&[EMBED_HISTORY, EMBED_DIM]));
    let n = EMBED_HISTORY * EMBED_DIM;
    let x = g.reshape_(x, vec![1, n as i64]);
    let h = linear(&mut g, x, "phrase.fc1", n, w.hidden);
    let h = g.relu(h);
    let logit = linear(&mut g, h, "phrase.fc2", w.hidden, 1);
    let out = g.sigmoid(logit);
    g.node_mut(out).name = Some("score".into());
    g.set_outputs(vec![out]);
    g
}

/// Export the openWakeWord directory `dir` as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let embedding = EmbeddingWeights::load(&dir.join("embedding.safetensors"))
        .context("embedding.safetensors")?;
    let keyword = std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|c| c["keyword"].as_str().map(String::from))
        .unwrap_or_else(|| "wake".into());
    let head = PhraseWeights::load(&dir.join("phrase.safetensors"), &keyword)
        .context("phrase.safetensors")?;
    let front = MelFrontend::new(MelConfig::default());
    let mut pkg = Package::new(
        &format!("openwakeword-{}", keyword.replace(' ', "-")),
        &opts.version,
        "openwakeword",
        "keyword-spotting",
        "rlx-openwakeword",
    );
    let (mg, _) = symbolic_in_lifted_many(&[(0, vec![37, 53, 20])], &[0], |n| {
        Ok(rlx_wake::export::mel(front.config(), n[0]))
    })?;
    pkg.function(mg);
    pkg.function(embed(&embedding));
    pkg.function(phrase(&head));
    pkg.dims = vec![("frames".into(), 0)];

    let mut params = rlx_wake::export::mel_params(&front);
    for (n, v) in [
        ("embed.conv1.weight", &embedding.conv1_w),
        ("embed.conv1.bias", &embedding.conv1_b),
        ("embed.conv2.weight", &embedding.conv2_w),
        ("embed.conv2.bias", &embedding.conv2_b),
        ("embed.conv3.weight", &embedding.conv3_w),
        ("embed.conv3.bias", &embedding.conv3_b),
        ("embed.fc.weight", &embedding.fc_w),
        ("embed.fc.bias", &embedding.fc_b),
        ("phrase.fc1.weight", &head.fc1_w),
        ("phrase.fc1.bias", &head.fc1_b),
        ("phrase.fc2.weight", &head.fc2_w),
        ("phrase.fc2.bias", &head.fc2_b),
    ] {
        params.insert(n.into(), v.clone());
    }
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .with_context(|| format!("no graph takes {n}"))?;
        pkg.hold_f32(n.clone(), d, &params[n]);
    }
    rlx_wake::export::detect_entry(
        &mut pkg,
        &front,
        json!({"embed": {
            "window": MEL_WINDOW,
            "step": MEL_STEP,
            "history": EMBED_HISTORY,
            "dim": EMBED_DIM,
        }}),
    )?;
    pkg.metadata("keyword", json!(keyword));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family openwakeword`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "openwakeword",
        "keyword-spotting",
        export_rlxp,
    ));
}
