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

//! Export a Piper voice as an `.rlxp` package rlx runs alone — text in,
//! speech out (`Model::synthesize`) — through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its graphs are rlx graphs symbolic in the phonemes and the frames: the
//! encoder and the decoder the native runner's split bundle (`enc_p.onnx`,
//! `flow_dec.onnx`) imported once, here, the constants the import bakes
//! from the length rebuilt from it ([`symbolic_in_lifted`]); the duration
//! predictor [`crate::sdp_graph`]; the prior `m + noise · exp(logs) ·
//! noise_scale` its own. Up to five ids the encoder's relative attention
//! is another structure: `encoder_3` … `encoder_5` are it at each. The rest of what [`crate::NativeVits`] does —
//! espeak-ng, the tokenizer, the flow noise, the length regulator — is the
//! package's program (`program/`, wasm32-wasip1), espeak-ng's English data
//! its files.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::json;

use crate::config::find_voice;
use crate::native::SPLIT_SUBDIR;

/// The symbols: the phonemes and the frames.
const PHONEMES: u32 = 0;
const FRAMES: u32 = 1;
const CHANNELS: usize = 192;
/// The most ids the encoder's relative attention (window 4) pads nothing
/// at: `encoder_3` … `encoder_5`, the encoder at each.
const SHORT: usize = 5;

/// The ONNX graph `file` imported at `len` as the rlx graph `name`, its
/// params kept the first time.
fn import(
    file: &Path,
    name: &str,
    len: usize,
    named: &[(&str, usize)],
    params: &RefCell<Option<HashMap<String, Vec<f32>>>>,
) -> Result<Graph> {
    let (g, weights) = rlx_tiny_tts::model::import_rlx_graph(file, name, len, named)?;
    params.borrow_mut().get_or_insert(weights);
    Ok(g)
}

/// `m + noise · exp(logs) · noise_scale` over `[1, 192, frames]`.
fn prior(frames: usize) -> Graph {
    let s = Shape::new(&[1, CHANNELS, frames], DType::F32);
    let mut g = Graph::new("prior");
    let m = g.input("m", s.clone());
    let logs = g.input("logs", s.clone());
    let noise = g.input("noise", s);
    let scale = g.input("noise_scale", Shape::new(&[1], DType::F32));
    let e = g.exp(logs);
    let n = g.mul(noise, e);
    let n = g.mul(n, scale);
    let z = g.add(m, n);
    g.node_mut(z).name = Some("z".into());
    g.set_outputs(vec![z]);
    g
}

/// Every file under `dir`, by its path relative to `base`.
fn files(dir: &Path, base: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            files(&p, base, out)?;
        } else {
            let rel = p.strip_prefix(base)?.to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
    Ok(())
}

/// Export the Piper voice in `dir` (`<voice>.onnx.json` and the split
/// bundle under `rlx-split/`) as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let (onnx, json_path) = find_voice(dir)?;
    let voice: serde_json::Value = serde_json::from_slice(&std::fs::read(&json_path)?)
        .with_context(|| json_path.display().to_string())?;
    let cfg = crate::config::PiperConfig::load(&json_path)?;
    let split = dir.join(SPLIT_SUBDIR);
    let name = onnx
        .file_stem()
        .and_then(|s| s.to_str())
        .context("the voice's name")?;
    let mut pkg = Package::new(name, &opts.version, "piper", "text-to-speech", "rlx-piper");
    pkg.model["license"] = json!("MIT");

    let held = RefCell::new(None);
    let (encoder, tables) = symbolic_in_lifted(PHONEMES, (8, 11, 14), 512, |n| {
        import(
            &split.join("enc_p.onnx"),
            "encoder",
            n,
            &[("phonemes", n)],
            &held,
        )
    })?;
    let mut params = held.take().context("the encoder's params")?;
    params.extend(tables);
    // Up to `window + 1` ids the relative attention pads nothing, a
    // structure of its own: the encoder at each such length, its weights
    // the same.
    let short: Vec<Graph> = (3..=SHORT)
        .map(|n| -> Result<Graph> {
            let (mut g, _) = rlx_tiny_tts::model::import_rlx_graph(
                &split.join("enc_p.onnx"),
                "encoder",
                n,
                &[("phonemes", n)],
            )?;
            g.name = format!("encoder_{n}");
            Ok(g)
        })
        .collect::<Result<_>>()?;
    let (decoder, tables) = symbolic_in_lifted(FRAMES, (20, 28, 36), 4096, |n| {
        import(
            &split.join("flow_dec.onnx"),
            "decoder",
            n,
            &[("frames", n), ("batch", 1)],
            &held,
        )
    })?;
    params.extend(held.take().context("the decoder's params")?);
    params.extend(tables);
    let (dims, dp) = crate::sdp_graph::weights(&split)?;
    params.extend(dp);
    let durations = symbolic_in(PHONEMES, (5, 7, 9), |t| {
        Ok(crate::sdp_graph::build(&dims, t))
    })?;
    let prior = symbolic_in(FRAMES, (5, 7, 9), |f| Ok(prior(f)))?;
    for g in [encoder, durations, prior, decoder]
        .into_iter()
        .chain(short)
    {
        pkg.function(g);
    }
    pkg.dims = vec![("phonemes".into(), PHONEMES), ("frames".into(), FRAMES)];
    let dims = pkg.param_dims();
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    for n in names {
        let d = dims
            .get(n)
            .cloned()
            .unwrap_or_else(|| vec![params[n].len()]);
        pkg.hold_f32(n.clone(), d, &params[n]);
    }

    let espeak = tempfile::tempdir()?;
    espeak_ng::install_bundled_language(espeak.path(), "en")
        .map_err(|e| anyhow::anyhow!("espeak-ng's English data: {e}"))?;
    let mut data = Vec::new();
    files(espeak.path(), espeak.path(), &mut data)?;
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
        &["synthesize"],
        "text-to-speech",
        json!({
            "short_encoders": SHORT,
            "phoneme_id_map": voice["phoneme_id_map"],
            "espeak_voice": cfg.espeak_voice,
            "espeak": "/package/espeak",
            "sample_rate": cfg.sample_rate,
            "noise_scale": cfg.noise_scale,
            "length_scale": cfg.length_scale,
            "noise_w": cfg.noise_w,
        }),
    );
    pkg.metadata("sample_rate", json!(cfg.sample_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("piper", "text-to-speech", export_rlxp).claims(&["piper"]));
}
