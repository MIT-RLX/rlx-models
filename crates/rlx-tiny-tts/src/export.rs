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

//! Export a TinyTTS / MeloTTS bundle as an `.rlxp` package rlx runs alone —
//! English text in, speech out (`Model::synthesize`) — through the one
//! exporter ([`rlx_core::export`]).
//!
//! Its graphs are rlx graphs: the text encoder, the duration predictor, the
//! flow and the decoder the bundle's ONNX imported once, here, symbolic in
//! the phones and the frames ([`symbolic_in_lifted`]); the frontend's
//! neural G2P (a GRU encoder-decoder for words no dictionary has, Rust in
//! rlx-inflect-nano) one step each, `g2p_encode` and `g2p_decode`. The rest
//! — the frontend's text rules and dictionaries, the VITS glue ([`crate::glue`])
//! — is the package's program (`program/`, wasm32-wasip1), the frontend's
//! files its files.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted;
use rlx_core::export::{Checkpoint, ExportOptions, ExportReport, Package, program};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use serde_json::json;

use crate::config::BundleConfig;

/// The symbols: the phones and the frames.
const PHONES: u32 = 0;
const FRAMES: u32 = 1;
/// The G2P's GRU width, and its phoneme vocabulary.
const HIDDEN: usize = 256;
const G2P_PHONEMES: usize = 74;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// A GRU cell over the embedding row `id` of `emb` from `h` `[1, H]`, the
/// weights `{p}_w_ih` … as PyTorch's (`r`, `z`, `n`).
fn gru_step(g: &mut Graph, p: &str, rows: usize) -> (NodeId, NodeId) {
    let id = g.input("id", Shape::new(&[1], DType::I64));
    let h = g.input("h", f32s(&[1, HIDDEN]));
    let emb = g.param(format!("g2p.{p}_emb"), f32s(&[rows, HIDDEN]));
    let x = g.gather_(emb, id, 0);
    let w_ih = g.param(format!("g2p.{p}_w_ih"), f32s(&[3 * HIDDEN, HIDDEN]));
    let w_hh = g.param(format!("g2p.{p}_w_hh"), f32s(&[3 * HIDDEN, HIDDEN]));
    let b_ih = g.param(format!("g2p.{p}_b_ih"), f32s(&[3 * HIDDEN]));
    let b_hh = g.param(format!("g2p.{p}_b_hh"), f32s(&[3 * HIDDEN]));
    let wt = g.transpose_(w_ih, vec![1, 0]);
    let gi = g.mm(x, wt);
    let gi = g.add(gi, b_ih);
    let wt = g.transpose_(w_hh, vec![1, 0]);
    let gh = g.mm(h, wt);
    let gh = g.add(gh, b_hh);
    let part = |g: &mut Graph, x: NodeId, k: usize| g.narrow_(x, 1, k * HIDDEN, HIDDEN);
    let (ir, iz, inn) = (part(g, gi, 0), part(g, gi, 1), part(g, gi, 2));
    let (hr, hz, hn) = (part(g, gh, 0), part(g, gh, 1), part(g, gh, 2));
    let r = g.add(ir, hr);
    let r = g.sigmoid(r);
    let z = g.add(iz, hz);
    let z = g.sigmoid(z);
    let rn = g.mul(r, hn);
    let n = g.add(inn, rn);
    let n = g.tanh(n);
    // h' = (1 − z)·n + z·h = n + z·(h − n)
    let d = g.sub(h, n);
    let zd = g.mul(z, d);
    let out = g.add(n, zd);
    (out, h)
}

/// The G2P encoder's step: `id`, `h` → `h`.
fn g2p_encode() -> Graph {
    let mut g = Graph::new("g2p_encode");
    let (h, _) = gru_step(&mut g, "enc", 29);
    g.node_mut(h).name = Some("h".into());
    g.set_outputs(vec![h]);
    g
}

/// The G2P decoder's step: `id` (the last phoneme), `h` → `h`, `logits`.
fn g2p_decode() -> Graph {
    let mut g = Graph::new("g2p_decode");
    let (h, _) = gru_step(&mut g, "dec", G2P_PHONEMES);
    let fc_w = g.param("g2p.fc_w", f32s(&[G2P_PHONEMES, HIDDEN]));
    let fc_b = g.param("g2p.fc_b", f32s(&[G2P_PHONEMES]));
    let wt = g.transpose_(fc_w, vec![1, 0]);
    let logits = g.mm(h, wt);
    let logits = g.add(logits, fc_b);
    g.node_mut(h).name = Some("h".into());
    g.node_mut(logits).name = Some("logits".into());
    g.set_outputs(vec![h, logits]);
    g
}

/// The G2P checkpoint's tensors, `g2p.`-named.
fn g2p_params(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut out = HashMap::new();
    for (name, t) in st.tensors() {
        anyhow::ensure!(
            t.dtype() == safetensors::Dtype::F32,
            "{name}: the G2P is f32"
        );
        let v = t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        out.insert(format!("g2p.{name}"), v);
    }
    Ok(out)
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

/// Export the TinyTTS / MeloTTS bundle in `dir` (`config.json`, `onnx/`,
/// `frontend/`) as the package `out`, of `family`.
pub fn export_family(
    dir: &Path,
    out: &Path,
    opts: &ExportOptions,
    family: &str,
) -> Result<ExportReport> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml");
    export_with(
        dir,
        out,
        opts,
        family,
        "rlx-tiny-tts",
        |_, _| Ok(json!({})),
        &manifest,
    )
}

/// What [`export_family`] writes, for a model built on TinyTTS (OpenVoice):
/// `extend` adds its functions and params and gives its program's config
/// besides TinyTTS'; `program` is its program crate (a `voice.rs` user).
pub fn export_with(
    dir: &Path,
    out: &Path,
    opts: &ExportOptions,
    family: &str,
    exported_by: &str,
    extend: impl FnOnce(&mut Package, &mut HashMap<String, Vec<f32>>) -> Result<serde_json::Value>,
    program: &Path,
) -> Result<ExportReport> {
    let ck = Checkpoint::open(dir)?;
    let cfg = BundleConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let onnx = dir.join("onnx");
    let mut pkg = Package::for_checkpoint(&ck, opts, family, "text-to-speech", exported_by);

    let held = RefCell::new(HashMap::new());
    let mut params = HashMap::new();
    for (name, sym) in [
        ("text_encoder", PHONES),
        ("duration_predictor", PHONES),
        ("flow", FRAMES),
        ("decoder", FRAMES),
    ] {
        let (g, tables) = symbolic_in_lifted(sym, (37, 45, 61), 8192, |n| {
            let (g, weights) =
                crate::model::import_rlx_graph(&onnx.join(format!("{name}.onnx")), name, n, &[])?;
            let mut h = held.borrow_mut();
            for (k, v) in weights {
                h.entry(k).or_insert(v);
            }
            Ok(g)
        })
        .with_context(|| name.to_string())?;
        params.extend(tables);
        pkg.function(g);
    }
    params.extend(held.take());
    pkg.function(g2p_encode());
    pkg.function(g2p_decode());
    let frontend = dir.join("frontend");
    params.extend(g2p_params(&frontend.join("g2p_checkpoint.safetensors"))?);
    pkg.dims = vec![("phones".into(), PHONES), ("frames".into(), FRAMES)];
    let extra = extend(&mut pkg, &mut params)?;
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

    // The frontend's files (its G2P's weights are the graphs' params).
    let mut data = Vec::new();
    files(&frontend, &frontend, &mut data)?;
    for (rel, p) in data {
        if rel.ends_with(".safetensors") {
            continue;
        }
        pkg.sidecar(
            &format!("frontend/{rel}"),
            "application/octet-stream",
            std::fs::read(p)?,
        );
    }

    let module = program::build_for(program, "wasm32-wasip1")?;
    let mut config = json!({
            "add_blank": cfg.add_blank,
            "speaker": cfg.default_speaker(),
            "sample_rate": cfg.sample_rate,
            "channels": cfg.inter_channels,
            "noise_scale": cfg.noise_scale,
            "length_scale": cfg.length_scale,
            "seed": crate::model::InferOpts::from_config(&cfg).seed,
    });
    if let (Some(c), Some(e)) = (config.as_object_mut(), extra.as_object()) {
        c.extend(e.clone());
    }
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize"],
        "text-to-speech",
        config,
    );
    pkg.metadata("sample_rate", json!(cfg.sample_rate));
    pkg.write(out, opts)
}

/// [`export_family`] as TinyTTS.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_family(dir, out, opts, "tiny-tts")
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(
        Exporter::new("tiny-tts", "text-to-speech", export_rlxp).claims(&["tinytts", "tiny-tts"]),
    );
}
