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

//! Export a [`WakeCnn`](crate::WakeCnn) wake word (nanowakeword, porcupine,
//! voxrt: one weight layout) as an `.rlxp` package rlx runs alone: 16 kHz
//! audio in, a score per 80 ms chunk and the detections out.
//!
//! Its graphs are the crate's: `mel` (the [`MelFrontend`](crate::MelFrontend)
//! frames, 400 samples every 160, windowed, their power through the mel
//! filters, `ln / 10 + 2`; symbolic in the frames) and `score_{n}` (the CNN over
//! a window of `n` mel frames, one graph each `n`: three convolutions, the time
//! mean, two linears, the sigmoid). The program (`program/`) streams the
//! audio as the engines do — `chunk` samples a push, each push scoring the
//! last `window` frames (`score_{n}`) once it has eight, none when it added no frame —
//! and fires at `threshold`, `cooldown_ms` apart (`options.threshold`,
//! `options.cooldown_ms`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::symbolic_in_lifted_many;
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::cnn::{WakeCnnConfig, WakeCnnWeights};
use crate::mel::{MelConfig, MelFrontend, OWW_CHUNK_SAMPLES};

/// The mel's frames, its one symbol.
const FRAMES: u32 = 0;
/// Mel frames the CNN keeps ([`crate::WakeCnn`]), and the fewest it scores.
const KEEP: usize = 40;
const LEAST: usize = 8;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn constant(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

/// `mel` at `t` frames: `signal` `[hop · (t + ⌈win / hop⌉ − 1)]` (the
/// samples, zeros past the last frame's end) → `mel` `[t, n_mels]`.
pub fn mel(cfg: &MelConfig, t: usize) -> Graph {
    let (hop, n_fft, n_mels) = (cfg.hop_length, cfg.n_fft, cfg.n_mels);
    let n_freq = n_fft / 2 + 1;
    let r = n_fft.div_ceil(hop);
    let mut g = Graph::new("mel");
    let x = g.input("signal", f32s(&[hop * (t + r - 1)]));
    let blocks = g.reshape_(x, vec![(t + r - 1) as i64, hop as i64]);
    let parts: Vec<NodeId> = (0..r).map(|j| g.narrow_(blocks, 0, j, t)).collect();
    let wide = g.concat_(parts, 1);
    let frames = g.narrow_(wide, 1, 0, n_fft);
    let bc = g.param("stft.cos", f32s(&[n_fft, n_freq]));
    let bs = g.param("stft.sin", f32s(&[n_fft, n_freq]));
    let re = g.mm(frames, bc);
    let im = g.mm(frames, bs);
    let re2 = g.mul(re, re);
    let im2 = g.mul(im, im);
    let power = g.add(re2, im2);
    let fb = g.param("mel.filters", f32s(&[n_freq, n_mels]));
    let m = g.mm(power, fb);
    let m = g.clamp_(m, 1e-10, f32::MAX);
    let ln = g.log(m);
    let tenth = constant(&mut g, 0.1);
    let scaled = g.mul(ln, tenth);
    let two = constant(&mut g, 2.0);
    let out = g.add(scaled, two);
    g.node_mut(out).name = Some("mel".into());
    g.set_outputs(vec![out]);
    g
}

/// One convolution over `[1, c_in, 1, t]` (kernel `k`, `stride`, padding
/// `k / 2`), its bias added and ReLU'd; the output and its length.
fn conv(
    g: &mut Graph,
    x: NodeId,
    name: &str,
    (c_in, c_out, k): (usize, usize, usize),
    stride: usize,
    t: usize,
) -> (NodeId, usize) {
    let w = g.param(format!("{name}.weight"), f32s(&[c_out, c_in, 1, k]));
    let y = g.conv2d(x, w, [1, k], [1, stride], [0, k / 2], [1, 1], 1);
    let b = g.param(format!("{name}.bias"), f32s(&[c_out]));
    let b = g.reshape_(b, vec![1, c_out as i64, 1, 1]);
    let y = g.add(y, b);
    (g.relu(y), (t + 2 * (k / 2) - k) / stride + 1)
}

/// `score_{t}`: `mel` `[t, n_mels]` → `score` `[1, 1]`.
fn score(cfg: &WakeCnnConfig, t: usize) -> Graph {
    let mut g = Graph::new("score");
    let x = g.input("mel", f32s(&[t, cfg.n_mels]));
    let x = g.transpose_(x, vec![1, 0]);
    let x = g.reshape_(x, vec![1, cfg.n_mels as i64, 1, t as i64]);
    let (x, t1) = conv(&mut g, x, "conv1", (cfg.n_mels, cfg.c1, cfg.k), 1, t);
    let (x, t2) = conv(&mut g, x, "conv2", (cfg.c1, cfg.c2, cfg.k), 2, t1);
    let (x, _) = conv(&mut g, x, "conv3", (cfg.c2, cfg.c3, cfg.k), 2, t2);
    let pooled = g.mean(x, vec![3], false);
    let pooled = g.reshape_(pooled, vec![1, cfg.c3 as i64]);
    let linear = |g: &mut Graph, x: NodeId, name: &str, i: usize, o: usize| {
        let w = g.param(format!("{name}.weight"), f32s(&[o, i]));
        let w = g.transpose_(w, vec![1, 0]);
        let y = g.mm(x, w);
        let b = g.param(format!("{name}.bias"), f32s(&[o]));
        let b = g.reshape_(b, vec![1, o as i64]);
        g.binary(BinaryOp::Add, y, b, f32s(&[1, o]))
    };
    let h = linear(&mut g, pooled, "fc1", cfg.c3, cfg.hidden);
    let h = g.relu(h);
    let logit = linear(&mut g, h, "fc2", cfg.hidden, 1);
    let out = g.sigmoid(logit);
    g.node_mut(out).name = Some("score".into());
    g.set_outputs(vec![out]);
    g
}

/// The mel's params: the windowed DFT bases and the filters, transposed.
pub fn mel_params(front: &MelFrontend) -> HashMap<String, Vec<f32>> {
    let cfg = front.config();
    let mut window = vec![0f32; cfg.n_fft];
    window[..cfg.win_length].copy_from_slice(front.window());
    let mut p = rlx_core::export::audio::stft_bases(cfg.n_fft, &window);
    let n_freq = cfg.n_fft / 2 + 1;
    let fb = front.filters();
    let mut t = vec![0f32; n_freq * cfg.n_mels];
    for m in 0..cfg.n_mels {
        for k in 0..n_freq {
            t[k * cfg.n_mels + m] = fb[m * n_freq + k];
        }
    }
    p.insert("mel.filters".into(), t);
    p
}

/// Export the wake CNN `weights` (a `.safetensors` of [`WakeCnnWeights`])
/// as the package `out`, of `family`.
pub fn export_family(
    family: &str,
    weights: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    let w = WakeCnnWeights::load(weights)
        .with_context(|| format!("{}: not a wake CNN", weights.display()))?;
    let cfg = w.cfg.clone();
    let front = MelFrontend::new(MelConfig::default());
    let name = weights
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("wake");
    let mut pkg = Package::new(
        &format!("{family}-{name}"),
        &opts.version,
        family,
        "keyword-spotting",
        "rlx-wake",
    );
    let (mg, _) = symbolic_in_lifted_many(&[(FRAMES, vec![37, 53, 20])], &[0], |n| {
        Ok(mel(front.config(), n[0]))
    })?;
    pkg.function(mg);
    // One graph a window length: its strided convolutions' lengths are
    // quotients no symbol fits.
    for w in LEAST..=KEEP {
        let mut g = score(&cfg, w);
        g.name = format!("score_{w}");
        pkg.function(g);
    }
    pkg.dims = vec![("frames".into(), FRAMES)];

    let mut params = mel_params(&front);
    for (n, v) in [
        ("conv1.weight", &w.conv1_w),
        ("conv1.bias", &w.conv1_b),
        ("conv2.weight", &w.conv2_w),
        ("conv2.bias", &w.conv2_b),
        ("conv3.weight", &w.conv3_w),
        ("conv3.bias", &w.conv3_b),
        ("fc1.weight", &w.fc1_w),
        ("fc1.bias", &w.fc1_b),
        ("fc2.weight", &w.fc2_w),
        ("fc2.bias", &w.fc2_b),
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

    detect_entry(&mut pkg, &front, json!({"window": KEEP, "least": LEAST}))?;
    pkg.write(out, opts)
}

/// The `detect` entry: the wake program over `front`'s frames and the
/// engines' defaults, `scoring` saying what scores a push (`{window,
/// least}`: the CNN's `score_{n}`; `{embed: {window, step, history}}`:
/// openWakeWord's `embed` and `phrase`).
pub fn detect_entry(
    pkg: &mut Package,
    front: &MelFrontend,
    scoring: serde_json::Value,
) -> Result<()> {
    let wake = crate::WakeConfig::default();
    let mc = front.config();
    let module = program::build(&Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"))
        .context("build the wake word's program")?;
    let mut config = json!({
        "sample_rate": mc.sample_rate,
        "hop": mc.hop_length,
        "win": mc.win_length,
        "n_mels": mc.n_mels,
        "chunk": OWW_CHUNK_SAMPLES,
        "threshold": wake.threshold,
        "cooldown_ms": wake.cooldown_ms,
    });
    if let (Some(c), Some(s)) = (config.as_object_mut(), scoring.as_object()) {
        c.extend(s.clone());
    }
    program::entry(
        pkg,
        "detect",
        module,
        &["detect", "run"],
        "keyword-spotting",
        config,
    );
    pkg.metadata("sample_rate", json!(mc.sample_rate));
    Ok(())
}

fn nanowakeword(w: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_family("nanowakeword", w, out, opts)
}

fn porcupine(w: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_family("porcupine", w, out, opts)
}

fn voxrt(w: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    export_family("voxrt", w, out, opts)
}

/// Register the wake CNN's exporters ([`rlx_core::export::register_exporter`])
/// under each engine's name (`--family nanowakeword|porcupine|voxrt`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new(
        "nanowakeword",
        "keyword-spotting",
        nanowakeword,
    ));
    register_exporter(Exporter::new("porcupine", "keyword-spotting", porcupine));
    register_exporter(Exporter::new("voxrt", "keyword-spotting", voxrt));
}
