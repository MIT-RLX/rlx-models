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

//! Export an Inflect-Nano bundle (`config.json`, `acoustic.safetensors`,
//! `vocoder.safetensors`, `frontend/`) as an `.rlxp` package rlx runs
//! alone: text in, 24 kHz speech out.
//!
//! Its graphs are [`crate::acoustic`] and the vocoder: `encode` (the
//! phone, tone and language embeddings, the speaker projection, the encoder
//! blocks, the prosody heads; the conditioned rows and their pitch, the
//! scales inputs), symbolic in the tokens; `frames` (the length-regulated
//! rows gathered at each frame's token and its neighbours, the frame
//! projection, the local context, the absolute position, the pitch
//! projection, the decoder blocks, the bidirectional GRU, the mel head and
//! postnet), symbolic in the tokens and the frames; `vocoder`
//! ([`crate::graph::vocoder_hir`]), symbolic in the frames. The program
//! (`program/`, wasm32-wasip1) is the crate's text frontend
//! ([`crate::frontend`], its assets under `frontend/`), the durations'
//! rounding, the frame layout and the output normalization.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_core::export::graph::{broadcast_params, symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_core::flow_util::{built_from_hir, graph_from_built};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::config::{AcousticConfig, BundleConfig};
use crate::weights::Weights;

/// The symbols: the tokens, the frames.
const TOKENS: u32 = 0;
const FRAMES: u32 = 1;
const LN_EPS: f32 = 1e-5;

type Built = (Graph, HashMap<String, Vec<f32>>);

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn ids(d: &[usize]) -> Shape {
    Shape::new(d, DType::I32)
}

/// The acoustic model's graph builder: params named `ac.<tensor>`.
struct Ac<'a> {
    g: Graph,
    w: &'a Weights,
    cfg: &'a AcousticConfig,
    params: HashMap<String, Vec<f32>>,
}

impl Ac<'_> {
    fn p(&mut self, key: &str, shape: &[usize]) -> Result<NodeId> {
        let data = self.w.data(key).with_context(|| key.to_string())?.to_vec();
        let name = format!("ac.{key}");
        self.params.insert(name.clone(), data);
        Ok(self.g.param(name, f32s(shape)))
    }

    /// `x · Wᵀ + b`, `x` `[rows, inp]`.
    fn linear(&mut self, x: NodeId, key: &str, out: usize, inp: usize) -> Result<NodeId> {
        let w = self.p(&format!("{key}.weight"), &[out, inp])?;
        let w = self.g.transpose_(w, vec![1, 0]);
        let y = self.g.mm(x, w);
        let b = self.p(&format!("{key}.bias"), &[out])?;
        let b = self.g.reshape_(b, vec![1, out as i64]);
        Ok(self.g.add(y, b))
    }

    fn layer_norm(&mut self, x: NodeId, key: &str) -> Result<NodeId> {
        let h = self.cfg.hidden;
        let g = self.p(&format!("{key}.weight"), &[h])?;
        let b = self.p(&format!("{key}.bias"), &[h])?;
        Ok(self.g.ln(x, g, b, LN_EPS))
    }

    /// LayerNorm → Linear(h, mid) → SiLU → Linear(mid, out).
    fn mlp_head(&mut self, x: NodeId, key: &str, mid: usize, out: usize) -> Result<NodeId> {
        let h = self.cfg.hidden;
        let y = self.layer_norm(x, &format!("{key}.0"))?;
        let y = self.linear(y, &format!("{key}.1"), mid, h)?;
        let y = self.g.silu(y);
        self.linear(y, &format!("{key}.3"), out, mid)
    }

    /// Linear(inp, mid) → SiLU → Linear(mid, out).
    fn mlp_proj(
        &mut self,
        x: NodeId,
        key: &str,
        inp: usize,
        mid: usize,
        out: usize,
    ) -> Result<NodeId> {
        let y = self.linear(x, &format!("{key}.0"), mid, inp)?;
        let y = self.g.silu(y);
        self.linear(y, &format!("{key}.2"), out, mid)
    }

    /// A 1-D convolution over `x` `[1, c_in, t, 1]` (weight `[c_out, c_in /
    /// groups, k]`), its bias added.
    fn conv(
        &mut self,
        x: NodeId,
        key: &str,
        (c_out, c_in_g, k): (usize, usize, usize),
        groups: usize,
    ) -> Result<NodeId> {
        let w = self.p(&format!("{key}.weight"), &[c_out, c_in_g, k, 1])?;
        let y = self
            .g
            .conv2d(x, w, [k, 1], [1, 1], [k / 2, 0], [1, 1], groups);
        let b = self.p(&format!("{key}.bias"), &[c_out])?;
        let b = self.g.reshape_(b, vec![1, c_out as i64, 1, 1]);
        Ok(self.g.add(y, b))
    }

    /// One ConvFFN block over `x` `[t, h]`.
    fn conv_ffn(&mut self, x: NodeId, key: &str, t: usize, ff_mid: usize) -> Result<NodeId> {
        let (h, k) = (self.cfg.hidden, self.cfg.kernel_size);
        let y = self.layer_norm(x, &format!("{key}.norm1"))?;
        let y = self.g.transpose_(y, vec![1, 0]);
        let y = self.g.reshape_(y, vec![1, h as i64, t as i64, 1]);
        let dd = self.conv(y, &format!("{key}.depth"), (2 * h, 1, k), h)?;
        let a = self.g.narrow_(dd, 1, 0, h);
        let gate = self.g.narrow_(dd, 1, h, h);
        let gate = self.g.sigmoid(gate);
        let gated = self.g.mul(a, gate);
        let pt = self.conv(gated, &format!("{key}.point"), (h, h, 1), 1)?;
        let pt = self.g.reshape_(pt, vec![h as i64, t as i64]);
        let pt = self.g.transpose_(pt, vec![1, 0]);
        let x = self.g.add(x, pt);
        let y = self.layer_norm(x, &format!("{key}.norm2"))?;
        let y = self.linear(y, &format!("{key}.ff.0"), ff_mid, h)?;
        let y = self.g.silu(y);
        let y = self.linear(y, &format!("{key}.ff.3"), h, ff_mid)?;
        Ok(self.g.add(x, y))
    }

    fn table(&mut self, key: &str, rows: usize, at: NodeId) -> Result<NodeId> {
        let width = self.w.shape(key)?[1];
        let t = self.p(key, &[rows, width])?;
        Ok(self.g.gather_(t, at, 0))
    }

    fn finish(mut self, outs: Vec<(NodeId, &str)>, name: &str) -> Built {
        for &(id, n) in &outs {
            self.g.node_mut(id).name = Some(n.into());
        }
        self.g.set_outputs(outs.iter().map(|o| o.0).collect());
        self.g.name = name.into();
        (self.g, self.params)
    }
}

/// `encode` at `s` tokens: `phone`, `tone`, `lang` `[s]` and `speaker`
/// `[1]` (i32), `energy_scale` and `pitch_scale` `[1]` → `log_duration`
/// `[s, 1]`, `conditioned` `[s, h]` and `pitch` `[s, 2]` (scaled, its
/// voicing clamped to [0, 1]).
fn encode(w: &Weights, cfg: &AcousticConfig, s: usize) -> Result<Built> {
    let h = cfg.hidden;
    let mut a = Ac {
        g: Graph::new("encode"),
        w,
        cfg,
        params: HashMap::new(),
    };
    let phone = a.g.input("phone", ids(&[s]));
    let tone = a.g.input("tone", ids(&[s]));
    let lang = a.g.input("lang", ids(&[s]));
    let speaker = a.g.input("speaker", ids(&[1]));
    let escale = a.g.input("energy_scale", f32s(&[1]));
    let pscale = a.g.input("pitch_scale", f32s(&[1]));
    let x = a.table("phone.weight", cfg.vocab_size, phone)?;
    let t = a.table("tone.weight", cfg.tone_size, tone)?;
    let l = a.table("lang.weight", cfg.lang_size, lang)?;
    let x = a.g.add(x, t);
    let x = a.g.add(x, l);
    let spk = a.table("speaker.weight", cfg.speaker_count, speaker)?;
    let spk = a.linear(spk, "speaker_proj", h, cfg.speaker_dim)?;
    let mut x = a.g.add(x, spk);
    for i in 0..cfg.encoder_layers {
        x = a.conv_ffn(x, &format!("encoder.{i}"), s, h * 4)?;
    }
    let log_dur = a.mlp_head(x, "duration_head", h, 1)?;
    let energy = a.mlp_head(x, "energy_head", h / 2, 1)?;
    let bright = a.mlp_head(x, "bright_head", h / 2, 1)?;
    let pitch = a.mlp_head(x, "pitch_head", h, 2)?;
    let energy = a.g.mul(energy, escale);
    let e = a.linear(energy, "energy_proj", h, 1)?;
    let b = a.linear(bright, "bright_proj", h, 1)?;
    let cond = a.g.add(x, e);
    let cond = a.g.add(cond, b);
    let p0 = a.g.narrow_(pitch, 1, 0, 1);
    let p0 = a.g.mul(p0, pscale);
    let p1 = a.g.narrow_(pitch, 1, 1, 1);
    let p1 = a.g.clamp_(p1, 0.0, 1.0);
    let pitch = a.g.concat_(vec![p0, p1], 1);
    Ok(a.finish(
        vec![
            (log_dur, "log_duration"),
            (cond, "conditioned"),
            (pitch, "pitch"),
        ],
        "encode",
    ))
}

/// `frames` at `s` tokens and `t` frames: `conditioned` `[s, h]` and
/// `pitch` `[s, 2]`, each frame's `token`, `prev`, `next` and absolute
/// position bin `abs` `[t]` (i32) and its `meta` `[t, 8]` → `mel` `[1,
/// n_mels, t, 1]`.
fn frames(w: &Weights, cfg: &AcousticConfig, s: usize, t: usize) -> Result<Built> {
    let h = cfg.hidden;
    let mut a = Ac {
        g: Graph::new("frames"),
        w,
        cfg,
        params: HashMap::new(),
    };
    let cond = a.g.input("conditioned", f32s(&[s, h]));
    let pitch = a.g.input("pitch", f32s(&[s, 2]));
    let token = a.g.input("token", ids(&[t]));
    let prev = a.g.input("prev", ids(&[t]));
    let next = a.g.input("next", ids(&[t]));
    let abs = a.g.input("abs", ids(&[t]));
    let meta = a.g.input("meta", f32s(&[t, 8]));
    let cur = a.g.gather_(cond, token, 0);
    let fp = a.mlp_proj(meta, "frame_proj", 8, h, h)?;
    let mut x = a.g.add(cur, fp);
    let before = a.g.gather_(cond, prev, 0);
    let after = a.g.gather_(cond, next, 0);
    let ctx = a.g.concat_(vec![before, cur, after], 1);
    let lc = a.linear(ctx, "local_ctx.0", 2 * h, 3 * h)?;
    let lc = a.g.silu(lc);
    let lc = a.linear(lc, "local_ctx.2", h, 2 * h)?;
    x = a.g.add(x, lc);
    let pos = a.table("abs_frame.weight", cfg.abs_frame_bins, abs)?;
    x = a.g.add(x, pos);
    if cfg.use_frame_pitch {
        let pf = a.g.gather_(pitch, token, 0);
        let pp = a.mlp_proj(pf, "pitch_proj", 2, h, h)?;
        x = a.g.add(x, pp);
    }
    let ff_mid = h * cfg.decoder_ff_mult;
    for i in 0..cfg.decoder_layers {
        x = a.conv_ffn(x, &format!("decoder.{i}"), t, ff_mid)?;
    }
    // The bidirectional GRU, its directions' weights packed.
    let hid = h / 2;
    let packed = |names: [&str; 2]| -> Result<Vec<f32>> {
        let mut v = Vec::new();
        for n in names {
            v.extend_from_slice(a.w.data(n).with_context(|| n.to_string())?);
        }
        Ok(v)
    };
    let gru: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
        (
            "ac.frame_gru.w_ih".into(),
            packed(["frame_gru.weight_ih_l0", "frame_gru.weight_ih_l0_reverse"])?,
            vec![6 * hid, h],
        ),
        (
            "ac.frame_gru.w_hh".into(),
            packed(["frame_gru.weight_hh_l0", "frame_gru.weight_hh_l0_reverse"])?,
            vec![6 * hid, hid],
        ),
        (
            "ac.frame_gru.b_ih".into(),
            packed(["frame_gru.bias_ih_l0", "frame_gru.bias_ih_l0_reverse"])?,
            vec![6 * hid],
        ),
        (
            "ac.frame_gru.b_hh".into(),
            packed(["frame_gru.bias_hh_l0", "frame_gru.bias_hh_l0_reverse"])?,
            vec![6 * hid],
        ),
    ];
    let mut gin = vec![a.g.reshape_(x, vec![1, t as i64, h as i64])];
    for (n, v, d) in gru {
        a.params.insert(n.clone(), v);
        gin.push(a.g.param(n, f32s(&d)));
    }
    let g = a.g.add_node(
        Op::Gru {
            hidden_size: hid,
            num_layers: 1,
            bidirectional: true,
            carry: false,
        },
        gin,
        f32s(&[1, t, 2 * hid]),
    );
    let g = a.g.reshape_(g, vec![t as i64, h as i64]);
    let x = a.g.add(x, g);
    let mel = a.mlp_head(x, "mel_head", h, cfg.n_mels)?;
    let mel = a.g.transpose_(mel, vec![1, 0]);
    let mel = a.g.reshape_(mel, vec![1, cfg.n_mels as i64, t as i64, 1]);
    let p = a.conv(mel, "postnet.0", (h, cfg.n_mels, 5), 1)?;
    let p = a.g.tanh(p);
    let p = a.conv(p, "postnet.2", (h, h, 5), 1)?;
    let p = a.g.tanh(p);
    let p = a.conv(p, "postnet.4", (cfg.n_mels, h, 5), 1)?;
    let scale = a.g.add_node(
        Op::Constant {
            data: cfg.postnet_scale.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let p = a.g.mul(p, scale);
    let mel = a.g.add(mel, p);
    Ok(a.finish(vec![(mel, "mel")], "frames"))
}

fn vocoder(w: &Weights, cfg: &BundleConfig, t: usize) -> Result<Built> {
    let (hir, params) = crate::graph::vocoder_hir(w, &cfg.vocoder, t);
    let (mut g, params) = graph_from_built(built_from_hir(hir, params)?)?;
    let out = g.outputs[0];
    g.node_mut(out).name = Some("audio".into());
    g.name = "vocoder".into();
    Ok((g, params))
}

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

/// Export the Inflect-Nano bundle in `dir` as the package `out`.
pub fn export_rlxp(dir: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let cfg = BundleConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).context("config.json")?,
    )?;
    let ac = Weights::load(&dir.join("acoustic.safetensors"))?;
    let voc = Weights::load(&dir.join("vocoder.safetensors"))?;
    let acfg = &cfg.acoustic;
    let mut pkg = Package::new(
        "inflect-nano-v1",
        &opts.version,
        "inflect-nano",
        "text-to-speech",
        "rlx-inflect-nano",
    );
    pkg.model["license"] = json!("apache-2.0");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let (g, _) = symbolic_in_lifted_many(&[(TOKENS, vec![37, 53, 20])], &[0], |n| {
        Ok(encode(&ac, acfg, n[0])?.0)
    })?;
    pkg.function(g);
    params.extend(encode(&ac, acfg, 20)?.1);
    let (g, _) = symbolic_in_lifted_many(
        &[(TOKENS, vec![9, 13, 7]), (FRAMES, vec![37, 53, 20])],
        &[0, 0],
        |n| Ok(frames(&ac, acfg, n[0], n[1])?.0),
    )?;
    pkg.function(g);
    params.extend(frames(&ac, acfg, 7, 20)?.1);
    // Its zero-insertion blocks are params as long as the frames: held as
    // one slice, expanded.
    let g = symbolic_in(FRAMES, (37, 53, 20), |t| Ok(vocoder(&voc, &cfg, t)?.0))?;
    let (at, mut vp) = vocoder(&voc, &cfg, 20)?;
    let g = broadcast_params(&g, &at, &mut vp, &[])?;
    pkg.function(g);
    params.extend(vp);
    pkg.dims = vec![("tokens".into(), TOKENS), ("frames".into(), FRAMES)];
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

    // The text frontend's assets, and the program.
    let mut data = Vec::new();
    let front = dir.join("frontend");
    files(&front, &front, &mut data)?;
    for (rel, p) in data {
        pkg.sidecar(
            &format!("frontend/{rel}"),
            "application/octet-stream",
            std::fs::read(p)?,
        );
    }
    let opts_default = crate::config::InferOpts::default();
    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )
    .context("build Inflect-Nano's program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize"],
        "text-to-speech",
        json!({
            "frontend": "/package/frontend",
            "add_blank": cfg.add_blank,
            "speakers": cfg.speakers,
            "speaker": cfg.default_speaker(),
            "tone_size": acfg.tone_size,
            "lang_size": acfg.lang_size,
            "max_frames": acfg.max_frames,
            "abs_frame_bins": acfg.abs_frame_bins,
            "min_duration": opts_default.min_duration,
            "max_duration": opts_default.max_duration,
            "sample_rate": cfg.sample_rate,
        }),
    );
    pkg.metadata("sample_rate", json!(cfg.sample_rate));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]),
/// taken by name (`--family inflect-nano`).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("inflect-nano", "text-to-speech", export_rlxp));
}
