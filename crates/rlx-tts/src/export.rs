//! Export RLX TTS (FastSpeech2 + WaveRNN) as an `.rlxp` package rlx runs
//! alone: text in, 24 kHz speech out, through the one exporter
//! ([`rlx_core::export`]).
//!
//! Its graphs, as [`crate::RlxTts`] computes on its portable path: `encode`
//! (the phones through the prenet, the positions, four FFT blocks; the
//! hidden states and the duration, pitch and energy predictors' outputs —
//! symbolic in the phones); `decode` (the pitch and energy rows added, each
//! frame's phone gathered, the dilated conv decoder: the mel — symbolic in
//! the phones and the frames); `wavernn` (one mel frame's 120 GRU steps as
//! an `Op::Scan`: the mel projection, the two sub-stream GRUs, their heads,
//! each sample the Gumbel-max of its logits and the given noise; the state
//! carried frame to frame). The program (`program/`) runs the crate's text
//! frontend (its rule files and G2P network as files), turns the predictors
//! into durations and pitch / energy buckets, draws the noise (the portable
//! BNNS AES-CTR stream the runner's `WaveRnnRng::Bnns` draws) and applies
//! the output post.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rlx_core::export::graph::{symbolic_in, symbolic_in_lifted_many};
use rlx_core::export::{ExportOptions, ExportReport, Package, program};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::json;

use crate::RlxTts;
use crate::weights::Weights;

const PHONES: u32 = 0;
const FRAMES: u32 = 1;
const HIDDEN: usize = 256;
const N_MELS: usize = 80;
const PHONE_VOCAB: usize = 84;
const BUCKETS: usize = 256;
/// Positional-encoding rows: past 256 phones the last row repeats.
const POS_ROWS: usize = 4096;
const DECODER_DILATIONS: [usize; 12] = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32];
const WR_HIDDEN: usize = 448;
const WR_GATES: usize = 1344;
const SUB: usize = 224;
const SUB_GATES: usize = 672;
const STEPS: usize = 120;
/// The scan's carry: the hidden state, the last fine sample, and the step's
/// coarse and fine samples.
const CARRY: usize = WR_HIDDEN + 3;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    )
}

/// The graphs' params, named after the bundle's tensors (`enc.` / `dec.` /
/// `wr.` before them).
struct Params<'a> {
    w: &'a Weights,
    prefix: &'static str,
    held: HashMap<String, Vec<f32>>,
}

impl Params<'_> {
    fn p(&mut self, g: &mut Graph, name: &str, shape: &[usize]) -> Result<NodeId> {
        let full = format!("{}{name}", self.prefix);
        let data = self.w.data(name).with_context(|| format!("no {name}"))?;
        anyhow::ensure!(
            data.len() == shape.iter().product::<usize>(),
            "{name}: {} values for {shape:?}",
            data.len()
        );
        self.held.insert(full.clone(), data.to_vec());
        Ok(g.param(full, f32s(shape)))
    }
}

/// `x` `[1, cin, t, 1]` through a same-padded conv `[cout, cin, k]` (+ bias).
#[allow(clippy::too_many_arguments)]
fn conv(
    g: &mut Graph,
    p: &mut Params,
    x: NodeId,
    w: &str,
    b: Option<&str>,
    cout: usize,
    cin: usize,
    k: usize,
    dil: usize,
) -> Result<NodeId> {
    let wp = p.p(g, w, &[cout, cin, k, 1])?;
    let pad = dil * (k - 1) / 2;
    let y = g.conv2d(x, wp, [k, 1], [1, 1], [pad, 0], [dil, 1], 1);
    Ok(match b {
        Some(b) => {
            let bp = p.p(g, b, &[cout])?;
            let bp = g.reshape_(bp, vec![1, cout as i64, 1, 1]);
            g.add(y, bp)
        }
        None => y,
    })
}

fn to_cf(g: &mut Graph, x: NodeId, t: usize, c: usize) -> NodeId {
    let x = g.transpose_(x, vec![1, 0]);
    g.reshape_(x, vec![1, c as i64, t as i64, 1])
}

fn to_tm(g: &mut Graph, x: NodeId, t: usize, c: usize) -> NodeId {
    let x = g.reshape_(x, vec![c as i64, t as i64]);
    g.transpose_(x, vec![1, 0])
}

fn ln(g: &mut Graph, p: &mut Params, x: NodeId, idx: usize) -> Result<NodeId> {
    let gamma = p.p(g, &format!("encoder.ln{idx}.gamma"), &[HIDDEN])?;
    let beta = p.p(g, &format!("encoder.ln{idx}.beta"), &[HIDDEN])?;
    Ok(g.ln(x, gamma, beta, 1e-6))
}

/// `encode` at `t` phones.
fn encode(w: &Weights, t: usize) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut p = Params {
        w,
        prefix: "enc.",
        held: HashMap::new(),
    };
    let mut g = Graph::new("encode");
    let phones = g.input("phones", Shape::new(&[t], DType::I32));
    let table = p.p(&mut g, "phone_embed.weight", &[PHONE_VOCAB, HIDDEN])?;
    let x = g.gather_(table, phones, 0);
    let x = to_cf(&mut g, x, t, HIDDEN);
    let x = conv(
        &mut g,
        &mut p,
        x,
        "encoder.prenet.weight",
        Some("encoder.prenet.bias"),
        HIDDEN,
        HIDDEN,
        9,
        1,
    )?;
    let x = g.relu(x);
    let x = to_tm(&mut g, x, t, HIDDEN);
    let mul = p.p(&mut g, "encoder.bn.mul", &[HIDDEN])?;
    let sub = p.p(&mut g, "encoder.bn.sub", &[HIDDEN])?;
    let x = g.mul(x, mul);
    let x = g.add(x, sub);
    let pe = g.param("enc.encoder.pos_enc.rows", f32s(&[POS_ROWS, HIDDEN]));
    let pe = g.narrow_(pe, 0, 0, t);
    let mut x = g.add(x, pe);
    for i in 0..4 {
        let lin = |g: &mut Graph, p: &mut Params, x: NodeId, n: &str, inp: usize, out: usize| {
            p.p(g, n, &[inp, out]).map(|w| g.mm(x, w))
        };
        let q = lin(
            &mut g,
            &mut p,
            x,
            &format!("encoder.block{i}.attn.q.weight"),
            HIDDEN,
            512,
        )?;
        let k = lin(
            &mut g,
            &mut p,
            x,
            &format!("encoder.block{i}.attn.k.weight"),
            HIDDEN,
            512,
        )?;
        let v = lin(
            &mut g,
            &mut p,
            x,
            &format!("encoder.block{i}.attn.v.weight"),
            HIDDEN,
            512,
        )?;
        let q = g.reshape_(q, vec![1, t as i64, 512]);
        let k = g.reshape_(k, vec![1, t as i64, 512]);
        let v = g.reshape_(v, vec![1, t as i64, 512]);
        let ctx = g.attention_kind(q, k, v, 2, 256, MaskKind::None, f32s(&[1, t, 512]));
        let ctx = g.reshape_(ctx, vec![t as i64, 512]);
        let a = lin(
            &mut g,
            &mut p,
            ctx,
            &format!("encoder.block{i}.attn.out.weight"),
            512,
            HIDDEN,
        )?;
        let ab = p.p(
            &mut g,
            &format!("encoder.block{i}.attn.out.bias"),
            &[HIDDEN],
        )?;
        let a = g.add(a, ab);
        let a = g.add(a, x);
        let x1 = ln(&mut g, &mut p, a, 2 * i)?;
        let h = to_cf(&mut g, x1, t, HIDDEN);
        let h = conv(
            &mut g,
            &mut p,
            h,
            &format!("encoder.block{i}.ffn0.weight"),
            Some(&format!("encoder.block{i}.ffn0.bias")),
            1024,
            HIDDEN,
            9,
            1,
        )?;
        let h = g.relu(h);
        let h = conv(
            &mut g,
            &mut p,
            h,
            &format!("encoder.block{i}.ffn1.weight"),
            Some(&format!("encoder.block{i}.ffn1.bias")),
            HIDDEN,
            1024,
            1,
            1,
        )?;
        let h = to_tm(&mut g, h, t, HIDDEN);
        let h = g.add(h, x1);
        x = ln(&mut g, &mut p, h, 2 * i + 1)?;
    }
    g.node_mut(x).name = Some("x".into());
    let mut outs = vec![x];
    for (name, (la, lb)) in [
        ("duration", (8, 11)),
        ("pitch", (9, 12)),
        ("energy", (10, 13)),
    ] {
        let h = to_cf(&mut g, x, t, HIDDEN);
        let h = conv(
            &mut g,
            &mut p,
            h,
            &format!("{name}.conv0.weight"),
            Some(&format!("{name}.conv0.bias")),
            HIDDEN,
            HIDDEN,
            3,
            1,
        )?;
        let h = g.relu(h);
        let h = to_tm(&mut g, h, t, HIDDEN);
        let h = ln(&mut g, &mut p, h, la)?;
        let h = to_cf(&mut g, h, t, HIDDEN);
        let h = conv(
            &mut g,
            &mut p,
            h,
            &format!("{name}.conv1.weight"),
            Some(&format!("{name}.conv1.bias")),
            HIDDEN,
            HIDDEN,
            3,
            1,
        )?;
        let h = g.relu(h);
        let h = to_tm(&mut g, h, t, HIDDEN);
        let h = ln(&mut g, &mut p, h, lb)?;
        let pw = p.p(&mut g, &format!("{name}.proj.weight"), &[HIDDEN, 1])?;
        let mut y = g.mm(h, pw);
        if w.contains(&format!("{name}.bias")) {
            let b = p.p(&mut g, &format!("{name}.bias"), &[1])?;
            y = g.add(y, b);
        }
        if name == "duration" {
            y = g.relu(y);
        }
        let y = g.reshape_(y, vec![t as i64]);
        g.node_mut(y).name = Some(name.into());
        outs.push(y);
    }
    g.set_outputs(outs);
    Ok((g, p.held))
}

/// `decode` at `t` phones and `f` frames.
fn decode(
    enc: &Weights,
    dec: &Weights,
    t: usize,
    f: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("decode");
    let x = g.input("x", f32s(&[t, HIDDEN]));
    let pitch = g.input("pitch_ids", Shape::new(&[t], DType::I32));
    let energy = g.input("energy_ids", Shape::new(&[t], DType::I32));
    let index = g.input("index", Shape::new(&[f], DType::I32));
    let mut pe = Params {
        w: enc,
        prefix: "enc.",
        held: HashMap::new(),
    };
    let pt = pe.p(&mut g, "pitch_embed.weight", &[BUCKETS, HIDDEN])?;
    let et = pe.p(&mut g, "energy_embed.weight", &[BUCKETS, HIDDEN])?;
    let pr = g.gather_(pt, pitch, 0);
    let er = g.gather_(et, energy, 0);
    let x = g.add(x, pr);
    let x = g.add(x, er);
    let x = g.gather_(x, index, 0);
    let mut h = to_cf(&mut g, x, f, HIDDEN);
    let mut pd = Params {
        w: dec,
        prefix: "dec.",
        held: HashMap::new(),
    };
    for (i, &dil) in DECODER_DILATIONS.iter().enumerate() {
        h = conv(
            &mut g,
            &mut pd,
            h,
            &format!("decoder.dilated.{i}.weight"),
            Some(&format!("decoder.dilated.{i}.bias")),
            HIDDEN,
            HIDDEN,
            3,
            dil,
        )?;
        h = g.relu(h);
    }
    let mel = conv(
        &mut g,
        &mut pd,
        h,
        "decoder.mel.weight",
        Some("decoder.mel.bias"),
        N_MELS,
        HIDDEN,
        3,
        1,
    )?;
    let mel = to_tm(&mut g, mel, f, N_MELS);
    g.node_mut(mel).name = Some("mel".into());
    g.set_outputs(vec![mel]);
    let mut held = pe.held;
    held.extend(pd.held);
    Ok((g, held))
}

/// `v` rounded to f16 and back (the portable path's activation quantization).
fn f16(g: &mut Graph, v: NodeId) -> NodeId {
    let shape = g.node(v).shape.clone();
    let h = g.add_node(
        Op::Cast { to: DType::F16 },
        vec![v],
        Shape::from_dims(shape.dims(), DType::F16),
    );
    g.add_node(Op::Cast { to: DType::F32 }, vec![h], shape)
}

/// `1 / (1 + exp(−x))`.
fn sigmoid(g: &mut Graph, x: NodeId) -> NodeId {
    let neg = g.neg(x);
    let e = g.exp(neg);
    let one = scalar(g, 1.0);
    let d = g.add(e, one);
    g.recip(d)
}

/// The runner's fused tanh, `2 · σ(2x) − 1`.
fn tanh_fused(g: &mut Graph, x: NodeId) -> NodeId {
    let two = scalar(g, 2.0);
    let x2 = g.mul(x, two);
    let s = sigmoid(g, x2);
    let s2 = g.mul(s, two);
    let one = scalar(g, 1.0);
    g.sub(s2, one)
}

/// One sub-stream GRU: `prev` `[1, 224]`, gate inputs `ig` / `rg` `[1, 672]`.
fn gru(g: &mut Graph, prev: NodeId, ig: NodeId, rg: NodeId) -> NodeId {
    let part = |g: &mut Graph, x: NodeId, i: usize| g.narrow_(x, 1, i * SUB, SUB);
    let (iz, ir, inn) = (part(g, ig, 0), part(g, ig, 1), part(g, ig, 2));
    let (rz, rr, rn) = (part(g, rg, 0), part(g, rg, 1), part(g, rg, 2));
    let z = g.add(iz, rz);
    let z = sigmoid(g, z);
    let r = g.add(ir, rr);
    let r = sigmoid(g, r);
    let n = g.mul(r, rn);
    let n = g.add(n, inn);
    let n = tanh_fused(g, n);
    // z · prev + (n − z · n)
    let zp = g.mul(z, prev);
    let zn = g.mul(z, n);
    let mix = g.sub(n, zn);
    g.add(zp, mix)
}

/// A head: `relu(f16(h) · w0 + b0)`, then `f16(·) · w1 + b1` → `[1, 256]`.
fn head(g: &mut Graph, h: NodeId, w0: NodeId, b0: NodeId, w1: NodeId, b1: NodeId) -> NodeId {
    let x = f16(g, h);
    let x = g.mm(x, w0);
    let x = g.add(x, b0);
    let x = g.relu(x);
    let x = f16(g, x);
    let x = g.mm(x, w1);
    g.add(x, b1)
}

/// The Gumbel-max sample of `logits` `[1, 256]` with `noise` `[1, 256]`,
/// as a sample in `[-1, 1]` (`[1, 1]`).
fn sample(g: &mut Graph, logits: NodeId, noise: NodeId) -> NodeId {
    let v = g.add(logits, noise);
    let idx = g.argmax(v, 1, true, f32s(&[1, 1]));
    let k = scalar(g, 2.0 / 255.0);
    let s = g.mul(idx, k);
    let one = scalar(g, 1.0);
    g.sub(s, one)
}

/// The bcast inputs, in order: `[name, shape]`.
fn wr_bcast() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("i_mel", vec![1, WR_GATES]),
        ("wavernn.bits_fc.weight", vec![2, WR_GATES]),
        ("wavernn.gru_hh.weight", vec![WR_HIDDEN, WR_GATES]),
        ("wavernn.gru_hh.bias", vec![1, WR_GATES]),
        ("wavernn.scale_const", vec![1, SUB_GATES]),
        ("wavernn.coarse_fc0.weight", vec![SUB, SUB]),
        ("wavernn.coarse_fc0.bias", vec![1, SUB]),
        ("wavernn.coarse_fc1.weight", vec![SUB, 256]),
        ("wavernn.coarse_fc1.bias", vec![1, 256]),
        ("wavernn.fine_fc0.weight", vec![SUB, SUB]),
        ("wavernn.fine_fc0.bias", vec![1, SUB]),
        ("wavernn.fine_fc1.weight", vec![SUB, 256]),
        ("wavernn.fine_fc1.bias", vec![1, 256]),
    ]
}

/// The scan body: `(carry [451], bcast…, noise [512]) → carry`.
fn wr_body() -> Graph {
    let mut g = Graph::new("wavernn_step");
    let carry = g.input("carry", f32s(&[CARRY]));
    let b: Vec<NodeId> = wr_bcast()
        .into_iter()
        .map(|(n, s)| g.input(n, f32s(&s)))
        .collect();
    let noise = g.input("noise", f32s(&[512]));
    let (i_mel, bits_w, hh_w, hh_b, inject) = (b[0], b[1], b[2], b[3], b[4]);
    let carry2 = g.reshape_(carry, vec![1, CARRY as i64]);
    let h = g.narrow_(carry2, 1, 0, WR_HIDDEN);
    let prev_fine = g.narrow_(carry2, 1, WR_HIDDEN, 1);
    // ig = i_mel + (f16(prev_fine) · w_bits[0] + f16(0) · w_bits[1]): the
    // coarse bit held at 0.
    let b0 = f16(&mut g, prev_fine);
    let w0 = g.narrow_(bits_w, 0, 0, 1);
    let bits = g.mul(b0, w0);
    let ig = g.add(i_mel, bits);
    let hf = f16(&mut g, h);
    let rg = g.mm(hf, hh_w);
    let rg = g.add(rg, hh_b);
    let hc = g.narrow_(h, 1, 0, SUB);
    let hfine = g.narrow_(h, 1, SUB, SUB);
    let igc = g.narrow_(ig, 1, 0, SUB_GATES);
    let igf = g.narrow_(ig, 1, SUB_GATES, SUB_GATES);
    let rgc = g.narrow_(rg, 1, 0, SUB_GATES);
    let rgf = g.narrow_(rg, 1, SUB_GATES, SUB_GATES);
    let h_coarse = gru(&mut g, hc, igc, rgc);
    let logits = head(&mut g, h_coarse, b[5], b[6], b[7], b[8]);
    let noise = g.reshape_(noise, vec![1, 512]);
    let nc = g.narrow_(noise, 1, 0, 256);
    let nf = g.narrow_(noise, 1, 256, 256);
    let coarse = sample(&mut g, logits, nc);
    let inj = g.mul(coarse, inject);
    let fig = g.add(igf, inj);
    let h_fine = gru(&mut g, hfine, fig, rgf);
    let logits = head(&mut g, h_fine, b[9], b[10], b[11], b[12]);
    let fine = sample(&mut g, logits, nf);
    let next = g.concat_(vec![h_coarse, h_fine, fine, coarse, fine], 1);
    let next = g.reshape_(next, vec![CARRY as i64]);
    g.set_outputs(vec![next]);
    g
}

/// `wavernn`: one mel `frame` `[1, 80]`, the `carry` `[451]` and the
/// frame's `noise` `[120, 512]` → the steps' carries `steps` `[120, 451]`.
fn wavernn(w: &Weights) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut p = Params {
        w,
        prefix: "wr.",
        held: HashMap::new(),
    };
    let mut g = Graph::new("wavernn");
    let frame = g.input("frame", f32s(&[1, N_MELS]));
    let carry = g.input("carry", f32s(&[CARRY]));
    let noise = g.input("noise", f32s(&[STEPS, 512]));
    let mw = p.p(&mut g, "wavernn.mel_fc.weight", &[N_MELS, WR_GATES])?;
    let mb = p.p(&mut g, "wavernn.mel_fc.bias", &[1, WR_GATES])?;
    let m = f16(&mut g, frame);
    let i_mel = g.mm(m, mw);
    let i_mel = g.add(i_mel, mb);
    let mut inputs = vec![carry, i_mel];
    for (name, shape) in wr_bcast().into_iter().skip(1) {
        inputs.push(p.p(&mut g, name, &shape)?);
    }
    inputs.push(noise);
    let steps = g.add_node(
        Op::Scan {
            body: Box::new(wr_body()),
            length: STEPS as u32,
            save_trajectory: true,
            num_bcast: (wr_bcast().len()) as u32,
            num_xs: 1,
            num_checkpoints: 0,
        },
        inputs,
        f32s(&[STEPS, CARRY]),
    );
    g.node_mut(steps).name = Some("steps".into());
    g.set_outputs(vec![steps]);
    Ok((g, p.held))
}

/// Every file of `dir` under it, by its path relative to `base`.
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

/// Export the RLX TTS bundle `bundle` (`rlx-tts.rlxp`, `rlx-tts.gguf`, or a
/// directory) as the package `out`.
pub fn export_rlxp(bundle: &Path, out: &Path, opts: &ExportOptions) -> Result<ExportReport> {
    let m = RlxTts::open(bundle)?;
    let (enc, dec, wr) = (
        m.encoder_weights(),
        m.decoder_weights(),
        m.wavernn_weights(),
    );
    let mut pkg = Package::new(
        "rlx-tts",
        &opts.version,
        "rlx-tts",
        "text-to-speech",
        "rlx-tts",
    );
    let mut held: HashMap<String, Vec<f32>> = HashMap::new();
    pkg.function(symbolic_in(PHONES, (5, 7, 3), |t| Ok(encode(enc, t)?.0))?);
    held.extend(encode(enc, 3)?.1);
    let (d, lifted) = symbolic_in_lifted_many(
        &[(PHONES, vec![5, 7, 3]), (FRAMES, vec![37, 53, 20])],
        &[POS_ROWS, POS_ROWS],
        |n| Ok(decode(enc, dec, n[0], n[1])?.0),
    )?;
    anyhow::ensure!(lifted.is_empty(), "the decoder lifted tables");
    pkg.function(d);
    held.extend(decode(enc, dec, 3, 20)?.1);
    let (wg, wp) = wavernn(wr)?;
    pkg.function(wg);
    held.extend(wp);
    // The positional encoding, its last row repeated past 256 phones.
    let pe = enc.data("encoder.pos_enc")?;
    let mut rows = Vec::with_capacity(POS_ROWS * HIDDEN);
    for i in 0..POS_ROWS {
        let r = i.min(255);
        rows.extend_from_slice(&pe[r * HIDDEN..(r + 1) * HIDDEN]);
    }
    held.insert("enc.encoder.pos_enc.rows".into(), rows);
    pkg.dims = vec![("phones".into(), PHONES), ("frames".into(), FRAMES)];
    let pdims = pkg.param_dims();
    let mut names: Vec<&String> = held.keys().collect();
    names.sort();
    for n in names {
        let d = pdims.get(n).cloned().unwrap_or_else(|| vec![held[n].len()]);
        pkg.hold_f32(n.clone(), d, &held[n]);
    }

    // The frontend's files, as the runner's bundle directory holds them.
    let dir = m.bundle_dir();
    let mut files = Vec::new();
    files_under(dir, dir, &mut files)?;
    for (rel, p) in files {
        if rel.ends_with(".safetensors") && !rel.starts_with("frontend/") {
            continue;
        }
        pkg.sidecar(
            &format!("bundle/{rel}"),
            "application/octet-stream",
            std::fs::read(p)?,
        );
    }
    let module = program::build_for(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("program/Cargo.toml"),
        "wasm32-wasip1",
    )
    .context("build RLX TTS' program")?;
    program::entry(
        &mut pkg,
        "speak",
        module,
        &["synthesize", "run"],
        "text-to-speech",
        json!({
            "bundle": "/package/bundle",
            "seed": 16_807,
            "gumbel_alpha": 1.0,
            "gumbel_beta": 0.01,
            "steps": STEPS,
            "carry": CARRY,
            "hidden": WR_HIDDEN,
            "n_mels": N_MELS,
            "sample_rate": m.sample_rate(),
            "post": crate::native::load_output_post(m.bundle_dir()).to_json(),
            "pause_min_duration_ms": m.frontend().pause_min_duration_ms,
            "phone_vocab": PHONE_VOCAB,
        }),
    );
    pkg.metadata("sample_rate", json!(m.sample_rate()));
    pkg.write(out, opts)
}

/// Register this crate's exporter ([`rlx_core::export::register_exporter`]).
pub fn register() {
    use rlx_core::export::{Exporter, register_exporter};
    register_exporter(Exporter::new("rlx-tts", "text-to-speech", export_rlxp));
}
