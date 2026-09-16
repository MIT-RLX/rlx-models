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

//! Runs the real Espresso graphs on the real installed weights.
//!
//! Every intermediate is checked against `.espresso.shape`, so a wiring or
//! layout mistake fails at the offending layer. Skips when nothing is installed.

use rlx_translate::assets::Assets;
use rlx_translate::espresso::Manifest;
use rlx_translate::exec::{Env, Value, run_checked};
use rlx_translate::net::Graph;
use rlx_translate::tensor::Tensor;
use std::path::PathBuf;

/// The declared sequence length of the shipped graphs.
const SEQ: usize = 64;

fn mt_dirs() -> Vec<PathBuf> {
    Assets::discover()
        .roots
        .iter()
        .map(|r| r.join("MT"))
        .filter(|p| p.is_dir())
        .collect()
}

fn manifest(dirs: &[PathBuf]) -> Option<Manifest> {
    dirs.iter()
        .map(|d| d.join("pyespresso.mdl.bin"))
        .find(|p| p.exists())
        .map(|p| Manifest::load(&p).expect("manifest parses"))
}

fn graph(dirs: &[PathBuf], net: &str) -> Option<Graph> {
    dirs.iter()
        .find(|d| d.join(net).exists())
        .map(|d| Graph::load(d, net).unwrap_or_else(|e| panic!("loading {net}: {e:#}")))
}

/// Largest valid token id for a graph's vocabulary gather.
///
/// Vocabulary size differs per bundle — 168 000 for the
/// en/es/de/it/fr/pt/nl model, 48 000 for others — so it is read from the
/// graph rather than assumed.
fn vocab_rows(g: &Graph) -> usize {
    g.layers
        .iter()
        .filter(|l| l.kind == "quantized_gather")
        .filter_map(|l| l.int("nRow"))
        .map(|v| v as usize)
        .max()
        .unwrap_or(1)
}

/// Token ids and positions for a nominal-length source.
fn embedding_inputs(m: &Manifest, vocab: usize) -> Env {
    let ids: Vec<f32> = (0..SEQ)
        .map(|i| ((i * 977) % vocab.max(1)) as f32)
        .collect();
    let pos: Vec<f32> = (0..SEQ).map(|i| i as f32).collect();
    let mut env = Env::new();
    env.insert(
        m.str("SourceInputStr").unwrap_or("src_tokens").to_string(),
        Value::F32(Tensor::new(vec![SEQ], ids).expect("tokens")),
    );
    env.insert(
        "positions".to_string(),
        Value::F32(Tensor::new(vec![SEQ], pos).expect("positions")),
    );
    env
}

#[test]
fn embedding_graph_runs_and_matches_declared_shapes() {
    let dirs = mt_dirs();
    let (Some(m), Some(g)) = (manifest(&dirs), graph(&dirs, "embedding.espresso.net")) else {
        eprintln!("skipping: embedding graph not installed");
        return;
    };
    let vocab = vocab_rows(&g);
    let env = run_checked(&g, embedding_inputs(&m, vocab)).expect("embedding runs");
    let out = env["embedding"].f32().expect("embedding is f32");
    assert_eq!(out.dims(), &[SEQ, 512], "embedding geometry");
    assert!(
        out.data().iter().all(|v| v.is_finite()),
        "embedding produced non-finite values"
    );
    // Scaled by sqrt(512) in the graph, so values are well away from zero.
    let peak = out.data().iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(peak > 0.1, "embedding looks empty (peak {peak})");
    eprintln!("embedding: {:?} peak {peak:.3} (vocab {vocab})", out.dims());
}

#[test]
fn full_encoder_stack_runs_on_real_weights() {
    let dirs = mt_dirs();
    let Some(m) = manifest(&dirs) else {
        eprintln!("skipping: no manifest installed");
        return;
    };
    let (Some(emb), Some(enc)) = (
        graph(&dirs, "embedding.espresso.net"),
        m.str("EncoderGraph").and_then(|f| graph(&dirs, f)),
    ) else {
        eprintln!("skipping: encoder graph not installed");
        return;
    };
    // Pick any installed source-language input net.
    let Some((lang, input_net)) = m
        .lang_graphs
        .get("InputLangGraph")
        .and_then(|m| m.iter().find(|(_, f)| graph(&dirs, f).is_some()))
    else {
        eprintln!("skipping: no input graph installed");
        return;
    };
    let inp = graph(&dirs, input_net).expect("input graph loads");

    // embedding -> input_<lang> -> encoder
    let vocab = vocab_rows(&emb);
    let env = run_checked(&emb, embedding_inputs(&m, vocab)).expect("embedding runs");
    let embedding = env["embedding"].clone();

    let mut ienv = Env::new();
    ienv.insert("embedding".to_string(), embedding);
    let ienv = run_checked(&inp, ienv).expect("input net runs");
    let bridge = m.str("InputNetValuesStr").unwrap_or("encoder.3.output");
    let mid = ienv
        .get(bridge)
        .unwrap_or_else(|| panic!("input net did not produce {bridge}"))
        .clone();

    let mut eenv = Env::new();
    eenv.insert(bridge.to_string(), mid);
    let eenv = run_checked(&enc, eenv).expect("encoder runs");

    let out_name = m.str("EncoderValuesStr").unwrap_or("encoder.15.output");
    let out = eenv[out_name].f32().expect("encoder output is f32");
    // A trailing reshape may leave a leading batch axis, so compare the
    // element count and feature width rather than the literal dims.
    assert_eq!(out.len(), SEQ * 512, "encoder output element count");
    assert_eq!(out.width(), 512, "encoder feature width");
    assert!(
        out.data().iter().all(|v| v.is_finite()),
        "encoder produced non-finite values"
    );
    // The stack ends in a LayerNorm, so per-row RMS should be near 1.
    let rms: f32 = (out.row(0).iter().map(|v| v * v).sum::<f32>() / 512.0).sqrt();
    assert!(
        rms > 0.1 && rms < 20.0,
        "encoder row RMS {rms} is implausible after a LayerNorm"
    );
    eprintln!("encoder({lang}): {:?} row0 RMS {rms:.3}", out.dims());
}

#[test]
fn handover_graph_produces_every_cross_attention_tensor() {
    let dirs = mt_dirs();
    let Some(m) = manifest(&dirs) else {
        eprintln!("skipping: no manifest installed");
        return;
    };
    let Some((lang, net)) = m
        .lang_graphs
        .get("HandoverLangGraph")
        .and_then(|g| g.iter().find(|(_, f)| graph(&dirs, f).is_some()))
    else {
        eprintln!("skipping: no handover graph installed");
        return;
    };
    let h = graph(&dirs, net).expect("handover loads");

    let enc_name = m.str("EncoderValuesStr").unwrap_or("encoder.15.output");
    let mut env = Env::new();
    env.insert(
        enc_name.to_string(),
        Value::F32(Tensor::new(vec![SEQ, 512], vec![0.01; SEQ * 512]).expect("encoder stand-in")),
    );
    let env = run_checked(&h, env).expect("handover runs");

    for name in m.csv("HandoverStrings") {
        let v = env
            .get(&name)
            .unwrap_or_else(|| panic!("handover did not produce {name}"));
        let t = v.f32().expect("handover tensor is f32");
        assert!(
            t.data().iter().all(|x| x.is_finite()),
            "{name} has non-finite values"
        );
    }
    eprintln!(
        "handover({lang}): produced {} tensors",
        m.csv("HandoverStrings").len()
    );
}

/// The export folds Espresso's `dynamic_quantize -> inner_product ->
/// dynamic_dequantize` triple into a single f32 linear
/// `y = x · (W_int8 / w_scale) + b`. That is an analytical argument about the
/// activation scale cancelling; this measures it on a real weight matrix.
///
/// The two paths are not bit-identical by construction — the int8 path rounds
/// activations to 8 bits and the folded path does not — so the check is cosine
/// similarity, which is how the rest of the workspace states parity.
#[test]
fn the_exported_f32_fold_matches_the_int8_path() {
    let dirs = mt_dirs();
    let Some(m) = manifest(&dirs) else {
        eprintln!("skipping: no manifest installed");
        return;
    };
    let Some(enc) = m.str("EncoderGraph").and_then(|f| graph(&dirs, f)) else {
        eprintln!("skipping: encoder graph not installed");
        return;
    };

    // First quantize/inner_product/dequantize triple in the encoder.
    let ip = enc
        .layers
        .iter()
        .find(|l| l.kind == "inner_product")
        .expect("encoder has a linear");
    let dq = enc
        .layers
        .iter()
        .find(|l| l.kind == "dynamic_dequantize" && l.bottoms.first() == Some(&ip.tops[0]))
        .expect("the linear has a dequantize");

    let n_in = ip.int("nB").expect("nB") as usize;
    let n_out = ip.int("nC").expect("nC") as usize;
    let w_scale = dq.float("w_quantization_scale").expect("w scale") as f32;
    let w_int8 = enc
        .weights
        .i8s(ip.blob("W_int8").expect("W"))
        .expect("weights");
    let bias = dq
        .blob("biases")
        .map(|b| enc.weights.f32s(b).expect("bias"))
        .unwrap_or_default();

    // A plausible activation: LayerNorm output, so roughly unit scale.
    let rows = 8usize;
    let x: Vec<f32> = (0..rows * n_in)
        .map(|i| ((i * 2654435761usize) % 2000) as f32 / 1000.0 - 1.0)
        .collect();

    // Path A — what the device does: quantize activations to int8.
    let peak = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let act = 127.0 / peak;
    let xq: Vec<i8> = x
        .iter()
        .map(|v| (v * act).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let mut a_out = vec![0.0f32; rows * n_out];
    for r in 0..rows {
        for o in 0..n_out {
            let mut acc = 0i32;
            for i in 0..n_in {
                acc += i32::from(xq[r * n_in + i]) * i32::from(w_int8[o * n_in + i]);
            }
            let mut v = acc as f32 / (act * w_scale);
            if !bias.is_empty() {
                v += bias[o];
            }
            if dq.flag("has_relu") && v < 0.0 {
                v = 0.0;
            }
            a_out[r * n_out + o] = v;
        }
    }

    // Path B — what the export emits: f32 weights, no activation quantization.
    let w_f32: Vec<f32> = w_int8.iter().map(|v| f32::from(*v) / w_scale).collect();
    let mut b_out = vec![0.0f32; rows * n_out];
    for r in 0..rows {
        for o in 0..n_out {
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += x[r * n_in + i] * w_f32[o * n_in + i];
            }
            if !bias.is_empty() {
                acc += bias[o];
            }
            if dq.flag("has_relu") && acc < 0.0 {
                acc = 0.0;
            }
            b_out[r * n_out + o] = acc;
        }
    }

    let dot: f64 = a_out
        .iter()
        .zip(&b_out)
        .map(|(p, q)| f64::from(*p) * f64::from(*q))
        .sum();
    let na: f64 = a_out
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b_out
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    let rel: f64 = {
        let num: f64 = a_out
            .iter()
            .zip(&b_out)
            .map(|(p, q)| (f64::from(*p) - f64::from(*q)).powi(2))
            .sum::<f64>()
            .sqrt();
        num / na.max(1e-12)
    };
    eprintln!(
        "fold parity on {}x{} linear (w_scale {w_scale:.3}): cosine {cos:.6}, \
         relative L2 {rel:.5}",
        n_out, n_in
    );
    assert!(
        cos > 0.999,
        "folded f32 linear diverges from the int8 path (cosine {cos:.6}); the \
         export's scale-cancellation argument does not hold"
    );
}

/// Validates every intermediate's FULL geometry against `.espresso.shape`.
///
/// Attention tensors all have the same element count and width, so only a
/// full-dims check can catch a wrong permutation there.
#[test]
fn input_net_matches_declared_geometry_exactly() {
    use rlx_translate::exec::run_strict;
    let dirs = mt_dirs();
    let Some(m) = manifest(&dirs) else {
        eprintln!("skipping: no manifest installed");
        return;
    };
    let Some(emb) = graph(&dirs, "embedding.espresso.net") else {
        eprintln!("skipping: embedding not installed");
        return;
    };
    let Some((lang, net)) = m
        .lang_graphs
        .get("InputLangGraph")
        .and_then(|g| g.iter().find(|(_, f)| graph(&dirs, f).is_some()))
    else {
        eprintln!("skipping: no input graph installed");
        return;
    };
    let inp = graph(&dirs, net).expect("input graph loads");

    let vocab = vocab_rows(&emb);
    let env = run_checked(&emb, embedding_inputs(&m, vocab)).expect("embedding runs");
    let embedded = env["embedding"].clone();
    let mut ienv = Env::new();
    ienv.insert("embedding".to_string(), embedded);

    match run_strict(&inp, ienv) {
        Ok(_) => eprintln!("input_{lang}: every intermediate matches its declared geometry"),
        Err(e) => panic!("geometry mismatch in input_{lang}: {e:#}"),
    }
}
