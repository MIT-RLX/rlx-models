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

//! Full encoder parity: RLX Wav2Vec2-BERT vs HuggingFace `Wav2Vec2BertModel`.
//!
//! Requires Python 3 + transformers + torch + the W2v-BERT 2.0 checkpoint.
//!
//! ```bash
//! RLX_W2V_BERT_DIR=/path/to/w2v-bert-2.0 \
//!   cargo test -p rlx-models wav2vec2_bert_hf_encoder_parity --release -- --nocapture
//! ```
//!
//! `RLX_W2V_BERT_DIR` may be a HF snapshot directory containing
//! `model.safetensors` + `config.json`, or omitted to let the reference
//! download `facebook/w2v-bert-2.0`.

mod compile_support;

use rlx_models::Wav2Vec2BertRunner;
use rlx_models::WeightMap;
use rlx_models::wav2vec2_bert::{
    W2vLayerStop, Wav2Vec2BertConfig, build_wav2vec2_bert_graph_probe,
};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_ABS_TOL: f32 = 5e-2;
const MEAN_ABS_TOL: f32 = 1e-2;
const COSINE_TOL: f32 = 0.999;

struct HfReference {
    batch: usize,
    seq: usize,
    hidden: usize,
    feat_dim: usize,
    features: Vec<f32>,
    mask: Vec<f32>,
    hidden_states: Vec<f32>,
}

fn model_dir() -> Option<PathBuf> {
    rlx_ir::env::var("RLX_W2V_BERT_DIR").map(PathBuf::from)
}

fn python_ok() -> bool {
    Command::new("python3")
        .args(["-c", "import transformers, torch"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run_hf_reference(
    model_dir: Option<&Path>,
    seq: usize,
    num_layers: Option<usize>,
    probe: Option<&str>,
) -> HfReference {
    let helper = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/wav2vec2_bert_parity_helpers/hf_reference.py");
    let mut cmd = Command::new("python3");
    cmd.arg(&helper).arg("--duration-sec").arg("1.0");
    if let Some(dir) = model_dir {
        cmd.arg("--model-dir").arg(dir);
    }
    if seq > 0 {
        cmd.arg("--seq").arg(seq.to_string());
    }
    if let Some(n) = num_layers {
        cmd.arg("--num-layers").arg(n.to_string());
    }
    if let Some(p) = probe {
        cmd.arg("--probe").arg(p);
    }
    let out = cmd.output().expect("python3 hf_reference");
    assert!(
        out.status.success(),
        "hf_reference failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse_reference(&String::from_utf8_lossy(&out.stdout))
}

fn parse_reference(text: &str) -> HfReference {
    let mut batch = 0usize;
    let mut seq = 0usize;
    let mut hidden = 0usize;
    let mut feat_dim = 0usize;
    let mut features = Vec::new();
    let mut mask = Vec::new();
    let mut hidden_states = Vec::new();

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("SHAPE ") {
            let mut it = rest.split_whitespace();
            batch = it.next().unwrap().parse().unwrap();
            seq = it.next().unwrap().parse().unwrap();
            hidden = it.next().unwrap().parse().unwrap();
            feat_dim = it.next().unwrap().parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("FEAT ") {
            features = rest
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
        } else if let Some(rest) = line.strip_prefix("MASK ") {
            mask = rest
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
        } else if let Some(rest) = line.strip_prefix("HIDDEN ") {
            hidden_states = rest
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
        }
    }
    assert!(
        batch > 0 && seq > 0 && hidden > 0 && feat_dim > 0,
        "missing SHAPE line"
    );
    HfReference {
        batch,
        seq,
        hidden,
        feat_dim,
        features,
        mask,
        hidden_states,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut sum = 0f64;
    let mut idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        sum += d as f64;
        if d > max {
            max = d;
            idx = i;
        }
    }
    ((sum / a.len() as f64) as f32, max, idx)
}

fn safetensors_path(dir: &Path) -> PathBuf {
    let st = dir.join("model.safetensors");
    assert!(
        st.exists(),
        "missing model.safetensors in {}",
        dir.display()
    );
    st
}

fn run_rlx(model_dir: &Path, seq: usize, num_layers: Option<usize>, hf: &HfReference) -> Vec<f32> {
    let mut cfg = Wav2Vec2BertConfig::from_file(&model_dir.join("config.json")).expect("config");
    if let Some(n) = num_layers {
        cfg.num_hidden_layers = n;
    }
    let weights = safetensors_path(model_dir);
    let mut runner = Wav2Vec2BertRunner::builder()
        .weights(&weights)
        .config_path(model_dir.join("config.json"))
        .preprocessor_config_path(model_dir.join("preprocessor_config.json"))
        .config(cfg)
        .batch(1)
        .seq(seq)
        .build()
        .expect("build runner");
    runner
        .encode_features(&hf.features, Some(&hf.mask))
        .expect("rlx forward")
}

fn run_rlx_probe(model_dir: &Path, seq: usize, hf: &HfReference, stop: W2vLayerStop) -> Vec<f32> {
    let mut cfg = Wav2Vec2BertConfig::from_file(&model_dir.join("config.json")).expect("config");
    cfg.num_hidden_layers = 1;
    let weights = safetensors_path(model_dir);
    let mut wm = WeightMap::from_file(weights.to_str().unwrap()).expect("weights");
    let (graph, params) =
        build_wav2vec2_bert_graph_probe(&cfg, &mut wm, 1, seq, 0, stop).expect("probe graph");
    let mut compiled = compile_support::compile_encoder(Device::Cpu, graph, params.clone());

    compiled
        .run(&[
            ("input_features", &hf.features),
            ("attention_mask", &hf.mask),
        ])
        .into_iter()
        .next()
        .expect("probe output")
}

#[test]
fn wav2vec2_bert_hf_layer0_substep_bisect() {
    if !python_ok() {
        eprintln!("skip: python3 + transformers + torch not available");
        return;
    }
    let model_dir = match model_dir() {
        Some(d) if d.join("model.safetensors").exists() => d,
        _ => {
            eprintln!("skip: set RLX_W2V_BERT_DIR");
            return;
        }
    };
    let seq = 128usize;
    let hf = run_hf_reference(Some(&model_dir), seq, Some(0), None);
    let probes = [
        (W2vLayerStop::AfterFfn1, "after_ffn1"),
        (W2vLayerStop::AfterAttn, "after_attn"),
        (W2vLayerStop::AfterConv, "after_conv"),
        (W2vLayerStop::AfterFfn2, "after_ffn2"),
        (W2vLayerStop::Final, "final"),
    ];
    for (stop, name) in probes {
        let hf_probe = run_hf_reference(Some(&model_dir), seq, Some(1), Some(name));
        let rlx = run_rlx_probe(&model_dir, seq, &hf, stop);
        let (mean, max, _) = diff_stats(&rlx, &hf_probe.hidden_states);
        let cos = cosine(&rlx, &hf_probe.hidden_states);
        eprintln!("{name}: cosine={cos:.6} mean_abs={mean:.6} max_abs={max:.6}");
    }
}

#[test]
fn wav2vec2_bert_hf_layer_bisect() {
    if !python_ok() {
        eprintln!("skip: python3 + transformers + torch not available");
        return;
    }
    let model_dir = match model_dir() {
        Some(d) if d.join("model.safetensors").exists() => d,
        _ => {
            eprintln!("skip: set RLX_W2V_BERT_DIR");
            return;
        }
    };

    let seq = 128usize;
    for layers in [Some(0usize), Some(1), Some(2), Some(4), Some(8), None] {
        let label = layers.map(|n| n.to_string()).unwrap_or_else(|| "24".into());
        let hf = run_hf_reference(Some(&model_dir), seq, layers, None);
        let rlx = run_rlx(&model_dir, seq, layers, &hf);
        let (mean, max, idx) = diff_stats(&rlx, &hf.hidden_states);
        let cos = cosine(&rlx, &hf.hidden_states);
        eprintln!(
            "layers={label}: cosine={cos:.6} mean_abs={mean:.6} max_abs={max:.6} worst_idx={idx}"
        );
    }
}

#[test]
fn wav2vec2_bert_hf_encoder_parity() {
    if !python_ok() {
        eprintln!("skip: python3 + transformers + torch not available");
        return;
    }

    let model_dir = match model_dir() {
        Some(d) if d.join("model.safetensors").exists() || d.join("config.json").exists() => d,
        _ => {
            eprintln!(
                "skip: set RLX_W2V_BERT_DIR to a HF snapshot dir with model.safetensors + config.json"
            );
            return;
        }
    };

    let seq = 128usize;
    let hf = run_hf_reference(Some(&model_dir), seq, None, None);
    assert_eq!(hf.batch, 1);
    assert_eq!(hf.seq, seq);
    assert_eq!(hf.hidden, 1024);
    assert_eq!(hf.feat_dim, 160);

    let cfg = Wav2Vec2BertConfig::from_file(&model_dir.join("config.json")).expect("config");
    let weights = safetensors_path(&model_dir);
    let mut runner = Wav2Vec2BertRunner::builder()
        .weights(&weights)
        .config_path(model_dir.join("config.json"))
        .preprocessor_config_path(model_dir.join("preprocessor_config.json"))
        .config(cfg.clone())
        .batch(1)
        .seq(seq)
        .build()
        .expect("build runner");

    let rlx = runner
        .encode_features(&hf.features, Some(&hf.mask))
        .expect("rlx forward");
    assert_eq!(rlx.len(), hf.hidden_states.len());

    let (mean, max, idx) = diff_stats(&rlx, &hf.hidden_states);
    let cos = cosine(&rlx, &hf.hidden_states);

    eprintln!(
        "wav2vec2-bert parity: cosine={cos:.6} mean_abs={mean:.6} max_abs={max:.6} worst_idx={idx}"
    );
    if max > MAX_ABS_TOL || mean > MEAN_ABS_TOL || cos < COSINE_TOL {
        let h = hf.hidden;
        let s = idx / h;
        let d = idx % h;
        eprintln!(
            "  hf[{s},{d}]={} rlx[{s},{d}]={}",
            hf.hidden_states[idx], rlx[idx]
        );
    }

    assert!(cos >= COSINE_TOL, "cosine similarity {cos} < {COSINE_TOL}");
    assert!(
        mean <= MEAN_ABS_TOL,
        "mean abs diff {mean} > {MEAN_ABS_TOL}"
    );
    assert!(
        max <= MAX_ABS_TOL,
        "max abs diff {max} > {MAX_ABS_TOL} at idx {idx}"
    );
}

#[test]
fn wav2vec2_bert_hf_end_to_end_waveform_parity() {
    if !python_ok() {
        eprintln!("skip: python3 + transformers + torch not available");
        return;
    }
    let model_dir = match model_dir() {
        Some(d) if d.join("model.safetensors").exists() => d,
        _ => {
            eprintln!("skip: set RLX_W2V_BERT_DIR");
            return;
        }
    };

    let seq = 128usize;
    let hf = run_hf_reference(Some(&model_dir), seq, None, None);
    let weights = safetensors_path(&model_dir);
    let mut runner = Wav2Vec2BertRunner::builder()
        .weights(&weights)
        .config_path(model_dir.join("config.json"))
        .preprocessor_config_path(model_dir.join("preprocessor_config.json"))
        .batch(1)
        .seq(seq)
        .build()
        .expect("build runner");

    // Same synthetic waveform as hf_reference.py
    let sr = 16_000usize;
    let seconds = 1.0f32;
    let n = (sr as f32 * seconds) as usize;
    let waveform: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.25
                + (880.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.1
                + (220.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.05
        })
        .collect();

    let mel = runner.extract_log_mel(&waveform);
    assert_eq!(mel.features.len(), seq * hf.feat_dim);
    assert_eq!(mel.attention_mask.len(), seq);

    let (mean_feat, max_feat, _) = diff_stats(&mel.features, &hf.features);
    eprintln!("preprocess parity: mean_abs={mean_feat:.6} max_abs={max_feat:.6}");
    assert!(max_feat <= 2e-2, "log-mel max diff {max_feat} > 2e-2");

    let rlx = runner.encode_waveform(&waveform).expect("encode");
    let (mean, max, idx) = diff_stats(&rlx, &hf.hidden_states);
    let cos = cosine(&rlx, &hf.hidden_states);
    eprintln!("e2e parity: cosine={cos:.6} mean_abs={mean:.6} max_abs={max:.6} worst_idx={idx}");
    assert!(cos >= COSINE_TOL, "e2e cosine {cos}");
    assert!(max <= MAX_ABS_TOL, "e2e max diff {max}");
}
