// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Streamed (layer-at-a-time) execution and training against the whole-graph
//! reference, on a synthetic checkpoint read through the real loader.
//!
//! * the streamed forward — canonical params, host-fixed routing over a compact
//!   expert bank, host-threaded Attention-Residual snapshots — reproduces the
//!   inference graph's logits;
//! * layer-by-layer VJP gradients equal whole-graph autodiff gradients, for full
//!   fine-tuning and for LoRA;
//! * whole-graph gradients agree with central finite differences;
//! * both trainers reduce the loss.

mod common;
use common::TempDir;
use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::flow::build_kimi_text_flow;
use rlx_kimi_k3::layer_exec::{LayerSpec, forward_head, forward_layer};
use rlx_kimi_k3::source::{CheckpointSource, WeightSource, load_flow_weights};
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_kimi_k3::train::{Batch, TrainSpec, batch_inputs, build_train_graph};
use rlx_kimi_k3::train_stream::{StreamOptions, StreamTrainer};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::Path;

const BATCH: usize = 1;
const SEQ: usize = 5;

fn synth_dir(tag: &str) -> TempDir {
    let dir = TempDir::new(&format!("rlx-kimi-synth-{tag}"));
    if !dir.join("config.json").exists() {
        write_synthetic_checkpoint(&dir, &SynthSpec::default())
            .expect("write synthetic checkpoint");
    }
    dir
}

fn open(dir: &Path) -> (CheckpointSource, rlx_kimi_k3::flow::FlowConfig) {
    let src = CheckpointSource::open(dir).expect("open checkpoint");
    let cfg = src.text_config().flow_config(BATCH, SEQ);
    (src, cfg)
}

fn batch(vocab: usize) -> Batch {
    let tokens: Vec<u32> = (0..SEQ).map(|i| ((i * 7 + 3) % vocab) as u32).collect();
    let labels: Vec<f32> = (0..SEQ).map(|i| ((i * 5 + 11) % vocab) as f32).collect();
    Batch {
        tokens,
        labels,
        mask: vec![1.0; SEQ],
    }
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn streamed_forward_matches_whole_graph() {
    let dir = synth_dir("fwd");
    let (mut src, cfg) = open(&dir);
    let (w, _embed) = load_flow_weights(&mut src, &cfg).expect("load weights");
    let b = batch(cfg.vocab);

    // Reference: the inference graph (fused params, in-graph routing).
    let mut hir = HirModule::new("ref");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[BATCH, SEQ, cfg.hidden], DType::F32));
    let mut params = HashMap::new();
    let logits = build_kimi_text_flow(&mut g, &mut params, h_in, &w, &cfg).expect("flow");
    g.set_outputs(vec![logits]);
    let mut c = compile_built(built_from_hir(hir, params).unwrap(), Device::Cpu).unwrap();
    let h0 = src.embed(&b.tokens, cfg.hidden).unwrap();
    let want = c.run(&[("h", h0.as_slice())]).remove(0);

    // Streamed.
    let store = HashMap::new();
    let spec = LayerSpec {
        cfg: &cfg,
        device: Device::Cpu,
        lora: None,
        overrides: &store,
        seed: 0,
        cache: None,
    };
    let (mut h, mut snaps) = (h0, Vec::new());
    let mut routed_layers = 0;
    for i in 0..src.num_layers() {
        let out = forward_layer(&mut src, i, &h, &snaps, &spec, true).expect("layer");
        if let Some(r) = &out.routing {
            routed_layers += 1;
            assert_eq!(r.idx.len(), BATCH * SEQ * cfg.moe.top_k);
        }
        assert!(
            out.taps
                .iter()
                .any(|(t, _)| t.name == format!("l{i}.ffn_out")),
            "layer {i} exposes its FFN output"
        );
        h = out.h_out;
        snaps = out.snaps_out;
    }
    assert!(routed_layers > 0, "the synthetic model has MoE layers");
    let head = src.head(&cfg).unwrap();
    let (got, taps) = forward_head(&head, &h, &snaps, &spec, true).unwrap();
    assert!(taps.iter().any(|(t, _)| t.name == "final_norm"));
    let err = max_rel(&got, &want);
    assert!(
        err < 1e-4,
        "streamed logits differ from the whole graph: rel {err}"
    );
}

/// Whole-graph loss + gradients of every trainable param.
fn whole_grads(
    src: &mut CheckpointSource,
    cfg: &rlx_kimi_k3::flow::FlowConfig,
    spec: &TrainSpec,
    b: &Batch,
) -> (f32, HashMap<String, Vec<f32>>) {
    let (w, embed) = load_flow_weights(src, cfg).unwrap();
    let tg = build_train_graph(&w, &embed, cfg, spec, &HashMap::new(), 0).unwrap();
    let ids: Vec<rlx_ir::NodeId> = tg.slots.iter().map(|s| s.node).collect();
    let bwd = rlx_autodiff::grad_with_loss(&tg.graph, &ids);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    for (n, v) in &tg.params {
        c.set_param(n, v);
    }
    let inputs = batch_inputs(b);
    let one = [1.0f32];
    let mut feeds: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    feeds.push(("d_output", &one));
    let outs = c.run(&feeds);
    let grads = tg
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.clone(), outs[1 + i].clone()))
        .collect();
    (outs[0][0], grads)
}

fn compare_grads(spec: TrainSpec, tag: &str) {
    let dir = synth_dir(tag);
    let (mut src, cfg) = open(&dir);
    let b = batch(cfg.vocab);
    let (want_loss, want) = whole_grads(&mut src, &cfg, &spec, &b);

    let (src2, _) = open(&dir);
    let mut st = StreamTrainer::new(
        Box::new(src2),
        cfg.clone(),
        spec,
        StreamOptions::default(),
        None,
    );
    let (loss, got) = st.loss_and_grads(&b).unwrap();
    assert!(
        (loss - want_loss).abs() < 1e-4 * want_loss.abs().max(1.0),
        "loss {loss} vs whole-graph {want_loss}"
    );
    assert_eq!(
        {
            let mut k: Vec<_> = got.keys().cloned().collect();
            k.sort();
            k
        },
        {
            let mut k: Vec<_> = want.keys().cloned().collect();
            k.sort();
            k
        },
        "streamed and whole-graph train the same params"
    );
    let mut nonzero = 0;
    for (name, w) in &want {
        let g = &got[name];
        assert_eq!(g.len(), w.len(), "{name}");
        if w.iter().all(|x| *x == 0.0) {
            assert!(
                g.iter().all(|x| x.abs() < 1e-7),
                "{name}: whole-graph grad is zero, streamed is not"
            );
            continue;
        }
        nonzero += 1;
        let err = max_rel(g, w);
        assert!(
            err < 2e-3,
            "{name}: streamed grad differs from whole-graph by rel {err}"
        );
    }
    assert!(nonzero > 0, "some gradient is nonzero");
}

#[test]
fn streamed_full_gradients_match_whole_graph() {
    compare_grads(TrainSpec::full(), "full");
}

#[test]
fn streamed_lora_gradients_match_whole_graph() {
    compare_grads(TrainSpec::lora(4, 8.0, &[]), "lora");
}

/// Central finite differences of the whole-graph loss agree with autodiff on
/// one element (the largest-gradient one) of each kind of weight: KDA
/// projection / conv / decay, MLA, the MoE router (through the differentiable
/// routing-weight gather), Attention Residuals, dense MLP and the LM head.
#[test]
fn whole_graph_gradients_match_finite_differences() {
    let dir = synth_dir("fd");
    let (mut src, cfg) = open(&dir);
    let spec = TrainSpec::full();
    let b = batch(cfg.vocab);
    let (_, grads) = whole_grads(&mut src, &cfg, &spec, &b);

    let (w, embed) = load_flow_weights(&mut src, &cfg).unwrap();
    let tg = build_train_graph(&w, &embed, &cfg, &spec, &HashMap::new(), 0).unwrap();
    let mut c = Session::new(Device::Cpu).compile(tg.graph.clone());
    for (n, v) in &tg.params {
        c.set_param(n, v);
    }
    let inputs = batch_inputs(&b);
    let feeds: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    let mut loss_at = |name: &str, idx: usize, delta: f32| -> f64 {
        let mut v = tg.params[name].clone();
        v[idx] += delta;
        c.set_param(name, &v);
        let l = c.run(&feeds)[0][0] as f64;
        c.set_param(name, &tg.params[name]);
        l
    };
    let checks = [
        "l1.self_attn.q_proj",
        "l1.self_attn.k_conv1d",
        "l1.self_attn.A_log",
        "l2.self_attn.kv_b_proj",
        "l2.block_sparse_moe.gate.weight",
        "l2.block_sparse_moe.routed_expert_up_proj",
        "l3.sa_res.proj",
        "l0.mlp.down_proj",
        "lm_head.weight",
    ];
    for name in checks {
        let g = grads
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not trainable"));
        let (idx, &an) = g
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap();
        // The loss is only piecewise smooth: a perturbation that flips a top-k
        // expert choice in a later MoE layer makes it jump, and a difference
        // across the jump is not a derivative. So take the reference at two step
        // sizes, shrinking until the two agree, and only then compare.
        let mut fd = |h: f32| (loss_at(name, idx, h) - loss_at(name, idx, -h)) / (2.0 * h as f64);
        let tol = 0.03 * an.abs() as f64 + 2e-4;
        let mut reference = None;
        let mut tried = Vec::new();
        for h in [1e-2f32, 2e-3, 5e-4] {
            let (f1, f2) = (fd(h), fd(h / 2.0));
            tried.push((h, f1, f2));
            if (f1 - f2).abs() < tol {
                reference = Some(f2);
                break;
            }
        }
        let fdv = reference
            .unwrap_or_else(|| panic!("{name}[{idx}]: finite differences never settle: {tried:?}"));
        assert!(
            (fdv - an as f64).abs() < tol,
            "{name}[{idx}]: autodiff {an} vs finite differences {fdv} ({tried:?})"
        );
    }
}

#[test]
fn both_trainers_reduce_the_loss() {
    let dir = synth_dir("fit");
    let (mut src, cfg) = open(&dir);
    let b = batch(cfg.vocab);

    // Whole graph, LoRA, via rlx_tune::Trainer.
    let spec = TrainSpec::lora(4, 8.0, &[]);
    let (w, embed) = load_flow_weights(&mut src, &cfg).unwrap();
    let tg = build_train_graph(&w, &embed, &cfg, &spec, &HashMap::new(), 0).unwrap();
    let dp = rlx_tune::DpConfig::new(1e-2);
    let mut wt = rlx_kimi_k3::train::WholeTrainer::new(tg, 20, None, &dp).unwrap();
    let first = wt.step(&b).unwrap().loss;
    let mut last = first;
    for _ in 0..19 {
        last = wt.step(&b).unwrap().loss;
    }
    assert!(
        last < 0.8 * first,
        "whole-graph LoRA: loss {first} -> {last}"
    );

    // Streamed, full fine-tune (attention + FFN of every layer, router, head).
    let (src2, _) = open(&dir);
    let opts = StreamOptions {
        lr: 3e-3,
        cache_layers: true,
        ..StreamOptions::default()
    };
    let mut st = StreamTrainer::new(Box::new(src2), cfg.clone(), TrainSpec::full(), opts, None);
    let first = st.train_step(std::slice::from_ref(&b)).unwrap().loss;
    let mut last = first;
    for _ in 0..9 {
        last = st.train_step(std::slice::from_ref(&b)).unwrap().loss;
    }
    assert!(last < 0.8 * first, "streamed full: loss {first} -> {last}");
}

/// Reusing compiled graphs across steps must not change a single bit: every
/// trained value and every compact expert bank is re-uploaded on a hit.
#[test]
fn graph_cache_is_exact() {
    let dir = synth_dir("cache");
    let (_, cfg) = open(&dir);
    let batches: Vec<Batch> = (0..3)
        .map(|k| Batch {
            tokens: (0..SEQ)
                .map(|i| ((i * 7 + k * 13 + 3) % cfg.vocab) as u32)
                .collect(),
            labels: (0..SEQ)
                .map(|i| ((i * 5 + k * 3 + 11) % cfg.vocab) as f32)
                .collect(),
            mask: vec![1.0; SEQ],
        })
        .collect();
    let run = |cache: bool, spec: TrainSpec| {
        let (src, _) = open(&dir);
        let opts = StreamOptions {
            lr: 3e-3,
            cache_layers: cache,
            ..StreamOptions::default()
        };
        let mut st = StreamTrainer::new(Box::new(src), cfg.clone(), spec, opts, None);
        let t = std::time::Instant::now();
        let losses: Vec<f32> = batches
            .iter()
            .map(|b| st.train_step(std::slice::from_ref(b)).unwrap().loss)
            .collect();
        (losses, st.trained().clone(), t.elapsed())
    };
    for spec in [TrainSpec::full(), TrainSpec::lora(4, 8.0, &[])] {
        let (l0, p0, t0) = run(false, spec.clone());
        let (l1, p1, t1) = run(true, spec);
        assert_eq!(l0, l1, "losses differ with the graph cache");
        assert_eq!(p0.len(), p1.len());
        for (k, v) in &p0 {
            assert_eq!(&p1[k], v, "{k} differs with the graph cache");
        }
        eprintln!("3 steps: uncached {t0:?}, cached {t1:?}");
    }
}
