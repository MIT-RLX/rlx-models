// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Weight inventory / dumps, activation traces and checkpoint export, on a
//! synthetic checkpoint read through the real loader.

mod common;
use common::TempDir;
use rlx_kimi_k3::export::{export_full, load_lora_adapter, save_lora_adapter};
use rlx_kimi_k3::inspect::{
    TraceOptions, dump_weights, inventory, logical_names, selected, summarize, trace,
};
use rlx_kimi_k3::layer_exec::{LayerSpec, forward_head, forward_layer};
use rlx_kimi_k3::source::{CheckpointSource, WeightSource, load_flow_weights};
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_kimi_k3::train::{Batch, TrainSpec, batch_inputs, build_train_graph, merge_lora};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

const SEQ: usize = 4;

fn tmp(tag: &str) -> TempDir {
    TempDir::new(&format!("rlx-kimi-ie-{tag}"))
}

fn synth(tag: &str) -> TempDir {
    let d = tmp(tag);
    write_synthetic_checkpoint(&d, &SynthSpec::default()).unwrap();
    d
}

fn read_st(path: &std::path::Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let v = st.tensor(name).unwrap();
    (
        v.shape().to_vec(),
        v.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[test]
fn inventory_and_weight_dump_cover_the_checkpoint() {
    let dir = synth("inv");
    let mut src = CheckpointSource::open(&dir).unwrap();
    let inv = inventory(&mut src.ck).unwrap();
    let sum = summarize(&inv);
    assert_eq!(sum.tensors, src.ck.tensor_names().len());
    assert!(sum.by_component.contains_key("layer.experts"));
    assert!(sum.by_component.contains_key("layer.attention"));
    assert!(sum.by_dtype.contains_key("U8") && sum.by_dtype.contains_key("BF16"));
    // Every tensor has a component (nothing classified "other").
    assert!(
        inv.iter().all(|e| e.component != "other"),
        "unclassified tensors"
    );

    let names: Vec<String> = logical_names(&inv)
        .into_iter()
        .filter(|n| selected(n, &[], Some(&(1..2))))
        .collect();
    assert!(names.iter().any(|n| n.ends_with("experts.0.w1.weight")));
    let out = tmp("inv-dump");
    let manifest = dump_weights(&mut src.ck, &names, &out, Some(1 << 30)).unwrap();
    for n in &names {
        let file = manifest[n]["file"].as_str().unwrap();
        let (shape, v) = read_st(&out.join(file), n);
        let (want_shape, want) = src.ck.tensor_f32_any(n).unwrap();
        assert_eq!(shape, want_shape, "{n}");
        assert_eq!(v, want, "{n}");
    }
    assert!(dump_weights(&mut src.ck, &names, &tmp("inv-cap"), Some(16)).is_err());
}

#[test]
fn trace_records_layer_activations_and_gradients() {
    let dir = synth("trace");
    let mut src = CheckpointSource::open(&dir).unwrap();
    let cfg = src.text_config().flow_config(1, SEQ);
    let tokens: Vec<u32> = vec![3, 9, 27, 17];
    let labels: Vec<u32> = vec![9, 27, 17, 51];
    let out = tmp("trace-out");
    let opts = TraceOptions {
        layers: 0..src.num_layers(),
        include: Vec::new(),
        tensors: true,
        grads: true,
        device: Device::Cpu,
    };
    let s = trace(&mut src, &cfg, &tokens, Some(&labels), &opts, Some(&out)).unwrap();
    assert!(out.join("manifest.json").exists());
    assert_eq!(s.top5.len(), 5);
    assert!(!s.experts.is_empty());
    for want in [
        "l0.in",
        "l1.self_attn.scan",
        "l1.block_sparse_moe.router",
        "l2.self_attn.attn",
        "l3.sa_res",
        "final_norm",
        "logits",
        "d l0.in",
    ] {
        assert!(
            s.entries.iter().any(|e| e.name == want),
            "trace has no {want}"
        );
    }
    assert!(
        s.entries
            .iter()
            .all(|e| e.stats.nan == 0 && e.stats.inf == 0)
    );

    // The traced final hidden state is the streamed forward's.
    let store = HashMap::new();
    let spec = LayerSpec {
        cfg: &cfg,
        device: Device::Cpu,
        lora: None,
        overrides: &store,
        seed: 0,
        cache: None,
    };
    let (mut h, mut snaps) = (src.embed(&tokens, cfg.hidden).unwrap(), Vec::new());
    for i in 0..src.num_layers() {
        let o = forward_layer(&mut src, i, &h, &snaps, &spec, false).unwrap();
        h = o.h_out;
        snaps = o.snaps_out;
    }
    let (logits, _) = forward_head(&src.head(&cfg).unwrap(), &h, &snaps, &spec, false).unwrap();
    let (_, traced) = read_st(&out.join("head.safetensors"), "logits");
    assert_eq!(traced, logits);

    // d l0.in is the loss gradient w.r.t. the embedding rows: compare with the
    // whole-graph embedding-table gradient (distinct tokens, so rows don't mix).
    let mut spec_t = TrainSpec::full();
    spec_t.train_embed = true;
    spec_t.only = vec!["embed_tokens".into()];
    let (w, embed) = load_flow_weights(&mut src, &cfg).unwrap();
    let tg = build_train_graph(&w, &embed, &cfg, &spec_t, &HashMap::new(), 0).unwrap();
    let ids: Vec<_> = tg.slots.iter().map(|s| s.node).collect();
    let mut c = Session::new(Device::Cpu).compile(rlx_autodiff::grad_with_loss(&tg.graph, &ids));
    for (n, v) in &tg.params {
        c.set_param(n, v);
    }
    let b = Batch {
        tokens: tokens.clone(),
        labels: labels.iter().map(|&t| t as f32).collect(),
        mask: vec![1.0; SEQ],
    };
    let inputs = batch_inputs(&b);
    let one = [1.0f32];
    let mut feeds: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    feeds.push(("d_output", &one));
    let outs = c.run(&feeds);
    assert!(
        (outs[0][0] - s.loss.unwrap()).abs() < 1e-4,
        "loss {} vs {:?}",
        outs[0][0],
        s.loss
    );
    let g_table = &outs[1];
    let (_, d_in) = read_st(&out.join("grad-layer-000.safetensors"), "d l0.in");
    let hid = cfg.hidden;
    let scale = d_in.iter().fold(0f32, |m, x| m.max(x.abs()));
    for (r, &t) in tokens.iter().enumerate() {
        for c in 0..hid {
            let (a, b) = (d_in[r * hid + c], g_table[t as usize * hid + c]);
            assert!((a - b).abs() <= 2e-3 * scale, "row {r} col {c}: {a} vs {b}");
        }
    }
}

#[test]
fn export_round_trips_through_the_loader() {
    let dir = synth("exp");
    let mut src = CheckpointSource::open(&dir).unwrap();
    let cfg = src.text_config().flow_config(1, SEQ);
    let (w, embed) = load_flow_weights(&mut src, &cfg).unwrap();
    let tg = build_train_graph(&w, &embed, &cfg, &TrainSpec::full(), &HashMap::new(), 0).unwrap();
    // "Trained" = every trainable weight nudged by a known amount.
    let trained: HashMap<String, Vec<f32>> = tg
        .slots
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                tg.params[&s.name].iter().map(|x| x + 0.25).collect(),
            )
        })
        .collect();
    let out = tmp("exp-out");
    export_full(&dir, &out, &trained, &tg.shapes, false).unwrap();
    let mut back = CheckpointSource::open(&out).unwrap();
    let (w2, embed2) = load_flow_weights(&mut back, &cfg).unwrap();
    let tg2 =
        build_train_graph(&w2, &embed2, &cfg, &TrainSpec::full(), &HashMap::new(), 0).unwrap();
    for (name, want) in &trained {
        assert_eq!(&tg2.params[name], want, "{name} did not round-trip");
    }
    // Untouched tensors (experts, embeddings) still come from the source.
    assert_eq!(embed2, embed);
    for (name, v) in &tg.params {
        if !trained.contains_key(name) {
            assert_eq!(&tg2.params[name], v, "{name} changed without being trained");
        }
    }
}

#[test]
fn lora_adapter_round_trips_and_merges() {
    let dir = synth("lora");
    let mut src = CheckpointSource::open(&dir).unwrap();
    let cfg = src.text_config().flow_config(1, SEQ);
    let (w, embed) = load_flow_weights(&mut src, &cfg).unwrap();
    let spec = TrainSpec::lora(4, 8.0, &[]);
    let tg = build_train_graph(&w, &embed, &cfg, &spec, &HashMap::new(), 0).unwrap();
    // Non-zero B so the adapter changes the model.
    let adapters: HashMap<String, Vec<f32>> = tg
        .params
        .iter()
        .filter(|(k, _)| k.starts_with("lora."))
        .map(|(k, v)| {
            let v = if k.ends_with(".b") {
                (0..v.len())
                    .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
                    .collect()
            } else {
                v.clone()
            };
            (k.clone(), v)
        })
        .collect();
    let out = tmp("lora-out");
    let lora = spec.lora_spec().unwrap();
    save_lora_adapter(
        &out,
        &adapters,
        &tg.shapes,
        lora.rank,
        lora.alpha,
        &lora.target_modules,
    )
    .unwrap();
    let back = load_lora_adapter(&out).unwrap();
    assert_eq!(back.rank, 4);
    assert_eq!(back.params.len(), adapters.len());
    for (k, v) in &adapters {
        assert_eq!(&back.params[k], v, "{k}");
    }

    // Loss of the adapted graph == loss of the merged dense weights.
    let b = Batch {
        tokens: vec![1, 5, 9, 13],
        labels: vec![5.0, 9.0, 13.0, 2.0],
        mask: vec![1.0; SEQ],
    };
    let loss = |tg: &rlx_kimi_k3::train::TrainGraph, over: &HashMap<String, Vec<f32>>| {
        let mut c = Session::new(Device::Cpu).compile(tg.graph.clone());
        for (n, v) in &tg.params {
            c.set_param(n, over.get(n).unwrap_or(v));
        }
        let inputs = batch_inputs(&b);
        let feeds: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        c.run(&feeds)[0][0]
    };
    let adapted = loss(&tg, &adapters);
    let full = build_train_graph(&w, &embed, &cfg, &TrainSpec::full(), &HashMap::new(), 0).unwrap();
    let mut base: HashMap<String, Vec<f32>> = full.params.clone();
    merge_lora(&mut base, &adapters, &tg.shapes, lora.scale()).unwrap();
    let merged = loss(&full, &base);
    let plain = loss(&full, &HashMap::new());
    assert!(
        (adapted - merged).abs() < 1e-4,
        "adapted {adapted} vs merged {merged}"
    );
    assert!(
        (adapted - plain).abs() > 1e-4,
        "the adapter must change the model"
    );
}
