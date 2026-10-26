// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The sizing model's parameter counts against what a checkpoint actually
//! stores, and its checkpoint-byte estimate against the measured bytes.

mod common;
use common::TempDir;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::sizing::Footprint;
use rlx_kimi_k3::source::CheckpointSource;
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};

#[test]
fn footprint_counts_every_stored_parameter() {
    let dir = TempDir::new("rlx-kimi-sizing");
    write_synthetic_checkpoint(&dir, &SynthSpec::default()).unwrap();
    let src = CheckpointSource::open(&dir).unwrap();
    let fp = Footprint::from_config(src.text_config());
    let mut ck = CheckpointLoader::open(&dir).unwrap();
    let (mut dense, mut experts) = (0u64, 0u64);
    for n in ck.tensor_names() {
        let (dtype, shape, _, _) = ck.tensor_meta(&n).unwrap();
        let elems: u64 = shape.iter().map(|&d| d as u64).product();
        if n.ends_with("weight_packed") {
            experts += elems * 2; // two E2M1 codes per byte
        } else if dtype != "U8" {
            dense += elems;
        }
    }
    assert_eq!(
        dense,
        fp.dense_params() + fp.embed_params,
        "dense parameters"
    );
    assert_eq!(experts, fp.experts_total_params, "expert parameters");
    // The byte estimate (bf16 dense, MXFP4 experts) is close to the measured one
    // (the synthetic checkpoint keeps conv / A_log / dt_bias / bias in f32).
    let est = fp.checkpoint_bytes;
    let measured = fp
        .clone()
        .with_checkpoint(&mut ck)
        .unwrap()
        .checkpoint_bytes;
    assert!(
        (est - measured).abs() / measured < 0.05,
        "estimate {est} vs measured {measured}"
    );
}

/// A config that disagrees with its checkpoint is refused at open.
#[test]
fn mismatched_config_is_refused() {
    let dir = TempDir::new("rlx-kimi-dims");
    write_synthetic_checkpoint(&dir, &SynthSpec::default()).unwrap();
    assert!(CheckpointSource::open(&dir).is_ok());
    // Drop the expert width: the config falls back to its default (3072).
    let p = dir.join("config.json");
    let mut cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    cfg["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("moe_intermediate_size");
    std::fs::write(&p, serde_json::to_vec(&cfg).unwrap()).unwrap();
    let err = CheckpointSource::open(&dir)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("experts.0.w1.weight_packed"), "{err}");
}
