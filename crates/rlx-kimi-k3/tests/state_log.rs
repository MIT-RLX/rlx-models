// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! KDA state log (feature `state-log`) on a synthetic checkpoint: logging does
//! not change the generation, and every KDA state rebuilt from sparse snapshots
//! equals the state the generation held, bit for bit.
#![cfg(feature = "state-log")]

mod common;
use common::TempDir;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::runner::run_generate;
use rlx_kimi_k3::source::CheckpointSource;
use rlx_kimi_k3::state_log::{StateLog, StateLogOptions, generate_logged};
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_runtime::Device;
use std::path::Path;

const PROMPT: [u32; 5] = [3, 14, 15, 9, 26];
const GEN: usize = 6;

fn tmp(tag: &str) -> TempDir {
    TempDir::new(&format!("rlx-kimi-statelog-{tag}"))
}

fn synth() -> TempDir {
    let d = tmp("ck");
    write_synthetic_checkpoint(&d, &SynthSpec::default()).unwrap();
    d
}

fn log(ck_dir: &Path, out: &Path, every: usize, stepwise: bool) -> Vec<u32> {
    let src = CheckpointSource::open(ck_dir).unwrap();
    let tc = src.text_config().clone();
    let mut ck = CheckpointLoader::open(ck_dir).unwrap();
    generate_logged(
        &mut ck,
        &tc,
        |s| tc.flow_config(1, s),
        &PROMPT,
        GEN,
        tc.num_hidden_layers,
        Device::Cpu,
        &StateLogOptions {
            dir: out.to_path_buf(),
            snapshot_every: every,
            conv: true,
            stepwise_prefill: stepwise,
        },
    )
    .unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn state_log_rebuilds_every_state_exactly() {
    let ck_dir = synth();

    // Logging must not change the generation.
    let src = CheckpointSource::open(&ck_dir).unwrap();
    let tc = src.text_config().clone();
    let mut ck = CheckpointLoader::open(&ck_dir).unwrap();
    let plain = run_generate(
        &mut ck,
        &tc,
        |s| tc.flow_config(1, s),
        &PROMPT,
        GEN,
        tc.num_hidden_layers,
        Device::Cpu,
    )
    .unwrap();
    let chunked_truth = tmp("chunk1");
    assert_eq!(
        log(&ck_dir, &chunked_truth, 1, false),
        plain,
        "logged run diverged"
    );

    // Stepwise prefill: every position's state is the run's own. Reference with a
    // snapshot at every token; a second run snapshots every 4.
    let truth_dir = tmp("step1");
    let sparse_dir = tmp("step4");
    let t1 = log(&ck_dir, &truth_dir, 1, true);
    let t4 = log(&ck_dir, &sparse_dir, 4, true);
    assert_eq!(t1, t4, "two identical runs must generate the same tokens");
    let truth = StateLog::open(&truth_dir).unwrap();
    let sparse = StateLog::open(&sparse_dir).unwrap();
    let n = truth.manifest.positions;
    assert_eq!(n, PROMPT.len() + GEN - 1);
    assert_eq!(sparse.manifest.snapshots, vec![4, 8]);
    assert!(!truth.manifest.kda_layers.is_empty());
    for p in 0..=n {
        let want = truth.snapshot(p).unwrap();
        // From the nearest sparse snapshot, and from scratch.
        for got in [
            sparse.state_at(p, Device::Cpu).unwrap(),
            sparse.state_from(0, p, Device::Cpu).unwrap(),
        ] {
            for (a, b) in got.iter().zip(&want) {
                assert_eq!(
                    bits(&a.scan),
                    bits(&b.scan),
                    "scan of layer {} at {p}",
                    a.layer
                );
                assert_eq!(
                    bits(&a.conv_q),
                    bits(&b.conv_q),
                    "conv_q of layer {} at {p}",
                    a.layer
                );
                assert_eq!(
                    bits(&a.conv_k),
                    bits(&b.conv_k),
                    "conv_k of layer {} at {p}",
                    a.layer
                );
                assert_eq!(
                    bits(&a.conv_v),
                    bits(&b.conv_v),
                    "conv_v of layer {} at {p}",
                    a.layer
                );
            }
        }
    }
    let last = truth.snapshot(n).unwrap();
    assert!(
        last.iter()
            .all(|l| l.scan.iter().any(|x| *x != 0.0) && l.conv_k.iter().any(|x| *x != 0.0)),
        "states must be non-trivial for the comparison to mean anything"
    );
    let rep = sparse.verify(Device::Cpu).unwrap();
    assert!(rep.exact && !rep.spans.is_empty(), "{rep:?}");

    // Chunked prefill: the prompt-end state is a snapshot; everything after it
    // rebuilds exactly from there.
    let chunked = tmp("chunk3");
    log(&ck_dir, &chunked, 3, false);
    let c3 = StateLog::open(&chunked).unwrap();
    let c1 = StateLog::open(&chunked_truth).unwrap();
    assert!(c3.manifest.snapshots.contains(&PROMPT.len()));
    for p in PROMPT.len()..=c3.manifest.positions {
        let (got, want) = (
            c3.state_at(p, Device::Cpu).unwrap(),
            c1.snapshot(p).unwrap(),
        );
        for (a, b) in got.iter().zip(&want) {
            assert_eq!(
                bits(&a.scan),
                bits(&b.scan),
                "chunked: scan of layer {} at {p}",
                a.layer
            );
        }
    }
    assert!(c3.verify(Device::Cpu).unwrap().exact);
}
