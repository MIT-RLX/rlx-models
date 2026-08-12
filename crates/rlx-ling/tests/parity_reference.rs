// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Numerical parity against the PyTorch reference forward.
//!
//! Generate the fixture, then point the test at it:
//!
//! ```sh
//! python3 scripts/ling3_reference.py .fixtures/ling3-parity
//! RLX_LING_PARITY_DIR=.fixtures/ling3-parity cargo test -p rlx-ling --test parity_reference
//! ```
//!
//! Without `RLX_LING_PARITY_DIR` the test skips — the fixture is ~1 MB of
//! generated weights and needs torch, so it is not committed.
//!
//! This is what actually pins the arithmetic: the KDA gate branch, the L2 norm,
//! the causal short conv, the delta-net recurrence and its `q/√n` readout, the
//! interleaved MLA RoPE, the head-wise output gate, the grouped `noaux_tc`
//! router and the shared expert. The synthetic-weight smoke test only proves the
//! graph is finite; this proves it computes Ling 3.0.

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_ling::{LingConfig, build_ling_text_flow, prepare_checkpoint};
use rlx_runtime::Device;
use std::path::PathBuf;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn fixture_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("RLX_LING_PARITY_DIR").ok()?);
    if dir.join("model.safetensors").is_file() {
        Some(dir)
    } else {
        panic!(
            "RLX_LING_PARITY_DIR={dir:?} has no model.safetensors — run scripts/ling3_reference.py first"
        );
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn matches_pytorch_reference() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: set RLX_LING_PARITY_DIR (see the module docs)");
        return;
    };

    let cfg = LingConfig::from_file(dir.join("config.json")).expect("config");
    let ids: Vec<u32> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("input_ids.json")).expect("ids"))
            .expect("ids json");
    let raw = std::fs::read(dir.join("logits.f32")).expect("reference logits");
    let expect: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let seq = ids.len();
    assert_eq!(expect.len(), seq * cfg.vocab_size, "reference logits shape");

    let mut wm = WeightMap::from_safetensors_dir(&dir).expect("load fixture weights");
    prepare_checkpoint(&cfg, &mut wm).expect("prepare checkpoint");
    let built = build_ling_text_flow(&cfg, &mut wm, seq, true).expect("build ling flow");
    let mut compiled = compile_built(built, dev()).expect("compile ling flow");

    let (cos, sin) = cfg.rope_tables(seq);
    let ids_f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let got = compiled
        .run(&[
            ("input_ids", ids_f.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ])
        .into_iter()
        .next()
        .expect("forward output");
    assert_eq!(got.len(), expect.len());

    let cos_sim = cosine(&got, &expect);
    let max_abs = got
        .iter()
        .zip(&expect)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = expect.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);

    // f32 accumulation order differs between the graph and torch; the tolerance
    // is set for that, not for a different formula. A wrong gate branch or a
    // transposed weight lands orders of magnitude outside it.
    //
    // CoreML/ANE is the exception: it evaluates in fp16 by design (fp16 eps is
    // ~1e-3, and 24 layers accumulate to a few e-3). Holding it to the f32 bound
    // would be testing the hardware's dtype, not this crate — so it gets an
    // fp16-appropriate bound while every other backend keeps the strict one.
    let (rel_bound, why) = match dev() {
        Device::Ane => (2e-2, "fp16 (ANE)"),
        _ => (1e-3, "f32"),
    };
    assert!(
        cos_sim > 0.999_99,
        "cosine {cos_sim:.8} — logits diverge from the reference"
    );
    assert!(
        max_abs / scale < rel_bound,
        "max abs diff {max_abs:.6} (rel {:.2e}) exceeds the {why} bound {rel_bound:.0e}",
        max_abs / scale
    );

    // Same argmax per position — the property a sampler actually depends on.
    for pos in 0..seq {
        let row = |v: &[f32]| {
            let s = &v[pos * cfg.vocab_size..(pos + 1) * cfg.vocab_size];
            s.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(row(&got), row(&expect), "argmax differs at position {pos}");
    }
    eprintln!("parity: cosine {cos_sim:.8}, max |Δ| {max_abs:.3e}");
}
