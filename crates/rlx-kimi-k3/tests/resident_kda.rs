// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Resident KDA state (the `scan` matrix kept in each cached decode graph
//! between tokens) is bit-identical to passing it in and reading it back every
//! token — with two sequences interleaved on one cache (so states are parked
//! and restored), a multi-token step after resident steps (so the host copy is
//! fetched back), and `sync_kda_state` reproducing the host-path state.

mod common;
use common::TempDir;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::runner::{AttnState, DecodeState, LayerCache, decode_forward_range};
use rlx_kimi_k3::source::{CheckpointSource, EMBED};
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_runtime::Device;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `RLX_TEST_DEVICE=metal|cuda|…` runs it on that backend (default CPU).
fn device() -> Device {
    rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .and_then(|d| rlx_cli::parse_device(&d).ok())
        .unwrap_or(Device::Cpu)
}

#[test]
fn resident_kda_state_is_bit_identical() {
    let dev = device();
    let dir = TempDir::new("rlx-kimi-resident");
    write_synthetic_checkpoint(&dir, &SynthSpec::default()).unwrap();
    let tc = CheckpointSource::open(&dir).unwrap().text_config().clone();
    let n = tc.num_hidden_layers;
    // Steps per sequence: (tokens fed this step). The 2-token step forces the
    // uncached path, which must fetch the resident state back first.
    let plan: [&[u32]; 6] = [&[5], &[17], &[3], &[40, 41], &[22], &[9]];
    let run = |resident: bool| {
        let mut ck = CheckpointLoader::open(&dir).unwrap();
        let mut cache = LayerCache::from_env().with_resident_kda(resident);
        let cfg1 = tc.flow_config(1, 1);
        let mut a = DecodeState::zeros(&tc, &cfg1);
        let mut b = DecodeState::zeros(&tc, &cfg1);
        let mut outs = Vec::new();
        for toks in plan {
            // Interleave two sequences on the same cached graphs.
            for (k, st) in [&mut a, &mut b].into_iter().enumerate() {
                let toks: Vec<u32> = toks.iter().map(|t| t + k as u32).collect();
                let cfg = tc.flow_config(1, toks.len());
                let h0 = ck.gather_embed(EMBED, &toks, cfg.hidden).unwrap();
                let (h, _) = decode_forward_range(
                    &mut ck,
                    &tc,
                    &cfg,
                    h0,
                    Vec::new(),
                    st,
                    0,
                    n,
                    &mut cache,
                    dev,
                )
                .unwrap();
                outs.push(h);
            }
        }
        let was_resident = a.has_resident_layers();
        cache.sync_kda_state(&mut a).unwrap();
        cache.sync_kda_state(&mut b).unwrap();
        (outs, a, b, was_resident)
    };
    let (plain, pa, pb, none) = run(false);
    let (res, ra, rb, some) = run(true);
    assert!(!none, "host path must not leave state resident");
    assert!(some, "the resident path was not taken");
    for (step, (x, y)) in plain.iter().zip(&res).enumerate() {
        assert_eq!(bits(x), bits(y), "hidden differs at step {step}");
    }
    for (p, r) in [(&pa, &ra), (&pb, &rb)] {
        for (i, (lp, lr)) in p.layers.iter().zip(&r.layers).enumerate() {
            if let (AttnState::Kda(x), AttnState::Kda(y)) = (lp, lr) {
                assert_eq!(bits(&x.scan), bits(&y.scan), "synced scan of layer {i}");
                assert!(x.scan.iter().any(|v| *v != 0.0));
            }
        }
    }
}
