// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Dense residency (`common::set_dense_residency`): bf16 holds the checkpoint's
//! own bf16 values, so decode matches f32 to accumulation order through every
//! layer; int8 is lossy — checked on the first (dense) layer, since through MoE
//! layers on random weights its error tips top-k choices and compounds. One
//! test in its own binary: the setting is process-wide.
//!
//! bf16 residency was silently wrong on CPU until rlx-cpu learned to
//! concatenate 2-byte types (`FuseSharedInputMatMul` concatenates the bf16
//! gate/up weights): relative error ~1.4 at the first layer.

mod common;
use common::TempDir;
use rlx_kimi_k3::common::{DenseResidency, set_dense_residency};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::runner::{DecodeState, LayerCache, decode_forward_range};
use rlx_kimi_k3::source::{CheckpointSource, EMBED};
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_runtime::Device;

#[test]
fn bf16_and_int8_residency_track_f32() {
    let dir = TempDir::new("rlx-kimi-dense");
    write_synthetic_checkpoint(&dir, &SynthSpec::default()).unwrap();
    let tc = CheckpointSource::open(&dir).unwrap().text_config().clone();
    let run = |r: DenseResidency, layers: usize| {
        set_dense_residency(Some(r));
        let mut ck = CheckpointLoader::open(&dir).unwrap();
        let cfg1 = tc.flow_config(1, 1);
        let mut state = DecodeState::zeros(&tc, &cfg1);
        let mut cache = LayerCache::from_env();
        let mut out = Vec::new();
        for t in [5u32, 17, 3, 40, 22, 9] {
            let h0 = ck.gather_embed(EMBED, &[t], cfg1.hidden).unwrap();
            let (h, _) = decode_forward_range(
                &mut ck,
                &tc,
                &cfg1,
                h0,
                Vec::new(),
                &mut state,
                0,
                layers,
                &mut cache,
                Device::Cpu,
            )
            .unwrap();
            out.push(h);
        }
        set_dense_residency(None);
        out
    };
    let n = tc.num_hidden_layers;
    let f32_ = run(DenseResidency::F32, n);
    let bf16 = run(DenseResidency::Bf16, n);
    let f32_1 = run(DenseResidency::F32, 1);
    let int8_1 = run(DenseResidency::Int8, 1);
    let rel = |a: &[f32], b: &[f32]| {
        let s = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / s
    };
    for (step, (a, b)) in f32_.iter().zip(&bf16).enumerate() {
        assert!(
            rel(b, a) < 1e-4,
            "bf16 vs f32 at step {step}: rel {}",
            rel(b, a)
        );
    }
    for (step, (a, c)) in f32_1.iter().zip(&int8_1).enumerate() {
        assert!(c.iter().all(|x| x.is_finite()));
        // Per-channel int8 on hidden-32 toy weights, carried by the KDA recurrence
        // across steps: a few percent.
        assert!(
            rel(c, a) < 0.1,
            "int8 vs f32 (layer 0) at step {step}: rel {}",
            rel(c, a)
        );
    }
}
