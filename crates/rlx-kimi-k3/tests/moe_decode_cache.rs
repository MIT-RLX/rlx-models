// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! Decode with the per-layer MoE cache (cached dense MoE weights, compiled
//! router / experts graphs, dequantized-expert LRU) is bit-identical to decode
//! without any cache, token by token, on a synthetic checkpoint.

mod common;
use common::TempDir;
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::runner::{DecodeState, LayerCache, decode_forward_range};
use rlx_kimi_k3::source::CheckpointSource;
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_runtime::Device;

#[test]
fn cached_moe_decode_is_bit_identical() {
    let dir = TempDir::new("rlx-kimi-moecache");
    write_synthetic_checkpoint(&dir, &SynthSpec::default()).unwrap();
    let src = CheckpointSource::open(&dir).unwrap();
    let tc = src.text_config().clone();
    let n = tc.num_hidden_layers;
    let run = |mut cache: LayerCache| {
        let mut ck = CheckpointLoader::open(&dir).unwrap();
        let cfg1 = tc.flow_config(1, 1);
        let mut state = DecodeState::zeros(&tc, &cfg1);
        let mut outs = Vec::new();
        for (step, tok) in [5u32, 17, 3, 40, 22, 9, 31, 12].into_iter().enumerate() {
            let h0 = ck
                .gather_embed(
                    "language_model.model.embed_tokens.weight",
                    &[tok],
                    cfg1.hidden,
                )
                .unwrap();
            let (h, snaps) = decode_forward_range(
                &mut ck,
                &tc,
                &cfg1,
                h0,
                Vec::new(),
                &mut state,
                0,
                n,
                &mut cache,
                Device::Cpu,
            )
            .unwrap();
            outs.push((step, h, snaps));
        }
        (outs, cache.moe_layers_cached())
    };
    let (plain, none) = run(LayerCache::disabled());
    let (cached, layers) = run(LayerCache::from_env().with_expert_cache_mb(64));
    assert_eq!(none, 0);
    assert!(layers > 0, "the MoE decode cache was not used");
    for ((s, h0, s0), (_, h1, s1)) in plain.iter().zip(&cached) {
        let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(b(h0), b(h1), "hidden differs at step {s}");
        for (a, c) in s0.iter().zip(s1) {
            assert_eq!(b(a), b(c), "snapshots differ at step {s}");
        }
    }
}
