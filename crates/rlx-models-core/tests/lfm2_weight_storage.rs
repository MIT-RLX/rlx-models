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

//! The LFM2 prefill graph with half-precision dense weights
//! (`DenseStore` / `WeightStorage`) on every compiled-in backend, against the
//! f32 graph over the same rounded weights on CPU.

use std::collections::{BTreeMap, HashMap};

use rlx_models_core::standard_decoder::{DenseStore, Lfm2Spec, build_lfm2_prefill_stored};
use rlx_models_core::weight_loader::SyntheticLoader;
use rlx_models_core::weight_storage::WeightStorage;
use rlx_runtime::{Device, Session};

fn spec() -> Lfm2Spec {
    Lfm2Spec {
        vocab_size: 1000,
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 64,
        conv_dim: 256,
        conv_kernel: 3,
        full_attn_layers: vec![1, 3],
        rope_theta: 1e6,
        rms_norm_eps: 1e-5,
    }
}

fn shapes(s: &Lfm2Spec) -> BTreeMap<String, Vec<usize>> {
    let (h, i, dh) = (s.hidden_size, s.intermediate_size, s.head_dim);
    let kv = s.num_key_value_heads * dh;
    let mut m = BTreeMap::new();
    m.insert("model.embedding_norm.weight".into(), vec![h]);
    for l in 0..s.num_hidden_layers {
        let p = format!("model.layers.{l}");
        m.insert(format!("{p}.operator_norm.weight"), vec![h]);
        m.insert(format!("{p}.ffn_norm.weight"), vec![h]);
        m.insert(format!("{p}.feed_forward.w1.weight"), vec![i, h]);
        m.insert(format!("{p}.feed_forward.w3.weight"), vec![i, h]);
        m.insert(format!("{p}.feed_forward.w2.weight"), vec![h, i]);
        if s.full_attn_layers.contains(&l) {
            m.insert(format!("{p}.self_attn.q_proj.weight"), vec![h, h]);
            m.insert(format!("{p}.self_attn.k_proj.weight"), vec![kv, h]);
            m.insert(format!("{p}.self_attn.v_proj.weight"), vec![kv, h]);
            m.insert(format!("{p}.self_attn.out_proj.weight"), vec![h, h]);
            m.insert(format!("{p}.self_attn.q_layernorm.weight"), vec![dh]);
            m.insert(format!("{p}.self_attn.k_layernorm.weight"), vec![dh]);
        } else {
            m.insert(format!("{p}.conv.in_proj.weight"), vec![3 * h, h]);
            m.insert(format!("{p}.conv.conv.weight"), vec![h, 1, 3]);
            m.insert(format!("{p}.conv.out_proj.weight"), vec![h, h]);
        }
    }
    m
}

fn devices() -> Vec<Device> {
    // `mut` only matters with a backend feature on.
    #[allow(unused_mut)]
    let mut d = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    d.push(Device::Metal);
    #[cfg(feature = "mlx")]
    d.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    d.push(Device::Gpu);
    #[cfg(feature = "cuda")]
    d.push(Device::Cuda);
    d
}

/// Final hidden states of the prefill graph on `device`.
fn run(device: Device, storage: WeightStorage, embeds: &[f32], seq: usize) -> Vec<f32> {
    let s = spec();
    let mut loader = SyntheticLoader::new(shapes(&s));
    let mut packed = HashMap::new();
    let mut store = DenseStore::new(storage);
    let (graph, params) =
        build_lfm2_prefill_stored(&s, &mut loader, seq, &mut packed, true, &mut store).unwrap();
    if storage != WeightStorage::F32 {
        assert!(
            !store.typed.is_empty(),
            "no projection was kept at half width"
        );
    }
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        device,
    );
    let mut c = Session::new(device).compile_with(graph, &opts);
    for (n, d) in &params {
        c.set_param(n, d);
    }
    for (n, (b, dt)) in &store.typed {
        c.set_param_typed(n, b, *dt);
    }
    c.run(&[("inputs_embeds", embeds)])
        .into_iter()
        .next()
        .unwrap()
}

#[test]
fn half_storage_prefill_matches_on_every_backend() {
    let seq = 24;
    let h = spec().hidden_size;
    let embeds: Vec<f32> = (0..seq * h)
        .map(|i| ((i * 29 % 113) as f32 - 56.0) / 90.0)
        .collect();
    let mut failures = Vec::new();
    for storage in [WeightStorage::Bf16, WeightStorage::F16] {
        let reference = run(Device::Cpu, storage, &embeds, seq);
        for dev in devices() {
            let got = run(dev, storage, &embeds, seq);
            let err = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("{dev:?} {storage}: max |d| vs CPU {err:.2e}");
            if err.is_nan() || err >= 1e-3 {
                failures.push(format!("{dev:?} {storage}: {err:.2e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "half-storage LFM2 prefill wrong on {failures:?}"
    );
}
