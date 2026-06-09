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

// Regression: Q6K GGUF tensors skip Op::DequantMatMul (broken in rlx-runtime 0.2.1).
//
//   cargo test -p rlx-models --test minicpm5_dequant_spot --release -- --nocapture

use rlx_core::flow_bridge::compile_options_for_packed_gguf_prefill;
use rlx_core::weight_loader::{GgufLoader, WeightLoader};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::Device;
use std::path::PathBuf;

fn gguf_path() -> Option<PathBuf> {
    PathBuf::from("/tmp/rlx-weights/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf")
        .is_file()
        .then_some(PathBuf::from(
            "/tmp/rlx-weights/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf",
        ))
}

#[test]
fn minicpm5_q6k_v_proj_packed_when_block_dequant_fixed() {
    let Some(path) = gguf_path() else {
        eprintln!("skip: fetch minicpm5 gguf Q4_K_M");
        return;
    };
    let mut loader = GgufLoader::from_file(path.to_str().unwrap()).expect("loader");
    let packed = loader
        .take_packed("model.layers.0.self_attn.v_proj.weight")
        .expect("probe");
    if rlx_core::dequant_matmul_supported(rlx_ir::quant::QuantScheme::GgufQ6K) {
        assert!(
            packed.is_some(),
            "Q6K v_proj should use DequantMatMul when rlx-gguf block dequant is fixed"
        );
    } else {
        assert!(
            packed.is_none(),
            "Q6K v_proj must fall back to F32 dequant until rlx-gguf block dequant is fixed"
        );
    }
}

#[test]
fn minicpm5_q4k_q_proj_dequant_matmul_matches_f32() {
    let Some(path) = gguf_path() else {
        eprintln!("skip");
        return;
    };
    let key = "model.layers.0.self_attn.q_proj.weight";
    let hidden = 1536usize;
    let out_dim = 2048usize;
    let seq = 64usize;

    let mut loader_f32 = GgufLoader::from_file(path.to_str().unwrap()).expect("loader");
    let (w_f32, _) = loader_f32.take_transposed(key).expect("f32");
    let mut loader_p = GgufLoader::from_file(path.to_str().unwrap()).expect("loader");
    let (bytes, scheme, _) = loader_p.take_packed(key).expect("packed").expect("q4k");

    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let mut ref_out = vec![0f32; seq * out_dim];
    for t in 0..seq {
        for o in 0..out_dim {
            for i in 0..hidden {
                ref_out[t * out_dim + o] += x[t * hidden + i] * w_f32[i * out_dim + o];
            }
        }
    }

    let mut g = Graph::new("q_spot");
    let x_id = g.input("x", Shape::new(&[1, seq, hidden], DType::F32));
    let w_id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
    let y_id = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_id, w_id],
        Shape::new(&[1, seq, out_dim], DType::F32),
    );
    g.set_outputs(vec![y_id]);
    let opts = compile_options_for_packed_gguf_prefill(Device::Cpu);
    let mut compiled = rlx_runtime::Session::new(Device::Cpu).compile_with(g, &opts);
    compiled.set_param_typed(key, &bytes, DType::U8);
    let got = &compiled.run(&[("x", x.as_slice())])[0];
    assert_eq!(got.len(), ref_out.len());
    let mut max_abs = 0f32;
    for (a, b) in ref_out.iter().zip(got.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!("q_proj Q4K DequantMatMul max_abs={max_abs:.6e}");
    assert!(max_abs < 1e-4, "Q4K DequantMatMul max_abs {max_abs}");
}
