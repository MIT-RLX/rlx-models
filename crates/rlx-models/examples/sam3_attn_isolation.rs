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

// Compare IR Op::Attention vs host multihead_attention for the SAME
// decoder self-attn inputs. Isolates whether the bug is in Op::Attention
// or somewhere else.

use anyhow::Result;
use rlx_ir::op::{BinaryOp, MaskKind};
use rlx_ir::shape;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::Device;

const D: usize = 256;
const NH: usize = 8;
const DH: usize = D / NH;

fn main() -> Result<()> {
    let weights = rlx_ir::env::var("RLX_SAM3_WEIGHTS")
        .ok_or_else(|| anyhow::anyhow!("set RLX_SAM3_WEIGHTS"))?;
    let model =
        rlx_models::sam3::Sam3::from_safetensors(&weights, rlx_models::sam3::Sam3Config::base())?;
    let dw = model.decoder_weights();
    let layer0 = &dw.layers[0];

    // Synthesize inputs for one self-attn step: [B=1, L=201, D=256].
    let l = 201;
    let mut sa_x = vec![0f32; l * D];
    let mut sa_pos = vec![0f32; l * D];
    for i in 0..(l * D) {
        sa_x[i] = ((i as f32) * 0.001).sin();
        sa_pos[i] = ((i as f32) * 0.002).cos() * 0.1;
    }

    // ── HOST path ──────────────────────────────────────────────────
    let mut qk = vec![0f32; l * D];
    for i in 0..(l * D) {
        qk[i] = sa_x[i] + sa_pos[i];
    }
    let host_attn = rlx_models::sam3::tensor::multihead_attention(
        &qk,
        &qk,
        &sa_x,
        &layer0.self_attn_in_w_t,
        &layer0.self_attn_in_b,
        &layer0.self_attn_out_w_t,
        &layer0.self_attn_out_b,
        1,
        l,
        l,
        D,
        NH,
        None,
    )?;

    // ── IR path ───────────────────────────────────────────────────
    let mut g = Graph::new("attn_iso");
    let f = DType::F32;
    let x_in = g.input("x", Shape::new(&[1, l, D], f));
    let pos_in = g.input("pos", Shape::new(&[1, l, D], f));
    let qk_node = g.binary(BinaryOp::Add, x_in, pos_in, Shape::new(&[1, l, D], f));

    // Split QKV from layer's in_proj.
    let (wq, wk, wv) = split_qkv(&layer0.self_attn_in_w_t, D);
    let bq = layer0.self_attn_in_b[0..D].to_vec();
    let bk = layer0.self_attn_in_b[D..2 * D].to_vec();
    let bv = layer0.self_attn_in_b[2 * D..3 * D].to_vec();

    let wq_id = g.param("wq", Shape::new(&[D, D], f));
    let wk_id = g.param("wk", Shape::new(&[D, D], f));
    let wv_id = g.param("wv", Shape::new(&[D, D], f));
    let bq_id = g.param("bq", Shape::new(&[D], f));
    let bk_id = g.param("bk", Shape::new(&[D], f));
    let bv_id = g.param("bv", Shape::new(&[D], f));
    let q_node = g.fused_matmul_bias_act(qk_node, wq_id, bq_id, None, Shape::new(&[1, l, D], f));
    let k_node = g.fused_matmul_bias_act(qk_node, wk_id, bk_id, None, Shape::new(&[1, l, D], f));
    let v_node = g.fused_matmul_bias_act(x_in, wv_id, bv_id, None, Shape::new(&[1, l, D], f));
    let attn = g.attention_kind(
        q_node,
        k_node,
        v_node,
        NH,
        DH,
        MaskKind::None,
        shape::attention_shape(g.shape(q_node)),
    );
    let proj_w_id = g.param("proj_w", Shape::new(&[D, D], f));
    let proj_b_id = g.param("proj_b", Shape::new(&[D], f));
    let proj = g.fused_matmul_bias_act(attn, proj_w_id, proj_b_id, None, Shape::new(&[1, l, D], f));
    g.set_outputs(vec![proj]);

    let mut compiled = rlx_models::flow_bridge::compile_graph_sam(Device::Cpu, g)?;
    compiled.set_param("wq", &wq);
    compiled.set_param("wk", &wk);
    compiled.set_param("wv", &wv);
    compiled.set_param("bq", &bq);
    compiled.set_param("bk", &bk);
    compiled.set_param("bv", &bv);
    compiled.set_param("proj_w", &layer0.self_attn_out_w_t);
    compiled.set_param("proj_b", &layer0.self_attn_out_b);
    let outputs = compiled.run(&[("x", sa_x.as_slice()), ("pos", sa_pos.as_slice())]);
    let ir_attn = outputs.into_iter().next().unwrap();

    // ── Compare ───────────────────────────────────────────────────
    let n = host_attn.len().min(ir_attn.len());
    let mut mad = 0f32;
    let mut idx = 0;
    for i in 0..n {
        let dlt = (host_attn[i] - ir_attn[i]).abs();
        if dlt > mad {
            mad = dlt;
            idx = i;
        }
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let a = host_attn[i] as f64;
        let b = ir_attn[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let cos = 1.0 - dot / (na * nb).sqrt();
    println!("len={n}  max_abs_diff={mad:.6}  cos_dist={cos:.3e}  idx={idx}");
    println!("host[0..8] = {:?}", &host_attn[..8]);
    println!("ir  [0..8] = {:?}", &ir_attn[..8]);
    Ok(())
}

fn split_qkv(w_t: &[f32], e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut wq = vec![0f32; e * e];
    let mut wk = vec![0f32; e * e];
    let mut wv = vec![0f32; e * e];
    for i in 0..e {
        for j in 0..e {
            wq[i * e + j] = w_t[i * 3 * e + j];
            wk[i * e + j] = w_t[i * 3 * e + e + j];
            wv[i * e + j] = w_t[i * 3 * e + 2 * e + j];
        }
    }
    (wq, wk, wv)
}
