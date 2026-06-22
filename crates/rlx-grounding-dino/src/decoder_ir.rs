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

//! One cross-modality decoder layer as a single on-device HIR graph:
//! query self-attention → text cross-attention → image multi-scale **deformable**
//! cross-attention (the fused [`crate::deform_op`] custom op) → FFN, all dispatched
//! through a [`Device`]. Combines the validated `ir::mha`/FFN blocks with the
//! custom op in one graph — the per-layer scalar glue (reference-point sine embed,
//! box refinement) stays on the host.

use crate::deform_attn::{DeformWeights, LevelShape};
use crate::deform_op::ensure_registered;
use crate::ir::{self, Params};
use anyhow::Result;
use rlx_ir::{DType, HirGraphExt, HirModule, HirMut, Shape};
use rlx_runtime::Device;

/// All learnable tensors of one decoder layer (PyTorch `[out, in]` layout).
#[derive(Clone)]
pub struct DecoderLayerWeights {
    // query self-attention
    pub sa_q_w: Vec<f32>,
    pub sa_q_b: Vec<f32>,
    pub sa_k_w: Vec<f32>,
    pub sa_k_b: Vec<f32>,
    pub sa_v_w: Vec<f32>,
    pub sa_v_b: Vec<f32>,
    pub sa_o_w: Vec<f32>,
    pub sa_o_b: Vec<f32>,
    pub sa_ln_w: Vec<f32>,
    pub sa_ln_b: Vec<f32>,
    // text cross-attention
    pub ta_q_w: Vec<f32>,
    pub ta_q_b: Vec<f32>,
    pub ta_k_w: Vec<f32>,
    pub ta_k_b: Vec<f32>,
    pub ta_v_w: Vec<f32>,
    pub ta_v_b: Vec<f32>,
    pub ta_o_w: Vec<f32>,
    pub ta_o_b: Vec<f32>,
    pub ta_ln_w: Vec<f32>,
    pub ta_ln_b: Vec<f32>,
    // deformable cross-attention (8 projection tensors)
    pub da_value_w: Vec<f32>,
    pub da_value_b: Vec<f32>,
    pub da_samp_w: Vec<f32>,
    pub da_samp_b: Vec<f32>,
    pub da_attw_w: Vec<f32>,
    pub da_attw_b: Vec<f32>,
    pub da_out_w: Vec<f32>,
    pub da_out_b: Vec<f32>,
    pub da_ln_w: Vec<f32>,
    pub da_ln_b: Vec<f32>,
    // FFN
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub final_ln_w: Vec<f32>,
    pub final_ln_b: Vec<f32>,
}

/// On-device decoder layer.
pub struct DecoderLayerIr {
    w: DecoderLayerWeights,
    d: usize,
    nh: usize,
    np: usize,
    eps: f32,
    device: Device,
}

impl DecoderLayerIr {
    pub fn new(w: DecoderLayerWeights, d: usize, nh: usize, np: usize, device: Device) -> Self {
        ensure_registered();
        Self {
            w,
            d,
            nh,
            np,
            eps: 1e-5,
            device,
        }
    }

    /// `hidden`/`query_pos` are `[nq, d]`, `ref_input` is `[nq, n_levels, 4]`,
    /// `memory` is `[seq, d]`, `text` is `[Lt, d]`, `text_bias` is `[1, nq, Lt]`
    /// (additive). Returns the layer output `[nq, d]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden: &[f32],
        query_pos: &[f32],
        ref_input: &[f32],
        memory: &[f32],
        text: &[f32],
        text_bias: &[f32],
        shapes: &[LevelShape],
    ) -> Result<Vec<f32>> {
        let d = self.d;
        let nh = self.nh;
        let nq = hidden.len() / d;
        let seq = memory.len() / d;
        let lt = text.len() / d;
        let nl = shapes.len();
        let w = &self.w;

        let mut hir = HirModule::new("gdino_decoder_layer");
        let mut params = Params::new();
        let mut g = HirMut::new(&mut hir);

        let hidden_n = g.input("hidden", Shape::new(&[nq, d], DType::F32));
        let qpos_n = g.input("query_pos", Shape::new(&[nq, d], DType::F32));
        let ref_n = g.input("ref", Shape::new(&[nq, nl, 4], DType::F32));
        let memory_n = g.input("memory", Shape::new(&[seq, d], DType::F32));
        let text_n = g.input("text", Shape::new(&[lt, d], DType::F32));
        let tbias_n = g.input("text_bias", Shape::new(&[1, nq, lt], DType::F32));
        let zsa_n = g.input("zero_sa", Shape::new(&[1, nq, nq], DType::F32));

        // 1. query self-attention (q = k = hidden + query_pos, v = hidden).
        let q_sa = g.add(hidden_n, qpos_n);
        let sa = ir::mha(
            &mut g,
            &mut params,
            "sa",
            q_sa,
            q_sa,
            hidden_n,
            nq,
            nq,
            d,
            nh,
            &w.sa_q_w,
            &w.sa_q_b,
            &w.sa_k_w,
            &w.sa_k_b,
            &w.sa_v_w,
            &w.sa_v_b,
            &w.sa_o_w,
            &w.sa_o_b,
            zsa_n,
        );
        let h1 = g.add(hidden_n, sa);
        let h1 = ir::layer_norm(
            &mut g,
            &mut params,
            "sa_ln",
            h1,
            &w.sa_ln_w,
            &w.sa_ln_b,
            self.eps,
        );

        // 2. text cross-attention (q = h1 + query_pos, k = v = text).
        let q_ta = g.add(h1, qpos_n);
        let ta = ir::mha(
            &mut g,
            &mut params,
            "ta",
            q_ta,
            text_n,
            text_n,
            nq,
            lt,
            d,
            nh,
            &w.ta_q_w,
            &w.ta_q_b,
            &w.ta_k_w,
            &w.ta_k_b,
            &w.ta_v_w,
            &w.ta_v_b,
            &w.ta_o_w,
            &w.ta_o_b,
            tbias_n,
        );
        let h2 = g.add(h1, ta);
        let h2 = ir::layer_norm(
            &mut g,
            &mut params,
            "ta_ln",
            h2,
            &w.ta_ln_w,
            &w.ta_ln_b,
            self.eps,
        );

        // 3. deformable image cross-attention (fused custom op). query = h2 + query_pos.
        // Uses 4-dim reference boxes; output is one row per decoder query (nq).
        let q_da = g.add(h2, qpos_n);
        let dw = DeformWeights {
            value_proj_w: &w.da_value_w,
            value_proj_b: &w.da_value_b,
            sampling_offsets_w: &w.da_samp_w,
            sampling_offsets_b: &w.da_samp_b,
            attention_weights_w: &w.da_attw_w,
            attention_weights_b: &w.da_attw_b,
            output_proj_w: &w.da_out_w,
            output_proj_b: &w.da_out_b,
        };
        let da = crate::deform_attn_ir::build_deform_node(
            &mut g,
            &mut params,
            "da.",
            q_da,
            memory_n,
            ref_n,
            &dw,
            d,
            nh,
            self.np,
            4,
            nq,
            shapes,
        );
        let h3 = g.add(h2, da);
        let h3 = ir::layer_norm(
            &mut g,
            &mut params,
            "da_ln",
            h3,
            &w.da_ln_w,
            &w.da_ln_b,
            self.eps,
        );

        // 4. FFN (relu).
        let inter = w.fc1_b.len();
        let f1 = ir::linear(
            &mut g,
            &mut params,
            "fc1",
            h3,
            d,
            inter,
            &w.fc1_w,
            &w.fc1_b,
            1.0,
        );
        let act = g.relu(f1);
        let f2 = ir::linear(
            &mut g,
            &mut params,
            "fc2",
            act,
            inter,
            d,
            &w.fc2_w,
            &w.fc2_b,
            1.0,
        );
        let h4 = g.add(h3, f2);
        let h4 = ir::layer_norm(
            &mut g,
            &mut params,
            "final_ln",
            h4,
            &w.final_ln_w,
            &w.final_ln_b,
            self.eps,
        );

        g.set_outputs(vec![h4]);

        let zsa = vec![0f32; nq * nq];
        let outs = ir::compile_and_run(
            hir,
            params,
            self.device,
            &[
                ("hidden", hidden),
                ("query_pos", query_pos),
                ("ref", ref_input),
                ("memory", memory),
                ("text", text),
                ("text_bias", text_bias),
                ("zero_sa", &zsa),
            ],
        )?;
        Ok(outs.into_iter().next().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deform_attn::{deform_forward, level_start_index};
    use crate::nn::{self, AttnBias};

    fn det(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 13 + seed * 7) % 17) as f32 - 8.0) * 0.02)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn native_layer(
        w: &DecoderLayerWeights,
        hidden: &[f32],
        query_pos: &[f32],
        ref_input: &[f32],
        memory: &[f32],
        text: &[f32],
        text_bias: &[f32],
        shapes: &[LevelShape],
        d: usize,
        nh: usize,
        np: usize,
        eps: f32,
    ) -> Vec<f32> {
        let nq = hidden.len() / d;
        let lt = text.len() / d;
        let starts = level_start_index(shapes);
        let zsa = vec![0f32; nq * nq];

        // self-attn
        let q_sa: Vec<f32> = hidden.iter().zip(query_pos).map(|(a, b)| a + b).collect();
        let sa = nn::mha(
            &q_sa,
            &q_sa,
            hidden,
            nq,
            nq,
            d,
            nh,
            &w.sa_q_w,
            &w.sa_q_b,
            &w.sa_k_w,
            &w.sa_k_b,
            &w.sa_v_w,
            &w.sa_v_b,
            &w.sa_o_w,
            &w.sa_o_b,
            AttnBias::Shared(&zsa),
        );
        let h1: Vec<f32> = hidden.iter().zip(&sa).map(|(a, b)| a + b).collect();
        let h1 = nn::layer_norm(&h1, &w.sa_ln_w, &w.sa_ln_b, d, eps);
        // text cross-attn
        let q_ta: Vec<f32> = h1.iter().zip(query_pos).map(|(a, b)| a + b).collect();
        let ta = nn::mha(
            &q_ta,
            text,
            text,
            nq,
            lt,
            d,
            nh,
            &w.ta_q_w,
            &w.ta_q_b,
            &w.ta_k_w,
            &w.ta_k_b,
            &w.ta_v_w,
            &w.ta_v_b,
            &w.ta_o_w,
            &w.ta_o_b,
            AttnBias::Shared(text_bias),
        );
        let h2: Vec<f32> = h1.iter().zip(&ta).map(|(a, b)| a + b).collect();
        let h2 = nn::layer_norm(&h2, &w.ta_ln_w, &w.ta_ln_b, d, eps);
        // deformable
        let q_da: Vec<f32> = h2.iter().zip(query_pos).map(|(a, b)| a + b).collect();
        let dw = DeformWeights {
            value_proj_w: &w.da_value_w,
            value_proj_b: &w.da_value_b,
            sampling_offsets_w: &w.da_samp_w,
            sampling_offsets_b: &w.da_samp_b,
            attention_weights_w: &w.da_attw_w,
            attention_weights_b: &w.da_attw_b,
            output_proj_w: &w.da_out_w,
            output_proj_b: &w.da_out_b,
        };
        let da = deform_forward(
            &q_da, memory, ref_input, 4, shapes, &starts, d, nh, np, &dw, None,
        );
        let h3: Vec<f32> = h2.iter().zip(&da).map(|(a, b)| a + b).collect();
        let h3 = nn::layer_norm(&h3, &w.da_ln_w, &w.da_ln_b, d, eps);
        // FFN
        let inter = w.fc1_b.len();
        let mut f1 = nn::linear(&h3, nq, d, &w.fc1_w, inter, &w.fc1_b);
        nn::relu(&mut f1);
        let f2 = nn::linear(&f1, nq, inter, &w.fc2_w, d, &w.fc2_b);
        let h4: Vec<f32> = h3.iter().zip(&f2).map(|(a, b)| a + b).collect();
        nn::layer_norm(&h4, &w.final_ln_w, &w.final_ln_b, d, eps)
    }

    fn synth_weights(
        d: usize,
        nh: usize,
        nl: usize,
        np: usize,
        inter: usize,
    ) -> DecoderLayerWeights {
        let attn = |s| det(d * d, s);
        DecoderLayerWeights {
            sa_q_w: attn(1),
            sa_q_b: vec![0.0; d],
            sa_k_w: attn(2),
            sa_k_b: vec![0.0; d],
            sa_v_w: attn(3),
            sa_v_b: vec![0.0; d],
            sa_o_w: attn(4),
            sa_o_b: vec![0.0; d],
            sa_ln_w: vec![1.0; d],
            sa_ln_b: vec![0.0; d],
            ta_q_w: attn(5),
            ta_q_b: vec![0.0; d],
            ta_k_w: attn(6),
            ta_k_b: vec![0.0; d],
            ta_v_w: attn(7),
            ta_v_b: vec![0.0; d],
            ta_o_w: attn(8),
            ta_o_b: vec![0.0; d],
            ta_ln_w: vec![1.0; d],
            ta_ln_b: vec![0.0; d],
            da_value_w: attn(9),
            da_value_b: vec![0.0; d],
            da_samp_w: det(nh * nl * np * 2 * d, 10),
            da_samp_b: det(nh * nl * np * 2, 11),
            da_attw_w: det(nh * nl * np * d, 12),
            da_attw_b: det(nh * nl * np, 13),
            da_out_w: attn(14),
            da_out_b: vec![0.0; d],
            da_ln_w: vec![1.0; d],
            da_ln_b: vec![0.0; d],
            fc1_w: det(inter * d, 15),
            fc1_b: vec![0.0; inter],
            fc2_w: det(d * inter, 16),
            fc2_b: vec![0.0; d],
            final_ln_w: vec![1.0; d],
            final_ln_b: vec![0.0; d],
        }
    }

    fn run(device: Device) -> (Vec<f32>, Vec<f32>) {
        let (d, nh, np) = (8usize, 2usize, 2usize);
        let shapes = [LevelShape { h: 4, w: 5 }, LevelShape { h: 2, w: 3 }];
        let nl = shapes.len();
        let seq: usize = shapes.iter().map(|s| s.h * s.w).sum();
        let inter = 16usize;
        let nq = 4usize;
        let lt = 5usize;
        let w = synth_weights(d, nh, nl, np, inter);

        let hidden = det(nq * d, 20);
        let query_pos = det(nq * d, 21);
        let memory = det(seq * d, 22);
        let text = det(lt * d, 23);
        let ref_input: Vec<f32> = (0..nq * nl * 4)
            .map(|i| 0.2 + 0.01 * (i % 9) as f32)
            .collect();
        let text_bias = vec![0f32; nq * lt];

        let native = native_layer(
            &w, &hidden, &query_pos, &ref_input, &memory, &text, &text_bias, &shapes, d, nh, np,
            1e-5,
        );
        let ir = DecoderLayerIr::new(w, d, nh, np, device);
        let got = ir
            .forward(
                &hidden, &query_pos, &ref_input, &memory, &text, &text_bias, &shapes,
            )
            .unwrap();
        (native, got)
    }

    #[test]
    fn decoder_layer_ir_matches_native() {
        let (native, got) = run(Device::Cpu);
        assert_eq!(native.len(), got.len());
        let e = native
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(e < 1e-4, "native vs IR decoder layer max_err={e}");
    }
}
