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

//! SAM2 memory attention IR.
//!
//! Default: five compiled subgraphs per layer with host axial RoPE between Q/K
//! projection and attention (best parity today).
//!
//! [`MemoryAttentionCompiled::compile_in_graph_rope`] fuses each layer into one
//! graph using [`Op::AxialRope2d`] (faster compile/run). Both paths share a
//! compiled stack final-norm subgraph (`skip_fusion` to avoid bad LN fusion).

use super::axial_rope::apply_axial_rope_2d;
use super::memory_attention::{
    Sam2MemoryAttentionLayerWeights, Sam2MemoryAttentionWeights, Sam2RoPEAttnWeights,
};
use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use std::collections::HashMap;

/// How each layer applies axial RoPE relative to attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerRopeMode {
    /// Host `apply_axial_rope_2d` between projection and attention subgraphs.
    HostBetweenGraphs,
    /// `Op::AxialRope2d` inside a single compiled layer graph.
    InGraph,
}

const LN_EPS: f32 = 1e-5;
const INPUT_POS_SCALE: f32 = 0.1;

/// Max prior-frame spatial memory banks in cross-attention (SAM2 ≈7 maskmem slots).
pub const MAX_MEMORY_FRAMES_IN_ATTN: usize = 7;

pub fn max_memory_slots(n_img: usize, max_obj_ptr_tokens: usize) -> usize {
    MAX_MEMORY_FRAMES_IN_ATTN * n_img + max_obj_ptr_tokens
}

struct MemoryAttentionLayerCompiled {
    mode: LayerRopeMode,
    /// Fused layer graph when `mode == InGraph`.
    fused: Option<CompiledGraph>,
    self_proj: Option<CompiledGraph>,
    self_attn: Option<CompiledGraph>,
    cross_proj: Option<CompiledGraph>,
    cross_attn: Option<CompiledGraph>,
    ffn: Option<CompiledGraph>,
    layer: Sam2MemoryAttentionLayerWeights,
}

pub struct MemoryAttentionCompiled {
    layers: Vec<MemoryAttentionLayerCompiled>,
    /// Stack final norm (compiled; `skip_fusion` at compile).
    final_norm: CompiledGraph,
    pub n_img: usize,
    pub max_n_mem: usize,
    pub d_model: usize,
    pub kv_in_dim: usize,
    pub max_obj_ptr_tokens: usize,
    pos_enc_at_input: bool,
}

impl MemoryAttentionCompiled {
    pub fn compile(
        w: &Sam2MemoryAttentionWeights,
        n_img: usize,
        max_n_mem: usize,
        max_obj_ptr_tokens: usize,
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(
            w,
            n_img,
            max_n_mem,
            max_obj_ptr_tokens,
            device,
            &rlx_flow::CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_with_profile(
        w: &Sam2MemoryAttentionWeights,
        n_img: usize,
        max_n_mem: usize,
        max_obj_ptr_tokens: usize,
        device: Device,
        profile: &rlx_flow::CompileProfile,
    ) -> Result<Self> {
        Self::compile_with_mode(
            w,
            n_img,
            max_n_mem,
            max_obj_ptr_tokens,
            device,
            LayerRopeMode::HostBetweenGraphs,
            profile,
        )
    }

    /// One compiled graph per layer (`Op::AxialRope2d` in-graph). Fusion is disabled
    /// so `FuseAttentionBlock` cannot reorder RoPE relative to attention.
    pub fn compile_in_graph_rope(
        w: &Sam2MemoryAttentionWeights,
        n_img: usize,
        max_n_mem: usize,
        max_obj_ptr_tokens: usize,
        device: Device,
    ) -> Result<Self> {
        Self::compile_in_graph_rope_with_profile(
            w,
            n_img,
            max_n_mem,
            max_obj_ptr_tokens,
            device,
            &rlx_flow::CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_in_graph_rope_with_profile(
        w: &Sam2MemoryAttentionWeights,
        n_img: usize,
        max_n_mem: usize,
        max_obj_ptr_tokens: usize,
        device: Device,
        profile: &rlx_flow::CompileProfile,
    ) -> Result<Self> {
        Self::compile_with_mode(
            w,
            n_img,
            max_n_mem,
            max_obj_ptr_tokens,
            device,
            LayerRopeMode::InGraph,
            profile,
        )
    }

    fn compile_with_mode(
        w: &Sam2MemoryAttentionWeights,
        n_img: usize,
        max_n_mem: usize,
        max_obj_ptr_tokens: usize,
        device: Device,
        mode: LayerRopeMode,
        profile: &rlx_flow::CompileProfile,
    ) -> Result<Self> {
        anyhow::ensure!(
            w.layers
                .iter()
                .all(|l| l.self_attn.num_heads == 1 && l.cross_attn.num_heads == 1),
            "memory_attention_ir currently requires num_heads=1"
        );
        let kv = w.layers[0].cross_attn.kv_in_dim;
        let mut layers = Vec::with_capacity(w.layers.len());
        for layer in &w.layers {
            layers.push(compile_layer(
                layer,
                n_img,
                max_n_mem,
                kv,
                max_obj_ptr_tokens,
                device,
                mode,
                profile,
            )?);
        }
        let (fn_g, fn_p) = build_final_norm_graph(&w.norm_g, &w.norm_b, n_img, w.d_model)?;
        let mut final_norm =
            Session::new(device).compile_with(fn_g, &compile_opts_no_fusion(device));
        for (n, d) in &fn_p {
            final_norm.set_param(n, d);
        }
        Ok(Self {
            layers,
            final_norm,
            n_img,
            max_n_mem,
            d_model: w.d_model,
            kv_in_dim: kv,
            max_obj_ptr_tokens,
            pos_enc_at_input: w.pos_enc_at_input,
        })
    }

    pub fn run(
        &mut self,
        curr: &[f32],
        curr_pos: &[f32],
        memory: &[f32],
        memory_pos: &[f32],
        active_n_mem: usize,
        num_obj_ptr_tokens: usize,
    ) -> Result<Vec<f32>> {
        let d = self.d_model;
        let kv = self.kv_in_dim;
        anyhow::ensure!(curr.len() == self.n_img * d);
        anyhow::ensure!(curr_pos.len() == self.n_img * d);
        anyhow::ensure!(memory.len() >= active_n_mem * kv);
        anyhow::ensure!(memory_pos.len() >= active_n_mem * kv);
        anyhow::ensure!(active_n_mem <= self.max_n_mem);
        anyhow::ensure!(num_obj_ptr_tokens <= self.max_obj_ptr_tokens);

        let mut tgt = curr.to_vec();
        if self.pos_enc_at_input {
            for i in 0..tgt.len() {
                tgt[i] += INPUT_POS_SCALE * curr_pos[i];
            }
        }

        let mut mem_pad = vec![0f32; self.max_n_mem * kv];
        let mut mem_pos_pad = vec![0f32; self.max_n_mem * kv];
        mem_pad[..active_n_mem * kv].copy_from_slice(&memory[..active_n_mem * kv]);
        mem_pos_pad[..active_n_mem * kv].copy_from_slice(&memory_pos[..active_n_mem * kv]);

        let nh = 1usize;
        let mut mask = vec![0f32; nh * self.n_img * self.max_n_mem];
        fill_cross_attn_bias(&mut mask, nh, self.n_img, self.max_n_mem, active_n_mem);

        for layer in &mut self.layers {
            tgt = match layer.mode {
                LayerRopeMode::InGraph => layer
                    .fused
                    .as_mut()
                    .expect("fused layer")
                    .run(&[
                        ("tgt", &tgt),
                        ("curr_pos", curr_pos),
                        ("memory", &mem_pad),
                        ("memory_pos", &mem_pos_pad),
                        ("mask_ca", &mask),
                    ])
                    .into_iter()
                    .next()
                    .expect("fused layer output"),
                LayerRopeMode::HostBetweenGraphs => layer.run_host_between(
                    &tgt,
                    curr_pos,
                    &mem_pad,
                    &mem_pos_pad,
                    active_n_mem,
                    num_obj_ptr_tokens,
                )?,
            };
        }

        let outs = self.final_norm.run(&[("tgt", &tgt)]);
        Ok(outs.into_iter().next().expect("memory_attention output"))
    }
}

impl MemoryAttentionLayerCompiled {
    fn run_host_between(
        &mut self,
        tgt: &[f32],
        curr_pos: &[f32],
        memory: &[f32],
        memory_pos: &[f32],
        active_n_mem: usize,
        num_obj_ptr_tokens: usize,
    ) -> Result<Vec<f32>> {
        let d = self.layer.d_model;
        let kv = self.layer.cross_attn.kv_in_dim;
        let n_img = tgt.len() / d;
        let max_n_mem = memory.len() / kv;
        let _id = self.layer.self_attn.internal_dim;

        let p = self
            .self_proj
            .as_mut()
            .expect("self_proj")
            .run(&[("tgt", tgt), ("curr_pos", curr_pos)]);
        let mut it = p.into_iter();
        let mut sa_q = it.next().expect("sa_q");
        let mut sa_k = it.next().expect("sa_k");
        let sa_v = it.next().expect("sa_v");
        host_rotate_qk(&mut sa_q, n_img, &self.layer.self_attn);
        host_rotate_qk(&mut sa_k, n_img, &self.layer.self_attn);

        let mut tgt = self
            .self_attn
            .as_mut()
            .expect("self_attn")
            .run(&[
                ("tgt", tgt),
                ("sa_q", &sa_q),
                ("sa_k", &sa_k),
                ("sa_v", &sa_v),
            ])
            .into_iter()
            .next()
            .expect("tgt after self");

        let c = self.cross_proj.as_mut().expect("cross_proj").run(&[
            ("tgt", &tgt),
            ("curr_pos", curr_pos),
            ("memory", memory),
            ("memory_pos", memory_pos),
        ]);
        let mut it = c.into_iter();
        let mut ca_q = it.next().expect("ca_q");
        let mut ca_k = it.next().expect("ca_k");
        host_rotate_qk(&mut ca_q, n_img, &self.layer.cross_attn);
        host_rotate_k_partial(
            &mut ca_k,
            max_n_mem,
            active_n_mem,
            num_obj_ptr_tokens,
            &self.layer.cross_attn,
        );

        let nh = self.layer.cross_attn.num_heads;
        let mut mask = vec![0f32; nh * n_img * max_n_mem];
        fill_cross_attn_bias(&mut mask, nh, n_img, max_n_mem, active_n_mem);

        tgt = self
            .cross_attn
            .as_mut()
            .expect("cross_attn")
            .run(&[
                ("tgt", &tgt),
                ("ca_q", &ca_q),
                ("ca_k", &ca_k),
                ("memory", memory),
                ("mask_ca", &mask),
            ])
            .into_iter()
            .next()
            .expect("tgt after cross");

        self.ffn
            .as_mut()
            .expect("ffn")
            .run(&[("tgt", &tgt)])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("ffn output missing"))
    }
}

fn compile_opts_no_fusion(device: Device) -> CompileOptions {
    rlx_core::flow_bridge::compile_options_sam2_memory_attention(device)
}

fn compile_layer(
    layer: &Sam2MemoryAttentionLayerWeights,
    n_img: usize,
    n_mem: usize,
    kv: usize,
    max_obj_ptr_tokens: usize,
    device: Device,
    mode: LayerRopeMode,
    profile: &rlx_flow::CompileProfile,
) -> Result<MemoryAttentionLayerCompiled> {
    let compile =
        |g: Graph, p: HashMap<String, Vec<f32>>, opts: &CompileOptions| -> Result<CompiledGraph> {
            let mut c = Session::new(device).compile_with(g, opts);
            for (n, d) in &p {
                c.set_param(n, d);
            }
            Ok(c)
        };
    match mode {
        LayerRopeMode::InGraph => {
            let opts = compile_opts_no_fusion(device);
            let (g, p) = build_layer_graph(layer, n_img, n_mem, kv, max_obj_ptr_tokens)?;
            Ok(MemoryAttentionLayerCompiled {
                mode,
                fused: Some(compile(g, p, &opts)?),
                self_proj: None,
                self_attn: None,
                cross_proj: None,
                cross_attn: None,
                ffn: None,
                layer: clone_layer(layer),
            })
        }
        LayerRopeMode::HostBetweenGraphs => {
            let opts = rlx_core::flow_bridge::compile_options_for_profile(profile, device);
            let (g1, p1) = build_self_proj_graph(layer, n_img)?;
            let (g2, p2) = build_self_attn_graph(layer, n_img)?;
            let (g3, p3) = build_cross_proj_graph(layer, n_img, n_mem, kv)?;
            let (g4, p4) = build_cross_attn_graph(layer, n_img, n_mem, kv)?;
            let (g5, p5) = build_ffn_graph(layer, n_img)?;
            Ok(MemoryAttentionLayerCompiled {
                mode,
                fused: None,
                self_proj: Some(compile(g1, p1, &opts)?),
                self_attn: Some(compile(g2, p2, &opts)?),
                cross_proj: Some(compile(g3, p3, &opts)?),
                cross_attn: Some(compile(g4, p4, &opts)?),
                ffn: Some(compile(g5, p5, &opts)?),
                layer: clone_layer(layer),
            })
        }
    }
}

fn fill_cross_attn_bias(
    out: &mut [f32],
    nh: usize,
    n_img: usize,
    max_n_mem: usize,
    active_n_mem: usize,
) {
    out.fill(0.0);
    for h in 0..nh {
        for qi in 0..n_img {
            for ki in active_n_mem..max_n_mem {
                out[(h * n_img + qi) * max_n_mem + ki] = -1e4;
            }
        }
    }
}

fn host_rotate_qk(seq: &mut [f32], n_tokens: usize, w: &Sam2RoPEAttnWeights) {
    let nh = w.num_heads;
    let dh = w.internal_dim / nh;
    let [ex, ey] = w.rope_feat_size;
    let out = apply_axial_rope_2d(seq, nh, n_tokens, dh, ex, ey, w.rope_theta, 1);
    seq.copy_from_slice(&out);
}

fn host_rotate_k_partial(
    k: &mut [f32],
    buf_tokens: usize,
    active_tokens: usize,
    num_k_exclude_rope: usize,
    w: &Sam2RoPEAttnWeights,
) {
    let nh = w.num_heads;
    let dh = w.internal_dim / nh;
    let [ex, ey] = w.rope_feat_size;
    let spatial = ex * ey;
    let num_k_rope = active_tokens.saturating_sub(num_k_exclude_rope);
    if num_k_rope == 0 {
        return;
    }
    let _ = buf_tokens;
    let r = if w.rope_k_repeat && num_k_rope >= spatial && num_k_rope.is_multiple_of(spatial) {
        num_k_rope / spatial
    } else {
        1
    };
    let prefix_len = nh * num_k_rope * dh;
    let out = apply_axial_rope_2d(
        &k[..prefix_len],
        nh,
        num_k_rope,
        dh,
        ex,
        ey,
        w.rope_theta,
        r,
    );
    k[..prefix_len].copy_from_slice(&out);
}

fn clone_layer(l: &Sam2MemoryAttentionLayerWeights) -> Sam2MemoryAttentionLayerWeights {
    Sam2MemoryAttentionLayerWeights {
        self_attn: clone_rope(&l.self_attn),
        cross_attn: clone_rope(&l.cross_attn),
        norm1_g: l.norm1_g.clone(),
        norm1_b: l.norm1_b.clone(),
        norm2_g: l.norm2_g.clone(),
        norm2_b: l.norm2_b.clone(),
        norm3_g: l.norm3_g.clone(),
        norm3_b: l.norm3_b.clone(),
        linear1_w: l.linear1_w.clone(),
        linear1_b: l.linear1_b.clone(),
        linear2_w: l.linear2_w.clone(),
        linear2_b: l.linear2_b.clone(),
        pos_enc_at_attn: l.pos_enc_at_attn,
        pos_enc_at_cross_attn_queries: l.pos_enc_at_cross_attn_queries,
        pos_enc_at_cross_attn_keys: l.pos_enc_at_cross_attn_keys,
        d_model: l.d_model,
    }
}

fn clone_rope(w: &Sam2RoPEAttnWeights) -> Sam2RoPEAttnWeights {
    Sam2RoPEAttnWeights {
        q_w: w.q_w.clone(),
        q_b: w.q_b.clone(),
        k_w: w.k_w.clone(),
        k_b: w.k_b.clone(),
        v_w: w.v_w.clone(),
        v_b: w.v_b.clone(),
        out_w: w.out_w.clone(),
        out_b: w.out_b.clone(),
        embedding_dim: w.embedding_dim,
        kv_in_dim: w.kv_in_dim,
        internal_dim: w.internal_dim,
        num_heads: w.num_heads,
        rope_theta: w.rope_theta,
        rope_feat_size: w.rope_feat_size,
        rope_k_repeat: w.rope_k_repeat,
    }
}

fn matmul_weight(w_out_in: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    let mut t = vec![0f32; in_d * out_d];
    for o in 0..out_d {
        for k in 0..in_d {
            t[k * out_d + o] = w_out_in[o * in_d + k];
        }
    }
    t
}

fn bind_linear(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    w: &[f32],
    b: &[f32],
    in_d: usize,
    out_d: usize,
) -> (NodeId, NodeId) {
    let f = DType::F32;
    let w_id = g.param(format!("{prefix}.w"), Shape::new(&[in_d, out_d], f));
    let b_id = g.param(format!("{prefix}.b"), Shape::new(&[out_d], f));
    params.insert(format!("{prefix}.w"), matmul_weight(w, in_d, out_d));
    params.insert(format!("{prefix}.b"), b.to_vec());
    (w_id, b_id)
}

fn linear(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    x: NodeId,
    w: &[f32],
    b: &[f32],
    in_d: usize,
    out_d: usize,
    seq: usize,
) -> NodeId {
    let f = DType::F32;
    let (w_id, b_id) = bind_linear(g, params, prefix, w, b, in_d, out_d);
    g.fused_matmul_bias_act(x, w_id, b_id, None, Shape::new(&[1, seq, out_d], f))
}

fn bind_ln(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    gamm: &[f32],
    bet: &[f32],
    e: usize,
) -> (NodeId, NodeId) {
    let f = DType::F32;
    let g_id = g.param(format!("{prefix}.g"), Shape::new(&[e], f));
    let b_id = g.param(format!("{prefix}.b"), Shape::new(&[e], f));
    params.insert(format!("{prefix}.g"), gamm.to_vec());
    params.insert(format!("{prefix}.b"), bet.to_vec());
    (g_id, b_id)
}

fn layer_norm(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    x: NodeId,
    gamm: &[f32],
    bet: &[f32],
    seq: usize,
    e: usize,
) -> NodeId {
    let f = DType::F32;
    let shape = Shape::new(&[1, seq, e], f);
    let (g_id, b_id) = bind_ln(g, params, prefix, gamm, bet, e);
    g.layer_norm(x, g_id, b_id, -1, LN_EPS, shape)
}

fn maybe_add_pos(
    g: &mut Graph,
    x: NodeId,
    pos: NodeId,
    seq: usize,
    e: usize,
    enabled: bool,
) -> NodeId {
    if enabled {
        let f = DType::F32;
        g.binary(BinaryOp::Add, x, pos, Shape::new(&[1, seq, e], f))
    } else {
        x
    }
}

fn apply_axial_rope_graph(
    g: &mut Graph,
    x: NodeId,
    w: &Sam2RoPEAttnWeights,
    _seq: usize,
    repeat_factor: usize,
) -> NodeId {
    let nh = w.num_heads;
    let dh = w.internal_dim / nh;
    let [ex, ey] = w.rope_feat_size;
    g.axial_rope2d(x, ex, ey, dh, nh, w.rope_theta, repeat_factor)
}

fn build_rope_attn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    w: &Sam2RoPEAttnWeights,
    q_in: NodeId,
    k_in: NodeId,
    v_in: NodeId,
    q_len: usize,
    k_len: usize,
    q_in_dim: usize,
    kv_in_dim: usize,
    num_k_exclude_rope: usize,
    bias: Option<NodeId>,
) -> NodeId {
    let d = w.embedding_dim;
    let id = w.internal_dim;
    let nh = w.num_heads;
    let dh = id / nh;
    let f = DType::F32;
    let [end_x, end_y] = w.rope_feat_size;
    let spatial = end_x * end_y;

    let q_proj = linear(
        g,
        params,
        &format!("{prefix}.q"),
        q_in,
        &w.q_w,
        &w.q_b,
        q_in_dim,
        id,
        q_len,
    );
    let k_proj = linear(
        g,
        params,
        &format!("{prefix}.k"),
        k_in,
        &w.k_w,
        &w.k_b,
        kv_in_dim,
        id,
        k_len,
    );
    let v_proj = linear(
        g,
        params,
        &format!("{prefix}.v"),
        v_in,
        &w.v_w,
        &w.v_b,
        kv_in_dim,
        id,
        k_len,
    );

    let q_rot = apply_axial_rope_graph(g, q_proj, w, q_len, 1);

    let num_k_rope = k_len.saturating_sub(num_k_exclude_rope);
    let k_rot = if num_k_rope == 0 {
        k_proj
    } else if num_k_rope == k_len {
        let r = if w.rope_k_repeat && num_k_rope >= spatial && num_k_rope.is_multiple_of(spatial) {
            num_k_rope / spatial
        } else {
            1
        };
        apply_axial_rope_graph(g, k_proj, w, k_len, r)
    } else {
        let k_prefix = g.narrow_(k_proj, 1, 0, num_k_rope);
        let k_suffix = g.narrow_(k_proj, 1, num_k_rope, k_len - num_k_rope);
        let r = if w.rope_k_repeat && num_k_rope >= spatial && num_k_rope.is_multiple_of(spatial) {
            num_k_rope / spatial
        } else {
            1
        };
        let k_pre_rot = apply_axial_rope_graph(g, k_prefix, w, num_k_rope, r);
        g.concat_(vec![k_pre_rot, k_suffix], 1)
    };

    let out_shape = Shape::new(&[1, q_len, id], f);
    let attn = if let Some(b) = bias {
        g.attention_bias(q_rot, k_rot, v_proj, b, nh, dh, out_shape.clone())
    } else {
        g.attention_kind(
            q_rot,
            k_rot,
            v_proj,
            nh,
            dh,
            MaskKind::None,
            out_shape.clone(),
        )
    };
    linear(
        g,
        params,
        &format!("{prefix}.o"),
        attn,
        &w.out_w,
        &w.out_b,
        id,
        d,
        q_len,
    )
}

fn build_layer_graph(
    layer: &Sam2MemoryAttentionLayerWeights,
    n_img: usize,
    n_mem: usize,
    kv_in_dim: usize,
    num_obj_ptr_tokens: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let d = layer.d_model;
    let f = DType::F32;
    let mut g = Graph::new("sam2_mem_attn_layer");
    let mut params = HashMap::new();

    let tgt = g.input("tgt", Shape::new(&[1, n_img, d], f));
    let curr_pos = g.input("curr_pos", Shape::new(&[1, n_img, d], f));
    let memory = g.input("memory", Shape::new(&[1, n_mem, kv_in_dim], f));
    let memory_pos = g.input("memory_pos", Shape::new(&[1, n_mem, kv_in_dim], f));
    let mask_ca = g.input(
        "mask_ca",
        Shape::new(&[1, layer.cross_attn.num_heads, n_img, n_mem], f),
    );

    let seq_shape = Shape::new(&[1, n_img, d], f);
    let mut tgt2 = layer_norm(
        &mut g,
        &mut params,
        "n1",
        tgt,
        &layer.norm1_g,
        &layer.norm1_b,
        n_img,
        d,
    );
    let q_sa = maybe_add_pos(&mut g, tgt2, curr_pos, n_img, d, layer.pos_enc_at_attn);
    let sa = build_rope_attn(
        &mut g,
        &mut params,
        "sa",
        &layer.self_attn,
        q_sa,
        tgt2,
        tgt2,
        n_img,
        n_img,
        d,
        d,
        0,
        None,
    );
    let mut out = g.binary(BinaryOp::Add, tgt, sa, seq_shape.clone());

    tgt2 = layer_norm(
        &mut g,
        &mut params,
        "n2",
        out,
        &layer.norm2_g,
        &layer.norm2_b,
        n_img,
        d,
    );
    let q_ca = maybe_add_pos(
        &mut g,
        tgt2,
        curr_pos,
        n_img,
        d,
        layer.pos_enc_at_cross_attn_queries,
    );
    let k_ca = maybe_add_pos(
        &mut g,
        memory,
        memory_pos,
        n_mem,
        kv_in_dim,
        layer.pos_enc_at_cross_attn_keys,
    );
    let ca = build_rope_attn(
        &mut g,
        &mut params,
        "ca",
        &layer.cross_attn,
        q_ca,
        k_ca,
        memory,
        n_img,
        n_mem,
        d,
        kv_in_dim,
        num_obj_ptr_tokens,
        Some(mask_ca),
    );
    out = g.binary(BinaryOp::Add, out, ca, seq_shape.clone());

    tgt2 = layer_norm(
        &mut g,
        &mut params,
        "n3",
        out,
        &layer.norm3_g,
        &layer.norm3_b,
        n_img,
        d,
    );
    let dim_ff = layer.linear1_b.len();
    let mid = linear(
        &mut g,
        &mut params,
        "ff1",
        tgt2,
        &layer.linear1_w,
        &layer.linear1_b,
        d,
        dim_ff,
        n_img,
    );
    let mid = g.activation(Activation::Relu, mid, Shape::new(&[1, n_img, dim_ff], f));
    let down = linear(
        &mut g,
        &mut params,
        "ff2",
        mid,
        &layer.linear2_w,
        &layer.linear2_b,
        dim_ff,
        d,
        n_img,
    );
    out = g.binary(BinaryOp::Add, out, down, seq_shape);

    g.set_outputs(vec![out]);
    Ok((g, params))
}

fn build_qkv_proj(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    w: &Sam2RoPEAttnWeights,
    q_in: NodeId,
    k_in: NodeId,
    v_in: NodeId,
    q_len: usize,
    k_len: usize,
    q_in_dim: usize,
    kv_in_dim: usize,
) -> (NodeId, NodeId, NodeId) {
    let id = w.internal_dim;
    let q = linear(
        g,
        params,
        &format!("{prefix}.q"),
        q_in,
        &w.q_w,
        &w.q_b,
        q_in_dim,
        id,
        q_len,
    );
    let k = linear(
        g,
        params,
        &format!("{prefix}.k"),
        k_in,
        &w.k_w,
        &w.k_b,
        kv_in_dim,
        id,
        k_len,
    );
    let v = linear(
        g,
        params,
        &format!("{prefix}.v"),
        v_in,
        &w.v_w,
        &w.v_b,
        kv_in_dim,
        id,
        k_len,
    );
    (q, k, v)
}

fn build_attention_out(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    w: &Sam2RoPEAttnWeights,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    q_len: usize,
    _k_len: usize,
    mask: Option<NodeId>,
) -> NodeId {
    let d = w.embedding_dim;
    let id = w.internal_dim;
    let nh = w.num_heads;
    let dh = id / nh;
    let f = DType::F32;
    let out_shape = Shape::new(&[1, q_len, id], f);
    let attn = if let Some(m) = mask {
        g.attention_bias(q, k, v, m, nh, dh, out_shape.clone())
    } else {
        g.attention_kind(q, k, v, nh, dh, MaskKind::None, out_shape.clone())
    };
    linear(
        g,
        params,
        &format!("{prefix}.o"),
        attn,
        &w.out_w,
        &w.out_b,
        id,
        d,
        q_len,
    )
}

fn build_self_proj_graph(
    layer: &Sam2MemoryAttentionLayerWeights,
    n_img: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let d = layer.d_model;
    let f = DType::F32;
    let mut g = Graph::new("sam2_mem_self_proj");
    let mut params = HashMap::new();
    let tgt = g.input("tgt", Shape::new(&[1, n_img, d], f));
    let curr_pos = g.input("curr_pos", Shape::new(&[1, n_img, d], f));
    let tgt2 = layer_norm(
        &mut g,
        &mut params,
        "n1",
        tgt,
        &layer.norm1_g,
        &layer.norm1_b,
        n_img,
        d,
    );
    let q_in = maybe_add_pos(&mut g, tgt2, curr_pos, n_img, d, layer.pos_enc_at_attn);
    let (sa_q, sa_k, sa_v) = build_qkv_proj(
        &mut g,
        &mut params,
        "sa",
        &layer.self_attn,
        q_in,
        tgt2,
        tgt2,
        n_img,
        n_img,
        d,
        d,
    );
    g.set_outputs(vec![sa_q, sa_k, sa_v]);
    Ok((g, params))
}

fn build_self_attn_graph(
    layer: &Sam2MemoryAttentionLayerWeights,
    n_img: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let d = layer.d_model;
    let f = DType::F32;
    let mut g = Graph::new("sam2_mem_self_attn");
    let mut params = HashMap::new();
    let tgt = g.input("tgt", Shape::new(&[1, n_img, d], f));
    let sa_q = g.input(
        "sa_q",
        Shape::new(&[1, n_img, layer.self_attn.internal_dim], f),
    );
    let sa_k = g.input(
        "sa_k",
        Shape::new(&[1, n_img, layer.self_attn.internal_dim], f),
    );
    let sa_v = g.input(
        "sa_v",
        Shape::new(&[1, n_img, layer.self_attn.internal_dim], f),
    );
    let sa = build_attention_out(
        &mut g,
        &mut params,
        "sa",
        &layer.self_attn,
        sa_q,
        sa_k,
        sa_v,
        n_img,
        n_img,
        None,
    );
    let out = g.binary(BinaryOp::Add, tgt, sa, Shape::new(&[1, n_img, d], f));
    g.set_outputs(vec![out]);
    Ok((g, params))
}

fn build_cross_proj_graph(
    layer: &Sam2MemoryAttentionLayerWeights,
    n_img: usize,
    n_mem: usize,
    kv: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let d = layer.d_model;
    let f = DType::F32;
    let mut g = Graph::new("sam2_mem_cross_proj");
    let mut params = HashMap::new();
    let tgt = g.input("tgt", Shape::new(&[1, n_img, d], f));
    let curr_pos = g.input("curr_pos", Shape::new(&[1, n_img, d], f));
    let memory = g.input("memory", Shape::new(&[1, n_mem, kv], f));
    let memory_pos = g.input("memory_pos", Shape::new(&[1, n_mem, kv], f));
    let tgt2 = layer_norm(
        &mut g,
        &mut params,
        "n2",
        tgt,
        &layer.norm2_g,
        &layer.norm2_b,
        n_img,
        d,
    );
    let q_in = maybe_add_pos(
        &mut g,
        tgt2,
        curr_pos,
        n_img,
        d,
        layer.pos_enc_at_cross_attn_queries,
    );
    let k_in = maybe_add_pos(
        &mut g,
        memory,
        memory_pos,
        n_mem,
        kv,
        layer.pos_enc_at_cross_attn_keys,
    );
    let (ca_q, ca_k, _) = build_qkv_proj(
        &mut g,
        &mut params,
        "ca",
        &layer.cross_attn,
        q_in,
        k_in,
        memory,
        n_img,
        n_mem,
        d,
        kv,
    );
    g.set_outputs(vec![ca_q, ca_k]);
    Ok((g, params))
}

fn build_cross_attn_graph(
    layer: &Sam2MemoryAttentionLayerWeights,
    n_img: usize,
    n_mem: usize,
    kv: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let d = layer.d_model;
    let f = DType::F32;
    let mut g = Graph::new("sam2_mem_cross_attn");
    let mut params = HashMap::new();
    let tgt = g.input("tgt", Shape::new(&[1, n_img, d], f));
    let ca_q = g.input(
        "ca_q",
        Shape::new(&[1, n_img, layer.cross_attn.internal_dim], f),
    );
    let ca_k = g.input(
        "ca_k",
        Shape::new(&[1, n_mem, layer.cross_attn.internal_dim], f),
    );
    let memory = g.input("memory", Shape::new(&[1, n_mem, kv], f));
    let mask_ca = g.input(
        "mask_ca",
        Shape::new(&[1, layer.cross_attn.num_heads, n_img, n_mem], f),
    );
    let ca = build_attention_out(
        &mut g,
        &mut params,
        "ca",
        &layer.cross_attn,
        ca_q,
        ca_k,
        memory,
        n_img,
        n_mem,
        Some(mask_ca),
    );
    let out = g.binary(BinaryOp::Add, tgt, ca, Shape::new(&[1, n_img, d], f));
    g.set_outputs(vec![out]);
    Ok((g, params))
}

fn build_ffn_graph(
    layer: &Sam2MemoryAttentionLayerWeights,
    n_img: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let d = layer.d_model;
    let f = DType::F32;
    let mut g = Graph::new("sam2_mem_ffn");
    let mut params = HashMap::new();
    let tgt = g.input("tgt", Shape::new(&[1, n_img, d], f));
    let seq_shape = Shape::new(&[1, n_img, d], f);
    let normed = layer_norm(
        &mut g,
        &mut params,
        "n3",
        tgt,
        &layer.norm3_g,
        &layer.norm3_b,
        n_img,
        d,
    );
    let dim_ff = layer.linear1_b.len();
    let mid = linear(
        &mut g,
        &mut params,
        "ff1",
        normed,
        &layer.linear1_w,
        &layer.linear1_b,
        d,
        dim_ff,
        n_img,
    );
    let mid = g.activation(Activation::Relu, mid, Shape::new(&[1, n_img, dim_ff], f));
    let down = linear(
        &mut g,
        &mut params,
        "ff2",
        mid,
        &layer.linear2_w,
        &layer.linear2_b,
        dim_ff,
        d,
        n_img,
    );
    let out = g.binary(BinaryOp::Add, tgt, down, seq_shape);
    g.set_outputs(vec![out]);
    Ok((g, params))
}

fn build_final_norm_graph(
    norm_g: &[f32],
    norm_b: &[f32],
    n_img: usize,
    d: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let f = DType::F32;
    let mut g = Graph::new("sam2_mem_attn_final");
    let mut params = HashMap::new();
    let tgt = g.input("tgt", Shape::new(&[1, n_img, d], f));
    let out = layer_norm(
        &mut g,
        &mut params,
        "out_norm",
        tgt,
        norm_g,
        norm_b,
        n_img,
        d,
    );
    g.set_outputs(vec![out]);
    Ok((g, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axial_rope::apply_axial_rope_2d;
    use crate::memory_attention::{
        Sam2MemoryAttentionLayerWeights, Sam2MemoryAttentionWeights, Sam2RoPEAttnWeights,
        memory_attention_forward, memory_attention_layer_forward,
    };
    use crate::transformer::layer_norm_last_cpu;
    use rlx_ir::Graph;

    #[test]
    fn axial_rope2d_op_matches_host_merged_layout() {
        let nh = 1usize;
        let n = 64usize;
        let dh = 256usize;
        let feat = [8usize, 8usize];
        let x: Vec<f32> = (0..n * nh * dh).map(|i| i as f32 * 0.001).collect();
        let host = apply_axial_rope_2d(&x, nh, n, dh, feat[0], feat[1], 10000.0, 1);

        let mut g = Graph::new("axial_rope_check");
        let f = rlx_ir::DType::F32;
        let inp = g.input("x", Shape::new(&[1, n, nh * dh], f));
        let out = g.axial_rope2d(inp, feat[0], feat[1], dh, nh, 10000.0, 1);
        g.set_outputs(vec![out]);
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_sam(Device::Cpu, g).expect("compile");
        let ir = compiled.run(&[("x", &x)]).into_iter().next().unwrap();

        let fd = host
            .iter()
            .zip(&ir)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 1e-5, "axial_rope2d op vs host max |Δ| = {fd:.3e}");
    }

    fn synth_rope_attn(d: usize, kv: usize, feat: [usize; 2]) -> Sam2RoPEAttnWeights {
        let id = d;
        Sam2RoPEAttnWeights {
            q_w: vec![0.01; id * d],
            q_b: vec![0.0; id],
            k_w: vec![0.02; id * kv],
            k_b: vec![0.0; id],
            v_w: vec![0.03; id * kv],
            v_b: vec![0.0; id],
            out_w: vec![0.04; d * id],
            out_b: vec![0.0; d],
            embedding_dim: d,
            kv_in_dim: kv,
            internal_dim: id,
            num_heads: 1,
            rope_theta: 10000.0,
            rope_feat_size: feat,
            rope_k_repeat: true,
        }
    }

    #[test]
    fn memory_attention_ir_matches_host_small_grid() {
        let d = 256usize;
        let kv = 64usize;
        let feat = [8usize, 8usize];
        let n_img = 64usize;
        let n_mem = 64usize;
        let layer = Sam2MemoryAttentionLayerWeights {
            self_attn: synth_rope_attn(d, d, feat),
            cross_attn: synth_rope_attn(d, kv, feat),
            norm1_g: vec![1.0; d],
            norm1_b: vec![0.0; d],
            norm2_g: vec![1.0; d],
            norm2_b: vec![0.0; d],
            norm3_g: vec![1.0; d],
            norm3_b: vec![0.0; d],
            linear1_w: vec![0.01; 2048 * d],
            linear1_b: vec![0.0; 2048],
            linear2_w: vec![0.02; d * 2048],
            linear2_b: vec![0.0; d],
            pos_enc_at_attn: false,
            pos_enc_at_cross_attn_queries: false,
            pos_enc_at_cross_attn_keys: true,
            d_model: d,
        };
        let w = Sam2MemoryAttentionWeights {
            layers: vec![layer],
            norm_g: vec![1.0; d],
            norm_b: vec![0.0; d],
            d_model: d,
            pos_enc_at_input: true,
        };

        let curr: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-4).collect();
        let curr_pos: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-5).collect();
        let memory: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-4).collect();
        let memory_pos: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-5).collect();

        let host = memory_attention_forward(
            &w,
            &curr,
            &curr_pos,
            &memory,
            &memory_pos,
            n_img,
            n_mem,
            kv,
            0,
        )
        .unwrap();

        let mut ir = MemoryAttentionCompiled::compile(&w, n_img, n_mem, 0, Device::Cpu).unwrap();
        let got = ir
            .run(&curr, &curr_pos, &memory, &memory_pos, n_mem, 0)
            .unwrap();

        let fd = host
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 3e-2, "memory attention max |Δ| = {fd:.3e}");
    }

    #[test]
    fn memory_attention_in_graph_rope_matches_host_small_grid() {
        let d = 256usize;
        let kv = 64usize;
        let feat = [8usize, 8usize];
        let n_img = 64usize;
        let n_mem = 64usize;
        let layer = Sam2MemoryAttentionLayerWeights {
            self_attn: synth_rope_attn(d, d, feat),
            cross_attn: synth_rope_attn(d, kv, feat),
            norm1_g: vec![1.0; d],
            norm1_b: vec![0.0; d],
            norm2_g: vec![1.0; d],
            norm2_b: vec![0.0; d],
            norm3_g: vec![1.0; d],
            norm3_b: vec![0.0; d],
            linear1_w: vec![0.01; 2048 * d],
            linear1_b: vec![0.0; 2048],
            linear2_w: vec![0.02; d * 2048],
            linear2_b: vec![0.0; d],
            pos_enc_at_attn: false,
            pos_enc_at_cross_attn_queries: false,
            pos_enc_at_cross_attn_keys: true,
            d_model: d,
        };
        let w = Sam2MemoryAttentionWeights {
            layers: vec![layer],
            norm_g: vec![1.0; d],
            norm_b: vec![0.0; d],
            d_model: d,
            pos_enc_at_input: true,
        };

        let curr: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-4).collect();
        let curr_pos: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-5).collect();
        let memory: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-4).collect();
        let memory_pos: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-5).collect();

        let host_mid = crate::memory_attention::memory_attention_forward_layers_only(
            &w,
            &curr,
            &curr_pos,
            &memory,
            &memory_pos,
            n_img,
            n_mem,
            kv,
            0,
        )
        .unwrap();

        let mut ir =
            MemoryAttentionCompiled::compile_in_graph_rope(&w, n_img, n_mem, 0, Device::Cpu)
                .unwrap();
        let got = ir
            .run(&curr, &curr_pos, &memory, &memory_pos, n_mem, 0)
            .unwrap();

        let mut mem_pad = vec![0f32; n_mem * kv];
        let mut mem_pos_pad = vec![0f32; n_mem * kv];
        mem_pad.copy_from_slice(&memory);
        mem_pos_pad.copy_from_slice(&memory_pos);
        let mut tgt = curr.clone();
        if w.pos_enc_at_input {
            for i in 0..tgt.len() {
                tgt[i] += INPUT_POS_SCALE * curr_pos[i];
            }
        }
        let nh = w.layers[0].cross_attn.num_heads;
        let mut mask = vec![0f32; nh * n_img * n_mem];
        fill_cross_attn_bias(&mut mask, nh, n_img, n_mem, n_mem);
        let layer_inputs = [
            ("tgt", tgt.as_slice()),
            ("curr_pos", curr_pos.as_slice()),
            ("memory", mem_pad.as_slice()),
            ("memory_pos", mem_pos_pad.as_slice()),
            ("mask_ca", mask.as_slice()),
        ];
        let ir_mid = ir.layers[0]
            .fused
            .as_mut()
            .expect("fused")
            .run(&layer_inputs)
            .into_iter()
            .next()
            .unwrap();
        let fd_layer = host_mid
            .iter()
            .zip(&ir_mid)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd_layer < 5e-2, "in-graph layer max |Δ| = {fd_layer:.3e}");

        let (fg, fp) = build_final_norm_graph(&w.norm_g, &w.norm_b, n_img, d).unwrap();
        let mut fn_alone =
            Session::new(Device::Cpu).compile_with(fg, &compile_opts_no_fusion(Device::Cpu));
        for (n, data) in &fp {
            fn_alone.set_param(n, data);
        }
        let ir_via_fn = fn_alone
            .run(&[("tgt", &ir_mid)])
            .into_iter()
            .next()
            .unwrap();
        let fd_got_fn = got
            .iter()
            .zip(&ir_via_fn)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            fd_got_fn < 1e-4,
            "pipeline output vs final_norm(ir_mid) max |Δ| = {fd_got_fn:.3e}"
        );
        // End-to-end vs host layers is dominated by stack LN sensitivity on ~2e-2
        // layer deltas; see `stack_final_norm_ir_matches_host_layer_output`.
    }

    #[test]
    fn cpu_layer_norm_row_matches_host_last_cpu() {
        let rows = 64usize;
        let h = 256usize;
        let x: Vec<f32> = (0..rows * h).map(|i| (i as f32) * 1e-3 - 0.5).collect();
        let g = vec![1.0; h];
        let b = vec![0.0; h];
        let mut host = x.clone();
        layer_norm_last_cpu(&mut host, rows, h, &g, &b, LN_EPS);
        let mut cpu = x.clone();
        for r in 0..rows {
            rlx_cpu::kernels::layer_norm_row(
                &x[r * h..(r + 1) * h],
                &g,
                &b,
                &mut cpu[r * h..(r + 1) * h],
                h,
                LN_EPS,
            );
        }
        let fd = host
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            fd < 1e-5,
            "cpu layer_norm_row vs host_last_cpu max |Δ| = {fd:.3e}"
        );
    }

    #[test]
    fn layer_norm_ir_matches_host_synthetic() {
        let n_img = 64usize;
        let d = 256usize;
        let x: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-3 - 0.5).collect();
        let norm_g = vec![1.0; d];
        let norm_b = vec![0.0; d];
        let mut host = x.clone();
        layer_norm_last_cpu(&mut host, n_img, d, &norm_g, &norm_b, LN_EPS);
        let (fg, fp) = build_final_norm_graph(&norm_g, &norm_b, n_img, d).unwrap();
        let mut compiled =
            Session::new(Device::Cpu).compile_with(fg, &compile_opts_no_fusion(Device::Cpu));
        for (n, data) in &fp {
            compiled.set_param(n, data);
        }
        let ir = compiled.run(&[("tgt", &x)]).into_iter().next().unwrap();
        let fd = host
            .iter()
            .zip(&ir)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            fd < 1e-4,
            "synthetic final norm IR vs host max |Δ| = {fd:.3e}"
        );
    }

    #[test]
    fn stack_final_norm_ir_matches_host_layer_output() {
        let d = 256usize;
        let n_img = 64usize;
        let layer_tgt: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-4).collect();
        let curr_pos: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-5).collect();
        let memory: Vec<f32> = (0..n_img * 64).map(|i| (i as f32) * 2e-4).collect();
        let memory_pos: Vec<f32> = (0..n_img * 64).map(|i| (i as f32) * 2e-5).collect();
        let layer = Sam2MemoryAttentionLayerWeights {
            self_attn: synth_rope_attn(d, d, [8, 8]),
            cross_attn: synth_rope_attn(d, 64, [8, 8]),
            norm1_g: vec![1.0; d],
            norm1_b: vec![0.0; d],
            norm2_g: vec![1.0; d],
            norm2_b: vec![0.0; d],
            norm3_g: vec![1.0; d],
            norm3_b: vec![0.0; d],
            linear1_w: vec![0.01; 2048 * d],
            linear1_b: vec![0.0; 2048],
            linear2_w: vec![0.02; d * 2048],
            linear2_b: vec![0.0; d],
            pos_enc_at_attn: false,
            pos_enc_at_cross_attn_queries: false,
            pos_enc_at_cross_attn_keys: true,
            d_model: d,
        };
        let host_layer = memory_attention_layer_forward(
            &layer,
            layer_tgt,
            &curr_pos,
            &memory,
            &memory_pos,
            n_img,
            n_img,
            64,
            0,
        )
        .unwrap();
        let mut host_final = host_layer.clone();
        let norm_g = vec![1.0; d];
        let norm_b = vec![0.0; d];
        layer_norm_last_cpu(&mut host_final, n_img, d, &norm_g, &norm_b, LN_EPS);

        let (fg, fp) = build_final_norm_graph(&norm_g, &norm_b, n_img, d).unwrap();
        let mut compiled =
            Session::new(Device::Cpu).compile_with(fg, &compile_opts_no_fusion(Device::Cpu));
        for (n, data) in &fp {
            compiled.set_param(n, data);
        }
        let ir_final = compiled
            .run(&[("tgt", &host_layer)])
            .into_iter()
            .next()
            .unwrap();
        let fd = host_final
            .iter()
            .zip(&ir_final)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 1e-4, "stack final norm IR vs host max |Δ| = {fd:.3e}");
    }

    /// Bisect monolithic layer: compare fused layer graph vs host `memory_attention_layer_forward`.
    #[test]
    fn memory_attention_layer_in_graph_rope_bisect() {
        let d = 256usize;
        let kv = 64usize;
        let feat = [8usize, 8usize];
        let n_img = 64usize;
        let n_mem = 64usize;
        let layer = Sam2MemoryAttentionLayerWeights {
            self_attn: synth_rope_attn(d, d, feat),
            cross_attn: synth_rope_attn(d, kv, feat),
            norm1_g: vec![1.0; d],
            norm1_b: vec![0.0; d],
            norm2_g: vec![1.0; d],
            norm2_b: vec![0.0; d],
            norm3_g: vec![1.0; d],
            norm3_b: vec![0.0; d],
            linear1_w: vec![0.01; 2048 * d],
            linear1_b: vec![0.0; 2048],
            linear2_w: vec![0.02; d * 2048],
            linear2_b: vec![0.0; d],
            pos_enc_at_attn: false,
            pos_enc_at_cross_attn_queries: false,
            pos_enc_at_cross_attn_keys: true,
            d_model: d,
        };

        let mut tgt: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-4).collect();
        for i in 0..tgt.len() {
            tgt[i] += INPUT_POS_SCALE * (i as f32) * 1e-5;
        }
        let curr_pos: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-5).collect();
        let memory: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-4).collect();
        let memory_pos: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-5).collect();

        let host = crate::memory_attention::memory_attention_layer_forward(
            &layer,
            tgt.clone(),
            &curr_pos,
            &memory,
            &memory_pos,
            n_img,
            n_mem,
            kv,
            0,
        )
        .unwrap();

        let (g, p) = build_layer_graph(&layer, n_img, n_mem, kv, 0).unwrap();
        let nh = layer.cross_attn.num_heads;
        let mut mask = vec![0f32; nh * n_img * n_mem];
        fill_cross_attn_bias(&mut mask, nh, n_img, n_mem, n_mem);
        let mut compiled =
            Session::new(Device::Cpu).compile_with(g, &compile_opts_no_fusion(Device::Cpu));
        for (n, data) in &p {
            compiled.set_param(n, data);
        }
        let got = compiled
            .run(&[
                ("tgt", &tgt),
                ("curr_pos", &curr_pos),
                ("memory", &memory),
                ("memory_pos", &memory_pos),
                ("mask_ca", &mask),
            ])
            .into_iter()
            .next()
            .unwrap();

        let fd = host
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(fd < 3e-2, "layer in-graph rope max |Δ| = {fd:.3e}");
    }

    /// Quick check: in-graph RoPE compiles and runs; log timing vs default (no hard SLA).
    #[test]
    fn memory_attention_in_graph_rope_timing_quick_check() {
        use std::time::Instant;

        let d = 256usize;
        let kv = 64usize;
        let feat = [8usize, 8usize];
        let n_img = 64usize;
        let n_mem = 64usize;
        let layer = Sam2MemoryAttentionLayerWeights {
            self_attn: synth_rope_attn(d, d, feat),
            cross_attn: synth_rope_attn(d, kv, feat),
            norm1_g: vec![1.0; d],
            norm1_b: vec![0.0; d],
            norm2_g: vec![1.0; d],
            norm2_b: vec![0.0; d],
            norm3_g: vec![1.0; d],
            norm3_b: vec![0.0; d],
            linear1_w: vec![0.01; 2048 * d],
            linear1_b: vec![0.0; 2048],
            linear2_w: vec![0.02; d * 2048],
            linear2_b: vec![0.0; d],
            pos_enc_at_attn: false,
            pos_enc_at_cross_attn_queries: false,
            pos_enc_at_cross_attn_keys: true,
            d_model: d,
        };
        let w = Sam2MemoryAttentionWeights {
            layers: vec![layer],
            norm_g: vec![1.0; d],
            norm_b: vec![0.0; d],
            d_model: d,
            pos_enc_at_input: true,
        };
        let curr: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-4).collect();
        let curr_pos: Vec<f32> = (0..n_img * d).map(|i| (i as f32) * 1e-5).collect();
        let memory: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-4).collect();
        let memory_pos: Vec<f32> = (0..n_mem * kv).map(|i| (i as f32) * 2e-5).collect();

        let t0 = Instant::now();
        let mut default =
            MemoryAttentionCompiled::compile(&w, n_img, n_mem, 0, Device::Cpu).unwrap();
        let compile_default_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let mut in_graph =
            MemoryAttentionCompiled::compile_in_graph_rope(&w, n_img, n_mem, 0, Device::Cpu)
                .unwrap();
        let compile_in_graph_ms = t1.elapsed().as_secs_f64() * 1000.0;

        const RUNS: usize = 5;
        let t2 = Instant::now();
        for _ in 0..RUNS {
            let _ = default
                .run(&curr, &curr_pos, &memory, &memory_pos, n_mem, 0)
                .unwrap();
        }
        let run_default_ms = t2.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        let t3 = Instant::now();
        for _ in 0..RUNS {
            let _ = in_graph
                .run(&curr, &curr_pos, &memory, &memory_pos, n_mem, 0)
                .unwrap();
        }
        let run_in_graph_ms = t3.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        eprintln!(
            "mem_attn compile ms: default={compile_default_ms:.2} in_graph={compile_in_graph_ms:.2}; \
             run ms (avg/{RUNS}): default={run_default_ms:.2} in_graph={run_in_graph_ms:.2}"
        );
        assert!(compile_in_graph_ms > 0.0 && run_in_graph_ms > 0.0);
    }
}
