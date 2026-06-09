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

//! Two-way transformer IR (`Op::Attention` + LayerNorm + ReLU MLP).

use anyhow::Result;
use rlx_flow::CompileProfile;
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-5;

/// Max extra sparse prompt tokens (points/boxes) compiled into the IR graph.
pub const MAX_SPARSE_PROMPT_TOKENS: usize = 32;

struct LayerMaskIds {
    self_attn: NodeId,
    t2i: NodeId,
    i2t: NodeId,
}

/// Attention weights in PyTorch `[out, in]` layout.
#[derive(Clone)]
pub struct AttentionSpec {
    pub q_w: Vec<f32>,
    pub q_b: Vec<f32>,
    pub k_w: Vec<f32>,
    pub k_b: Vec<f32>,
    pub v_w: Vec<f32>,
    pub v_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
    pub num_heads: usize,
    pub embed_dim: usize,
    pub internal_dim: usize,
}

#[derive(Clone)]
pub struct TwoWayBlockSpec {
    pub self_attn: AttentionSpec,
    pub norm1_g: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub cross_token_to_image: AttentionSpec,
    pub norm2_g: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub mlp_lin1_w: Vec<f32>,
    pub mlp_lin1_b: Vec<f32>,
    pub mlp_lin2_w: Vec<f32>,
    pub mlp_lin2_b: Vec<f32>,
    pub norm3_g: Vec<f32>,
    pub norm3_b: Vec<f32>,
    pub cross_image_to_token: AttentionSpec,
    pub norm4_g: Vec<f32>,
    pub norm4_b: Vec<f32>,
    pub skip_first_layer_pe: bool,
}

#[derive(Clone)]
pub struct TwoWayTransformerSpec {
    pub layers: Vec<TwoWayBlockSpec>,
    pub final_attn: AttentionSpec,
    pub norm_final_g: Vec<f32>,
    pub norm_final_b: Vec<f32>,
    pub embed_dim: usize,
}

pub struct TwoWayTransformerCompiled {
    graph: CompiledGraph,
    /// Compiled query-token slots (`base + MAX_SPARSE` when `masked`).
    pub max_q_n: usize,
    pub k_n: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// When true, pass per-attention masks and pad queries to `max_q_n`.
    pub masked: bool,
}

impl TwoWayTransformerCompiled {
    pub fn compile(
        spec: &TwoWayTransformerSpec,
        q_n: usize,
        k_n: usize,
        device: Device,
    ) -> Result<Self> {
        Self::compile_with_profile(
            spec,
            q_n,
            k_n,
            device,
            false,
            &CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_with_profile(
        spec: &TwoWayTransformerSpec,
        q_n: usize,
        k_n: usize,
        device: Device,
        masked: bool,
        profile: &CompileProfile,
    ) -> Result<Self> {
        Self::compile_inner(spec, q_n, k_n, device, masked, profile)
    }

    /// `base_q_n` + up to [`MAX_SPARSE_PROMPT_TOKENS`] padded query slots (masked attention).
    pub fn compile_with_sparse_slots(
        spec: &TwoWayTransformerSpec,
        base_q_n: usize,
        k_n: usize,
        device: Device,
    ) -> Result<Self> {
        let max_q = base_q_n + MAX_SPARSE_PROMPT_TOKENS;
        Self::compile_with_profile(
            spec,
            max_q,
            k_n,
            device,
            true,
            &CompileProfile::sam_encoder(),
        )
    }

    pub fn compile_with_sparse_slots_profile(
        spec: &TwoWayTransformerSpec,
        base_q_n: usize,
        k_n: usize,
        device: Device,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let max_q = base_q_n + MAX_SPARSE_PROMPT_TOKENS;
        Self::compile_with_profile(spec, max_q, k_n, device, true, profile)
    }

    fn compile_inner(
        spec: &TwoWayTransformerSpec,
        max_q_n: usize,
        k_n: usize,
        device: Device,
        masked: bool,
        profile: &CompileProfile,
    ) -> Result<Self> {
        let nh = spec
            .layers
            .first()
            .map(|l| l.self_attn.num_heads)
            .unwrap_or(spec.final_attn.num_heads);
        let (graph, params) = build_transformer_graph(spec, max_q_n, k_n, masked)?;
        let mut compiled =
            rlx_core::flow_bridge::compile_graph_with_profile(device, graph, profile)?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            graph: compiled,
            max_q_n,
            k_n,
            embed_dim: spec.embed_dim,
            num_heads: nh,
            num_layers: spec.layers.len(),
            masked,
        })
    }

    /// Fill `[1, H, max_q, max_k]` mask (1 = attend, 0 = mask out).
    pub fn fill_attn_mask(
        out: &mut [f32],
        num_heads: usize,
        max_q: usize,
        max_k: usize,
        active_q: usize,
        active_k: usize,
    ) {
        out.fill(0.0);
        for h in 0..num_heads {
            for qi in 0..active_q.min(max_q) {
                for s in 0..active_k.min(max_k) {
                    let idx = (h * max_q + qi) * max_k + s;
                    out[idx] = 1.0;
                }
            }
        }
    }

    /// NCHW `[E, H, W]` → sequence `[H*W, E]` (same layout as host `two_way_transformer_forward`).
    pub fn nchw_to_seq(nchw: &[f32], e: usize, h: usize, w: usize) -> Vec<f32> {
        let k_n = h * w;
        let mut seq = vec![0f32; k_n * e];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..e {
                    let src = ch * h * w + y * w + x;
                    let dst = (y * w + x) * e + ch;
                    seq[dst] = nchw[src];
                }
            }
        }
        seq
    }

    /// `tokens`: `[q_n, E]`; image tensors NCHW `[E, g, g]`.
    pub fn run_nchw(
        &mut self,
        tokens: &[f32],
        image_nchw: &[f32],
        image_pe_nchw: &[f32],
        grid: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let e = self.embed_dim;
        let image_seq = Self::nchw_to_seq(image_nchw, e, grid, grid);
        let image_pe = Self::nchw_to_seq(image_pe_nchw, e, grid, grid);
        if self.masked {
            self.run_nchw_masked(tokens, tokens.len() / e, image_nchw, image_pe_nchw, grid)
        } else {
            self.run(tokens, &image_seq, &image_pe)
        }
    }

    /// Padded-query path: `active_q_n` real tokens, rest masked out.
    pub fn run_nchw_masked(
        &mut self,
        tokens: &[f32],
        active_q_n: usize,
        image_nchw: &[f32],
        image_pe_nchw: &[f32],
        grid: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        anyhow::ensure!(
            self.masked,
            "run_nchw_masked requires compile_with_sparse_slots"
        );
        anyhow::ensure!(
            active_q_n <= self.max_q_n,
            "active_q_n {active_q_n} > compiled max_q_n {}",
            self.max_q_n
        );
        let e = self.embed_dim;
        let image_seq = Self::nchw_to_seq(image_nchw, e, grid, grid);
        let image_pe = Self::nchw_to_seq(image_pe_nchw, e, grid, grid);
        let mut padded = vec![0f32; self.max_q_n * e];
        padded[..tokens.len()].copy_from_slice(tokens);
        let (q, k) = self.run_masked(&padded, active_q_n, &image_seq, &image_pe)?;
        Ok((q, k))
    }

    /// `tokens` / `query_pe`: `[q_n, E]`; `image_seq` / `image_pe_seq`: `[k_n, E]` row-major.
    pub fn run(
        &mut self,
        tokens: &[f32],
        image_seq: &[f32],
        image_pe_seq: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let e = self.embed_dim;
        anyhow::ensure!(!self.masked, "use run_masked for masked compile");
        anyhow::ensure!(tokens.len() == self.max_q_n * e, "tokens len mismatch");
        anyhow::ensure!(image_seq.len() == self.k_n * e, "image_seq len mismatch");
        anyhow::ensure!(
            image_pe_seq.len() == self.k_n * e,
            "image_pe_seq len mismatch"
        );
        let outs = self.graph.run(&[
            ("tokens", tokens),
            ("image_seq", image_seq),
            ("image_pe", image_pe_seq),
        ]);
        let mut it = outs.into_iter();
        let queries = it.next().expect("queries_out");
        let keys = it.next().expect("keys_out");
        Ok((queries, keys))
    }

    pub fn run_masked(
        &mut self,
        tokens_padded: &[f32],
        active_q_n: usize,
        image_seq: &[f32],
        image_pe_seq: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let e = self.embed_dim;
        let nh = self.num_heads;
        let max_q = self.max_q_n;
        let max_k = self.k_n;
        let plane = max_q * max_k;
        let mut mask_buf = vec![0f32; nh * plane];

        let mut owned: Vec<(String, Vec<f32>)> = vec![
            ("tokens".into(), tokens_padded.to_vec()),
            ("image_seq".into(), image_seq.to_vec()),
            ("image_pe".into(), image_pe_seq.to_vec()),
        ];
        for i in 0..self.num_layers {
            Self::fill_attn_mask(&mut mask_buf, nh, max_q, max_q, active_q_n, active_q_n);
            owned.push((format!("mask_L{i}_self"), mask_buf.clone()));
            Self::fill_attn_mask(&mut mask_buf, nh, max_q, max_k, active_q_n, max_k);
            owned.push((format!("mask_L{i}_t2i"), mask_buf.clone()));
            Self::fill_attn_mask(&mut mask_buf, nh, max_k, max_q, max_k, active_q_n);
            owned.push((format!("mask_L{i}_i2t"), mask_buf.clone()));
        }
        Self::fill_attn_mask(&mut mask_buf, nh, max_q, max_k, active_q_n, max_k);
        owned.push(("mask_final_t2i".into(), mask_buf.clone()));

        let feeds: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        let outs = self.graph.run(&feeds);
        let mut it = outs.into_iter();
        let queries_full = it.next().expect("queries_out");
        let keys = it.next().expect("keys_out");
        let mut queries = vec![0f32; active_q_n * e];
        queries.copy_from_slice(&queries_full[..active_q_n * e]);
        Ok((queries, keys))
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

fn build_attention(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    spec: &AttentionSpec,
    q_in: NodeId,
    k_in: NodeId,
    v_in: NodeId,
    q_len: usize,
    k_len: usize,
    mask: Option<NodeId>,
) -> NodeId {
    let e = spec.embed_dim;
    let id = spec.internal_dim;
    let nh = spec.num_heads;
    let dh = id / nh;
    let f = DType::F32;

    let q_proj = linear(
        g,
        params,
        &format!("{prefix}.q"),
        q_in,
        &spec.q_w,
        &spec.q_b,
        e,
        id,
        q_len,
    );
    let k_proj = linear(
        g,
        params,
        &format!("{prefix}.k"),
        k_in,
        &spec.k_w,
        &spec.k_b,
        e,
        id,
        k_len,
    );
    let v_proj = linear(
        g,
        params,
        &format!("{prefix}.v"),
        v_in,
        &spec.v_w,
        &spec.v_b,
        e,
        id,
        k_len,
    );
    let out_shape = Shape::new(&[1, q_len, id], f);
    let attn = if let Some(m) = mask {
        g.attention(q_proj, k_proj, v_proj, m, nh, dh, out_shape.clone())
    } else {
        g.attention_kind(
            q_proj,
            k_proj,
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
        &spec.out_w,
        &spec.out_b,
        id,
        e,
        q_len,
    )
}

fn build_block(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    block: &TwoWayBlockSpec,
    queries: NodeId,
    keys: NodeId,
    query_pe: NodeId,
    key_pe: NodeId,
    q_n: usize,
    k_n: usize,
    e: usize,
    masks: Option<&LayerMaskIds>,
) -> (NodeId, NodeId) {
    let f = DType::F32;
    let q_shape = Shape::new(&[1, q_n, e], f);
    let k_shape = Shape::new(&[1, k_n, e], f);

    let m_self = masks.map(|m| m.self_attn);
    let m_t2i = masks.map(|m| m.t2i);
    let m_i2t = masks.map(|m| m.i2t);

    let mut q = if block.skip_first_layer_pe {
        build_attention(
            g,
            params,
            &format!("{prefix}.self"),
            &block.self_attn,
            queries,
            queries,
            queries,
            q_n,
            q_n,
            m_self,
        )
    } else {
        let q_pe_sum = g.binary(BinaryOp::Add, queries, query_pe, q_shape.clone());
        let attn = build_attention(
            g,
            params,
            &format!("{prefix}.self"),
            &block.self_attn,
            q_pe_sum,
            q_pe_sum,
            queries,
            q_n,
            q_n,
            m_self,
        );
        g.binary(BinaryOp::Add, queries, attn, q_shape.clone())
    };
    q = layer_norm(
        g,
        params,
        &format!("{prefix}.n1"),
        q,
        &block.norm1_g,
        &block.norm1_b,
        q_n,
        e,
    );

    let q_pe_sum = g.binary(BinaryOp::Add, q, query_pe, q_shape.clone());
    let k_pe_sum = g.binary(BinaryOp::Add, keys, key_pe, k_shape.clone());
    let cross_t = build_attention(
        g,
        params,
        &format!("{prefix}.t2i"),
        &block.cross_token_to_image,
        q_pe_sum,
        k_pe_sum,
        keys,
        q_n,
        k_n,
        m_t2i,
    );
    q = g.binary(BinaryOp::Add, q, cross_t, q_shape.clone());
    q = layer_norm(
        g,
        params,
        &format!("{prefix}.n2"),
        q,
        &block.norm2_g,
        &block.norm2_b,
        q_n,
        e,
    );

    let mlp_dim = block.mlp_lin1_b.len();
    let mid = linear(
        g,
        params,
        &format!("{prefix}.mlp1"),
        q,
        &block.mlp_lin1_w,
        &block.mlp_lin1_b,
        e,
        mlp_dim,
        q_n,
    );
    let mid_relu = g.activation(Activation::Relu, mid, Shape::new(&[1, q_n, mlp_dim], f));
    let mlp_out = linear(
        g,
        params,
        &format!("{prefix}.mlp2"),
        mid_relu,
        &block.mlp_lin2_w,
        &block.mlp_lin2_b,
        mlp_dim,
        e,
        q_n,
    );
    q = g.binary(BinaryOp::Add, q, mlp_out, q_shape.clone());
    q = layer_norm(
        g,
        params,
        &format!("{prefix}.n3"),
        q,
        &block.norm3_g,
        &block.norm3_b,
        q_n,
        e,
    );

    let q_pe2 = g.binary(BinaryOp::Add, q, query_pe, q_shape.clone());
    let k_pe2 = g.binary(BinaryOp::Add, keys, key_pe, k_shape.clone());
    let cross_i = build_attention(
        g,
        params,
        &format!("{prefix}.i2t"),
        &block.cross_image_to_token,
        k_pe2,
        q_pe2,
        q,
        k_n,
        q_n,
        m_i2t,
    );
    let keys_out = g.binary(BinaryOp::Add, keys, cross_i, k_shape);
    let keys_out = layer_norm(
        g,
        params,
        &format!("{prefix}.n4"),
        keys_out,
        &block.norm4_g,
        &block.norm4_b,
        k_n,
        e,
    );
    (q, keys_out)
}

fn build_transformer_graph(
    spec: &TwoWayTransformerSpec,
    q_n: usize,
    k_n: usize,
    masked: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let e = spec.embed_dim;
    let f = DType::F32;
    let mut g = Graph::new("twoway_transformer");
    let mut params = HashMap::new();
    let nh0 = spec
        .layers
        .first()
        .map(|l| l.self_attn.num_heads)
        .unwrap_or(spec.final_attn.num_heads);

    let tokens = g.input("tokens", Shape::new(&[1, q_n, e], f));
    let image_seq = g.input("image_seq", Shape::new(&[1, k_n, e], f));
    let image_pe = g.input("image_pe", Shape::new(&[1, k_n, e], f));
    let query_pe = tokens;

    let mut layer_masks = Vec::new();
    if masked {
        for i in 0..spec.layers.len() {
            let nh = spec.layers[i].self_attn.num_heads;
            layer_masks.push(LayerMaskIds {
                self_attn: g.input(format!("mask_L{i}_self"), Shape::new(&[1, nh, q_n, q_n], f)),
                t2i: g.input(format!("mask_L{i}_t2i"), Shape::new(&[1, nh, q_n, k_n], f)),
                i2t: g.input(format!("mask_L{i}_i2t"), Shape::new(&[1, nh, k_n, q_n], f)),
            });
        }
    }
    let final_mask = if masked {
        Some(g.input("mask_final_t2i", Shape::new(&[1, nh0, q_n, k_n], f)))
    } else {
        None
    };

    let mut queries = tokens;
    let mut keys = image_seq;
    for (i, layer) in spec.layers.iter().enumerate() {
        let masks = if masked { Some(&layer_masks[i]) } else { None };
        let (q, k) = build_block(
            &mut g,
            &mut params,
            &format!("L{i}"),
            layer,
            queries,
            keys,
            query_pe,
            image_pe,
            q_n,
            k_n,
            e,
            masks,
        );
        queries = q;
        keys = k;
    }

    let q_shape = Shape::new(&[1, q_n, e], f);
    let k_shape = Shape::new(&[1, k_n, e], f);
    let q_pe_f = g.binary(BinaryOp::Add, queries, query_pe, q_shape.clone());
    let k_pe_f = g.binary(BinaryOp::Add, keys, image_pe, k_shape.clone());
    let final_attn = build_attention(
        &mut g,
        &mut params,
        "final",
        &spec.final_attn,
        q_pe_f,
        k_pe_f,
        keys,
        q_n,
        k_n,
        final_mask,
    );
    let queries_out = g.binary(BinaryOp::Add, queries, final_attn, q_shape.clone());
    let queries_out = layer_norm(
        &mut g,
        &mut params,
        "final_ln",
        queries_out,
        &spec.norm_final_g,
        &spec.norm_final_b,
        q_n,
        e,
    );

    g.set_outputs(vec![queries_out, keys]);
    Ok((g, params))
}
