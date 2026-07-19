// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! A generic, differentiable Vision Transformer forward built directly at the
//! `rlx_ir::Graph` level.
//!
//! Building at the `Graph` level (rather than through `rlx-flow`'s inference
//! `ModelFlow`) is deliberate: every weight is a named `Op::Param`, so the
//! graph composes with [`rlx_autodiff::grad_with_loss`] (SnapViT gradients,
//! GLARE training) and with `rlx-tune`'s `Trainer`. Two extra inputs
//! `head_mask [L·H]` and `ffn_mask [L·inner]` are multiplied per-channel into
//! the attention output (before `proj`) and the FFN inner (before the down
//! projection) — feeding all-ones is a numerical identity, so the same
//! compiled graph serves both a bit-exact reference forward and every
//! SnapViT pruning candidate with no recompilation.
//!
//! Each block mirrors the reference ViT / DINOv2 topology:
//! ```text
//!   x = x + ls1 · proj(head_mask ⊙ attn(norm1(x)))
//!   x = x + ls2 · fc2(ffn_mask ⊙ act(fc1(norm2(x))))
//! ```
//! (`ls*` present only when `cfg.layer_scale`). Weights are the canonical
//! matmul-ready (`[in, out]`) layout produced by [`super::weights`].

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use super::config::{FfnKind, VitConfig};

const F: DType = DType::F32;

/// A declared weight parameter: its canonical name, graph node, and dims.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: String,
    pub node: NodeId,
    pub dims: Vec<usize>,
}

/// A built ViT forward graph plus the handles needed to run, differentiate,
/// and mask it.
pub struct VitGraph {
    /// The compute graph. Its single output is the post-final-norm token
    /// sequence `[B, seq, H]`; loss builders may append ops and re-`set_outputs`.
    pub graph: Graph,
    pub cfg: VitConfig,
    pub batch: usize,
    /// Post-final-norm token sequence `[B, seq, H]`. The pooled `[CLS]` feature
    /// is extracted lazily by loss builders via [`extract_cls`] — the base
    /// forward stays free of the CLS `transpose`/`narrow` (dead code in a
    /// pure-inference runner; it clobbered batched Metal forward via the arena).
    pub output: NodeId,
    /// `"hidden"` input `[B, seq, H]` — the host-assembled patch/token tensor.
    pub hidden_input: NodeId,
    /// `"head_mask"` input `[L·H]` — per-attention-channel keep factors.
    pub head_mask_input: NodeId,
    /// `"ffn_mask"` input `[L·inner]` — per-FFN-channel keep factors.
    pub ffn_mask_input: NodeId,
    /// Every backbone weight parameter, in creation order.
    pub params: Vec<ParamSpec>,
    /// UniAdapter parameters (empty unless built with [`AdapterOpts`]). These
    /// are GLARE's *only* trainable params (the backbone stays frozen).
    pub adapter_params: Vec<ParamSpec>,
}

/// UniAdapter insertion options (GLARE): `x' = x + s·ReLU(x·W_down)·W_up`,
/// inserted after each attention sub-block.
#[derive(Clone, Copy, Debug)]
pub struct AdapterOpts {
    pub rank: usize,
    pub scale: f32,
}

impl VitGraph {
    /// Names of all weight params (for `Trainer` / `set_param`).
    pub fn param_names(&self) -> Vec<String> {
        self.params.iter().map(|p| p.name.clone()).collect()
    }
    /// All-ones head mask (`[L·H]`) — the no-pruning identity.
    pub fn ones_head_mask(&self) -> Vec<f32> {
        vec![1.0; self.cfg.num_hidden_layers * self.cfg.hidden_size]
    }
    /// All-ones FFN mask (`[L·inner]`) — the no-pruning identity.
    pub fn ones_ffn_mask(&self) -> Vec<f32> {
        vec![1.0; self.cfg.num_hidden_layers * self.cfg.ffn_inner()]
    }
}

/// Extract the pooled `[CLS]` feature `[batch, hidden]` (row 0 of a
/// `[batch, seq, hidden]` token sequence). Uses a LAST-axis narrow (transpose
/// seq to the end first): a middle-axis (seq) narrow's backward is NaN on Metal,
/// whereas last-axis narrows (as in the qkv split) are correct everywhere.
pub fn extract_cls(g: &mut Graph, output: NodeId, batch: usize, hidden: usize) -> NodeId {
    let out_t = g.transpose_(output, vec![0, 2, 1]); // [B, H, seq]
    let cls_t = g.narrow_(out_t, 2, 0, 1); // [B, H, 1]
    g.reshape_(cls_t, vec![batch as i64, hidden as i64]) // [B, H]
}

fn declare_param(
    g: &mut Graph,
    params: &mut Vec<ParamSpec>,
    name: String,
    dims: &[usize],
) -> NodeId {
    let node = g.param(name.clone(), Shape::new(dims, F));
    params.push(ParamSpec {
        name,
        node,
        dims: dims.to_vec(),
    });
    node
}

/// Build the differentiable, maskable ViT forward for `cfg` at `batch`.
pub fn build_vit_graph(cfg: &VitConfig, batch: usize) -> VitGraph {
    build_vit_graph_with(cfg, batch, None)
}

/// [`build_vit_graph`] with optional UniAdapters inserted after each attention
/// sub-block (GLARE).
pub fn build_vit_graph_with(
    cfg: &VitConfig,
    batch: usize,
    adapter: Option<AdapterOpts>,
) -> VitGraph {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let inner = cfg.ffn_inner();
    let seq = cfg.seq_len();
    let depth = cfg.num_hidden_layers;
    let eps = cfg.layer_norm_eps as f32;

    let mut g = Graph::new("vit-elastic");
    let mut params: Vec<ParamSpec> = Vec::new();
    let mut adapter_params: Vec<ParamSpec> = Vec::new();

    let hidden = g.input("hidden", Shape::new(&[batch, seq, h], F));
    let head_mask = g.input("head_mask", Shape::new(&[depth * h], F));
    let ffn_mask = g.input("ffn_mask", Shape::new(&[depth * inner], F));
    // All-ones (no-padding) attention mask `[batch, seq]`, exactly as the
    // Metal/MLX/wgpu-verified `rlx-uni2` reference — the kernel-synthesized
    // `MaskKind::None` path is not bit-safe on all backends (NaN on Metal).
    let attn_mask = g.full(&[batch, seq], 1.0, F);

    let mut x = hidden;
    for li in 0..depth {
        let lp = format!("blocks.{li}");

        // ---- attention sub-block ----
        let resid = x;
        let n1w = declare_param(&mut g, &mut params, format!("{lp}.norm1.weight"), &[h]);
        let n1b = declare_param(&mut g, &mut params, format!("{lp}.norm1.bias"), &[h]);
        let xn = g.ln(x, n1w, n1b, eps);

        let qkv_w = declare_param(
            &mut g,
            &mut params,
            format!("{lp}.attn.qkv.weight"),
            &[h, 3 * h],
        );
        let qkv_b = declare_param(&mut g, &mut params, format!("{lp}.attn.qkv.bias"), &[3 * h]);
        let qkv = g.mm(xn, qkv_w);
        let qkv = g.add(qkv, qkv_b);
        let q = g.narrow_(qkv, 2, 0, h);
        let k = g.narrow_(qkv, 2, h, h);
        let v = g.narrow_(qkv, 2, 2 * h, h);
        let attn = g.attention_(q, k, v, attn_mask, nh, hd);

        // Per-head mask: this layer's [li·H, (li+1)·H) slice, broadcast over [B,seq,H].
        let hm = g.narrow_(head_mask, 0, li * h, h);
        let attn = g.mul(attn, hm);

        let proj_w = declare_param(
            &mut g,
            &mut params,
            format!("{lp}.attn.proj.weight"),
            &[h, h],
        );
        let proj_b = declare_param(&mut g, &mut params, format!("{lp}.attn.proj.bias"), &[h]);
        let attn = g.mm(attn, proj_w);
        let attn = g.add(attn, proj_b);
        let attn = if cfg.layer_scale {
            let ls1 = declare_param(&mut g, &mut params, format!("{lp}.ls1.gamma"), &[h]);
            g.mul(attn, ls1)
        } else {
            attn
        };
        let mut x1 = g.add(resid, attn);

        // ---- UniAdapter (GLARE): x1 += s·ReLU(x1·W_down)·W_up ----
        if let Some(a) = adapter {
            let wd = g.param(
                format!("{lp}.adapter.down.weight"),
                Shape::new(&[h, a.rank], F),
            );
            adapter_params.push(ParamSpec {
                name: format!("{lp}.adapter.down.weight"),
                node: wd,
                dims: vec![h, a.rank],
            });
            let wu = g.param(
                format!("{lp}.adapter.up.weight"),
                Shape::new(&[a.rank, h], F),
            );
            adapter_params.push(ParamSpec {
                name: format!("{lp}.adapter.up.weight"),
                node: wu,
                dims: vec![a.rank, h],
            });
            let z = g.mm(x1, wd);
            let z = g.relu(z);
            let z = g.mm(z, wu);
            let scale = g.constant(a.scale as f64, F);
            let z = g.mul(z, scale);
            x1 = g.add(x1, z);
        }

        // ---- FFN sub-block ----
        let resid2 = x1;
        let n2w = declare_param(&mut g, &mut params, format!("{lp}.norm2.weight"), &[h]);
        let n2b = declare_param(&mut g, &mut params, format!("{lp}.norm2.bias"), &[h]);
        let xn2 = g.ln(x1, n2w, n2b, eps);
        let fm = g.narrow_(ffn_mask, 0, li * inner, inner);

        let y = match cfg.ffn_kind {
            FfnKind::Gelu => {
                let fc1w = declare_param(
                    &mut g,
                    &mut params,
                    format!("{lp}.mlp.fc1.weight"),
                    &[h, inner],
                );
                let fc1b =
                    declare_param(&mut g, &mut params, format!("{lp}.mlp.fc1.bias"), &[inner]);
                let hmid = g.mm(xn2, fc1w);
                let hmid = g.add(hmid, fc1b);
                let hmid = g.gelu(hmid);
                let hmid = g.mul(hmid, fm);
                let fc2w = declare_param(
                    &mut g,
                    &mut params,
                    format!("{lp}.mlp.fc2.weight"),
                    &[inner, h],
                );
                let fc2b = declare_param(&mut g, &mut params, format!("{lp}.mlp.fc2.bias"), &[h]);
                let y = g.mm(hmid, fc2w);
                g.add(y, fc2b)
            }
            FfnKind::PackedSwiGLU => {
                let vw = declare_param(
                    &mut g,
                    &mut params,
                    format!("{lp}.mlp.fc1_value.weight"),
                    &[h, inner],
                );
                let vb = declare_param(
                    &mut g,
                    &mut params,
                    format!("{lp}.mlp.fc1_value.bias"),
                    &[inner],
                );
                let gw = declare_param(
                    &mut g,
                    &mut params,
                    format!("{lp}.mlp.fc1_gate.weight"),
                    &[h, inner],
                );
                let gb = declare_param(
                    &mut g,
                    &mut params,
                    format!("{lp}.mlp.fc1_gate.bias"),
                    &[inner],
                );
                let val = g.mm(xn2, vw);
                let val = g.add(val, vb);
                let gate = g.mm(xn2, gw);
                let gate = g.add(gate, gb);
                let act = g.silu(val);
                let gated = g.mul(act, gate);
                let gated = g.mul(gated, fm);
                let fc2w = declare_param(
                    &mut g,
                    &mut params,
                    format!("{lp}.mlp.fc2.weight"),
                    &[inner, h],
                );
                let fc2b = declare_param(&mut g, &mut params, format!("{lp}.mlp.fc2.bias"), &[h]);
                let y = g.mm(gated, fc2w);
                g.add(y, fc2b)
            }
        };
        let y = if cfg.layer_scale {
            let ls2 = declare_param(&mut g, &mut params, format!("{lp}.ls2.gamma"), &[h]);
            g.mul(y, ls2)
        } else {
            y
        };
        x = g.add(resid2, y);
    }

    // Final norm + CLS pooling.
    let nw = declare_param(&mut g, &mut params, "norm.weight".to_string(), &[h]);
    let nb = declare_param(&mut g, &mut params, "norm.bias".to_string(), &[h]);
    let out = g.ln(x, nw, nb, eps);
    g.set_outputs(vec![out]);

    VitGraph {
        graph: g,
        cfg: cfg.clone(),
        batch,
        output: out,
        hidden_input: hidden,
        head_mask_input: head_mask,
        ffn_mask_input: ffn_mask,
        params,
        adapter_params,
    }
}
