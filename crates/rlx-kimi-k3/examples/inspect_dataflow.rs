//! `inspect_dataflow` — print the op-level DATA FLOW of every repeating unit of
//! Kimi-K3 using rlx's graph inspector (`rlx_ir::inspect_hir`), or (with
//! `RLX_INSPECT_DUMP=1`) emit the lowered MIR graph as an opscope edge-list
//! (`# title` + `idx op_name in0 in1 …`) for `rlx-opscope`'s dataflow
//! repeated-pattern miner. Tiny synthetic dims at the real config *shapes*.
//!
//!   cargo run -p rlx-kimi-k3 --example inspect_dataflow [vision|kda|mla|moe|all]
//!   RLX_INSPECT_DUMP=1 cargo run ... -- kda > kda.txt   # then: opscope-graph kda.txt

use rlx_core::flow_util::graph_from_hir;
use rlx_ir::graph::NodeId;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Op, Shape, inspect_hir};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights, build_kda_layer};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights, build_mla_layer};
use rlx_kimi_k3::moe::{MoeDims, MoeWeights, build_latent_moe};
use rlx_kimi_k3::vision::{VisionBlockWeights, VisionDims, VisionWeights, build_vision};
use std::collections::HashMap;

type P = HashMap<String, Vec<f32>>;

fn v(n: usize) -> Vec<f32> {
    vec![0.02; n]
}

/// opscope-style op label — kind-level so repeated sub-DAGs collapse.
fn op_name(op: &Op) -> String {
    match op {
        Op::Activation(a) => format!("{a:?}"),
        Op::Binary(b) => format!("{b:?}"),
        Op::Input { .. } => "in".into(),
        Op::Param { .. } => "w".into(),
        Op::Constant { .. } => "k".into(),
        other => format!("{:?}", other.kind()),
    }
}

/// Either pretty-print the HIR flow, or (RLX_INSPECT_DUMP) emit the lowered
/// MIR graph as an opscope edge-list.
fn emit(hir: HirModule, params: P, title: &str) {
    if std::env::var_os("RLX_INSPECT_DUMP").is_some() {
        let (g, _) = graph_from_hir(hir, params).expect("lower");
        let nodes = g.nodes();
        let idx: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();
        println!("# {title}");
        for (i, n) in nodes.iter().enumerate() {
            let ins: Vec<String> = n
                .inputs
                .iter()
                .filter_map(|id| idx.get(id))
                .map(|x| x.to_string())
                .collect();
            println!("{i} {} {}", op_name(&n.op), ins.join(" "));
        }
    } else {
        println!("\n{}", "═".repeat(78));
        println!("  {title}");
        println!("{}", "═".repeat(78));
        print!("{}", inspect_hir(&hir));
    }
}

fn vision() {
    let d = VisionDims {
        hidden: 8,
        qkv_hidden: 8,
        num_heads: 2,
        head_dim: 4,
        inter: 16,
        merge: 2,
        text_hidden: 8,
        proj_mid: 8,
        eps: 1e-5,
        grid_h: 2,
        grid_w: 2,
    };
    let (l, hid, qh, hd) = (d.seq_len(), d.hidden, d.qkv_hidden, d.head_dim);
    let block = VisionBlockWeights {
        norm0: v(hid),
        wqkv: v(hid * 3 * qh),
        wo: v(qh * hid),
        norm1: v(hid),
        fc0: v(hid * d.inter),
        fc1: v(d.inter * hid),
    };
    let w = VisionWeights {
        blocks: vec![block],
        final_norm: v(hid),
        proj0: v(d.merge_in() * d.proj_mid),
        proj2: v(d.proj_mid * d.text_hidden),
        post_norm: v(d.text_hidden),
    };
    let mut hir = HirModule::new("vision_block");
    let mut g = HirMut::new(&mut hir);
    let hh = g.input("hidden", Shape::new(&[1, l, hid], DType::F32));
    let cos = g.input("cos", Shape::new(&[l, hd / 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[l, hd / 2], DType::F32));
    let mut p = HashMap::new();
    let out = build_vision(&mut g, &mut p, hh, cos, sin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    emit(hir, p, "MoonViT vision block x1 + patchmergerv2 projector");
}

fn kda() {
    let d = KdaDims {
        hidden: 8,
        num_heads: 2,
        head_dim: 4,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq: 3,
    };
    let (h, nh, hd) = (d.hidden, d.num_heads, d.head_dim);
    let proj = nh * hd;
    let w = KdaWeights {
        q_proj: v(h * proj),
        k_proj: v(h * proj),
        v_proj: v(h * proj),
        q_conv: v(proj * d.conv_kernel),
        k_conv: v(proj * d.conv_kernel),
        v_conv: v(proj * d.conv_kernel),
        f_a: v(h * hd),
        f_b: v(hd * proj),
        dt_bias: v(proj),
        a_log: v(hd),
        b_proj: v(h * nh),
        g_proj: v(h * proj),
        o_norm: v(hd),
        o_proj: v(proj * h),
    };
    let mut hir = HirModule::new("kda_layer");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[d.batch, d.seq, h], DType::F32));
    let mut p = HashMap::new();
    let out = build_kda_layer(&mut g, &mut p, "kda", hin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    emit(
        hir,
        p,
        "KimiLinear KDA layer (gated delta-net linear attn, 69/93 layers)",
    );
}

fn mla() {
    let d = MlaDims {
        hidden: 8,
        num_heads: 2,
        q_lora_rank: 4,
        kv_lora_rank: 4,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 2,
        v_head_dim: 3,
        eps: 1e-5,
        batch: 1,
        seq: 3,
    };
    let nh = d.num_heads;
    let qk = d.qk_nope_head_dim + d.qk_rope_head_dim;
    let w = MlaWeights {
        q_a_proj: v(d.hidden * d.q_lora_rank),
        q_a_layernorm: v(d.q_lora_rank),
        q_b_proj: v(d.q_lora_rank * nh * qk),
        kv_a_proj_with_mqa: v(d.hidden * (d.kv_lora_rank + d.qk_rope_head_dim)),
        kv_a_layernorm: v(d.kv_lora_rank),
        kv_b_proj: v(d.kv_lora_rank * nh * (d.qk_nope_head_dim + d.v_head_dim)),
        g_proj: v(d.hidden * nh * d.v_head_dim),
        o_proj: v(nh * d.v_head_dim * d.hidden),
    };
    let mut hir = HirModule::new("mla_layer");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[d.batch, d.seq, d.hidden], DType::F32));
    let mut p = HashMap::new();
    let out = build_mla_layer(&mut g, &mut p, "mla", hin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    emit(
        hir,
        p,
        "MLA NoPE latent attention (asymmetric v_head_dim, 24/93 layers)",
    );
}

fn moe() {
    let d = MoeDims {
        hidden: 8,
        latent: 6,
        moe_inter: 4,
        num_experts: 4,
        top_k: 2,
        num_shared: 1,
        routed_scaling: 1.0,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq: 3,
    };
    let (h, ll, i, e, s) = (d.hidden, d.latent, d.moe_inter, d.num_experts, d.num_shared);
    let w = MoeWeights {
        router: v(h * e),
        e_score_bias: v(e),
        down_latent: v(h * ll),
        up_latent: v(ll * h),
        routed_norm: v(ll),
        experts_gate_up: v(e * ll * 2 * i),
        experts_down: v(e * i * ll),
        shared_gate: v(h * s * i),
        shared_up: v(h * s * i),
        shared_down: v(s * i * h),
    };
    let mut hir = HirModule::new("latent_moe");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("h", Shape::new(&[d.batch, d.seq, h], DType::F32));
    let mut p = HashMap::new();
    let out = build_latent_moe(&mut g, &mut p, "moe", hin, &w, d).unwrap();
    g.set_outputs(vec![out]);
    emit(
        hir,
        p,
        "LatentMoE (batched: router + GroupedMatMul experts + shared)",
    );
}

fn main() {
    let unit = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    match unit.as_str() {
        "vision" => vision(),
        "kda" => kda(),
        "mla" => mla(),
        "moe" => moe(),
        _ => {
            vision();
            kda();
            mla();
            moe();
        }
    }
}
