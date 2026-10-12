//! Follow a non-finite training gradient upstream to the op that produces it.
//!
//! `every_trainable_gradient_is_finite` says *that* a backend poisons the
//! adapter gradients; this says *where*. It compiles the backward graph, then
//! walks from a bad gradient output into its inputs, re-rooting the graph at
//! each candidate, and descends into the first input that is already
//! non-finite. The op it stops at is the one that manufactured the NaN — its
//! inputs are all finite and its output is not.
//!
//! ```text
//! train_nan_probe [device]          # default: metal
//! ```
use anyhow::Result;
use rlx_autodiff::grad_with_loss;
use rlx_ir::{Graph, NodeId};
use rlx_kev::api::{Record, RecordQuestion};
use rlx_kev::encode::{EncodeOpts, Encoder, SPECIAL, TokenizerLike};
use rlx_kev::train::{
    LoraTargets, TrainConfig, TrainExample, TrainShape, build_feeds, build_train_graph,
};
use rlx_qwen35::synth::{synth_weights, tiny_cfg};
use rlx_qwen35::{MatWeight, Proj, Qwen35LayerFfn, Qwen35TrunkLayer, Qwen35Weights};
use rlx_runtime::{Device, Session};
use std::collections::HashSet;

struct FakeTok;
impl TokenizerLike for FakeTok {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        SPECIAL
            .iter()
            .position(|s| *s == token)
            .map(|i| 27 + i as u32)
    }
    fn encode_no_special(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.chars().map(|c| (c as u32) % 20 + 1).collect())
    }
}

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

fn randomize(w: &mut Qwen35Weights) {
    let fill = |v: &mut Vec<f32>, seed: u64| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = 0.5 * hashed(seed, i);
        }
    };
    let fill_proj = |p: &mut Proj, seed: u64| {
        for (j, m) in p.mats_mut().enumerate() {
            if let MatWeight::F32(v) = m {
                fill(v, seed.wrapping_add(j as u64 * 7919));
            }
        }
    };
    let fill_norm = |v: &mut Vec<f32>, seed: u64| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = 1.0 + 0.3 * hashed(seed, i);
        }
    };
    for (li, layer) in w.trunk_layers.iter_mut().enumerate() {
        let s = |k: u64| (li as u64 + 1).wrapping_mul(0x9E37_79B1).wrapping_add(k);
        let ffn = match layer {
            Qwen35TrunkLayer::Linear(l) => {
                fill_norm(&mut l.attn_norm, s(1));
                fill_norm(&mut l.attn_post_norm, s(2));
                fill_proj(&mut l.attn_qkv, s(3));
                fill_proj(&mut l.attn_gate, s(4));
                fill_proj(&mut l.ssm_beta, s(5));
                fill_proj(&mut l.ssm_alpha, s(6));
                fill_proj(&mut l.ssm_out, s(7));
                fill_norm(&mut l.ssm_norm, s(8));
                &mut l.ffn
            }
            Qwen35TrunkLayer::FullAttn(l) => {
                fill_norm(&mut l.attn_norm, s(1));
                fill_norm(&mut l.attn_post_norm, s(2));
                fill_proj(&mut l.attn_q_gate, s(3));
                fill_proj(&mut l.attn_k, s(4));
                fill_proj(&mut l.attn_v, s(5));
                fill_proj(&mut l.attn_output, s(6));
                fill_norm(&mut l.attn_q_norm, s(7));
                fill_norm(&mut l.attn_k_norm, s(8));
                &mut l.ffn
            }
        };
        if let Qwen35LayerFfn::Dense { gate, up, down } = ffn {
            fill_proj(gate, s(20));
            fill_proj(up, s(21));
            fill_proj(down, s(22));
        }
    }
    fill_norm(&mut w.output_norm, 0xFEED);
}

fn main() -> Result<()> {
    let dev = std::env::args()
        .nth(1)
        .map_or(Ok(Device::Metal), |d| rlx_runtime::parse_device(&d))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let cfg = tiny_cfg();
    let mut w = synth_weights(&cfg);
    randomize(&mut w);
    let shape = TrainShape {
        rows: 1,
        seq: 64,
        max_options: 3,
    };
    let tcfg = TrainConfig {
        lora_rank: 4,
        lora_alpha: 8.0,
        targets: LoraTargets::All,
        head_dim: 8,
        lora_dropout: 0.0,
        checkpoint_segments: 1,
        ..TrainConfig::default()
    };
    let e = Encoder::new(FakeTok)?;
    let ex = TrainExample {
        record: Record {
            state: "late parcel".into(),
            questions: vec![RecordQuestion {
                instr: "which team".into(),
                options: vec!["ret".into(), "ship".into(), "bill".into()],
                label: 1,
            }],
        },
    };
    let feeds = build_feeds(&e, &cfg, shape, &[ex], EncodeOpts::default())?;
    let g = build_train_graph(&cfg, w, shape, &tcfg)?;
    let names: Vec<String> = g.wrt.iter().map(|s| s.name.clone()).collect();
    let wrt: Vec<NodeId> = g.wrt.iter().map(|s| s.node).collect();
    let params = g.params.clone();
    let bwd = grad_with_loss(&g.forward, &wrt);

    // Every feed the backward graph needs, including the seed cotangent.
    let mut named = feeds.named.clone();
    named.push(("d_output".into(), vec![1.0]));

    let eval = |target: NodeId| -> Option<Vec<f32>> {
        let mut gg: Graph = bwd.clone();
        gg.set_outputs(vec![target]);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut c = Session::new(dev).compile(gg);
            for (n, v) in &params {
                c.set_param(n, v);
            }
            let feed: Vec<(&str, &[f32])> = named
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_slice()))
                .collect();
            c.run(&feed)[0].clone()
        }))
        .ok()
    };
    let finite = |id: NodeId| eval(id).is_some_and(|v| v.iter().all(|x| x.is_finite()));

    if let Ok(spec) = std::env::var("RLX_KEV_NODE") {
        for tok in spec.split(',') {
            let id = NodeId(tok.trim().parse().expect("node id"));
            let node = bwd.node(id);
            let v = eval(id).unwrap_or_default();
            let zeros = v.iter().filter(|x| **x == 0.0).count();
            let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
            let mn = v
                .iter()
                .cloned()
                .filter(|x| x.is_finite())
                .fold(f32::INFINITY, f32::min);
            let mx = v.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
            println!(
                "{dev:?} node {id:?} {:?} dims {:?}: n={} zeros={zeros} nonfinite={nonfinite} min={mn:.3e} max|v|={mx:.3e}",
                node.op,
                node.shape.dims(),
                v.len()
            );
            for (j, i) in node.inputs.iter().enumerate() {
                println!("    in[{j}] = {:?} {:?}", i, bwd.node(*i).op);
            }
            if v.len() > 512 {
                for (ci, ch) in v.chunks(512).enumerate().take(10) {
                    let cmx = ch.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
                    let cz = ch.iter().filter(|x| **x == 0.0).count();
                    println!(
                        "      chunk {ci} [{}..]: max|v|={cmx:.3e} zeros={cz}/{}",
                        ci * 512,
                        ch.len()
                    );
                }
            }
        }
        return Ok(());
    }

    // Pick a bad gradient output: bwd.outputs = [loss, grad(wrt[0]), ...].
    let mut start = None;
    for (i, name) in names.iter().enumerate() {
        let out = bwd.outputs[i + 1];
        if !finite(out) {
            println!("bad gradient: {name} (node {out:?})");
            start = Some(out);
            break;
        }
    }
    let Some(start) = start else {
        println!("every gradient is finite on {dev:?} — nothing to trace");
        return Ok(());
    };

    // Descend into the first non-finite input until every input is finite.
    let mut cur = start;
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(cur) {
            println!("cycle at {cur:?}");
            break;
        }
        let node = bwd.node(cur);
        let bad_input = node.inputs.iter().copied().find(|i| !finite(*i));
        match bad_input {
            Some(next) => {
                println!(
                    "  through {:?} {:?} {:?}",
                    node.op,
                    node.shape.dtype(),
                    node.shape.dims()
                );
                cur = next;
            }
            None => {
                let node = bwd.node(cur);
                println!("\nORIGIN: {:?}  (node {cur:?})", node.op);
                println!(
                    "  dtype {:?} dims {:?}",
                    node.shape.dtype(),
                    node.shape.dims()
                );
                for (j, i) in node.inputs.iter().enumerate() {
                    let n = bwd.node(*i);
                    let v = eval(*i).unwrap_or_default();
                    let mx = v.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
                    println!("  in[{j}] {:?} {:?} max|v|={mx:.3e}", n.op, n.shape.dims());
                }
                break;
            }
        }
    }
    Ok(())
}
