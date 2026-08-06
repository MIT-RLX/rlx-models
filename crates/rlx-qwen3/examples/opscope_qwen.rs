// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Drive the **real rlx-qwen3 decoder** (GQA + RMSNorm + RoPE + qk-norm + fused
//! attention) end-to-end through every opscope tool, incl. the new fused-
//! `Op::Attention` tap. Tiny random config, no disk. Requires `just link-local`.
//!   cd rlx-models && cargo run -p rlx-qwen3 --example opscope_qwen --release -- /tmp/qwen.csv
//!   cd ../rlx     && cargo run -p rlx-opscope --bin opscope-mine -- /tmp/qwen.csv

use std::collections::HashMap;
use std::path::Path;

use rlx_core::weight_map::WeightMap;
use rlx_ir::Philox4x32;
use rlx_opscope::dataflow::repeated_flow_patterns;
use rlx_opscope::shapes::{DEFAULT_RIDGE, gemm_shape_histogram, op_costs, roofline_class};
use rlx_opscope::{Recorder, StatConfig, inject_attention_stats, inject_matmul_stats};
use rlx_qwen3::{Qwen3Config, build_qwen3_graph_sized};
use rlx_runtime::{Device, Session};

/// Full Qwen3 weight set, randomly initialized (RMSNorm γ=1, others N(0,0.02²)).
fn random_weights(cfg: &Qwen3Config, rng: &mut Philox4x32) -> WeightMap {
    let (h, q_dim, kv_dim, int_dim, dh) = (
        cfg.hidden_size,
        cfg.q_proj_dim(),
        cfg.kv_proj_dim(),
        cfg.intermediate_size,
        cfg.head_dim,
    );
    let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let add = |t: &mut HashMap<_, _>, key: String, shape: Vec<usize>, rng: &mut Philox4x32| {
        let n: usize = shape.iter().product();
        let mut d = vec![0f32; n];
        if key.contains("norm") {
            d.fill(1.0); // RMSNorm / q_norm / k_norm weights
        } else {
            rng.fill_normal(&mut d);
            for v in &mut d {
                *v *= 0.02;
            }
        }
        t.insert(key, (d, shape));
    };
    add(
        &mut t,
        "model.embed_tokens.weight".into(),
        vec![cfg.vocab_size, h],
        rng,
    );
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        add(&mut t, format!("{lp}.input_layernorm.weight"), vec![h], rng);
        add(
            &mut t,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.self_attn.q_proj.weight"),
            vec![q_dim, h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.self_attn.o_proj.weight"),
            vec![h, q_dim],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.self_attn.q_norm.weight"),
            vec![dh],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.self_attn.k_norm.weight"),
            vec![dh],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.mlp.gate_proj.weight"),
            vec![int_dim, h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.mlp.up_proj.weight"),
            vec![int_dim, h],
            rng,
        );
        add(
            &mut t,
            format!("{lp}.mlp.down_proj.weight"),
            vec![h, int_dim],
            rng,
        );
    }
    add(&mut t, "model.norm.weight".into(), vec![h], rng);
    WeightMap::from_tensors(t)
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

/// Load the weight set: real trained qwen3-0.6B safetensors, or a tiny random
/// config. Returns `(cfg, WeightMap)` — used both for graph building and for the
/// per-layer tensor dump.
fn qwen3_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 4, // GQA (kv < q)
        head_dim: 8,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: true,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    }
}

/// True if `key` is a `model.layers.{i}.*` weight with `i < max_layers`.
fn within_layer_cap(key: &str, max_layers: usize) -> bool {
    match key.strip_prefix("model.layers.") {
        Some(rest) => rest
            .split('.')
            .next()
            .and_then(|n| n.parse::<usize>().ok())
            .map(|i| i < max_layers)
            .unwrap_or(false),
        None => false, // non-layer weights (embed/norm/lm_head) excluded from the cap set
    }
}

/// `... <dir> dump [real] [max_layers|all]` — dump 2-D weights to
/// `<dir>/<key>.tensor` (via `probe::save_tensor`) for `opscope-layers <dir>`
/// mining. `cap = Some(n)` restricts to the first `n` transformer blocks;
/// `cap = None` ("all") dumps EVERY 2-D weight including the huge tied
/// embedding/lm_head table (the single biggest tensor — ~26% of a 0.6B model).
/// Real mode needs the safetensors.
fn dump_tensors(dir: &str, real: bool, cap: Option<usize>) {
    use rlx_opscope::probe::save_tensor;
    std::fs::create_dir_all(dir).expect("create dump dir");
    let wm = if real {
        let base = Path::new("/Users/Shared/rlx-models/weights/lm/qwen3-0.6b");
        WeightMap::from_file(base.join("model.safetensors").to_str().unwrap()).expect("safetensors")
    } else {
        let mut rng = Philox4x32::new(0x9E37);
        random_weights(&qwen3_config(), &mut rng)
    };
    let keys: Vec<String> = wm.keys().map(|k| k.to_string()).collect();
    let (mut n, mut bytes) = (0usize, 0usize);
    for key in keys {
        // `None` (all) keeps every weight; `Some(n)` keeps only blocks < n.
        if let Some(nl) = cap {
            if !within_layer_cap(&key, nl) {
                continue;
            }
        }
        if let Some((data, shape)) = wm.get(&key) {
            if shape.len() != 2 || shape[0] < 2 || shape[1] < 2 {
                continue; // only 2-D matmul weights are decomposable
            }
            let (rows, cols) = (shape[0], shape[1]);
            let path = format!("{dir}/{key}.tensor");
            save_tensor(&path, rows, cols, data).expect("save_tensor");
            n += 1;
            bytes += data.len() * 4;
        }
    }
    let scope = cap
        .map(|n| format!("first {n} layers"))
        .unwrap_or_else(|| "ALL weights".into());
    println!(
        "dumped {n} 2-D weights ({scope}, {:.1} MB) → {dir}",
        bytes as f64 / 1e6
    );
    println!(
        "mine them: cd ../rlx && cargo run --release -p rlx-opscope --bin opscope-layers -- {dir} --quant"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // `... <dir> dump [real] [max_layers|all]` → per-layer tensor dumper.
    if args.get(2).map(|s| s == "dump").unwrap_or(false) {
        let dir = args.get(1).cloned().unwrap_or_else(|| "qwen_layers".into());
        let real = args.get(3).map(|s| s == "real").unwrap_or(false);
        // 4th arg: "all" → every weight; a number → that many blocks; default 4.
        let cap = match args.get(4).map(|s| s.as_str()) {
            Some("all") => None,
            Some(n) => Some(n.parse().unwrap_or(4usize)),
            None => Some(4usize),
        };
        dump_tensors(&dir, real, cap);
        return;
    }
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope_qwen.csv".into());
    // `... <csv> real` loads the trained qwen3-0.6B safetensors instead of a
    // tiny random config → genuine attention sinks + activation sparsity.
    let real = std::env::args().nth(2).as_deref() == Some("real");
    let mut rng = Philox4x32::new(0x9E37);

    // `... <csv> decode` → inspect the m=1 KV-cache DECODE graph (the one skill
    // actually optimizes) instead of prefill. Structural dataflow / op-cost /
    // roofline / fusion-pattern analysis is what matters; the graph isn't run.
    let want_decode = std::env::args().any(|a| a == "decode");
    let (cfg, graph, params, seq) = if want_decode {
        let cfg = qwen3_config();
        let past = 32usize;
        let mut wm = random_weights(&cfg, &mut rng);
        let (g, p) = rlx_qwen3::build_qwen3_decode_graph_sized(&cfg, &mut wm, 1, past)
            .expect("build qwen3 decode graph");
        (cfg, g, p, 1usize)
    } else if real {
        let dir = Path::new("/Users/Shared/rlx-models/weights/lm/qwen3-0.6b");
        let cfg = Qwen3Config::from_file(&dir.join("config.json")).expect("qwen3 config.json");
        // WeightMap::from_file auto-converts BF16 → f32; pass the FILE not the dir.
        let mut loader = WeightMap::from_file(dir.join("model.safetensors").to_str().unwrap())
            .expect("safetensors");
        let seq = 16usize; // shorter prefill — 28 layers × hidden 1024 on CPU
        let (g, p) = build_qwen3_graph_sized(&cfg, &mut loader, 1, seq, true, false)
            .expect("build qwen3 graph (real)");
        (cfg, g, p, seq)
    } else {
        let cfg = qwen3_config();
        let seq = 32usize;
        let mut wm = random_weights(&cfg, &mut rng);
        let (g, p) =
            build_qwen3_graph_sized(&cfg, &mut wm, 1, seq, true, false).expect("build qwen3 graph");
        (cfg, g, p, seq)
    };
    println!(
        "real rlx-qwen3: {} layers, seq {seq}, hidden {}, {}/{} q/kv heads (GQA) — {} nodes, {} params\n",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        graph.nodes().len(),
        params.len()
    );

    // ── Tier 3 ──
    let costs = op_costs(&graph);
    let (tf, tb): (u64, u64) = (
        costs.iter().map(|c| c.flops).sum(),
        costs.iter().map(|c| c.bytes).sum(),
    );
    let (mut mem, mut comp) = (0u64, 0u64);
    for c in &costs {
        match roofline_class(c, DEFAULT_RIDGE) {
            "memory-bound" => mem += c.flops,
            "compute-bound" => comp += c.flops,
            _ => {}
        }
    }
    println!(
        "[Tier3] {:.3} GFLOP, {:.2} MB, {:.1} FLOP/byte — {:.0}% memory-bound",
        tf as f64 / 1e9,
        tb as f64 / 1e6,
        tf as f64 / tb.max(1) as f64,
        mem as f64 / (mem + comp).max(1) as f64 * 100.0
    );
    print!("[Tier3] hot GEMM shapes: ");
    for ((m, k, n), (ct, _)) in gemm_shape_histogram(&costs).iter().take(5) {
        print!("{m}×{k}×{n}(×{ct}) ");
    }
    println!("\n");

    // ── Structural ──
    println!("[struct] top repeated dataflow blocks (fusion candidates):");
    for p in repeated_flow_patterns(&graph, 3, 5, 2).iter().take(12) {
        println!("   ×{} d{}  {}", p.count, p.depth, p.tree);
    }
    println!();

    // Decode graph inputs (input_ids[1,1] + rope + past_k/v) differ from the
    // prefill-shaped stat-injection run below, so stop after the static dataflow
    // inspect (op-costs / roofline / GEMM shapes / fusion patterns) for decode.
    if want_decode {
        return;
    }

    // ── Inject value + attention sketches, then run ──
    let scfg = StatConfig::default();
    let (g1, mm) = inject_matmul_stats(&graph, &scfg);
    let (g2, inf) = inject_attention_stats(&g1, &scfg);
    // Profile the UNFUSED graph: every matmul stays a tappable site, and stat
    // taps on gate/up matmuls don't break SwiGLU/bias-act fusion rewrites.
    let mut opts = rlx_runtime::CompileOptions::default();
    opts.fusion_opts.skip_fusion = true;
    let mut c = Session::new(Device::Cpu).compile_with(g2, &opts);
    for (name, data) in &params {
        c.set_param(name, data);
    }
    let ids: Vec<f32> = (0..seq)
        .map(|_| (rng.next_u32() % cfg.vocab_size as u32) as f32)
        .collect();
    let outs = c.run(&[("input_ids", &ids)]);

    // ── Tier 1: attention concentration from the fused Op::Attention ──
    println!("[Tier1] attention concentration (decomposed from fused Op::Attention):");
    for s in inf.iter().filter(|s| s.stat == "attn_qmax") {
        let qmax = &outs[s.out_idx];
        let sink = inf
            .iter()
            .find(|x| x.stat == "attn_krecv" && x.site == s.site)
            .map(|x| {
                let kr = &outs[x.out_idx];
                kr.iter().cloned().fold(0.0f32, f32::max) / kr.iter().sum::<f32>().max(1.0)
            })
            .unwrap_or(0.0);
        println!(
            "   {:<10} mean per-query peak {:.2}   hottest key gets {:.0}% of its head's mass",
            s.site,
            mean(qmax),
            sink * 100.0
        );
    }
    println!();

    let mut rec = Recorder::create(&out).expect("csv");
    rec.record(
        0,
        0,
        "cpu",
        "qwen",
        seq,
        cfg.hidden_size,
        cfg.vocab_size,
        &mm,
        &outs,
    )
    .unwrap();
    rec.flush().unwrap();
    println!("[value] wrote {} matmul-site sketches → {out}", mm.len());
    println!("        next (from ../rlx): opscope-mine {out}  |  opscope-plan {out}");
}
