//! `record_forward` — run a REAL single-token (or short-prompt) forward through the
//! WHOLE Kimi-K3 model (all 93 layers, real streamed weights) and record the
//! dataflow with the rlx **opscope** ops-inspector. Unlike `record_dataflow` (which
//! records each layer in isolation on RANDOM input), this chains the ACTUAL
//! activations layer-to-layer — so the per-matmul stats (density / per-channel
//! outliers / histogram / temporal drift) reflect the true forward, and the run
//! ends with the real next-token. Records the attention/pre-FFN + dense-MLP matmuls
//! per layer (the routed MoE experts are MXFP4-paged inside `run_moe_paged`).
//!
//!   cargo run -p rlx-kimi-k3 --features cluster --example record_forward -- \
//!       out.csv [tokens_csv] [n_layers] [model_dir]
//!   (from ../rlx) cargo run -p rlx-opscope --bin opscope-mine -- out.csv

use rlx_core::flow_util::graph_from_hir;
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, HirGraphExt, Shape};
use rlx_kimi_k3::config::{KimiK3Config, KimiLinearConfig};
use rlx_kimi_k3::flow::{FfnWeights, FlowConfig, build_layer_pre_ffn};
use rlx_kimi_k3::kda::KdaDims;
use rlx_kimi_k3::mla::MlaDims;
use rlx_kimi_k3::moe::{MoeDims, build_dense_mlp};
use rlx_kimi_k3::runner::{apply_head, load_layer_backbone, run_moe_paged};
use rlx_opscope::{Recorder, StatConfig, inject_matmul_stats};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::Path;

const EMB: &str = "language_model.model.embed_tokens.weight";

fn kimi_flow_cfg(tc: &KimiLinearConfig, seq: usize) -> FlowConfig {
    let hidden = tc.hidden_size;
    FlowConfig {
        hidden,
        vocab: tc.vocab_size,
        attn_res_block_size: tc.attn_res_block_size.unwrap_or(12),
        eps: 1e-5,
        kda: KdaDims {
            hidden,
            num_heads: 96,
            head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound: Some(-5.0),
            eps: 1e-5,
            batch: 1,
            seq,
        },
        mla: MlaDims {
            hidden,
            num_heads: 96,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            eps: 1e-5,
            batch: 1,
            seq,
        },
        moe: MoeDims {
            hidden,
            latent: 3584,
            moe_inter: 3072,
            num_experts: 896,
            top_k: 16,
            num_shared: 2,
            routed_scaling: 2.5,
            eps: 1e-5,
            situ_beta: 4.0,
            situ_linear_beta: Some(25.0),
            batch: 1,
            seq,
        },
        dense_inter: 33792,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq,
    }
}

fn main() -> Result<(), String> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "kimi_forward.csv".into());
    let tokens: Vec<u32> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "1".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let model_dir = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "/Volumes/FOUR/kimi".into());
    if !Path::new(&model_dir).join("config.json").exists() {
        eprintln!("skip: {model_dir}/config.json not found");
        return Ok(());
    }
    let kc =
        KimiK3Config::load(Path::new(&model_dir).join("config.json")).map_err(|e| e.to_string())?;
    let tc = &kc.text_config;
    let n_layers: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(tc.num_hidden_layers);
    let (hidden, vocab, seq) = (tc.hidden_size, tc.vocab_size, tokens.len().max(1));
    let cfg = kimi_flow_cfg(tc, seq);
    let dev = Device::Cpu;

    let mut ck =
        rlx_kimi_k3::loader::CheckpointLoader::open(&model_dir).map_err(|e| e.to_string())?;
    let mut h = ck
        .gather_embed(EMB, &tokens, hidden)
        .map_err(|e| e.to_string())?;
    let mut snaps: Vec<Vec<f32>> = Vec::new();
    let mut rec = Recorder::create(&out).map_err(|e| e.to_string())?;
    let stat = StatConfig::default();
    eprintln!(
        "[opscope] REAL forward: {seq}-token prompt {tokens:?} through {n_layers}/{} layers -> {out}",
        tc.num_hidden_layers
    );

    for i in 0..n_layers {
        let t0 = std::time::Instant::now();
        let lp = format!("language_model.model.layers.{i}");
        let is_moe = tc.is_moe_layer(i);
        let lw = load_layer_backbone(&mut ck, tc, &cfg, i).map_err(|e| e.to_string())?;

        // build this layer's HIR (pre-FFN attention + dense-MLP, mirroring the runner).
        let mut hir = HirModule::new("layer");
        let mut g = HirMut::new(&mut hir);
        let h_node = g.input("h", Shape::new(&[1, seq, hidden], DType::F32));
        let snap_nodes: Vec<_> = (0..snaps.len())
            .map(|j| {
                g.input(
                    format!("snap_{j}"),
                    Shape::new(&[1, seq, hidden], DType::F32),
                )
            })
            .collect();
        let mut params: HashMap<String, Vec<f32>> = HashMap::new();
        let (mn, stream, snaps_out) =
            build_layer_pre_ffn(&mut g, &mut params, i, h_node, snap_nodes, &lw, &cfg)
                .map_err(|e| e.to_string())?;
        let mut outputs = Vec::new();
        if is_moe {
            outputs.push(mn);
            outputs.push(stream);
        } else {
            let FfnWeights::Dense(dw) = &lw.ffn else {
                unreachable!()
            };
            let ffn = build_dense_mlp(
                &mut g,
                &mut params,
                &format!("l{i}.mlp"),
                mn,
                dw,
                hidden,
                cfg.dense_inter,
                1,
                seq,
                cfg.situ_beta,
                cfg.situ_linear_beta,
            )
            .map_err(|e| e.to_string())?;
            outputs.push(g.add(stream, ffn));
        }
        let n_real = outputs.len() + snaps_out.len();
        outputs.extend(snaps_out);
        g.set_outputs(outputs);

        // opscope: convert → inject matmul stat-taps → compile → run REAL activations.
        let (graph, pvec) = graph_from_hir(hir, params).map_err(|e| e.to_string())?;
        let (ginj, specs) = inject_matmul_stats(&graph, &stat);
        let mut compiled = Session::new(dev).compile(ginj);
        for (name, data) in &pvec {
            compiled.set_param(name, data);
        }
        let snap_names: Vec<String> = (0..snaps.len()).map(|j| format!("snap_{j}")).collect();
        let mut inputs: Vec<(&str, &[f32])> = vec![("h", h.as_slice())];
        for (j, sn) in snaps.iter().enumerate() {
            inputs.push((snap_names[j].as_str(), sn.as_slice()));
        }
        let outs = compiled.run(&inputs);
        rec.record(
            i as u64,
            0,
            "cpu",
            &format!("L{i:03}"),
            1,
            hidden,
            0,
            &specs,
            &outs,
        )
        .map_err(|e| e.to_string())?;

        // continue the REAL forward from the (non-tap) outputs.
        let real = &outs[..n_real];
        if is_moe {
            let mn_v = &real[0];
            let stream_v = &real[1];
            snaps = real[2..].to_vec();
            let moe = run_moe_paged(&mut ck, &lp, mn_v, cfg.moe, dev).map_err(|e| e.to_string())?;
            h = stream_v.iter().zip(&moe).map(|(a, b)| a + b).collect();
        } else {
            h = real[0].clone();
            snaps = real[1..].to_vec();
        }
        eprintln!(
            "  L{i:>3} {} recorded ({:.1}s)",
            if is_moe { "MoE " } else { "dense" },
            t0.elapsed().as_secs_f64()
        );
    }
    rec.flush().map_err(|e| e.to_string())?;

    if n_layers == tc.num_hidden_layers {
        let logits = apply_head(&mut ck, &cfg, &h, &snaps, dev).map_err(|e| e.to_string())?;
        let last = &logits[(seq - 1) * vocab..seq * vocab];
        let tok = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        eprintln!(
            "\nreal next-token (greedy) = {tok}  (logit {:.3}), finite={}",
            last[tok],
            logits.iter().all(|v| v.is_finite())
        );
    }
    eprintln!(
        "[opscope] done -> {out}  (mine: cargo run -p rlx-opscope --bin opscope-mine -- {out})"
    );
    Ok(())
}
