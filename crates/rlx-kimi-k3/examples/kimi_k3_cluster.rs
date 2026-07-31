// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! `kimi_k3_cluster` — feature-flagged (`--features cluster`) multi-node runner
//! for **Kimi-K3** across this Mac + `ssh msi` + `ssh amd`.
//!
//! Kimi-K3 is 2.8T params / 1.56 TB on disk, but only ~8% of the bytes (a 114 GB
//! BF16 backbone: KDA/MLA attention, router, shared experts, embed, norms) is
//! dense and needed every token; the other 92% (1.45 TB of MXFP4 routed experts)
//! is sparse — 16 of 896 per layer per token — so it is modelled as **disk-paged**
//! rather than resident. This example does the two pieces that are runnable today:
//!
//!   plan  --config <cluster.toml> --model-dir <dir>
//!       Scan the real checkpoint (config.json + safetensors headers), split the
//!       114 GB resident backbone across the cluster's nodes (RAM-balanced, via
//!       the shared `rlx-distributed` placement planner), and report the per-node
//!       layer assignment + resident bytes + the per-token expert-paging volume
//!       and an NVMe-bound decode estimate.
//!
//!   local [--device cpu|metal|mlx|gpu|coreml] [--layers N] [--seq N]
//!       Build a small synthetic KimiLinear text flow (hybrid KDA/MLA + LatentMoE
//!       + Attention-Residuals + lm_head) and run one forward on the chosen
//!       backend — proof the model graph compiles & runs end to end.
//!
//!   pipeline [--device …] [--layers N] [--seq N] [--stages N]
//!       Split that flow into block-aligned stages, compile+run each as its own
//!       graph while threading the hidden state AND the AttnRes snapshots across
//!       the boundaries, and assert the result matches the single-graph forward.
//!       This validates the `build_kimi_text_stage` decomposition — the in-graph
//!       analogue of the distributed `build_kimi_k3_stage` seam.
//!
//! The full streaming/paged distributed forward still needs the weight side of
//! `build_kimi_k3_stage` (a `WeightLoader`-backed stage) + MXFP4 expert paging
//! (see the `kimi-k3-cluster-plan` memory). The stage *math* + snapshot carry is
//! done and verified here; what remains is streaming real weights into it.

use anyhow::{Context, Result, bail};
use rlx_distributed::cluster::{ClusterConfig, ModelCost, NodeCaps, NodeConfig, plan_placement};
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("plan") => cmd_plan(&args[2..]),
        Some("design") => cmd_design(&args[2..]),
        Some("local") => cmd_local(&args[2..]),
        Some("pipeline") => cmd_pipeline(&args[2..]),
        Some("spec") => cmd_spec(&args[2..]),
        _ => {
            eprintln!(
                "usage:\n  kimi_k3_cluster plan     --config <cluster.toml> --model-dir <dir>\n  \
                 kimi_k3_cluster design   --config <cluster.toml> --model-dir <dir>\n  \
                 kimi_k3_cluster local    [--device cpu|metal|mlx|gpu|coreml] [--layers N] [--seq N]\n  \
                 kimi_k3_cluster pipeline [--device …] [--layers N] [--seq N] [--stages N]\n  \
                 kimi_k3_cluster spec     [--device …] [--layers N] [--draft-layers N] [--k N] [--gen N]"
            );
            std::process::exit(2);
        }
    }
}

/// `--flag value` lookup in an argv slice.
fn opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

// ─────────────────────────────────────────────────────────────────────────────
// plan — real-checkpoint placement across the cluster
// ─────────────────────────────────────────────────────────────────────────────

/// Resident-vs-paged byte split of a sharded safetensors checkpoint.
struct Sizes {
    backbone: u64, // non-expert tensors (attn/router/shared/embed/head/norms)
    experts: u64,  // routed-expert tensors (the paged part)
    embed: u64,
    head: u64,
}

/// Read a safetensors file's JSON header (the leading u64 length + that many
/// bytes) without mapping the tensor data.
fn read_st_header(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)?;
    let v: serde_json::Value = serde_json::from_slice(&buf)?;
    Ok(v.as_object().cloned().unwrap_or_default())
}

/// Sum exact tensor bytes (via `data_offsets`) across every shard, split into the
/// resident backbone and the paged routed experts.
fn scan_checkpoint(model_dir: &Path) -> Result<Sizes> {
    let idx_path = model_dir.join("model.safetensors.index.json");
    let idx: serde_json::Value = serde_json::from_reader(
        File::open(&idx_path).with_context(|| format!("open {}", idx_path.display()))?,
    )?;
    let wm = idx
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("index has no weight_map")?;
    // unique shard files
    let mut shards: Vec<String> = wm
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shards.sort();
    shards.dedup();

    let (mut backbone, mut experts, mut embed, mut head) = (0u64, 0u64, 0u64, 0u64);
    for shard in &shards {
        let hdr = read_st_header(&model_dir.join(shard))?;
        for (name, meta) in &hdr {
            if name == "__metadata__" {
                continue;
            }
            let off = meta.get("data_offsets").and_then(|v| v.as_array());
            let bytes = match off {
                Some(a) if a.len() == 2 => a[1]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_sub(a[0].as_u64().unwrap_or(0)),
                _ => 0,
            };
            if name.contains(".experts.") {
                experts += bytes;
            } else {
                backbone += bytes;
                if name.contains("embed_tokens") {
                    embed += bytes;
                } else if name.contains("lm_head") {
                    head += bytes;
                }
            }
        }
    }
    Ok(Sizes {
        backbone,
        experts,
        embed,
        head,
    })
}

/// Fabricate a `NodeCaps` from a cluster-TOML node (RAM from `max_ram_gb`, no GPU
/// mem ceiling → RAM-bounded, which is what the CPU/Metal nodes want). Avoids
/// needing a probe binary deployed to the remotes just to *plan*.
fn caps_for(n: &NodeConfig) -> NodeCaps {
    let ram = (n.max_ram_gb.unwrap_or(8.0) * 1e9) as u64;
    serde_json::from_value(serde_json::json!({
        "addr": n.addr,
        "os": "linux",
        "cores": 8usize,
        "ram_total": ram,
        "ram_avail": ram,
        "disk_free": 0u64,
        "devices": [],
        "gflops": 100.0,
        "io_mbps": 0.0,
    }))
    .expect("NodeCaps json")
}

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

fn cmd_plan(args: &[String]) -> Result<()> {
    let config = opt(args, "--config").context("--config <cluster.toml> required")?;
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let model_dir = Path::new(model_dir);

    let cfg = ClusterConfig::from_path(config)?;
    let kc = rlx_kimi_k3::config::KimiK3Config::load(model_dir.join("config.json"))?;
    let t = &kc.text_config;

    eprintln!("scanning checkpoint shards for the backbone/expert byte split…");
    let sz = scan_checkpoint(model_dir)?;
    let n_layers = t.num_hidden_layers;
    let n_moe = (0..n_layers).filter(|&i| t.is_moe_layer(i)).count();
    let num_experts = t.num_experts.unwrap_or(0);

    // Resident model = backbone only; experts are paged (not counted per-layer).
    let per_layer = (sz.backbone.saturating_sub(sz.embed + sz.head)) / n_layers.max(1) as u64;
    let model = ModelCost {
        n_layers,
        per_layer_bytes: per_layer,
        embed_bytes: sz.embed,
        head_bytes: sz.head,
        per_layer_flops: 1.0,
    };

    let nodes: Vec<(NodeCaps, NodeConfig)> =
        cfg.nodes.iter().map(|n| (caps_for(n), n.clone())).collect();
    let reserve = (cfg.reserve_ram_gb * 1e9) as u64;

    println!("\nKimi-K3 checkpoint  {}", model_dir.display());
    println!(
        "  {n_layers} layers ({} MoE), hidden {}, vocab {}, {num_experts} experts / {} active + {} shared",
        n_moe, t.hidden_size, t.vocab_size, t.num_experts_per_token, t.num_shared_experts
    );
    println!(
        "  on-disk {:.1} GB = backbone {:.1} GB (RESIDENT) + experts {:.1} GB (PAGED, {:.1}%)",
        gb(sz.backbone + sz.experts),
        gb(sz.backbone),
        gb(sz.experts),
        sz.experts as f64 / (sz.backbone + sz.experts) as f64 * 100.0,
    );
    println!(
        "  resident per-layer backbone {:.2} GB, embed {:.2} GB, head {:.2} GB",
        gb(per_layer),
        gb(sz.embed),
        gb(sz.head)
    );

    match plan_placement(&model, &nodes, cfg.placement.policy, reserve) {
        Ok(plan) => {
            println!(
                "\n  placement (policy {:?}, reserve {:.0} GB/node):",
                cfg.placement.policy, cfg.reserve_ram_gb
            );
            println!(
                "    {:<22} {:>10} {:>8} {:>10} {:>8}",
                "node", "layers", "count", "resident", "budget"
            );
            for a in &plan {
                let tag = if a.first && a.last {
                    "single"
                } else if a.first {
                    "first(embed)"
                } else if a.last {
                    "last(head)"
                } else {
                    "middle"
                };
                println!(
                    "    {:<22} {:>10} {:>8} {:>8.1}G {:>7.0}G  [{tag}]",
                    format!(
                        "{}{}",
                        a.addr,
                        a.ssh
                            .as_deref()
                            .map(|s| format!(" ({s})"))
                            .unwrap_or_default()
                    ),
                    format!("{}..{}", a.layers.start, a.layers.end),
                    a.layers.len(),
                    gb(a.est_bytes),
                    gb(a.budget_bytes),
                );
            }
        }
        Err(e) => {
            println!("\n  ⚠ backbone does not fit even as resident-only: {e}");
        }
    }

    // Per-token expert paging (the decode bottleneck) — and how topology changes it.
    if n_moe > 0 && num_experts > 0 {
        let n_nodes = cfg.nodes.len().max(1);
        let per_expert = sz.experts / (num_experts as u64 * n_moe as u64).max(1);
        let active = (t.num_experts_per_token as u64) * n_moe as u64;
        let per_tok = active * per_expert;
        let node_bw = 5.0e9; // ~5 GB/s NVMe per node
        let agg_bw = node_bw * n_nodes as f64;
        // Free RAM across the cluster after the resident backbone → expert cache.
        let sum_ram: f64 = cfg.nodes.iter().filter_map(|n| n.max_ram_gb).sum::<f64>() * 1e9;
        let cache_bytes = (sum_ram - sz.backbone as f64).max(0.0);
        let cache_frac = (cache_bytes / sz.experts as f64).clamp(0.0, 1.0);

        println!(
            "\n  expert paging (decode) — {:.1} GB read/token",
            gb(per_tok)
        );
        println!(
            "    {} active experts/token ({} MoE layers × {} top-k), ~{:.1} MB each",
            active,
            n_moe,
            t.num_experts_per_token,
            per_expert as f64 / 1e6
        );
        // Pipeline serializes the nodes for a single stream (stage 0→1→2), so the
        // NVMes never read at once → effective bandwidth is ONE node's.
        println!(
            "    pipeline, single stream : ÷ {:.0} GB/s (nodes serialized) ≈ {:.1} s/token",
            node_bw / 1e9,
            per_tok as f64 / node_bw
        );
        // Expert-parallel (or a pipelined micro-batch): all NVMes read concurrently.
        println!(
            "    expert-parallel / batched: ÷ {:.0} GB/s ({} NVMes concurrent) ≈ {:.1} s/token  ({}× faster)",
            agg_bw / 1e9,
            n_nodes,
            per_tok as f64 / agg_bw,
            n_nodes
        );
        println!(
            "    + RAM expert cache: only ~{:.0} GB free after the {:.0} GB backbone ⇒ ~{:.1}% of experts cacheable (router skew raises the hit rate)",
            cache_bytes / 1e9,
            gb(sz.backbone),
            cache_frac * 100.0
        );
        println!(
            "    prefill is compute-bound & amortizes (each needed expert read once for the whole prompt) → GPUs help there"
        );
    }
    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// design — the max-parallelism execution architecture (target for the runner)
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_design(args: &[String]) -> Result<()> {
    let config = opt(args, "--config").context("--config <cluster.toml> required")?;
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let model_dir = Path::new(model_dir);
    let cfg = ClusterConfig::from_path(config)?;
    let kc = rlx_kimi_k3::config::KimiK3Config::load(model_dir.join("config.json"))?;
    let t = &kc.text_config;
    eprintln!("scanning checkpoint shards…");
    let sz = scan_checkpoint(model_dir)?;

    let n_nodes = cfg.nodes.len().max(1);
    let n_layers = t.num_hidden_layers;
    let n_moe = (0..n_layers).filter(|&i| t.is_moe_layer(i)).count();
    let num_experts = t.num_experts.unwrap_or(0);

    // Backbone: layer-sharded, RESIDENT (needed every token — can't page it).
    let per_layer = (sz.backbone.saturating_sub(sz.embed + sz.head)) / n_layers.max(1) as u64;
    let model = ModelCost {
        n_layers,
        per_layer_bytes: per_layer,
        embed_bytes: sz.embed,
        head_bytes: sz.head,
        per_layer_flops: 1.0,
    };
    let nodes: Vec<(NodeCaps, NodeConfig)> =
        cfg.nodes.iter().map(|n| (caps_for(n), n.clone())).collect();
    let reserve = (cfg.reserve_ram_gb * 1e9) as u64;
    let plan = plan_placement(&model, &nodes, cfg.placement.policy, reserve).ok();

    // Experts: expert-sharded across the nodes' DISKS (by expert id, all layers).
    let expert_shard = sz.experts / n_nodes as u64;
    let per_expert = sz.experts / (num_experts as u64 * n_moe as u64).max(1);
    let per_tok = (t.num_experts_per_token as u64) * n_moe as u64 * per_expert;
    let node_bw = 5.0e9;
    let agg_bw = node_bw * n_nodes as f64;
    let cache_gb = ((cfg.nodes.iter().filter_map(|n| n.max_ram_gb).sum::<f64>() * 1e9)
        - sz.backbone as f64)
        .max(0.0)
        / 1e9;

    println!("\nKimi-K3 MAX-PARALLELISM design — {n_nodes} nodes");
    println!(
        "  backbone {:.0} GB → RESIDENT, layer-sharded   ·   experts {:.0} GB → DISK, expert-sharded (~{:.0} GB/node)",
        gb(sz.backbone),
        gb(sz.experts),
        gb(expert_shard)
    );

    println!("\n  per-node roles (backbone-owner ⟂ expert-owner — decoupled):");
    for (i, (_, n)) in nodes.iter().enumerate() {
        let range = plan
            .as_ref()
            .and_then(|p| p.get(i))
            .map(|a| format!("L{}..{}", a.layers.start, a.layers.end))
            .unwrap_or_else(|| "—".into());
        let host = format!(
            "{}{}",
            n.addr,
            n.ssh
                .as_deref()
                .map(|s| format!(" ({s})"))
                .unwrap_or_default()
        );
        println!(
            "    {:<22} backbone {:<9} | expert shard {i}/{n_nodes} (~{:.0} GB disk) | device {}",
            host,
            range,
            gb(expert_shard),
            n.device
        );
    }

    println!(
        "\n  per-MoE-layer dataflow (decode): router → all-to-all DISPATCH the {} active tokens to\n  \
         their expert-OWNER nodes → each owner reads+MXFP4-dequants+matmuls its expert from LOCAL NVMe\n  \
         (all {n_nodes} disks at once) → all-to-all COMBINE. Expert WEIGHTS never cross the net.",
        t.num_experts_per_token
    );

    println!("\n  parallelism stack (decode), highest-leverage first:");
    println!(
        "    1 expert-parallel      {} reads/token over {n_nodes} NVMes AT ONCE → {:.1} GB ÷ {:.0} GB/s\n      \
         (vs ÷{:.0} GB/s serialized in a layer-pipeline = {n_nodes}× worse)",
        t.num_experts_per_token * n_moe,
        gb(per_tok),
        agg_bw / 1e9,
        node_bw / 1e9
    );
    println!(
        "    2 intra-node overlap   disk-read ∥ dequant ∥ matmul ∥ net, double-buffered → cheap compute\n      \
         + network hide under dominant I/O; GPU runs the dense backbone + expert matmul"
    );
    println!(
        "    3 speculative (DSpark) draft K, verify in ONE target pass → K correlated tokens reuse the\n      \
         SAME expert reads (fewer bytes/token) and fill pipeline bubbles  [see `spec` mode]"
    );
    println!(
        "    4 RAM expert cache     ~{:.0} GB LRU across nodes + skewed routing → hot experts stay resident",
        cache_gb
    );
    println!(
        "    ⇒ decode FLOOR ≈ {:.1} s/token (aggregate-NVMe bound); layers 3–4 push below it.",
        per_tok as f64 / agg_bw
    );
    println!(
        "\n  prefill: token-parallel — batch the whole prompt, each expert read ONCE and applied to every\n  \
         token routing to it, GPU matmul → compute-bound, fast TTFT (not disk-bound like decode)."
    );
    println!(
        "\n  to realize (beyond the wired layer-pipeline): an EXPERT-PARALLEL runner — per-layer all-to-all\n  \
         dispatch/combine + a disk-resident expert ParamSource keyed by (layer, expert). Backbone reuses\n  \
         the layer-shard + streaming loader already in rlx-distributed."
    );
    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// local — reduced synthetic forward, proves the graph runs on a backend
// ─────────────────────────────────────────────────────────────────────────────

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::flow::{
    AttnWeights, FfnWeights, FlowConfig, FlowWeights, LayerWeights, build_kimi_text_flow,
    build_kimi_text_stage,
};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights};
use rlx_kimi_k3::mla::{MlaDims, MlaWeights};
use rlx_kimi_k3::moe::{DenseMlpWeights, MoeDims, MoeWeights};
use rlx_runtime::{CompiledGraph, Device};
use std::collections::HashMap;
use std::time::Instant;

fn parse_device(s: &str) -> Device {
    match s {
        "metal" | "mtl" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "coreml" | "ane" => Device::Ane,
        "cuda" => Device::Cuda,
        "vulkan" | "vk" => Device::Vulkan,
        _ => Device::Cpu,
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.15
        })
        .collect()
}

fn kda_w(d: KdaDims, sd: u64) -> KdaWeights {
    let (hidden, h, hd, proj, k) = (d.hidden, d.num_heads, d.head_dim, d.proj(), d.conv_kernel);
    KdaWeights {
        q_proj: fill(hidden * proj, sd + 1),
        k_proj: fill(hidden * proj, sd + 2),
        v_proj: fill(hidden * proj, sd + 3),
        q_conv: fill(proj * k, sd + 4),
        k_conv: fill(proj * k, sd + 5),
        v_conv: fill(proj * k, sd + 6),
        f_a: fill(hidden * hd, sd + 7),
        f_b: fill(hd * proj, sd + 8),
        dt_bias: fill(proj, sd + 9),
        a_log: fill(hd, sd + 10),
        b_proj: fill(hidden * h, sd + 11),
        g_proj: fill(hidden * proj, sd + 12),
        o_norm: vec![1.0; hd],
        o_proj: fill(proj * hidden, sd + 13),
    }
}

fn mla_w(d: MlaDims, sd: u64) -> MlaWeights {
    let (hidden, h, ql, kvl, nope, rope, vd, qk) = (
        d.hidden,
        d.num_heads,
        d.q_lora_rank,
        d.kv_lora_rank,
        d.qk_nope_head_dim,
        d.qk_rope_head_dim,
        d.v_head_dim,
        d.qk(),
    );
    MlaWeights {
        q_a_proj: fill(hidden * ql, sd + 1),
        q_a_layernorm: vec![1.0; ql],
        q_b_proj: fill(ql * h * qk, sd + 2),
        kv_a_proj_with_mqa: fill(hidden * (kvl + rope), sd + 3),
        kv_a_layernorm: vec![1.0; kvl],
        kv_b_proj: fill(kvl * h * (nope + vd), sd + 4),
        g_proj: fill(hidden * h * vd, sd + 5),
        o_proj: fill(h * vd * hidden, sd + 6),
    }
}

fn moe_w(d: MoeDims, sd: u64) -> MoeWeights {
    let (hidden, l, mi, e, si) = (
        d.hidden,
        d.latent,
        d.moe_inter,
        d.num_experts,
        d.num_shared * d.moe_inter,
    );
    MoeWeights {
        router: fill(hidden * e, sd + 1),
        e_score_bias: fill(e, sd + 2),
        down_latent: fill(hidden * l, sd + 3),
        up_latent: fill(l * hidden, sd + 4),
        routed_norm: vec![1.0; l],
        experts_gate_up: fill(e * l * 2 * mi, sd + 5),
        experts_down: fill(e * mi * l, sd + 6),
        shared_gate: fill(hidden * si, sd + 7),
        shared_up: fill(hidden * si, sd + 8),
        shared_down: fill(si * hidden, sd + 9),
    }
}

fn layer(hidden: usize, attn: AttnWeights, ffn: FfnWeights, sd: u64) -> LayerWeights {
    LayerWeights {
        input_ln: vec![1.0; hidden],
        post_ln: vec![1.0; hidden],
        sa_res_norm: vec![1.0; hidden],
        sa_res_proj: fill(hidden, sd + 1),
        mlp_res_norm: vec![1.0; hidden],
        mlp_res_proj: fill(hidden, sd + 2),
        attn,
        ffn,
    }
}

/// A small synthetic KimiLinear text flow that mirrors the real topology
/// (hybrid KDA/MLA + LatentMoE + dense L0 + AttnRes block 2) at toy dims.
fn synth_flow(n_layers: usize, seq: usize) -> (FlowWeights, FlowConfig) {
    let (batch, hidden, vocab) = (1usize, 16usize, 20usize);
    let kda = KdaDims {
        hidden,
        num_heads: 2,
        head_dim: 8,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch,
        seq,
    };
    let mla = MlaDims {
        hidden,
        num_heads: 2,
        q_lora_rank: 8,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 2,
        v_head_dim: 4,
        eps: 1e-5,
        batch,
        seq,
    };
    let moe = MoeDims {
        hidden,
        latent: 12,
        moe_inter: 8,
        num_experts: 4,
        top_k: 2,
        num_shared: 1,
        routed_scaling: 1.0,
        eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch,
        seq,
    };
    let dense_inter = 24usize;

    // L0 KDA+dense (first_k_dense_replace=1); MoE elsewhere, MLA every 3rd layer.
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let attn = if i % 3 == 2 {
            AttnWeights::Mla(Box::new(mla_w(mla, 300 + i as u64 * 10)))
        } else {
            AttnWeights::Kda(Box::new(kda_w(kda, 100 + i as u64 * 10)))
        };
        let ffn = if i == 0 {
            FfnWeights::Dense(Box::new(DenseMlpWeights {
                gate: fill(hidden * dense_inter, 900),
                up: fill(hidden * dense_inter, 901),
                down: fill(dense_inter * hidden, 902),
            }))
        } else {
            FfnWeights::Moe(Box::new(moe_w(moe, 200 + i as u64 * 10)))
        };
        layers.push(layer(hidden, attn, ffn, 10 + i as u64));
    }

    let w = FlowWeights {
        layers,
        final_norm: vec![1.0; hidden],
        out_res_norm: vec![1.0; hidden],
        out_res_proj: fill(hidden, 800),
        lm_head: fill(hidden * vocab, 801),
    };
    let cfg = FlowConfig {
        hidden,
        vocab,
        attn_res_block_size: 2,
        eps: 1e-5,
        kda,
        mla,
        moe,
        dense_inter,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch,
        seq,
    };
    (w, cfg)
}

/// Split `n_layers` into `n_stages` contiguous ranges aligned to `block` (AttnRes
/// block size) so a pipeline boundary never lands mid-block. Returns each stage's
/// `[start, end)` layer range.
fn stage_ranges(n_layers: usize, block: usize, n_stages: usize) -> Vec<(usize, usize)> {
    let n_blocks = n_layers.div_ceil(block);
    let n_stages = n_stages.clamp(1, n_blocks.max(1));
    let mut out = Vec::with_capacity(n_stages);
    for si in 0..n_stages {
        let b0 = si * n_blocks / n_stages;
        let b1 = (si + 1) * n_blocks / n_stages;
        let start = (b0 * block).min(n_layers);
        let end = (b1 * block).min(n_layers);
        if start < end {
            out.push((start, end));
        }
    }
    out
}

/// `pipeline` — split the flow into block-aligned stages, run them in sequence
/// (each its own compiled graph), threading the hidden state + AttnRes snapshots
/// across the boundaries, and assert the final logits match the single-graph
/// forward. This validates the `build_kimi_text_stage` seam that a real
/// distributed `build_kimi_k3_stage` is built on — snapshot carry and all —
/// before any streaming / MXFP4 / network.
fn cmd_pipeline(args: &[String]) -> Result<()> {
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let seq: usize = opt(args, "--seq").and_then(|s| s.parse().ok()).unwrap_or(4);
    let n_stages: usize = opt(args, "--stages")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let (w, cfg) = synth_flow(n_layers, seq);
    let (batch, hidden, vocab) = (cfg.batch, cfg.hidden, cfg.vocab);
    let hin = fill(batch * seq * hidden, 7);

    // ── reference: whole model as one graph ──
    let mut hir = HirModule::new("kimi_ref");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[batch, seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let logits = build_kimi_text_flow(&mut g, &mut params, h_in, &w, &cfg).context("ref build")?;
    g.set_outputs(vec![logits]);
    let built = built_from_hir(hir, params).context("ref built")?;
    let mut compiled = compile_built(built, device).context("ref compile")?;
    let reference = compiled.run(&[("h", hin.as_slice())]).remove(0);

    // ── staged: one compiled graph per block-aligned stage ──
    let stages = stage_ranges(n_layers, cfg.attn_res_block_size, n_stages);
    let mut carry_hidden = hin.clone();
    let mut carry_snaps: Vec<Vec<f32>> = Vec::new();
    let mut staged_logits: Vec<f32> = Vec::new();
    let mut layout: Vec<String> = Vec::new();

    for (si, &(start, end)) in stages.iter().enumerate() {
        let last = si == stages.len() - 1;
        let mut hir = HirModule::new("kimi_stage");
        let mut g = HirMut::new(&mut hir);
        let hidden_in = g.input("hidden_in", Shape::new(&[batch, seq, hidden], DType::F32));
        let snaps_in: Vec<_> = (0..carry_snaps.len())
            .map(|j| {
                g.input(
                    &format!("snap_{j}"),
                    Shape::new(&[batch, seq, hidden], DType::F32),
                )
            })
            .collect();
        let mut params = HashMap::new();
        let (out, snaps_out) = build_kimi_text_stage(
            &mut g,
            &mut params,
            hidden_in,
            snaps_in,
            &w.layers[start..end],
            start,
            last,
            &w,
            &cfg,
        )
        .context("stage build")?;
        let mut outputs = vec![out];
        outputs.extend(snaps_out.iter().copied());
        g.set_outputs(outputs);
        let built = built_from_hir(hir, params).context("stage built")?;
        let mut compiled = compile_built(built, device).context("stage compile")?;

        // bind hidden_in + carried snapshots by name
        let snap_names: Vec<String> = (0..carry_snaps.len())
            .map(|j| format!("snap_{j}"))
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = vec![("hidden_in", carry_hidden.as_slice())];
        for (j, sn) in carry_snaps.iter().enumerate() {
            inputs.push((snap_names[j].as_str(), sn.as_slice()));
        }
        let mut results = compiled.run(&inputs);

        layout.push(format!(
            "s{si}:L{start}..{end}({})",
            if last { "head" } else { "hidden" }
        ));
        if last {
            staged_logits = results.remove(0);
        } else {
            carry_hidden = results.remove(0);
            carry_snaps = results; // remaining outputs are the snapshots, in order
        }
    }

    // ── parity ──
    let max_abs = reference
        .iter()
        .zip(&staged_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "Kimi-K3 pipeline on {device:?}: {n_layers} layers → {} stages [{}]",
        stages.len(),
        layout.join(" ")
    );
    println!("  logits [{seq},{vocab}] vs single-graph reference: max_abs={max_abs:.3e}");
    if reference.len() != staged_logits.len() {
        bail!(
            "length mismatch {} vs {}",
            reference.len(),
            staged_logits.len()
        );
    }
    // CPU is bit-exact; GPU backends carry tiny fp reassociation noise.
    let tol = if matches!(device, Device::Cpu) {
        1e-6
    } else {
        2e-3
    };
    if max_abs > tol {
        bail!("pipeline parity failed: max_abs={max_abs:.3e} > {tol:.0e}");
    }
    println!("  ✓ pipeline matches single-graph forward (tol {tol:.0e})");
    Ok(())
}

fn cmd_local(args: &[String]) -> Result<()> {
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let seq: usize = opt(args, "--seq").and_then(|s| s.parse().ok()).unwrap_or(4);
    let (w, cfg) = synth_flow(n_layers, seq);
    let (batch, hidden, vocab) = (cfg.batch, cfg.hidden, cfg.vocab);

    let mut hir = HirModule::new("kimi_k3_local");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[batch, seq, hidden], DType::F32));
    let mut params = HashMap::new();
    let logits = build_kimi_text_flow(&mut g, &mut params, h_in, &w, &cfg).context("build flow")?;
    g.set_outputs(vec![logits]);

    let built = built_from_hir(hir, params).context("build model")?;
    let mut compiled =
        compile_built(built, device).with_context(|| format!("compile on {device:?}"))?;

    let hin = fill(batch * seq * hidden, 7);
    let y = compiled
        .run(&[("h", hin.as_slice())])
        .into_iter()
        .next()
        .context("no output")?;
    if y.len() != batch * seq * vocab {
        bail!("logits len {} != {}", y.len(), batch * seq * vocab);
    }
    if !y.iter().all(|v| v.is_finite()) {
        bail!("non-finite logits");
    }
    // argmax of the last position
    let last = &y[(seq - 1) * vocab..seq * vocab];
    let (tok, _) =
        last.iter().enumerate().fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        );
    println!(
        "Kimi-K3 local forward OK on {device:?}: {n_layers} layers, seq {seq}, hidden {hidden} → logits [{seq},{vocab}] finite, argmax(last)={tok}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// spec — speculative decoding (cheap draft + one-pass target verify).
// Exact for greedy: the output is token-identical to plain greedy decode; the
// draft only changes SPEED (how many tokens each target pass yields). This is the
// parallelism-stack layer 3 (DSpark-style) that fills pipeline bubbles and reuses
// expert reads across the K drafted tokens.
// ─────────────────────────────────────────────────────────────────────────────

fn embed_tokens(tokens: &[u32], embed: &[f32], hidden: usize) -> Vec<f32> {
    let mut h = Vec::with_capacity(tokens.len() * hidden);
    for &tk in tokens {
        let o = tk as usize * hidden;
        h.extend_from_slice(&embed[o..o + hidden]);
    }
    h
}

fn argmax_row(logits: &[f32], pos: usize, vocab: usize) -> u32 {
    logits[pos * vocab..(pos + 1) * vocab]
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            },
        )
        .0 as u32
}

/// Forward flow `w` over `tokens` (embedded via `embed`); returns per-position
/// logits `[len, vocab]`. Rebuilt for this exact length (the flow is seq-shaped)
/// — fine for the tiny demo model. The layer count comes from `w`, so the same
/// helper runs both the deep target and the shallow draft.
fn run_flow(w: &FlowWeights, embed: &[f32], tokens: &[u32], device: Device) -> Result<Vec<f32>> {
    let len = tokens.len();
    let (_, cfg) = synth_flow(1, len); // cfg carries seq=len; layer count comes from `w`
    let hidden = cfg.hidden;
    let mut hir = HirModule::new("spec_flow");
    let mut g = HirMut::new(&mut hir);
    let h_in = g.input("h", Shape::new(&[1, len, hidden], DType::F32));
    let mut params = HashMap::new();
    let logits = build_kimi_text_flow(&mut g, &mut params, h_in, w, &cfg).context("spec build")?;
    g.set_outputs(vec![logits]);
    let built = built_from_hir(hir, params).context("spec built")?;
    let mut compiled = compile_built(built, device).context("spec compile")?;
    let h = embed_tokens(tokens, embed, hidden);
    Ok(compiled.run(&[("h", h.as_slice())]).remove(0))
}

/// A decode graph compiled ONCE at `max_len` and reused for every step. Valid
/// because the flow is causal — position `p`'s logits depend only on positions
/// `≤ p`, so padding the tail past the real tokens never changes a real row. This
/// removes the per-step recompile that dominates `run_flow` (the realistic
/// decoder pattern: compile once, run many).
struct Decoder {
    compiled: CompiledGraph,
    max_len: usize,
    hidden: usize,
}

impl Decoder {
    fn new(w: &FlowWeights, max_len: usize, device: Device) -> Result<Self> {
        let (_, cfg) = synth_flow(1, max_len);
        let hidden = cfg.hidden;
        let mut hir = HirModule::new("decoder");
        let mut g = HirMut::new(&mut hir);
        let h_in = g.input("h", Shape::new(&[1, max_len, hidden], DType::F32));
        let mut params = HashMap::new();
        let logits =
            build_kimi_text_flow(&mut g, &mut params, h_in, w, &cfg).context("decoder build")?;
        g.set_outputs(vec![logits]);
        let built = built_from_hir(hir, params).context("decoder built")?;
        let compiled = compile_built(built, device).context("decoder compile")?;
        Ok(Self {
            compiled,
            max_len,
            hidden,
        })
    }

    /// Per-position logits `[max_len, vocab]` for `tokens` (len ≤ max_len),
    /// reusing the compiled graph; read rows `< tokens.len()`. The tail is
    /// causal-padded with zeros.
    fn logits(&mut self, embed: &[f32], tokens: &[u32]) -> Vec<f32> {
        let mut h = embed_tokens(tokens, embed, self.hidden);
        h.resize(self.max_len * self.hidden, 0.0);
        self.compiled.run(&[("h", h.as_slice())]).remove(0)
    }
}

fn cmd_spec(args: &[String]) -> Result<()> {
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let draft_layers: usize = opt(args, "--draft-layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
        .clamp(1, n_layers);
    let n_gen: usize = opt(args, "--gen")
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let k: usize = opt(args, "--k")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .max(1);

    // Target model + a cheaper EARLY-EXIT draft: the target's first `draft_layers`
    // layers + the SAME head/embedding. `synth_flow` uses per-index seeds, so
    // `synth_flow(draft_layers)` is byte-identical to the target's leading layers.
    let (w, cfg) = synth_flow(n_layers, 1);
    let (hidden, vocab) = (cfg.hidden, cfg.vocab);
    let embed = fill(vocab * hidden, 4242);
    let (w_draft, _) = synth_flow(draft_layers, 1);
    let prompt: Vec<u32> = vec![3, 7];
    let max_len = prompt.len() + n_gen + k; // the verify block never exceeds this

    // ── baseline: greedy, recompiling the graph every step (what run_flow does) ──
    let t0 = Instant::now();
    let mut greedy_recompile = prompt.clone();
    for _ in 0..n_gen {
        let l = run_flow(&w, &embed, &greedy_recompile, device)?;
        greedy_recompile.push(argmax_row(&l, greedy_recompile.len() - 1, vocab));
    }
    let ms_recompile = t0.elapsed().as_secs_f64() * 1e3;

    // ── speedup 1: greedy, graph compiled ONCE and reused (causal-padded) ──
    let mut tgt = Decoder::new(&w, max_len, device)?;
    let t1 = Instant::now();
    let mut greedy = prompt.clone();
    for _ in 0..n_gen {
        let l = tgt.logits(&embed, &greedy);
        greedy.push(argmax_row(&l, greedy.len() - 1, vocab));
    }
    let ms_once = t1.elapsed().as_secs_f64() * 1e3;
    if greedy != greedy_recompile {
        bail!("compile-once decoder diverged from the recompile path (non-causal?)");
    }

    // ── speedup 2: speculative decode on top of the compile-once decoders ──
    let mut draft_dec = Decoder::new(&w_draft, max_len, device)?;
    let t2 = Instant::now();
    let mut toks = prompt.clone();
    let (mut target_calls, mut draft_calls, mut proposed, mut accepted) =
        (0usize, 0usize, 0usize, 0usize);
    while toks.len() < prompt.len() + n_gen {
        // draft K tokens autoregressively with the cheap model
        let mut draft = Vec::with_capacity(k);
        let mut dt = toks.clone();
        for _ in 0..k {
            let dl = draft_dec.logits(&embed, &dt);
            draft_calls += 1;
            let n = argmax_row(&dl, dt.len() - 1, vocab);
            dt.push(n);
            draft.push(n);
        }
        // verify: ONE target forward over prefix ++ draft
        let block: Vec<u32> = toks.iter().chain(&draft).copied().collect();
        let tl = tgt.logits(&embed, &block);
        target_calls += 1;
        let base = toks.len(); // target's next-after-prefix is row base-1
        let mut accept = 0;
        for (i, &d) in draft.iter().enumerate() {
            if argmax_row(&tl, base - 1 + i, vocab) == d {
                accept += 1;
            } else {
                break;
            }
        }
        proposed += k;
        accepted += accept;
        for &d in &draft[..accept] {
            toks.push(d);
        }
        // target's own token at the accept boundary — the guaranteed-correct bonus
        toks.push(argmax_row(&tl, base - 1 + accept, vocab));
    }
    toks.truncate(prompt.len() + n_gen);
    let ms_spec = t2.elapsed().as_secs_f64() * 1e3;
    if toks != greedy {
        bail!("speculative output diverged from greedy — model non-causal or verify bug");
    }

    // ── report: two independent, composable speedups, both token-exact ──
    println!(
        "Kimi-K3 decode speedups on {device:?}: target {n_layers}L · draft {draft_layers}L · K={k} · gen={n_gen}"
    );
    println!("  greedy, recompile / step : {ms_recompile:8.1} ms   (baseline)");
    println!(
        "  greedy, compile ONCE     : {ms_once:8.1} ms   → {:.1}× faster (removes per-step recompile)",
        ms_recompile / ms_once.max(1e-6)
    );
    println!(
        "  + speculative decode     : {ms_spec:8.1} ms   → {:.1}× vs baseline  ({target_calls} target passes vs {n_gen}, {accepted}/{proposed} draft accepted = {:.0}%)",
        ms_recompile / ms_spec.max(1e-6),
        accepted as f64 / proposed.max(1) as f64 * 100.0
    );
    println!("  (+ {draft_calls} cheap {draft_layers}-layer draft forwards)");
    println!("  ✓ all three produce token-identical output");
    Ok(())
}
