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
use std::net::ToSocketAddrs;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("plan") => cmd_plan(&args[2..]),
        Some("design") => cmd_design(&args[2..]),
        Some("cache") => cmd_cache(&args[2..]),
        Some("local") => cmd_local(&args[2..]),
        Some("pipeline") => cmd_pipeline(&args[2..]),
        Some("spec") => cmd_spec(&args[2..]),
        Some("run") => cmd_run(&args[2..]),
        Some("generate") => cmd_generate(&args[2..]),
        Some("logits") => cmd_logits(&args[2..]),
        Some("quantize-backbone") => cmd_quantize_backbone(&args[2..]),
        Some("read-bench") => cmd_read_bench(&args[2..]),
        Some("load-cmp") => cmd_load_cmp(&args[2..]),
        Some("page-bench") => cmd_page_bench(&args[2..]),
        Some("specgen") => cmd_specgen(&args[2..]),
        Some("vision") => cmd_vision(&args[2..]),
        Some("vlm") => cmd_vlm(&args[2..]),
        Some("worker") => cmd_worker(&args[2..]),
        Some("dist") => cmd_dist(&args[2..]),
        Some("dworker") => cmd_dworker(&args[2..]),
        Some("dgen") => cmd_dgen(&args[2..]),
        Some("expert-worker") => cmd_expert_worker(&args[2..]),
        Some("expert-selfcheck") => cmd_expert_selfcheck(&args[2..]),
        Some("expert-run") => cmd_expert_run(&args[2..]),
        _ => {
            eprintln!(
                "usage:\n  kimi_k3_cluster plan     --config <cluster.toml> --model-dir <dir>\n  \
                 kimi_k3_cluster design   --config <cluster.toml> --model-dir <dir>\n  \
                 kimi_k3_cluster local    [--device cpu|metal|mlx|gpu|coreml] [--layers N] [--seq N]\n  \
                 kimi_k3_cluster pipeline [--device …] [--layers N] [--seq N] [--stages N]\n  \
                 kimi_k3_cluster spec     [--device …] [--layers N] [--draft-layers N] [--k N] [--gen N]\n  \
                 kimi_k3_cluster cache    --config <cluster.toml> --model-dir <dir> [--vram-gb N]\n  \
                 kimi_k3_cluster run      --model-dir <dir> [--tokens 1,100,5000] [--layers N] [--device …]"
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
// cache — how the DISTRIBUTED memory hierarchy improves the expert cache
// ─────────────────────────────────────────────────────────────────────────────

/// Generalized harmonic number H(m, z) = Σ_{r=1}^{m} r^{-z}.
fn harmonic(m: usize, z: f64) -> f64 {
    (1..=m).map(|r| (r as f64).powf(-z)).sum()
}

fn cmd_cache(args: &[String]) -> Result<()> {
    let config = opt(args, "--config").context("--config <cluster.toml> required")?;
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let model_dir = Path::new(model_dir);
    // Fast tier: a GPU node's VRAM used purely as a hot-expert cache (read-only,
    // ~17 MB/expert), even when that node computes on CPU. Default = msi 3080Ti.
    let vram_gb: f64 = opt(args, "--vram-gb")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16.0);

    let cfg = ClusterConfig::from_path(config)?;
    let kc = rlx_kimi_k3::config::KimiK3Config::load(model_dir.join("config.json"))?;
    let t = &kc.text_config;
    eprintln!("scanning checkpoint shards…");
    let sz = scan_checkpoint(model_dir)?;

    let n_nodes = cfg.nodes.len().max(1);
    let n_layers = t.num_hidden_layers;
    let n_moe = (0..n_layers).filter(|&i| t.is_moe_layer(i)).count();
    let e = t.num_experts.unwrap_or(1).max(1);
    let topk = t.num_experts_per_token;

    let per_unit = sz.experts / (e as u64 * n_moe as u64).max(1); // bytes per (layer,expert)
    let per_tok = (topk as u64) * n_moe as u64 * per_unit; // 25.8 GB with no cache
    let node_bw = 5.0e9;
    let agg_bw = node_bw * n_nodes as f64;

    // Distributed cache tiers: aggregate free RAM after the resident backbone
    // (split across nodes, each caching its own expert shard) + GPU VRAM.
    let ram_free = ((cfg.nodes.iter().filter_map(|n| n.max_ram_gb).sum::<f64>() * 1e9)
        - sz.backbone as f64)
        .max(0.0);
    let vram = vram_gb * 1e9;
    let cache_units = ((ram_free + vram) / per_unit as f64) as usize; // cached (layer,expert) units
    let per_layer_cached = (cache_units / n_moe.max(1)).min(e); // cached experts per MoE layer

    println!("\nKimi-K3 DISTRIBUTED expert cache — {n_nodes} nodes");
    println!(
        "  experts {:.0} GB = {e} × {n_moe} layers, ~{:.1} MB each · {} active/token = {:.1} GB read/token",
        gb(sz.experts),
        per_unit as f64 / 1e6,
        topk * n_moe,
        gb(per_tok)
    );
    println!(
        "  cache tiers: RAM ~{:.0} GB (free after {:.0} GB backbone, expert-sharded to owner nodes) + VRAM {:.0} GB @ ~900 GB/s",
        ram_free / 1e9,
        gb(sz.backbone),
        vram_gb
    );
    println!(
        "  ⇒ hold ~{cache_units} hot expert-units resident = ~{per_layer_cached} of {e} per layer ({:.1}%)",
        per_layer_cached as f64 / e as f64 * 100.0
    );

    // Popularity-aware pinning: with Zipf-skewed routing, the hottest experts
    // serve a disproportionate share. Hit rate = H(cached, z) / H(experts, z).
    // (`noaux_tc` routing has no load-balancing loss, so real skew tends high —
    // profile to calibrate `z`.)
    println!("\n  decode s/token vs routing skew (Zipf z; higher = more skewed):");
    println!(
        "    {:>6}  {:>10}  {:>10}  {:>10}",
        "z", "hit%", "s/token", "vs no-cache"
    );
    let base_s = per_tok as f64 / agg_bw;
    let hn: Vec<(f64, f64)> = [0.5f64, 0.8, 1.0, 1.3]
        .iter()
        .map(|&z| (z, harmonic(e, z)))
        .collect();
    for (z, h_all) in hn {
        let hit = harmonic(per_layer_cached, z) / h_all;
        let s = per_tok as f64 * (1.0 - hit) / agg_bw;
        println!(
            "    {z:>6.1}  {:>9.0}%  {s:>8.2} s  {:>8.2}×",
            hit * 100.0,
            base_s / s.max(1e-9)
        );
    }
    println!("    (no cache: {base_s:.2} s/token)");

    println!("\n  distributed levers (why 3 nodes beat one pool):");
    println!("    • VRAM tier — msi's 16 GB 3080Ti is idle for compute (device=cpu) but is a");
    println!("      180× faster cache than NVMe; the hottest experts served from it are ~free.");
    println!("    • expert-affinity — each node caches ONLY its own expert shard, co-located with");
    println!(
        "      the disk that owns it → no cross-node cache traffic, cache = compute = storage."
    );
    println!(
        "    • popularity pinning — pin the globally-hottest experts (power-law) instead of a"
    );
    println!("      reactive LRU; a small hot-set captures a large share when skew is high.");
    println!(
        "    • speculative prefetch — the draft pass reveals the experts the verify pass will"
    );
    println!("      likely need; each node prefetches its share from NVMe during drafting.");
    println!();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// local — reduced synthetic forward, proves the graph runs on a backend
// ─────────────────────────────────────────────────────────────────────────────

use rlx_core::flow_util::{built_from_hir, compile_built};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{DType, Shape};
use rlx_kimi_k3::config::{KimiK3Config, KimiLinearConfig};
use rlx_kimi_k3::dist::{
    run_distributed_generate, run_distributed_prefix, serve_decode_worker, serve_worker,
};
use rlx_kimi_k3::flow::{
    AttnWeights, FfnWeights, FlowConfig, FlowWeights, LayerWeights, build_kimi_text_flow,
    build_kimi_text_stage,
};
use rlx_kimi_k3::kda::{KdaDims, KdaWeights};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::mla::{MlaDims, MlaWeights};
use rlx_kimi_k3::moe::{DenseMlpWeights, MoeDims, MoeWeights};
use rlx_kimi_k3::runner::{
    DecodeState, apply_head, decode_forward, run_generate, run_prefix_logits,
    run_speculative_generate,
};
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
        "rocm" | "hip" => Device::Rocm,
        "xdna" | "npu" => Device::Xdna,
        "oneapi" | "levelzero" => Device::OneApi,
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
    // Backbone device-parity: recompute the SAME synthetic forward on CPU and report
    // cosine + max|Δ| vs `device`. This isolates whether a GPU backbone is NUMERICALLY
    // correct (cos≈1 → any token flip is just borderline argmax) from a real miscompute
    // (cos≪1), independent of the flaky 4-layer toy argmax and the expert offload.
    if !matches!(device, Device::Cpu) {
        let mut hir2 = HirModule::new("kimi_k3_local_cpu");
        let mut g2 = HirMut::new(&mut hir2);
        let h2 = g2.input("h", Shape::new(&[batch, seq, hidden], DType::F32));
        let mut p2 = HashMap::new();
        let lg2 = build_kimi_text_flow(&mut g2, &mut p2, h2, &w, &cfg).context("build cpu ref")?;
        g2.set_outputs(vec![lg2]);
        let mut cpu =
            compile_built(built_from_hir(hir2, p2)?, Device::Cpu).context("cpu ref compile")?;
        let yc = cpu
            .run(&[("h", hin.as_slice())])
            .into_iter()
            .next()
            .context("no cpu out")?;
        let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in y.iter().zip(&yc) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
            mx = mx.max((*a - *b).abs() as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        let ct = {
            let l = &yc[(seq - 1) * vocab..seq * vocab];
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                )
                .0
        };
        println!(
            "  {device:?}-vs-CPU backbone logits: cos={cos:.6}  max|Δ|={mx:.3e}  argmax {device:?}={tok} CPU={ct}"
        );
        println!(
            "  → {}",
            if cos > 0.9999 {
                "CORRECT (cos≈1; token flips are borderline-argmax noise)"
            } else {
                "MISCOMPUTE (cos≪1 — real backbone bug on this device)"
            }
        );
    }
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

// ─────────────────────────────────────────────────────────────────────────────
// run — REAL streaming inference on the actual checkpoint → next-token logits
// ─────────────────────────────────────────────────────────────────────────────

/// `generate` — O(1)/token decode: prefill the prompt (establishing the per-layer
/// KDA/MLA decode state), then greedily generate `--gen` tokens one at a time.
fn cmd_generate(args: &[String]) -> Result<()> {
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let prompt: Vec<u32> = opt(args, "--tokens")
        .unwrap_or("5000")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let n_gen: usize = opt(args, "--gen").and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut ck = CheckpointLoader::open(model_dir)?;
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config.clone();
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(tc.num_hidden_layers);
    eprintln!(
        "generate: prompt {prompt:?} + {n_gen} tokens, {n_layers}/{} layers on {device:?}…",
        tc.num_hidden_layers
    );
    let t0 = std::time::Instant::now();
    let toks = run_generate(
        &mut ck,
        &tc,
        |seq| kimi_flow_cfg(&tc, seq),
        &prompt,
        n_gen,
        n_layers,
        device,
    )?;
    eprintln!(
        "generated {toks:?}  ({:.1}s, {:.1}s/token)",
        t0.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64() / n_gen.max(1) as f64
    );
    rlx_kimi_k3::io_opt::report();
    Ok(())
}

/// `logits` — run ONE forward (quant scheme read from the env: `RLX_KIMI_QUANT`
/// for weight-only int8/mxfp4, `RLX_KIMI_SCALED_BACKBONE` for fp8/mxfp4 W×A8) and
/// write the last-token logit vector (f32 LE) to `--out`. An external driver runs
/// this per-config and compares each dump against the bf16 baseline (cosine /
/// rel-L2 / argmax match) — the accuracy half of the quant bench.
fn cmd_logits(args: &[String]) -> Result<()> {
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let out = opt(args, "--out").context("--out <path> required")?;
    let prompt: Vec<u32> = opt(args, "--tokens")
        .unwrap_or("1,100,5000")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let mut ck = CheckpointLoader::open(model_dir)?;
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config.clone();
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(tc.num_hidden_layers);
    let t0 = std::time::Instant::now();
    let logits = run_prefix_logits(
        &mut ck,
        &tc,
        &kimi_flow_cfg(&tc, prompt.len()),
        &prompt,
        n_layers,
        device,
    )?;
    let seq = prompt.len();
    let vocab = if logits.len() % seq == 0 && logits.len() / seq > 1024 {
        logits.len() / seq
    } else {
        logits.len()
    };
    let last = &logits[logits.len() - vocab..];
    let (am, _) =
        last.iter().enumerate().fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            },
        );
    let bytes: Vec<u8> = last.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, &bytes)?;
    eprintln!(
        "logits: vocab={vocab} argmax={am} layers={n_layers} dev={device:?} \
         ({:.1}s) → {out}",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `load-cmp` — isolate the per-load CPU cost the current loader pays. For each
/// big backbone weight in layers `0..N` it times **(a)** `linear_t` (the real path:
/// mmap → `bf16→f32` upcast → transpose → owned Vec) vs **(b)** `mmap`-ing the
/// pre-quantized int8 `.bin` and touching every page (zero-copy: no upcast, no
/// transpose). Same page-cache state, so the delta is exactly the upcast+transpose
/// CPU that quantized+pretransposed+mmapped weights eliminate.
fn cmd_load_cmp(args: &[String]) -> Result<()> {
    use memmap2::Mmap;
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let nvme = opt(args, "--nvme").unwrap_or("/Users/Shared/rlx-models/.nvme");
    let q8 = format!("{nvme}/kimi-q8");
    let mut ck = CheckpointLoader::open(model_dir)?;
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config.clone();
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut names: Vec<String> = Vec::new();
    for l in 0..n_layers {
        let mut v = ck.layer_backbone_names(l);
        v.sort();
        names.extend(v);
    }
    let (mut t_bf16, mut t_i8) = (0f64, 0f64);
    let (mut b_bf16, mut b_i8) = (0u64, 0u64);
    let mut nw = 0usize;
    for name in &names {
        let shape = match ck.tensor_shape(name) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if shape.len() != 2 || shape[0] * shape[1] < 4096 {
            continue;
        }
        let (out, inn) = (shape[0], shape[1]);
        // (a) the real bf16 load: mmap + upcast + transpose → owned f32 Vec.
        let t = std::time::Instant::now();
        let w = ck.linear_t(name, out, inn)?;
        t_bf16 += t.elapsed().as_secs_f64();
        b_bf16 += (out * inn * 2) as u64; // bf16 bytes on disk
        std::hint::black_box(&w);
        // (b) mmap the int8 codes, touch every page (zero-copy load).
        let p = format!("{q8}/{}.bin", sanitize(name));
        if let Ok(f) = std::fs::File::open(&p) {
            let t = std::time::Instant::now();
            let m = unsafe { Mmap::map(&f)? };
            let mut acc = 0u64;
            for c in m.chunks(4096) {
                acc = acc.wrapping_add(c[0] as u64);
            }
            std::hint::black_box(acc);
            t_i8 += t.elapsed().as_secs_f64();
            b_i8 += m.len() as u64;
        }
        nw += 1;
    }
    let gb = |b: u64| b as f64 / 1e9;
    println!("\n── load-cmp ({n_layers} layers, {nw} weights) ──");
    println!(
        "  (a) bf16 linear_t  (mmap+upcast+transpose): {:.2} GB in {:.2}s = {:.2} GB/s",
        gb(b_bf16),
        t_bf16,
        gb(b_bf16) / t_bf16.max(1e-9)
    );
    println!(
        "  (b) int8 mmap+touch (zero-copy)           : {:.2} GB in {:.2}s = {:.2} GB/s",
        gb(b_i8),
        t_i8,
        gb(b_i8) / t_i8.max(1e-9)
    );
    println!(
        "  ⇒ mmapped-int8 load is {:.1}× faster (per-layer bf16 {:.2}s → int8 {:.2}s)",
        t_bf16 / t_i8.max(1e-9),
        t_bf16 / n_layers as f64,
        t_i8 / n_layers as f64
    );
    Ok(())
}

/// `page-bench` — clean cold A/B of the two expert-range read methods on DISJOINT
/// unlikely-fired experts (so both are cold, no purge needed): the old
/// mmap-whole-shard + page-fault vs the new `pread` of the contiguous range. Reports
/// GB/s each. `--base` picks an expert offset unlikely to be in the OS cache.
fn cmd_page_bench(args: &[String]) -> Result<()> {
    use memmap2::Mmap;
    use std::os::unix::fs::FileExt;
    use std::os::unix::io::AsRawFd;
    let model_dir = opt(args, "--model-dir").unwrap_or("/Volumes/FOUR/kimi");
    let layer: usize = opt(args, "--layer")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let n: usize = opt(args, "--experts")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let base: usize = opt(args, "--base")
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let mut ck = CheckpointLoader::open(model_dir)?;
    let lp = format!("language_model.model.layers.{layer}");
    // two disjoint cold expert sets: [base, base+n) for mmap, [base+n, base+2n) for pread
    let mut set_a = Vec::new();
    let mut set_b = Vec::new();
    for e in base..base + n {
        set_a.push(ck.expert_ranges(&lp, e)?);
    }
    for e in base + n..base + 2 * n {
        set_b.push(ck.expert_ranges(&lp, e)?);
    }
    let bytes = |s: &[[(std::path::PathBuf, usize, usize); 6]]| -> u64 {
        s.iter().flatten().map(|(_, a, b)| (b - a) as u64).sum()
    };
    // A: old path — mmap the whole shard + to_vec the range (cold page-fault).
    let mut acc = 0u64;
    let t = std::time::Instant::now();
    for r in &set_a {
        for (path, a, b) in r {
            let f = std::fs::File::open(path)?;
            let m = unsafe { Mmap::map(&f)? };
            let v = m[*a..*b].to_vec();
            acc = acc.wrapping_add(v[0] as u64);
        }
    }
    let ta = t.elapsed().as_secs_f64();
    // B: new path — open each distinct shard once, pread the range (F_NOCACHE=cold).
    let t = std::time::Instant::now();
    for r in &set_b {
        let mut files: std::collections::HashMap<&std::path::Path, std::fs::File> =
            std::collections::HashMap::new();
        for (path, _, _) in r {
            if !files.contains_key(path.as_path()) {
                let f = std::fs::File::open(path)?;
                // F_NOCACHE is macOS-only (cold reads); no-op on Linux workers —
                // `read-bench`/`load-cmp` are Mac diagnostic subcommands.
                #[cfg(target_os = "macos")]
                unsafe {
                    libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1)
                };
                files.insert(path.as_path(), f);
            }
        }
        for (path, a, b) in r {
            let mut buf = vec![0u8; b - a];
            files[path.as_path()].read_exact_at(&mut buf, *a as u64)?;
            acc = acc.wrapping_add(buf[0] as u64);
        }
    }
    let tb = t.elapsed().as_secs_f64();
    let ga = bytes(&set_a) as f64 / 1e9;
    let gb = bytes(&set_b) as f64 / 1e9;
    println!("page-bench layer {layer}, {n} experts/method (cold, disjoint)");
    println!(
        "  A mmap+fault (old): {ga:.3} GB in {ta:.2}s = {:.2} GB/s",
        ga / ta.max(1e-9)
    );
    println!(
        "  B pread F_NOCACHE (new): {gb:.3} GB in {tb:.2}s = {:.2} GB/s",
        gb / tb.max(1e-9)
    );
    println!(
        "  → per-GB: mmap {:.2}s/GB vs pread {:.2}s/GB  ({:.2}x)  (chk {})",
        ta / ga,
        tb / gb,
        (ta / ga) / (tb / gb),
        acc & 0xff
    );
    Ok(())
}

/// Per-output-row symmetric int8 of a native `[n,k]` weight: `q[o,i]=round(w/s[o])`,
/// `s[o]=amax_i/127`. Returns `(codes [n·k] i8-as-u8, scales [n] f32)`.
fn quant_int8_row(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    use rayon::prelude::*;
    let mut codes = vec![0u8; n * k];
    let mut scales = vec![0f32; n];
    codes
        .par_chunks_mut(k)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(o, (crow, s))| {
            let base = o * k;
            let mut amax = 0f32;
            for i in 0..k {
                amax = amax.max(w[base + i].abs());
            }
            let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            *s = sc;
            for i in 0..k {
                crow[i] = ((w[base + i] / sc).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        });
    (codes, scales)
}

/// Per-output-row **MXFP4** (FP4 e2m1 + e8m0 block-32 along k) of a native `[n,k]`
/// weight — the model's own expert encoding. Returns `(codes [n·⌈k/2⌉] packed 2
/// nibbles/byte low-first, scales [n·⌈k/32⌉] e8m0)`.
fn quant_mxfp4_row(w: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    use rayon::prelude::*;
    use rlx_ir::lowp_codec::{e8m0_to_f32, encode, f32_to_e8m0, max_finite};
    use rlx_ir::quant::ScaledFormat;
    const GS: usize = 32;
    let fmt = ScaledFormat::F4E2M1;
    let mxf = max_finite(fmt);
    let nb = k.div_ceil(GS);
    let cbytes = k.div_ceil(2);
    let mut codes = vec![0u8; n * cbytes];
    let mut scales = vec![0u8; n * nb];
    codes
        .par_chunks_mut(cbytes)
        .zip(scales.par_chunks_mut(nb))
        .enumerate()
        .for_each(|(o, (crow, srow))| {
            let base = o * k;
            for b in 0..nb {
                let lo = b * GS;
                let hi = (lo + GS).min(k);
                let mut amax = 0f32;
                for i in lo..hi {
                    amax = amax.max(w[base + i].abs());
                }
                let scale = if amax > 0.0 { amax / mxf } else { 1.0 };
                let e8 = f32_to_e8m0(scale);
                srow[b] = e8;
                let sdec = e8m0_to_f32(e8).max(f32::MIN_POSITIVE);
                for i in lo..hi {
                    let q = encode(fmt, w[base + i] / sdec) & 0x0f;
                    let bi = i / 2;
                    if i % 2 == 0 {
                        crow[bi] |= q;
                    } else {
                        crow[bi] |= q << 4;
                    }
                }
            }
        });
    (codes, scales)
}

fn sanitize(name: &str) -> String {
    name.replace('/', "_")
}

/// `quantize-backbone` — read each backbone weight (bf16) from `--model-dir`,
/// encode to int8 (per-row) AND mxfp4 (block-32, the model's expert format), and
/// write to `--q8`/`--q4` dirs (default under `--nvme`). One `.bin` per weight
/// (`[codes][scales]`). Reports source vs quantized bytes + per-scheme throughput.
/// This is the "quantize + move to NVMe" half of the quant bench.
fn cmd_quantize_backbone(args: &[String]) -> Result<()> {
    use std::io::Write;
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let nvme = opt(args, "--nvme").unwrap_or("/Users/Shared/rlx-models/.nvme");
    let q8_dir = format!("{nvme}/kimi-q8");
    let q4_dir = format!("{nvme}/kimi-q4");
    let scheme = opt(args, "--scheme").unwrap_or("both");
    std::fs::create_dir_all(&q8_dir)?;
    std::fs::create_dir_all(&q4_dir)?;
    let mut ck = CheckpointLoader::open(model_dir)?;
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config.clone();
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(tc.num_hidden_layers);

    let mut names: Vec<String> = Vec::new();
    if opt(args, "--no-head").is_none() {
        for n in ["lm_head.weight", "language_model.model.embed_tokens.weight"] {
            names.push(n.to_string());
        }
    }
    for l in 0..n_layers {
        let mut v = ck.layer_backbone_names(l);
        v.sort();
        names.extend(v);
    }

    let (mut src_b, mut q8_b, mut q4_b) = (0u64, 0u64, 0u64);
    let (mut t_read, mut t_q8, mut t_q4, mut t_wr) = (0f64, 0f64, 0f64, 0f64);
    let (mut n_w, mut n_skip) = (0usize, 0usize);
    let t_all = std::time::Instant::now();
    let write_bin = |dir: &str, name: &str, codes: &[u8], sc_bytes: &[u8]| -> Result<()> {
        let mut f = std::fs::File::create(format!("{dir}/{}.bin", sanitize(name)))?;
        f.write_all(codes)?;
        f.write_all(sc_bytes)?;
        Ok(())
    };
    for (i, name) in names.iter().enumerate() {
        let t = std::time::Instant::now();
        let (shape, w) = match ck.tensor_f32_shaped(name) {
            Ok(x) => x,
            Err(_) => continue,
        };
        t_read += t.elapsed().as_secs_f64();
        src_b += (w.len() * 2) as u64;
        if shape.len() != 2 || w.len() < 4096 {
            n_skip += 1;
            continue; // norms / tiny 1-D — negligible, keep out of the bench
        }
        let (n, k) = (shape[0], shape[1]);
        n_w += 1;
        if scheme != "mxfp4" {
            let t = std::time::Instant::now();
            let (codes, scales) = quant_int8_row(&w, n, k);
            t_q8 += t.elapsed().as_secs_f64();
            let sb: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
            q8_b += (codes.len() + sb.len()) as u64;
            let t = std::time::Instant::now();
            write_bin(&q8_dir, name, &codes, &sb)?;
            t_wr += t.elapsed().as_secs_f64();
        }
        if scheme != "int8" {
            let t = std::time::Instant::now();
            let (codes, scales) = quant_mxfp4_row(&w, n, k);
            t_q4 += t.elapsed().as_secs_f64();
            q4_b += (codes.len() + scales.len()) as u64;
            let t = std::time::Instant::now();
            write_bin(&q4_dir, name, &codes, &scales)?;
            t_wr += t.elapsed().as_secs_f64();
        }
        if i % 50 == 0 {
            eprintln!(
                "  [{i}/{}] {name}  src={:.1}GB q8={:.1}GB q4={:.1}GB",
                names.len(),
                src_b as f64 / 1e9,
                q8_b as f64 / 1e9,
                q4_b as f64 / 1e9
            );
        }
    }
    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!("\n── backbone quantize ({n_layers} layers, {n_w} weights, {n_skip} skipped) ──");
    println!(
        "  source  bf16 : {:.2} GB  (read {:.1}s from {model_dir})",
        gb(src_b),
        t_read
    );
    if scheme != "mxfp4" {
        println!(
            "  int8    W8   : {:.2} GB  ({:.2}× smaller, encode {:.1}s)  → {q8_dir}",
            gb(q8_b),
            src_b as f64 / q8_b.max(1) as f64,
            t_q8
        );
    }
    if scheme != "int8" {
        println!(
            "  mxfp4   W4   : {:.2} GB  ({:.2}× smaller, encode {:.1}s)  → {q4_dir}",
            gb(q4_b),
            src_b as f64 / q4_b.max(1) as f64,
            t_q4
        );
    }
    println!(
        "  write {:.1}s, total {:.1}s",
        t_wr,
        t_all.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `read-bench` — measure **cold** sequential read bandwidth of a directory of
/// files (`--dir`). Each fd is `fcntl(F_NOCACHE)`'d so reads bypass the unified
/// buffer cache and hit the device — a true storage-bandwidth number independent
/// of page-cache state (macOS won't let us `purge` without sudo). Reports GB, s,
/// GB/s. Point `--dir` at an external checkpoint dir to get its bf16 read speed too.
fn cmd_read_bench(args: &[String]) -> Result<()> {
    use std::io::Read;
    use std::os::unix::io::AsRawFd;
    let dir = opt(args, "--dir").context("--dir <path> required")?;
    let ext = opt(args, "--ext"); // optional filename-suffix filter (e.g. "safetensors")
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .filter(|p| ext.is_none_or(|e| p.to_string_lossy().ends_with(e)))
        .collect();
    files.sort();
    let cap: u64 = opt(args, "--cap-gb")
        .and_then(|s| s.parse::<f64>().ok())
        .map(|g| (g * 1e9) as u64)
        .unwrap_or(u64::MAX);
    let mut buf = vec![0u8; 8 << 20]; // 8 MiB
    let t = std::time::Instant::now();
    let mut total = 0u64;
    let mut acc = 0u64;
    'outer: for p in &files {
        let f = std::fs::File::open(p)?;
        // F_NOCACHE=48 on macOS: bypass the buffer cache for this fd → cold reads.
        // macOS-only; no-op on Linux workers (Mac diagnostic subcommand).
        #[cfg(target_os = "macos")]
        unsafe {
            libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1)
        };
        let mut f = f;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            total += n as u64;
            acc = acc.wrapping_add(buf[0] as u64);
            if total >= cap {
                break 'outer;
            }
        }
    }
    let s = t.elapsed().as_secs_f64();
    let gb = total as f64 / 1e9;
    println!(
        "read-bench {dir}: {} files, {:.2} GB (decimal) in {:.2}s = {:.2} GB/s COLD  (chk {})",
        files.len(),
        gb,
        s,
        gb / s.max(1e-9),
        acc & 0xff
    );
    Ok(())
}

/// Speculative decode + a correctness check that it equals greedy `run_generate`.
/// `--draft <L>` = draft depth (first L layers), `--k <K>` = tokens proposed/round.
fn cmd_specgen(args: &[String]) -> Result<()> {
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let prompt: Vec<u32> = opt(args, "--tokens")
        .unwrap_or("5000")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let n_gen: usize = opt(args, "--gen").and_then(|s| s.parse().ok()).unwrap_or(6);
    let k: usize = opt(args, "--k").and_then(|s| s.parse().ok()).unwrap_or(4);

    let mut ck = CheckpointLoader::open(model_dir)?;
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config.clone();
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(tc.num_hidden_layers);
    let n_draft: usize = opt(args, "--draft")
        .and_then(|s| s.parse().ok())
        .unwrap_or((n_layers / 2).max(1));
    eprintln!(
        "specgen: prompt {prompt:?} + {n_gen} tokens, target {n_layers}/{} layers, \
         draft {n_draft} layers, k={k} on {device:?}…",
        tc.num_hidden_layers
    );
    let t0 = std::time::Instant::now();
    let (toks, accepted) = run_speculative_generate(
        &mut ck,
        &tc,
        |seq| kimi_flow_cfg(&tc, seq),
        &prompt,
        n_gen,
        n_layers,
        n_draft,
        k,
        device,
    )?;
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "spec  {toks:?}  ({dt:.1}s, {:.1}s/token, {accepted} drafts accepted)",
        dt / n_gen.max(1) as f64
    );

    // correctness invariant: spec output MUST equal greedy output.
    let g0 = std::time::Instant::now();
    let greedy = run_generate(
        &mut ck,
        &tc,
        |seq| kimi_flow_cfg(&tc, seq),
        &prompt,
        n_gen,
        n_layers,
        device,
    )?;
    eprintln!("greedy {greedy:?}  ({:.1}s)", g0.elapsed().as_secs_f64());
    if toks == greedy {
        eprintln!("✓ speculative output IDENTICAL to greedy");
    } else {
        anyhow::bail!("✗ spec {toks:?} != greedy {greedy:?}");
    }
    Ok(())
}

/// `vision` — load the **MoonViT vision tower + patchmergerv2** REAL weights and
/// run it over a synthetic `--grid`×`--grid` patch grid (the upstream patch-embed /
/// image preprocessing is separate), producing projected image tokens. Proves the
/// vision tower is real-weight runnable (the loader was the gap). `--grid` small.
fn cmd_vision(args: &[String]) -> Result<()> {
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let grid: usize = opt(args, "--grid")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let vc = kc
        .vision_config
        .context("config.json has no vision_config")?;
    let mut ck = CheckpointLoader::open(model_dir)?;
    let (w, mut d) = ck.load_vision(&vc)?;
    // override the patch grid for a light smoke run (must be merge-aligned).
    let g_al = grid - grid % d.merge.max(1);
    d.grid_h = g_al.max(d.merge);
    d.grid_w = g_al.max(d.merge);
    let (l, hid, hd) = (d.seq_len(), d.hidden, d.head_dim);
    eprintln!(
        "vision: {} MoonViT blocks (hidden {hid}, qkv {}, heads {}×{hd}, inter {}), \
         merge {}, proj {}→{}, grid {}×{} = {l} patches on {device:?}…",
        w.blocks.len(),
        d.qkv_hidden,
        d.num_heads,
        d.inter,
        d.merge,
        d.proj_mid,
        d.text_hidden,
        d.grid_h,
        d.grid_w
    );

    let mut hir = HirModule::new("vision");
    let mut g = HirMut::new(&mut hir);
    let hidden = g.input("hidden", Shape::new(&[1, l, hid], DType::F32));
    let cos = g.input("cos", Shape::new(&[l, hd / 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[l, hd / 2], DType::F32));
    let mut params = std::collections::HashMap::new();
    let out = rlx_kimi_k3::vision::build_vision(&mut g, &mut params, hidden, cos, sin, &w, d)?;
    g.set_outputs(vec![out]);
    let mut compiled = compile_built(built_from_hir(hir, params)?, device)?;

    // synthetic normalized patch states + 2D-RoPE tables (patch-embed is upstream).
    let synth = |n: usize, s: u64| -> Vec<f32> {
        let mut x = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((x >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
            })
            .collect()
    };
    let t0 = Instant::now();
    let y = compiled
        .run(&[
            ("hidden", synth(l * hid, 1).as_slice()),
            ("cos", synth(l * (hd / 2), 2).as_slice()),
            ("sin", synth(l * (hd / 2), 3).as_slice()),
        ])
        .remove(0);
    let n_tokens = (d.grid_h / d.merge) * (d.grid_w / d.merge);
    eprintln!(
        "vision tower REAL weights OK: {n_tokens} image tokens × {} finite={} ({:.1}s)",
        d.text_hidden,
        y.iter().all(|v| v.is_finite()) && y.len() == n_tokens * d.text_hidden,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `vlm` — FULL multimodal forward: synthetic image → host patch-embed → 2D-RoPE →
/// MoonViT tower + projector → splice the projected image tokens into a text prompt's
/// placeholder rows → decode text. Proves the image→text path runs end-to-end on real
/// weights. (HF-parity of the vision features / RoPE schedule is NOT verified — no
/// reference; this is a structural/runnability milestone.) `--grid` small, `--layers`.
fn cmd_vlm(args: &[String]) -> Result<()> {
    use rlx_kimi_k3::vision::{build_vision, patch_embed, vision_rope_2d};
    use rlx_kimi_k3::wrapper::merge_text_and_vision_embds;
    const EMB: &str = "language_model.model.embed_tokens.weight";
    let argmax = |v: &[f32]| -> u32 {
        v.iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |m, (i, &x)| if x > m.1 { (i, x) } else { m },
            )
            .0 as u32
    };

    let model_dir = opt(args, "--model-dir").context("--model-dir required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let grid: usize = opt(args, "--grid")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n_gen: usize = opt(args, "--gen").and_then(|s| s.parse().ok()).unwrap_or(3);
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config.clone();
    let vc = kc.vision_config.clone().context("no vision_config")?;
    let ph_id = kc
        .media_placeholder_token_id
        .context("no media_placeholder_token_id")?;
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(tc.num_hidden_layers);
    let hidden = tc.hidden_size; // LM hidden = vision projector out (7168)
    let patch = vc.patch_size;

    let mut ck = CheckpointLoader::open(model_dir)?;

    // ── synthetic image [3, H, W], grid merge-aligned so M vision tokens = (gh/2)² ──
    let g_al = (grid - grid % vc.merge_kernel_size.first().copied().unwrap_or(2)).max(2);
    let (h_img, w_img) = (g_al * patch, g_al * patch);
    let synth = |n: usize, s: u64| -> Vec<f32> {
        let mut x = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((x >> 33) as f32) / (u32::MAX as f32) - 0.5) * 2.0 // ~[-1,1] normalized pixels
            })
            .collect()
    };
    let image = synth(3 * h_img * w_img, 42);

    // ── patch-embed (host conv + bilinear pos-emb) → patch hidden [L, vis_hidden] ──
    let (conv, pos_emb, pos_h, pos_w) = ck.load_patch_embed()?;
    let (patch_hidden, gh, gw) = patch_embed(
        &image,
        h_img,
        w_img,
        &conv,
        &pos_emb,
        pos_h,
        pos_w,
        patch,
        vc.hidden_size,
    );

    // ── vision tower + projector → image tokens [M, hidden] ──
    let (w, mut d) = ck.load_vision(&vc)?;
    d.grid_h = gh;
    d.grid_w = gw;
    let (l, vh, hd) = (gh * gw, d.hidden, d.head_dim);
    let (cosv, sinv) = vision_rope_2d(gh, gw, hd);
    let mut hir = HirModule::new("vlm_vision");
    let mut g = HirMut::new(&mut hir);
    let hin = g.input("hidden", Shape::new(&[1, l, vh], DType::F32));
    let cos = g.input("cos", Shape::new(&[l, hd / 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[l, hd / 2], DType::F32));
    let mut vp = std::collections::HashMap::new();
    let vout = build_vision(&mut g, &mut vp, hin, cos, sin, &w, d)?;
    g.set_outputs(vec![vout]);
    let mut vcompiled = compile_built(built_from_hir(hir, vp)?, device)?;
    let vision_tokens = vcompiled
        .run(&[
            ("hidden", patch_hidden.as_slice()),
            ("cos", cosv.as_slice()),
            ("sin", sinv.as_slice()),
        ])
        .remove(0);
    let m = vision_tokens.len() / hidden;
    eprintln!(
        "vlm: image {h_img}×{w_img} → {gh}×{gw} patches → {m} vision tokens (× {hidden}) on {device:?}"
    );

    // ── text prompt with M placeholders: [txt, PH×M, txt] → embed → splice ──
    let mut prompt: Vec<u32> = vec![5000];
    prompt.extend(std::iter::repeat_n(ph_id as u32, m));
    prompt.push(5001);
    let ids: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
    let mut embeds = ck.gather_embed(EMB, &prompt, hidden)?;
    merge_text_and_vision_embds(&mut embeds, &ids, hidden, &vision_tokens, ph_id)?;

    // ── decode the spliced sequence → text ──
    let cfg1 = kimi_flow_cfg(&tc, 1);
    let cfg_p = kimi_flow_cfg(&tc, prompt.len());
    let mut state = DecodeState::zeros(&tc, &cfg1);
    let t0 = Instant::now();
    let (h, snaps) = decode_forward(&mut ck, &tc, &cfg_p, embeds, &mut state, n_layers, device)?;
    let last = prompt.len() - 1;
    let sl = |v: &[f32]| v[last * hidden..(last + 1) * hidden].to_vec();
    let snaps_last: Vec<Vec<f32>> = snaps.iter().map(|s| sl(s)).collect();
    let mut tok = argmax(&apply_head(&mut ck, &cfg1, &sl(&h), &snaps_last, device)?);
    let mut out = vec![tok];
    for _ in 1..n_gen {
        let hin = ck.gather_embed(EMB, &[tok], hidden)?;
        let (h, snaps) = decode_forward(&mut ck, &tc, &cfg1, hin, &mut state, n_layers, device)?;
        tok = argmax(&apply_head(&mut ck, &cfg1, &h, &snaps, device)?);
        out.push(tok);
    }
    eprintln!(
        "VLM image→text OK: {m} vision tokens spliced into {}-tok prompt, {n_layers} layers → generated {out:?} ({:.1}s)",
        prompt.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<()> {
    let model_dir = opt(args, "--model-dir").context("--model-dir <dir> required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let tokens: Vec<u32> = opt(args, "--tokens")
        .unwrap_or("1,100,5000")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let seq = tokens.len().max(1);

    let mut ck = CheckpointLoader::open(model_dir)?;
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = &kc.text_config;
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(tc.num_hidden_layers);
    let (hidden, vocab) = (tc.hidden_size, tc.vocab_size);

    let kda = KdaDims {
        hidden,
        num_heads: 96,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: Some(-5.0),
        eps: 1e-5,
        batch: 1,
        seq,
    };
    let mla = MlaDims {
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
    };
    let moe = MoeDims {
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
    };
    let cfg = FlowConfig {
        hidden,
        vocab,
        attn_res_block_size: tc.attn_res_block_size.unwrap_or(12),
        eps: 1e-5,
        kda,
        mla,
        moe,
        dense_inter: 33792,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        batch: 1,
        seq,
    };

    eprintln!(
        "streaming {n_layers}/{} layers on {device:?}, prompt {tokens:?} (external-drive IO-bound)…",
        tc.num_hidden_layers
    );
    let t0 = Instant::now();
    let logits = run_prefix_logits(&mut ck, tc, &cfg, &tokens, n_layers, device)?;
    let secs = t0.elapsed().as_secs_f64();

    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_unstable_by(|&a, &b| last[b].total_cmp(&last[a]));
    let finite = logits.iter().all(|v| v.is_finite());
    println!(
        "\nKimi-K3 streaming inference: {n_layers} layers, {seq} tokens → logits [{seq},{vocab}] finite={finite} in {secs:.1}s"
    );
    println!(
        "  next-token (greedy) = {}  (logit {:.3})",
        idx[0], last[idx[0]]
    );
    println!(
        "  top-5: {:?}",
        idx[..5].iter().map(|&i| (i, last[i])).collect::<Vec<_>>()
    );
    if n_layers < tc.num_hidden_layers {
        println!(
            "  (partial: {n_layers}/{} layers — pass --layers {} for the real next token)",
            tc.num_hidden_layers, tc.num_hidden_layers
        );
    }
    rlx_kimi_k3::io_opt::report();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// worker / dist — the layer-pipeline distributed runner (fast NVMe per node)
// ─────────────────────────────────────────────────────────────────────────────

/// Build the real Kimi-K3 `FlowConfig` for a given sequence length.
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

/// `worker` — serve a layer range from the LOCAL checkpoint over TCP.
fn cmd_worker(args: &[String]) -> Result<()> {
    let addr = opt(args, "--addr").context("--addr host:port required")?;
    let model_dir = opt(args, "--model-dir").context("--model-dir required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config;
    serve_worker(
        addr,
        model_dir,
        &tc,
        |seq| kimi_flow_cfg(&tc, seq),
        device,
        0,
    )
}

/// `dworker` — stateful DECODE worker: holds its layer-range decode state resident
/// across tokens (O(1)/token). Same flags as `worker`.
fn cmd_dworker(args: &[String]) -> Result<()> {
    let addr = opt(args, "--addr").context("--addr host:port required")?;
    let model_dir = opt(args, "--model-dir").context("--model-dir required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config;
    serve_decode_worker(
        addr,
        model_dir,
        &tc,
        |seq| kimi_flow_cfg(&tc, seq),
        device,
        0,
    )
}

/// `dgen` — distributed GENERATION coordinator: prefill + O(1) decode over the
/// resident-state `dworker`s (`--workers host:port:start:end,…`). Output equals
/// single-node `run_generate`. `--gen N` tokens.
fn cmd_dgen(args: &[String]) -> Result<()> {
    let model_dir = opt(args, "--model-dir").context("--model-dir required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let prompt: Vec<u32> = opt(args, "--tokens")
        .unwrap_or("5000")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let n_gen: usize = opt(args, "--gen").and_then(|s| s.parse().ok()).unwrap_or(4);
    let stages: Vec<(String, usize, usize)> = opt(args, "--workers")
        .context("--workers host:port:start:end,… required")?
        .split(',')
        .map(|w| {
            let p: Vec<&str> = w.split(':').collect();
            (
                format!("{}:{}", p[0], p[1]),
                p[2].parse().unwrap(),
                p[3].parse().unwrap(),
            )
        })
        .collect();

    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config;
    let mut ck = CheckpointLoader::open(model_dir)?;
    eprintln!(
        "dgen: prompt {prompt:?} + {n_gen} tokens → {} stages {stages:?}",
        stages.len()
    );
    let t0 = Instant::now();
    let toks = run_distributed_generate(
        &mut ck,
        |seq| kimi_flow_cfg(&tc, seq),
        &prompt,
        n_gen,
        &stages,
        device,
    )?;
    eprintln!(
        "distributed generated {toks:?}  ({:.1}s, {:.1}s/token)",
        t0.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64() / n_gen.max(1) as f64
    );
    Ok(())
}

/// `dist` — coordinator: embed locally, pipeline the boundary state through the
/// `--workers addr:start:end,…`, then apply the head → next-token logits.
fn cmd_dist(args: &[String]) -> Result<()> {
    let model_dir = opt(args, "--model-dir").context("--model-dir required")?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let tokens: Vec<u32> = opt(args, "--tokens")
        .unwrap_or("5000")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let stages: Vec<(String, usize, usize)> = opt(args, "--workers")
        .context("--workers addr:start:end,… required")?
        .split(',')
        .map(|w| {
            let p: Vec<&str> = w.split(':').collect();
            (
                format!("{}:{}", p[0], p[1]),
                p[2].parse().unwrap(),
                p[3].parse().unwrap(),
            )
        })
        .collect();

    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config;
    let cfg = kimi_flow_cfg(&tc, tokens.len());
    let mut ck = CheckpointLoader::open(model_dir)?;

    eprintln!(
        "coordinator: prompt {tokens:?} → {} stages {stages:?}",
        stages.len()
    );
    let t0 = Instant::now();
    let (h, snaps) = run_distributed_prefix(&mut ck, &cfg, &tokens, &stages)?;
    let logits = apply_head(&mut ck, &cfg, &h, &snaps, device)?;
    let secs = t0.elapsed().as_secs_f64();

    let (seq, vocab) = (tokens.len(), cfg.vocab);
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let (tok, val) =
        last.iter().enumerate().fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        );
    println!(
        "\nKimi-K3 DISTRIBUTED inference: logits [{seq},{vocab}] finite={} in {secs:.1}s",
        logits.iter().all(|v| v.is_finite())
    );
    println!("  next-token (greedy) = {tok} (logit {val:.3})");
    Ok(())
}

/// Parse `host:port,host:port,...` (index = rank) into peer addresses.
fn parse_peers(s: &str) -> Result<Vec<std::net::SocketAddr>> {
    s.split(',')
        .map(|p| {
            p.trim()
                .to_socket_addrs()
                .ok()
                .and_then(|mut it| it.next())
                .with_context(|| format!("bad peer {p:?}"))
        })
        .collect()
}

/// `expert-worker` — serve this node's routed-expert shard `[--lo,--hi)` over the
/// transport (rank `--rank` of `--peers`), paging from `--dest`. Runs until the
/// orchestrator (rank 0) sends the shutdown sentinel.
fn cmd_expert_worker(args: &[String]) -> Result<()> {
    use rlx_distributed::{TcpTransport, serve_expert_worker};
    use rlx_kimi_k3::dist_experts::KimiExpertProvider;
    use std::collections::HashSet;
    let peers = parse_peers(opt(args, "--peers").context("--peers required")?)?;
    let rank: u32 = opt(args, "--rank").context("--rank")?.parse()?;
    let dest = opt(args, "--dest").unwrap_or("/Volumes/FOUR/kimi");
    let lo: usize = opt(args, "--lo").context("--lo")?.parse()?;
    let hi: usize = opt(args, "--hi").context("--hi")?.parse()?;
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let world = peers.len() as u32;
    let kc = KimiK3Config::load(Path::new(dest).join("config.json"))?;
    let d = kimi_flow_cfg(&kc.text_config, 1).moe;
    eprintln!("[expert-worker rank {rank}/{world}] experts [{lo},{hi}) from {dest} on {device:?}");
    let t = TcpTransport::bind(rank, world, peers, 64 << 20)?;
    let owned: HashSet<usize> = (lo..hi).collect();
    let mut p = KimiExpertProvider::open(dest, d, owned, device)?;
    serve_expert_worker(&t, 0, &mut p)?;
    let tm = p.timing();
    let total = tm.experts_paged + tm.cache_hits;
    let hit_pct = if total > 0 {
        100.0 * tm.cache_hits as f64 / total as f64
    } else {
        0.0
    };
    eprintln!(
        "[expert-worker rank {rank}] shutdown | {} reqs on {device:?}: {} paged + {} cache-hits ({hit_pct:.0}% hit) | PAGING {:.2}s + COMPUTE {:.2}s (compile {:.2}s [{} graph-hits] + run {:.2}s)",
        tm.calls,
        tm.experts_paged,
        tm.cache_hits,
        tm.page_ms / 1e3,
        tm.graph_ms / 1e3,
        tm.compile_ms / 1e3,
        tm.graph_hits,
        tm.run_ms / 1e3
    );
    Ok(())
}

/// `expert-selfcheck` — per-node PRECISION/ERROR alignment proof. Runs this shard's
/// [`KimiExpertProvider::compute`] on the requested `--device` (GPU) AND on the CPU
/// for the SAME deterministic synthetic input (fired experts restricted to `[lo,hi)`),
/// then reports max|Δ| and rel-L2 between the two partials. This is the numeric
/// contract the distributed run relies on: the GPU expert math must match CPU f32
/// within GEMM-accumulation tolerance. No transport, no orchestrator — pure local.
fn cmd_expert_selfcheck(args: &[String]) -> Result<()> {
    use rlx_distributed::ExpertProvider;
    use rlx_ir::Philox4x32;
    use rlx_kimi_k3::dist_experts::KimiExpertProvider;
    use std::collections::HashSet;
    let dest = opt(args, "--dest").unwrap_or("/Volumes/FOUR/kimi");
    let lo: usize = opt(args, "--lo").context("--lo")?.parse()?;
    let hi: usize = opt(args, "--hi").context("--hi")?.parse()?;
    let device = parse_device(opt(args, "--device").unwrap_or("cuda"));
    let layer: u32 = opt(args, "--layer")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let rows: usize = opt(args, "--rows")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let tol: f64 = opt(args, "--tol")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2e-3);
    let seed: u64 = opt(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xC0FFEE);

    let kc = KimiK3Config::load(Path::new(dest).join("config.json"))?;
    let d = kimi_flow_cfg(&kc.text_config, 1).moe;
    let (l, k, span) = (d.latent, d.top_k, hi - lo);
    // deterministic latent FFN input h_lat [rows, L].
    let mut rng = Philox4x32::new(seed);
    let mut h_lat = vec![0f32; rows * l];
    rng.fill_normal(&mut h_lat);
    // fired routing: k distinct OWNED experts per row, normalized positive probs.
    let mut ids = vec![0u32; rows * k];
    let mut probs = vec![0f32; rows * k];
    let mut praw = vec![0f32; rows * k];
    rng.fill_normal(&mut praw);
    for r in 0..rows {
        let mut acc = 0f32;
        for j in 0..k {
            ids[r * k + j] = (lo + (r * k + j) % span) as u32;
            let p = praw[r * k + j].abs() + 1e-3;
            probs[r * k + j] = p;
            acc += p;
        }
        for j in 0..k {
            probs[r * k + j] /= acc; // per-row softmax-like normalization
        }
    }
    let owned: HashSet<usize> = (lo..hi).collect();
    // --packed → compare the PACKED (MXFP4 DequantGroupedMatMulMlx) path vs the f32 path,
    // both on `--device` (validates the packed wiring). Else compare `--device` vs CPU.
    let packed = args.iter().any(|a| a == "--packed");
    let scaled = args.iter().any(|a| a == "--scaled");
    let (ref_dev, lhs, rhs) = if scaled {
        (
            device,
            format!("{device:?}/scaled-W4A8"),
            format!("{device:?}/f32"),
        )
    } else if packed {
        (
            device,
            format!("{device:?}/packed"),
            format!("{device:?}/f32"),
        )
    } else {
        (Device::Cpu, format!("{device:?}"), "Cpu".to_string())
    };
    eprintln!("[selfcheck] experts [{lo},{hi}) L={l} k={k} rows={rows} | {lhs} vs {rhs}");

    let mut gpu = KimiExpertProvider::open(dest, d, owned.clone(), device)?;
    let mut cpu = KimiExpertProvider::open(dest, d, owned, ref_dev)?;
    if scaled {
        gpu.set_scaled(true);
        cpu.set_scaled(false);
    } else if packed {
        gpu.set_packed(true);
        cpu.set_packed(false);
    }
    let og = gpu.compute(layer, &h_lat, rows, l, &ids, &probs)?;
    let oc = cpu.compute(layer, &h_lat, rows, l, &ids, &probs)?;
    let (mut maxd, mut num, mut den) = (0f64, 0f64, 0f64);
    for (a, b) in og.iter().zip(&oc) {
        let dlt = (*a - *b) as f64;
        maxd = maxd.max(dlt.abs());
        num += dlt * dlt;
        den += (*b as f64) * (*b as f64);
    }
    let rel = (num / den.max(1e-30)).sqrt();
    let gmax = og.iter().fold(0f32, |m, &x| m.max(x.abs()));
    eprintln!(
        "[selfcheck] {lhs} vs {rhs} partial [{rows},{l}]: max|Δ|={maxd:.3e}  rel-L2={rel:.3e}  |out|max={gmax:.3e}"
    );
    if rel <= tol && maxd.is_finite() {
        eprintln!(
            "[selfcheck] PASS (rel-L2 {rel:.3e} ≤ tol {tol:.1e}) — GPU expert math aligned with CPU"
        );
        Ok(())
    } else {
        bail!(
            "[selfcheck] FAIL: rel-L2 {rel:.3e} > tol {tol:.1e} (or non-finite) — precision NOT aligned"
        );
    }
}

/// `expert-run` — the ORCHESTRATOR (rank 0): install the cluster MoE context (shard
/// map + Mac-local overflow), then generate. MoE layers offload to the workers.
fn cmd_expert_run(args: &[String]) -> Result<()> {
    use rlx_distributed::{ExpertShards, TcpTransport, Transport, shutdown_expert_workers};
    use rlx_kimi_k3::dist_experts::{
        ClusterMoe, KimiExpertProvider, install_cluster_moe, take_cluster_moe,
    };
    use rlx_kimi_k3::runner::run_generate;
    use std::collections::HashSet;
    use std::sync::Arc;
    let peers = parse_peers(opt(args, "--peers").context("--peers required")?)?;
    let model_dir = opt(args, "--model-dir").unwrap_or("/Volumes/FOUR/kimi");
    let device = parse_device(opt(args, "--device").unwrap_or("cpu"));
    let n_layers: usize = opt(args, "--layers")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n_gen: usize = opt(args, "--gen").and_then(|s| s.parse().ok()).unwrap_or(1);
    let prompt: Vec<u32> = opt(args, "--tokens")
        .unwrap_or("1,100,5000")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let world = peers.len() as u32;
    let mut ck = CheckpointLoader::open(model_dir)?;
    let kc = KimiK3Config::load(Path::new(model_dir).join("config.json"))?;
    let tc = kc.text_config.clone();
    let d = kimi_flow_cfg(&tc, 1).moe;

    // Shard map: assign each routed expert to a WORKER rank; the unassigned complement
    // is served Mac-local (overflow). One rank == one physical compute ENGINE (a GPU,
    // an iGPU pinned by HIP_VISIBLE_DEVICES, a CPU pool, an NPU), so N ranks across the
    // nodes light up N engines concurrently (dispatch_experts fans out to all owners).
    //   --shards "rank:lo-hi,rank:lo-hi,..."   general N-engine fleet
    //   --msi lo-hi --amd lo-hi                legacy 2-worker convenience (default)
    let mut rank_of = vec![ExpertShards::LOCAL; d.num_experts];
    if let Some(spec) = opt(args, "--shards") {
        for part in spec.split(',').filter(|s| !s.is_empty()) {
            let (r, range) = part
                .split_once(':')
                .with_context(|| format!("bad shard {part:?}"))?;
            let (lo, hi) = range
                .split_once('-')
                .with_context(|| format!("bad range {range:?}"))?;
            let (r, lo, hi): (u32, usize, usize) = (r.parse()?, lo.parse()?, hi.parse()?);
            for e in lo..hi.min(d.num_experts) {
                rank_of[e] = r;
            }
        }
    } else {
        let (m_lo, m_hi) = opt(args, "--msi")
            .unwrap_or("0-430")
            .split_once('-')
            .unwrap();
        let (a_lo, a_hi) = opt(args, "--amd")
            .unwrap_or("466-896")
            .split_once('-')
            .unwrap();
        let (m_lo, m_hi): (usize, usize) = (m_lo.parse()?, m_hi.parse()?);
        let (a_lo, a_hi): (usize, usize) = (a_lo.parse()?, a_hi.parse()?);
        for e in m_lo..m_hi {
            rank_of[e] = 1;
        }
        for e in a_lo..a_hi {
            rank_of[e] = 2;
        }
    }
    let local_owned: HashSet<usize> = (0..d.num_experts)
        .filter(|&e| rank_of[e] == ExpertShards::LOCAL)
        .collect();
    let mut per_rank: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for &r in &rank_of {
        if r != ExpertShards::LOCAL {
            *per_rank.entry(r).or_default() += 1;
        }
    }
    let map_str = per_rank
        .iter()
        .map(|(r, c)| format!("rank{r}:{c}"))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!(
        "[expert-run] shard map: {map_str} | Mac-local {}",
        local_owned.len()
    );
    let local = KimiExpertProvider::open(model_dir, d, local_owned, device)?;

    let t: Arc<dyn Transport> = Arc::new(TcpTransport::bind(0, world, peers, 64 << 20)?);
    install_cluster_moe(ClusterMoe {
        transport: t.clone(),
        shards: ExpertShards { rank_of },
        local: Some(local),
    });

    // `--repeat N` runs the generate N times in ONE process (workers + transport stay
    // alive, page cache warms after iter 0) → rigorous A/B: median the `[bench]` backbone
    // over the WARM iters (skip iter 0). Each iter drains its own orch timing.
    let repeat: usize = opt(args, "--repeat")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    for iter in 0..repeat {
        let t0 = std::time::Instant::now();
        let toks = run_generate(
            &mut ck,
            &tc,
            |seq| kimi_flow_cfg(&tc, seq),
            &prompt,
            n_gen,
            n_layers,
            device,
        )?;
        let total_s = t0.elapsed().as_secs_f64();
        let mt = rlx_kimi_k3::dist_experts::orch_timing_take();
        let moe_s = (mt.phase1_ms + mt.dispatch_ms + mt.local_ms + mt.tail_ms) / 1e3;
        let backbone_s = (total_s - moe_s).max(0.0);
        let warm = if iter == 0 { "cold" } else { "warm" };
        eprintln!(
            "[bench] iter={iter} {warm} total={total_s:.2}s backbone={backbone_s:.2}s moe={moe_s:.2}s dispatch={:.2}s tok={toks:?}",
            mt.dispatch_ms / 1e3
        );
    }
    let _ = take_cluster_moe();
    let workers: Vec<u32> = (1..world).collect();
    shutdown_expert_workers(&*t, &workers)?;
    Ok(())
}
