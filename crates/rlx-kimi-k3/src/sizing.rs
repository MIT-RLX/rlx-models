// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! How many machines does Kimi-K3 need, and how fast will it go?
//!
//! [`Footprint`] counts every parameter from the config alone (or reads exact
//! byte counts from a checkpoint index), split into the parts that behave
//! differently: the dense backbone every token touches, the routed experts of
//! which only `top_k / num_experts` are touched per token, the embedding (a
//! gather, not a matmul), and the per-sequence state (KDA recurrent state, MLA
//! KV cache). [`estimate`] combines it with a [`Workload`] and a [`NodeSpec`]
//! into per-resource totals, the node count each resource forces, and a
//! roofline per-token time: for each byte or FLOP, the time at the node's
//! achievable rate, the slowest resource being the bottleneck.
//!
//! It is a model, not a measurement. Rates are nominal hardware figures scaled
//! by [`Workload::efficiency`]; routing is assumed uniform (real routing is
//! skewed, which makes an expert cache hit more often than estimated); compute,
//! memory traffic, disk and network are assumed to overlap within a stage.
//! Every number it prints says where it came from.

use crate::config::KimiLinearConfig;
use anyhow::Result;
use rlx_distributed::cluster::sizing::{StepWork, time_step};
use serde::Serialize;

/// Bytes per routed-expert parameter in the checkpoint: MXFP4 = 4-bit codes plus
/// one 8-bit scale per 32 values.
pub const MXFP4_BYTES: f64 = 0.53125;

/// Parameter and state counts of one model config.
#[derive(Debug, Clone, Serialize)]
pub struct Footprint {
    pub layers: usize,
    pub kda_layers: usize,
    pub mla_layers: usize,
    pub moe_layers: usize,
    pub dense_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    /// Embedding table (gathered, not multiplied).
    pub embed_params: u64,
    /// LM head + final norms.
    pub head_params: u64,
    /// Every non-expert, non-embedding weight in the layers.
    pub layer_backbone_params: u64,
    /// One routed expert (w1, w3, w2).
    pub expert_params: u64,
    /// Width of the space routed experts run in (the latent dim of LatentMoE).
    pub expert_latent: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// All routed experts.
    pub experts_total_params: u64,
    /// KDA state floats per layer per sequence (scan + conv windows).
    pub kda_state_floats: u64,
    /// MLA cache floats per layer per token as this crate's decode caches it
    /// (expanded K and V).
    pub mla_kv_floats: u64,
    /// Attention-Residual block size (number of snapshots carried ≈ layers / block).
    pub attn_res_block: usize,
    /// Checkpoint bytes, when read from an index (else estimated: bf16 backbone,
    /// MXFP4 experts).
    pub checkpoint_bytes: f64,
    pub checkpoint_measured: bool,
}

impl Footprint {
    /// Count everything from the config.
    pub fn from_config(tc: &KimiLinearConfig) -> Self {
        let cfg = tc.flow_config(1, 1);
        let h = tc.hidden_size as u64;
        let n = tc.num_hidden_layers;
        let kda = &cfg.kda;
        let (heads, hd, kk) = (
            kda.num_heads as u64,
            kda.head_dim as u64,
            kda.conv_kernel as u64,
        );
        let proj = heads * hd;
        let kda_attn = 4 * h * proj
            + h * hd
            + hd * proj
            + h * heads
            + proj * h
            + 3 * proj * kk
            + proj
            + 2 * hd;
        let m = &cfg.mla;
        let (mh, ql, kvl, nope, rope, vd) = (
            m.num_heads as u64,
            m.q_lora_rank as u64,
            m.kv_lora_rank as u64,
            m.qk_nope_head_dim as u64,
            m.qk_rope_head_dim as u64,
            m.v_head_dim as u64,
        );
        let qk = nope + rope;
        let mla_attn = h * ql
            + ql
            + ql * mh * qk
            + h * (kvl + rope)
            + kvl
            + kvl * mh * (nope + vd)
            + h * mh * vd
            + mh * vd * h;
        let norms = 6 * h;
        let dense = 3 * h * tc.intermediate_size as u64;
        let moe = &cfg.moe;
        let (e, l, mi) = (
            moe.num_experts as u64,
            moe.latent as u64,
            moe.moe_inter as u64,
        );
        let si = moe.num_shared as u64 * mi;
        let moe_dense = h * e + e + 2 * h * l + l + 3 * h * si;
        let expert = 3 * l * mi;
        let (mut kda_n, mut moe_n) = (0usize, 0usize);
        let mut backbone = 0u64;
        for i in 0..n {
            let is_kda = tc.is_kda_layer(i);
            let is_moe = tc.is_moe_layer(i);
            kda_n += usize::from(is_kda);
            moe_n += usize::from(is_moe);
            backbone += norms + if is_kda { kda_attn } else { mla_attn };
            backbone += if is_moe { moe_dense } else { dense };
        }
        let vocab = tc.vocab_size as u64;
        let embed = vocab * h;
        let head = vocab * h + 3 * h;
        let experts_total = expert * e * moe_n as u64;
        Self {
            layers: n,
            kda_layers: kda_n,
            mla_layers: n - kda_n,
            moe_layers: moe_n,
            dense_layers: n - moe_n,
            hidden: tc.hidden_size,
            vocab: tc.vocab_size,
            embed_params: embed,
            head_params: head,
            layer_backbone_params: backbone,
            expert_params: expert,
            expert_latent: moe.latent,
            num_experts: moe.num_experts,
            top_k: moe.top_k,
            experts_total_params: experts_total,
            kda_state_floats: heads * hd * hd + 3 * (kk - 1) * proj,
            mla_kv_floats: mh * (qk + vd),
            attn_res_block: cfg.attn_res_block_size,
            checkpoint_bytes: 2.0 * (backbone + embed + head) as f64
                + MXFP4_BYTES * experts_total as f64,
            checkpoint_measured: false,
        }
    }

    /// Replace the checkpoint-size estimate with the exact bytes in an index's
    /// shard headers.
    pub fn with_checkpoint(mut self, ck: &mut crate::loader::CheckpointLoader) -> Result<Self> {
        let mut total = 0usize;
        for n in ck.tensor_names() {
            total += ck.tensor_meta(&n)?.2;
        }
        self.checkpoint_bytes = total as f64;
        self.checkpoint_measured = true;
        Ok(self)
    }

    /// Dense weights every token touches (layers + head; the embedding is a row gather).
    pub fn dense_params(&self) -> u64 {
        self.layer_backbone_params + self.head_params
    }

    pub fn total_params(&self) -> u64 {
        self.dense_params() + self.embed_params + self.experts_total_params
    }

    /// Parameters a token multiplies through: dense weights + `top_k` experts per MoE layer.
    pub fn active_params(&self) -> u64 {
        self.dense_params() + self.top_k as u64 * self.expert_params * self.moe_layers as u64
    }

    /// Distinct experts a batch of `b` tokens touches per MoE layer (uniform routing).
    pub fn unique_experts(&self, b: f64) -> f64 {
        let (e, k) = (self.num_experts as f64, self.top_k as f64);
        if e == 0.0 {
            return 0.0;
        }
        e * (1.0 - (1.0 - k / e).powf(b))
    }

    /// Boundary state crossing a pipeline cut, bytes per token: hidden + the
    /// Attention-Residual snapshots (up to one per block).
    pub fn boundary_bytes(&self) -> f64 {
        let snaps = self.layers.div_ceil(self.attn_res_block.max(1)) as f64;
        (1.0 + snaps) * self.hidden as f64 * 4.0
    }
}

/// Kimi-K3's text config for sizing without a checkpoint: the dims this crate
/// runs (hidden 7168, 93 layers, 896 experts / 16 active / 2 shared in a 3584
/// latent, MLA ranks 1536 / 512, KDA 96 × 128, vocab 163840). The counts of KDA
/// (69) and MLA (24) layers are exact; *which* layers are which is approximated
/// (every 4th, plus the last, is MLA), which no total here depends on. Size
/// from the real `config.json` when you have it.
pub fn kimi_k3_reference() -> KimiLinearConfig {
    let mla: Vec<usize> = (1..=93).filter(|l| l % 4 == 0 || *l == 93).collect();
    let kda: Vec<usize> = (1..=93).filter(|l| !mla.contains(l)).collect();
    serde_json::from_value(serde_json::json!({
        "vocab_size": 163840,
        "hidden_size": 7168,
        "intermediate_size": 33792,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": true,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "routed_scaling_factor": 2.5,
        "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584,
        "first_k_dense_replace": 1,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 12,
        "linear_attn_config": {
            "kda_layers": kda,
            "full_attn_layers": mla,
            "head_dim": 128,
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0
        }
    }))
    .expect("reference config parses")
}

/// What to run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Mode {
    /// Greedy decode (one token per sequence per step).
    Infer,
    /// LoRA fine-tune (streamed: experts frozen and paged).
    TrainLora { rank: usize },
    /// Full fine-tune of the dense weights (experts frozen), AdamW.
    TrainFull,
}

/// Where routed experts live.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Experts {
    /// In node memory (MXFP4-packed).
    Resident,
    /// On local disk, read per step; `cache_gb` of node memory (total over the
    /// cluster) holds recently used ones.
    Paged { cache_gb: f64 },
}

/// The job being sized.
#[derive(Debug, Clone, Serialize)]
pub struct Workload {
    pub mode: Mode,
    pub experts: Experts,
    pub parallel: Parallel,
    /// Resident precision of the dense weights, bytes per param (bf16 = 2,
    /// int8 = 1, f32 = 4).
    pub dense_bytes: f64,
    /// Context length the KV cache must hold (inference) / sequence length (training).
    pub context: usize,
    /// Concurrent sequences (inference) / sequences per micro-batch (training).
    pub batch: usize,
    /// Fraction of nominal FLOP/s and bandwidth actually achieved (0..1].
    pub efficiency: f64,
}

pub use rlx_distributed::cluster::sizing::{NodeSpec, Parallel, preset, presets};

/// One resource's total requirement and the node count it alone would force.
#[derive(Debug, Clone, Serialize)]
pub struct Requirement {
    pub resource: &'static str,
    pub total: f64,
    pub unit: &'static str,
    pub per_node_capacity: f64,
    pub nodes: usize,
    pub note: String,
}

pub use rlx_distributed::cluster::sizing::StepTime;

/// The result of [`estimate`].
#[derive(Debug, Clone, Serialize)]
pub struct Estimate {
    pub footprint: Footprint,
    pub workload: Workload,
    pub node: NodeSpec,
    pub requirements: Vec<Requirement>,
    /// Nodes needed: the largest of the per-resource counts.
    pub nodes: usize,
    pub limiting: &'static str,
    /// Per-step costs (one decode step for `batch` sequences, or one training
    /// micro-batch of `batch × context` tokens).
    pub flops_per_step: f64,
    pub memory_bytes_per_step: f64,
    pub disk_bytes_per_step: f64,
    pub network_bytes_per_step: f64,
    pub step: StepTime,
    /// Tokens per second over the whole cluster (decode: `batch / step`;
    /// training: `batch·context / step`).
    pub tokens_per_s: f64,
    /// Seconds per token for one sequence (inference).
    pub latency_s: f64,
}

/// Size `w` on nodes of type `node`, at the fewest nodes that hold it.
pub fn estimate(fp: &Footprint, w: &Workload, node: &NodeSpec) -> Estimate {
    estimate_at(fp, w, node, None)
}

/// [`estimate`] at `nodes` nodes (never fewer than the job needs).
pub fn estimate_at(
    fp: &Footprint,
    w: &Workload,
    node: &NodeSpec,
    nodes: Option<usize>,
) -> Estimate {
    const GB: f64 = 1e9;
    let eff = w.efficiency.clamp(0.01, 1.0);
    let b = w.batch.max(1) as f64;
    let t = w.context.max(1) as f64;
    let dense_bytes = fp.dense_params() as f64 * w.dense_bytes;
    let embed_bytes = fp.embed_params as f64 * 2.0;
    let experts_bytes = fp.experts_total_params as f64 * MXFP4_BYTES;
    let expert_bytes = fp.expert_params as f64 * MXFP4_BYTES;

    // ── per-sequence state ──
    let kda_state = fp.kda_layers as f64 * fp.kda_state_floats as f64 * 4.0;
    let kv_per_token = fp.mla_layers as f64 * fp.mla_kv_floats as f64 * 4.0;

    // ── tokens per step ──
    let training = !matches!(w.mode, Mode::Infer);
    let tokens = if training { b * t } else { b };

    // ── memory (RAM + VRAM) ──
    let mut mem = dense_bytes + embed_bytes;
    let mut notes = vec![format!(
        "dense weights {:.1} GB at {} B/param + embedding {:.1} GB",
        dense_bytes / GB,
        w.dense_bytes,
        embed_bytes / GB
    )];
    match w.experts {
        Experts::Resident => {
            mem += experts_bytes;
            notes.push(format!(
                "experts resident {:.0} GB (MXFP4)",
                experts_bytes / GB
            ));
        }
        Experts::Paged { cache_gb } => {
            mem += cache_gb * GB;
            if cache_gb > 0.0 {
                notes.push(format!("expert cache {cache_gb:.0} GB"));
            }
        }
    }
    let lora_params = match w.mode {
        // Every attention / shared / dense projection: rank·(in + out) each,
        // ≈ rank × (sum of in+out) ≈ rank × 2·hidden × projections.
        Mode::TrainLora { rank } => rank as f64 * 2.0 * fp.hidden as f64 * 7.0 * fp.layers as f64,
        _ => 0.0,
    };
    match w.mode {
        Mode::Infer => {
            let s = b * (kda_state + kv_per_token * t);
            mem += s;
            notes.push(format!(
                "state for {} seq × {} ctx: KDA {:.2} GB + MLA KV {:.2} GB",
                w.batch,
                w.context,
                b * kda_state / GB,
                b * kv_per_token * t / GB
            ));
        }
        Mode::TrainFull => {
            // f32 master + grad + Adam m, v for every dense weight (experts frozen).
            let opt = fp.dense_params() as f64 * 16.0;
            mem += opt;
            notes.push(format!(
                "full fine-tune: f32 weights+grads+AdamW {:.0} GB",
                opt / GB
            ));
        }
        Mode::TrainLora { rank } => {
            let opt = lora_params * 16.0;
            mem += opt;
            notes.push(format!(
                "LoRA r{rank}: ~{:.0} M adapter params, {:.2} GB with grads+AdamW",
                lora_params / 1e6,
                opt / GB
            ));
        }
    }
    if training {
        // Streamed backprop: every layer's input (+ the snapshots) kept, one
        // layer's activations live (~166k floats/token measured from the taps,
        // ×3 for forward + backward intermediates).
        let stash = fp.layers as f64 * tokens * fp.boundary_bytes() / 2.0;
        let live = tokens * 166_000.0 * 4.0 * 3.0;
        mem += stash + live;
        notes.push(format!(
            "activations: stash {:.1} GB + one layer live {:.1} GB",
            stash / GB,
            live / GB
        ));
    }

    // ── storage ──
    // Pipeline: each node stores its layers' share (+1 copy of embed/head on the
    // coordinator); expert-parallel: same total, sharded by expert.
    let storage = fp.checkpoint_bytes
        + if w.mode == Mode::Infer {
            0.0
        } else {
            dense_bytes * 2.0
        };

    let usable = node.usable_gb() * GB;
    let n_mem = (mem / usable.max(1.0)).ceil().max(1.0) as usize;
    let n_disk = (storage / (node.disk_gb * GB).max(1.0)).ceil().max(1.0) as usize;
    let mut requirements = vec![
        Requirement {
            resource: "memory (RAM+VRAM)",
            total: mem / GB,
            unit: "GB",
            per_node_capacity: usable / GB,
            nodes: n_mem,
            note: notes.join("; "),
        },
        Requirement {
            resource: "storage",
            total: storage / GB,
            unit: "GB",
            per_node_capacity: node.disk_gb,
            nodes: n_disk,
            note: format!(
                "checkpoint {:.0} GB ({}){}",
                fp.checkpoint_bytes / GB,
                if fp.checkpoint_measured {
                    "measured"
                } else {
                    "estimated"
                },
                if training {
                    " + training state / export"
                } else {
                    ""
                }
            ),
        },
    ];
    let (min_nodes, mut limiting) = if n_mem >= n_disk {
        (n_mem, "memory")
    } else {
        (n_disk, "storage")
    };
    let nodes = match nodes {
        Some(n) if n > min_nodes => {
            limiting = "throughput target";
            n
        }
        _ => min_nodes,
    };
    let nf = nodes as f64;

    // ── per-step work, reported to the generic roofline ──
    // Pipeline micro-batches: `m` groups in flight (training keeps the pipeline
    // full by accumulation).
    let m = if training { nf } else { b.min(nf) };
    let (fmul, passes) = if training { (4.0, 3.0) } else { (1.0, 1.0) };
    let hit = match w.experts {
        Experts::Resident => 1.0,
        Experts::Paged { cache_gb } => (cache_gb * GB / experts_bytes.max(1.0)).min(1.0),
    };
    let work = |tok: f64| -> StepWork {
        let attn = fp.mla_layers as f64
            * 2.0
            * fp.mla_kv_floats as f64
            * if training { t / 2.0 } else { t }
            * tok
            + fp.kda_layers as f64 * 4.0 * fp.kda_state_floats as f64 * tok;
        StepWork {
            // Training: forward + recompute + backward ≈ 4× forward FLOPs and
            // 3 weight passes.
            dense_flops: fmul * (2.0 * fp.dense_params() as f64 * tok + attn),
            expert_flops: fmul
                * 2.0
                * (fp.top_k as u64 * fp.expert_params * fp.moe_layers as u64) as f64
                * tok,
            dense_bytes: dense_bytes * passes
                + if training {
                    0.0
                } else {
                    tok * (2.0 * kda_state + kv_per_token * t)
                },
            expert_bytes: fp.unique_experts(tok) * expert_bytes * fp.moe_layers as f64 * passes,
            expert_in_memory: hit,
            boundary_bytes_per_token: fp.boundary_bytes() * if training { 2.0 } else { 1.0 },
            // Each token's k routed-latent vectors go to their owners and back
            // per MoE layer.
            alltoall_bytes_per_token: fp.moe_layers as f64
                * fp.top_k as f64
                * fp.expert_latent as f64
                * 4.0
                * 2.0
                * passes,
            moe_layers: fp.moe_layers,
        }
    };
    let step = time_step(&work, tokens, m, mem / nf, node, nodes, w.parallel, eff);
    let (flops, mem_traffic, disk, net) = (
        step.flops,
        step.memory_bytes,
        step.disk_bytes,
        step.network_bytes,
    );
    let uniq = fp.unique_experts(tokens / m);
    let tokens_per_s = step.tokens_per_s;
    let latency_s = step.latency_s;
    requirements.push(Requirement {
        resource: "network",
        total: net / tokens / 1e6,
        unit: "MB/token",
        per_node_capacity: node.net_gbps,
        nodes: 1,
        note: match w.parallel {
            Parallel::Pipeline => format!(
                "pipeline boundary {:.0} KB/token per hop",
                fp.boundary_bytes() / 1e3
            ),
            Parallel::ExpertParallel => "expert all-to-all dispatch + combine".into(),
        },
    });
    requirements.push(Requirement {
        resource: "disk read",
        total: disk / tokens / GB,
        unit: "GB/token",
        per_node_capacity: node.disk_read_gbs,
        nodes: 1,
        note: match w.experts {
            Experts::Resident => "experts resident: no paging".into(),
            Experts::Paged { .. } => format!(
                "{:.1} distinct experts/layer/step × {:.1} MB each{}",
                uniq,
                expert_bytes / 1e6,
                if training { " × 3 passes" } else { "" }
            ),
        },
    });
    requirements.push(Requirement {
        resource: "compute",
        total: flops / tokens / 1e9,
        unit: "GFLOP/token",
        per_node_capacity: if node.gpu_tflops > 0.0 {
            node.gpu_tflops
        } else {
            node.cpu_tflops
        },
        nodes: 1,
        note: format!(
            "{:.1} B active params/token{}",
            fp.active_params() as f64 / 1e9,
            if training { ", ×4 for train" } else { "" }
        ),
    });

    Estimate {
        footprint: fp.clone(),
        workload: w.clone(),
        node: node.clone(),
        requirements,
        nodes,
        limiting,
        flops_per_step: flops,
        memory_bytes_per_step: mem_traffic,
        disk_bytes_per_step: disk,
        network_bytes_per_step: net,
        step,
        tokens_per_s,
        latency_s,
    }
}

/// The smallest node count (from what the job needs up to `max`) whose
/// throughput reaches `target` tokens/s, or `None`. In a pipeline more nodes add
/// throughput only with more sequences than stages in flight; one sequence's
/// token time does not shrink. Expert parallelism spreads each token's expert
/// reads and compute over every node.
pub fn nodes_for_throughput(
    fp: &Footprint,
    w: &Workload,
    node: &NodeSpec,
    target: f64,
    max: usize,
) -> Option<Estimate> {
    let base = estimate(fp, w, node);
    (base.nodes..=max.max(base.nodes))
        .map(|n| estimate_at(fp, w, node, Some(n)))
        .find(|e| e.tokens_per_s >= target)
}

impl std::fmt::Display for Estimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fp = &self.footprint;
        let w = &self.workload;
        writeln!(
            f,
            "model: {} layers ({} KDA, {} MLA; {} MoE, {} dense), {:.1} B params total, {:.1} B active/token",
            fp.layers,
            fp.kda_layers,
            fp.mla_layers,
            fp.moe_layers,
            fp.dense_layers,
            fp.total_params() as f64 / 1e9,
            fp.active_params() as f64 / 1e9
        )?;
        writeln!(
            f,
            "  dense {:.1} B · embedding {:.2} B · experts {:.1} B ({} × {:.1} M, {} active)",
            fp.dense_params() as f64 / 1e9,
            fp.embed_params as f64 / 1e9,
            fp.experts_total_params as f64 / 1e9,
            fp.num_experts,
            fp.expert_params as f64 / 1e6,
            fp.top_k
        )?;
        writeln!(
            f,
            "  per sequence: KDA state {:.0} MB, MLA KV {:.0} KB/token · pipeline boundary {:.0} KB/token",
            fp.kda_layers as f64 * fp.kda_state_floats as f64 * 4.0 / 1e6,
            fp.mla_layers as f64 * fp.mla_kv_floats as f64 * 4.0 / 1e3,
            fp.boundary_bytes() / 1e3
        )?;
        writeln!(
            f,
            "workload: {:?}, experts {:?}, {:?}, dense at {} B/param, context {}, batch {}, efficiency {:.0}%",
            w.mode,
            w.experts,
            w.parallel,
            w.dense_bytes,
            w.context,
            w.batch,
            w.efficiency * 100.0
        )?;
        let n = &self.node;
        writeln!(
            f,
            "node: {} — RAM {:.0} GB @ {:.0} GB/s{}, disk {:.0} GB @ {:.1} GB/s, net {:.0} Gb/s, {:.0} TFLOP/s",
            n.name,
            n.ram_gb,
            n.ram_bw_gbs,
            if n.vram_gb > 0.0 {
                format!(", VRAM {:.0} GB @ {:.0} GB/s", n.vram_gb, n.vram_bw_gbs)
            } else {
                String::new()
            },
            n.disk_gb,
            n.disk_read_gbs,
            n.net_gbps,
            if n.gpu_tflops > 0.0 {
                n.gpu_tflops
            } else {
                n.cpu_tflops
            }
        )?;
        writeln!(
            f,
            "\n{:<18} {:>20} {:>16} {:>6}  note",
            "resource", "needed", "per node", "nodes"
        )?;
        for r in &self.requirements {
            let nodes = if r.unit == "GB" {
                r.nodes.to_string()
            } else {
                "—".into()
            };
            let cap_unit = match r.unit {
                "GB" => "GB",
                "MB/token" => "Gb/s",
                "GB/token" => "GB/s",
                _ => "TFLOP/s",
            };
            writeln!(
                f,
                "{:<18} {:>10.2} {:<9} {:>8.1} {:<7} {:>6}  {}",
                r.resource, r.total, r.unit, r.per_node_capacity, cap_unit, nodes, r.note
            )?;
        }
        writeln!(
            f,
            "\n⇒ {} node(s) of {} (set by {})",
            self.nodes, n.name, self.limiting
        )?;
        let s = &self.step;
        writeln!(
            f,
            "  per step: compute {:.3}s · memory {:.3}s · disk {:.3}s · network {:.3}s → {:.3}s, bound by {}",
            s.compute_s, s.memory_s, s.disk_s, s.network_s, s.total_s, s.bottleneck
        )?;
        match w.mode {
            Mode::Infer => writeln!(
                f,
                "  ≈ {:.2} tokens/s total ({} seq), {:.3} s/token per sequence",
                self.tokens_per_s, w.batch, s.total_s
            ),
            _ => writeln!(
                f,
                "  ≈ {:.0} training tokens/s ({} × {} per micro-batch, {:.1} s each)",
                self.tokens_per_s, w.batch, w.context, s.total_s
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn infer(experts: Experts) -> Workload {
        Workload {
            mode: Mode::Infer,
            experts,
            parallel: Parallel::Pipeline,
            dense_bytes: 2.0,
            context: 4096,
            batch: 1,
            efficiency: 0.6,
        }
    }

    #[test]
    fn reference_footprint_matches_the_known_checkpoint() {
        let fp = Footprint::from_config(&kimi_k3_reference());
        assert_eq!((fp.kda_layers, fp.mla_layers, fp.moe_layers), (69, 24, 92));
        // Checkpoint as recorded when it was on disk: ~114 GB bf16 backbone
        // (embedding + head included), ~1.45 TB MXFP4 experts.
        let backbone_gb = 2.0 * (fp.dense_params() + fp.embed_params) as f64 / 1e9;
        let experts_tb = fp.experts_total_params as f64 * MXFP4_BYTES / 1e12;
        assert!(
            (105.0..125.0).contains(&backbone_gb),
            "backbone {backbone_gb} GB"
        );
        assert!(
            (1.40..1.50).contains(&experts_tb),
            "experts {experts_tb} TB"
        );
        let total_t = fp.total_params() as f64 / 1e12;
        assert!((2.6..2.9).contains(&total_t), "{total_t} T params");
    }

    #[test]
    fn estimates_move_the_right_way() {
        let fp = Footprint::from_config(&kimi_k3_reference());
        let mini = preset("mac-mini-m4pro-64").unwrap();
        let studio = preset("mac-studio-m3ultra-512").unwrap();
        let paged = infer(Experts::Paged { cache_gb: 0.0 });
        let resident = infer(Experts::Resident);
        let (pm, ps) = (estimate(&fp, &paged, &mini), estimate(&fp, &paged, &studio));
        assert!(ps.nodes < pm.nodes, "more RAM per node, fewer nodes");
        let rm = estimate(&fp, &resident, &mini);
        assert!(
            rm.nodes > pm.nodes,
            "resident experts need more nodes than paged"
        );
        assert_eq!(pm.step.bottleneck, "disk", "uncached paging is disk-bound");
        assert!(
            rm.tokens_per_s > pm.tokens_per_s,
            "resident experts are faster"
        );
        // More concurrent sequences than stages fill the pipeline.
        let mut busy = resident.clone();
        busy.batch = rm.nodes * 2;
        assert!(estimate(&fp, &busy, &mini).tokens_per_s > rm.tokens_per_s);
        // A throughput target the minimum cluster misses is met with more nodes
        // in expert-parallel paging (more disks reading at once).
        let mut ep = paged.clone();
        ep.parallel = Parallel::ExpertParallel;
        let base = estimate(&fp, &ep, &mini);
        let want = base.tokens_per_s * 1.5;
        let more = nodes_for_throughput(&fp, &ep, &mini, want, 200).expect("reachable");
        assert!(more.nodes > base.nodes && more.tokens_per_s >= want);
        // Full fine-tuning needs far more memory than LoRA.
        let mut full = paged.clone();
        full.mode = Mode::TrainFull;
        full.context = 512;
        let mut lora = full.clone();
        lora.mode = Mode::TrainLora { rank: 16 };
        assert!(estimate(&fp, &full, &mini).nodes > 4 * estimate(&fp, &lora, &mini).nodes);
    }
}
