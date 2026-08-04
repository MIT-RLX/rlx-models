// RLX — versatile ML compiler + runtime. GPLv3.
//! **(d) Run the REAL DeepSeek-V4-Flash GA checkpoint through `PagedGroupedMoe`.**
//! Loads one real MoE layer's per-expert MXFP4 weights (`layers.{L}.ffn.experts.
//! {e}.w{1,3,2}`) from the MXFP4-MLX checkpoint, routes a small batch with the real
//! `ffn.gate`, and runs the GPU grouped-MoE path on those weights — validating
//! real-weight parity vs the host S3 path + finiteness + perf. This is the paged
//! MoE on genuine weights WITHOUT loading the full 156 GB model (only the layer's
//! active experts touch memory).
//!
//!   dsv4_real_moe_run --ckpt /Volumes/FOUR/DeepSeek/DeepSeek-V4-Flash-0731-MXFP4-MLX \
//!                     --layer 3 --batch 8 --device metal    # (build --features metal)

use anyhow::{Context, Result};
use rlx_ir::quant::QuantScheme;
use rlx_models_core::standard_decoder::{
    DeepseekSpec, ExpertSource, PackedExpertSource, PagedGroupedMoe, RopeScaling,
    dsv4_ref_expert_key, paged_moe_forward_batched, paged_moe_route,
};
use rlx_models_core::weight_loader::{MlxLoader, MlxPackedLinear, WeightLoader};
use rlx_runtime::Device;
use std::collections::HashMap;
use std::time::Instant;

fn flag(a: &[String], k: &str) -> Option<String> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .cloned()
}
fn parse(a: &[String], k: &str, d: usize) -> usize {
    flag(a, k).and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn rnd(seed: usize) -> f32 {
    ((seed.wrapping_mul(2654435761) % 1000) as f32) / 500.0 - 1.0
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = flag(&a, "--ckpt")
        .unwrap_or_else(|| "/Volumes/FOUR/DeepSeek/DeepSeek-V4-Flash-0731-MXFP4-MLX".into());
    let il = parse(&a, "--layer", 3);
    let batch = parse(&a, "--batch", 8);
    let dev_s = flag(&a, "--device").unwrap_or_else(|| "metal".into());

    // Real GA config (DeepSeek-V4-Flash).
    let (h, inter, n, top_k, gs) = (4096usize, 2048usize, 256usize, 6usize, 32usize);
    let scheme = QuantScheme::MlxMxfp4 {
        group_size: gs as u32,
    };
    let spec = DeepseekSpec {
        vocab_size: 0,
        hidden_size: h,
        num_hidden_layers: 43,
        num_attention_heads: 1,
        q_lora_rank: 0,
        absorbed_mla: false,
        kv_lora_rank: 0,
        qk_nope_head_dim: 0,
        qk_rope_head_dim: 0,
        v_head_dim: 0,
        intermediate_size: inter,
        moe_intermediate_size: inter,
        n_routed_experts: n,
        num_experts_per_tok: top_k,
        n_shared_experts: 1,
        first_k_dense_replace: 0,
        routed_scaling_factor: 1.5,
        norm_topk_prob: true,
        sigmoid_gate: false,
        sqrtsoftplus_gate: true,
        swiglu_limit: 10.0,
        rope_theta: 10000.0,
        rope_scaling: RopeScaling::None,
        attn_score_scale: None,
        rope_neox: true,
        rms_norm_eps: 1e-6,
    };

    eprintln!(
        "[dsv4-real-moe] ckpt={ckpt}\n  layer={il} batch={batch} h={h} inter={inter} experts={n} top_k={top_k} device={dev_s}"
    );
    let mut loader = MlxLoader::open_lazy(&ckpt).context("open checkpoint (lazy)")?;

    // Real router weight for this layer → score-route a synthetic batch.
    let router_key = format!("layers.{il}.ffn.gate.weight");
    let (router_w, rshape) = loader
        .take(&router_key)
        .with_context(|| format!("router {router_key}"))?;
    eprintln!(
        "  router {router_key} shape={rshape:?} ({} f32)",
        router_w.len()
    );
    // paged_moe_route wants router_w as [n, h] row-major (router_w[e*h + i]).
    let router_w = if rshape == vec![h, n] {
        // transpose [h,n] -> [n,h]
        let mut t = vec![0f32; n * h];
        for i in 0..h {
            for e in 0..n {
                t[e * h + i] = router_w[i * n + e];
            }
        }
        t
    } else {
        router_w
    };

    let x: Vec<f32> = (0..batch * h).map(|i| rnd(i + 1) * 0.5).collect();
    let routes: Vec<Vec<(usize, f32)>> = (0..batch)
        .map(|b| {
            let (top, w) = paged_moe_route(&spec, &x[b * h..b * h + h], &router_w, None, None);
            top.into_iter().zip(w).collect()
        })
        .collect();
    let mut active: Vec<usize> = routes.iter().flatten().map(|&(e, _)| e).collect();
    active.sort_unstable();
    active.dedup();
    eprintln!(
        "  routed {batch} tokens → {} distinct active experts",
        active.len()
    );

    // Fetch each active expert's REAL packed w1/w3/w2 ONCE (paging: only these touch
    // memory) into an in-memory bank shared by both paths.
    let t = Instant::now();
    let mut bank: HashMap<usize, [MlxPackedLinear; 3]> = HashMap::new();
    let mut bytes = 0usize;
    for &e in &active {
        let g = loader
            .take_packed_mlx(&dsv4_ref_expert_key(il, e, "gate_proj"))?
            .context("gate")?;
        let u = loader
            .take_packed_mlx(&dsv4_ref_expert_key(il, e, "up_proj"))?
            .context("up")?;
        let d = loader
            .take_packed_mlx(&dsv4_ref_expert_key(il, e, "down_proj"))?
            .context("down")?;
        bytes += g.w_q.len() + u.w_q.len() + d.w_q.len();
        bank.insert(e, [g, u, d]);
    }
    eprintln!(
        "  fetched {} experts ({:.0} MB packed codes) in {:.2}s",
        active.len(),
        bytes as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );

    // Sources over the bank: dequant (host) + packed (GPU).
    struct Bank<'a>(&'a HashMap<usize, [MlxPackedLinear; 3]>, usize);
    let proj_j = |proj: &str| match proj {
        "gate_proj" => 0,
        "up_proj" => 1,
        _ => 2,
    };
    impl ExpertSource for Bank<'_> {
        fn fetch(&mut self, _il: usize, e: usize, proj: &str) -> Result<Vec<f32>> {
            let j = match proj {
                "gate_proj" => 0,
                "up_proj" => 1,
                _ => 2,
            };
            let p = &self.0[&e][j];
            let (out, inn) = (p.out_shape[0], p.out_shape[1]);
            rlx_mlx_io::dequant_mxfp4_f32(&p.w_q, &p.scales, self.1 as u32, out, inn / self.1)
        }
    }
    impl PackedExpertSource for Bank<'_> {
        fn fetch_packed(&mut self, _il: usize, e: usize, proj: &str) -> Result<MlxPackedLinear> {
            let j = match proj {
                "gate_proj" => 0,
                "up_proj" => 1,
                _ => 2,
            };
            Ok(self.0[&e][j].clone())
        }
    }
    let _ = proj_j; // (documentation of the mapping used above)

    // Host S3 (dequant + CPU GEMM) on the REAL weights.
    let t = Instant::now();
    let mut hs = Bank(&bank, gs);
    let host =
        paged_moe_forward_batched(&spec, il, &x, batch, &router_w, None, None, &mut hs, None)?;
    let host_ms = t.elapsed().as_secs_f64() * 1e3;
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    let l2 = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    eprintln!(
        "  [host CPU S3] {host_ms:.1} ms, finite={}, out_rms={:.4}",
        finite(&host),
        l2(&host)
    );

    // GPU grouped path on the REAL weights.
    let dev = match dev_s.as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cpu" => Device::Cpu,
        _ => Device::Cpu,
    };
    let mut moe = PagedGroupedMoe::new(
        dev,
        active.len(),
        batch * top_k,
        h,
        inter,
        gs,
        spec.swiglu_limit,
        scheme,
    );
    let mut ps = Bank(&bank, gs);
    let t = Instant::now();
    let gpu = moe.forward(il, &x, batch, &routes, &mut ps)?; // cold: compile + first upload
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;
    // Warm: residency cache means 0 re-uploads; measures steady-state grouped GEMM.
    let iters = 8usize;
    let t = Instant::now();
    for _ in 0..iters {
        let mut s = Bank(&bank, gs);
        std::hint::black_box(moe.forward(il, &x, batch, &routes, &mut s)?);
    }
    let warm_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!(
        "  [{dev_s} grouped] cold {cold_ms:.1} ms (compile+upload), warm {warm_ms:.1} ms/batch \
         ({:.3} ms/token), finite={}, out_rms={:.4}, uploads={}",
        warm_ms / batch as f64,
        finite(&gpu),
        l2(&gpu),
        moe.uploads
    );

    let err = host
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mag = host.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
    eprintln!(
        "  REAL-WEIGHT parity host vs {dev_s}: rel_err {:e} ({})",
        err / mag,
        if err / mag < 1e-2 { "OK" } else { "MISMATCH" }
    );
    Ok(())
}
