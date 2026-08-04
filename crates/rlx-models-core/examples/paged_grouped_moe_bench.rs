// RLX — versatile ML compiler + runtime. GPLv3.
//! **Paged grouped-MoE GPU bench** — times the winning S3 reduce wired onto the
//! native `DequantGroupedMatMulMlx` kernel ([`PagedGroupedMoe`]) against the pure-CPU
//! host path ([`paged_moe_forward_batched`], which dequants each expert to f32 and
//! GEMMs on CPU). Same synthetic MXFP4 per-expert weights + routing for both, so the
//! only difference is WHERE the dequant+matmul runs. Reports ms/token, GFLOP/s,
//! speedup, and parity (host vs device).
//!
//!   paged_grouped_moe_bench --device metal --batch 128    # (build --features metal)
//!   paged_grouped_moe_bench --device cpu   --batch 128     # grouped op on CPU
//!
//! `--device host` runs only the CPU host path (no graph); any other value picks the
//! device the grouped graph compiles for (cpu/metal/mlx/…).

use rlx_ir::quant::QuantScheme;
use rlx_models_core::standard_decoder::{
    DeepseekSpec, ExpertSource, PackedExpertSource, PagedGroupedMoe, RopeScaling,
    paged_moe_forward_batched, paged_moe_route,
};
use rlx_models_core::weight_loader::MlxPackedLinear;
use rlx_runtime::Device;
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let h = parse(&a, "--hidden", 2048);
    let inter = parse(&a, "--inter", 1024);
    let n = parse(&a, "--experts", 64);
    let top_k = parse(&a, "--topk", 8);
    let batch = parse(&a, "--batch", 128);
    let iters = parse(&a, "--iters", 20);
    let gs = parse(&a, "--group", 32);
    let dev_s = flag(&a, "--device").unwrap_or_else(|| "cpu".into());
    let scheme = QuantScheme::MlxMxfp4 {
        group_size: gs as u32,
    };

    let spec = DeepseekSpec {
        vocab_size: 16,
        hidden_size: h,
        num_hidden_layers: 1,
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
        n_shared_experts: 0,
        first_k_dense_replace: 0,
        routed_scaling_factor: 1.0,
        norm_topk_prob: true,
        sigmoid_gate: false,
        sqrtsoftplus_gate: true,
        swiglu_limit: 7.0,
        rope_theta: 10000.0,
        rope_scaling: RopeScaling::None,
        attn_score_scale: None,
        rope_neox: true,
        rms_norm_eps: 1e-6,
    };

    // Synthetic per-expert MXFP4 weights. gate/up: [inter,h]; down: [h,inter].
    let mk = |e: usize, proj: u8, out: usize, inn: usize| -> MlxPackedLinear {
        let ng = inn / gs;
        let w_q: Vec<u8> = (0..out * (inn / 2))
            .map(|i| ((i * 31 + e * 17 + proj as usize * 7 + 3) % 256) as u8)
            .collect();
        let scales: Vec<u8> = (0..out * ng)
            .map(|i| (0x7c + ((i + e) % 6)) as u8)
            .collect();
        MlxPackedLinear {
            w_q,
            scales,
            biases: Vec::new(),
            scheme,
            out_shape: vec![out, inn],
        }
    };
    let bank: Vec<[MlxPackedLinear; 3]> = (0..n)
        .map(|e| [mk(e, 0, inter, h), mk(e, 1, inter, h), mk(e, 2, h, inter)])
        .collect();

    struct PackedBank<'a>(&'a [[MlxPackedLinear; 3]]);
    impl PackedExpertSource for PackedBank<'_> {
        fn fetch_packed(
            &mut self,
            _il: usize,
            e: usize,
            proj: &str,
        ) -> anyhow::Result<MlxPackedLinear> {
            let j = match proj {
                "gate_proj" => 0,
                "up_proj" => 1,
                _ => 2,
            };
            Ok(self.0[e][j].clone())
        }
    }
    struct DequantBank<'a> {
        bank: &'a [[MlxPackedLinear; 3]],
        gs: usize,
    }
    impl ExpertSource for DequantBank<'_> {
        fn fetch(&mut self, _il: usize, e: usize, proj: &str) -> anyhow::Result<Vec<f32>> {
            let j = match proj {
                "gate_proj" => 0,
                "up_proj" => 1,
                _ => 2,
            };
            let p = &self.bank[e][j];
            let (out, inn) = (p.out_shape[0], p.out_shape[1]);
            Ok(
                rlx_mlx_io::dequant_mxfp4_f32(
                    &p.w_q,
                    &p.scales,
                    self.gs as u32,
                    out,
                    inn / self.gs,
                )
                .unwrap(),
            )
        }
    }

    let x: Vec<f32> = (0..batch * h).map(|i| rnd(i + 1) * 0.5).collect();
    let router_w: Vec<f32> = (0..n * h).map(|i| rnd(i + 100)).collect();
    let routes: Vec<Vec<(usize, f32)>> = (0..batch)
        .map(|b| {
            let (top, w) = paged_moe_route(&spec, &x[b * h..b * h + h], &router_w, None, None);
            top.into_iter().zip(w).collect()
        })
        .collect();
    let active: std::collections::HashSet<usize> =
        routes.iter().flatten().map(|&(e, _)| e).collect();
    let flop_tok = top_k as f64 * 3.0 * (h * inter) as f64 * 2.0;

    eprintln!(
        "[paged-grouped-moe] h={h} inter={inter} experts={n} top_k={top_k} batch={batch} \
         active_experts={} device={dev_s}",
        active.len()
    );

    // ── Host CPU path (dequant + GEMM on CPU) ──
    let mut hsrc = DequantBank { bank: &bank, gs };
    let host =
        paged_moe_forward_batched(&spec, 0, &x, batch, &router_w, None, None, &mut hsrc, None)
            .unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let mut s = DequantBank { bank: &bank, gs };
        std::hint::black_box(
            paged_moe_forward_batched(&spec, 0, &x, batch, &router_w, None, None, &mut s, None)
                .unwrap(),
        );
    }
    let host_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!(
        "  [host CPU S3   ] {host_ms:.2} ms/batch, {:.3} ms/token, {:.1} GFLOP/s",
        host_ms / batch as f64,
        flop_tok * batch as f64 / (host_ms / 1e3) / 1e9
    );

    if dev_s == "host" {
        return;
    }

    // ── Device grouped path (dequant + GEMM on-device) ──
    let dev = match dev_s.as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "vulkan" => Device::Vulkan,
        "gpu" => Device::Gpu,
        _ => Device::Cpu,
    };
    let mut moe = PagedGroupedMoe::new(
        dev,
        n,
        batch * top_k,
        h,
        inter,
        gs,
        spec.swiglu_limit,
        scheme,
    );
    let mut psrc = PackedBank(&bank);
    let gpu = moe.forward(0, &x, batch, &routes, &mut psrc).unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let mut s = PackedBank(&bank);
        std::hint::black_box(moe.forward(0, &x, batch, &routes, &mut s).unwrap());
    }
    let dev_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!(
        "  [{dev_s} grouped] {dev_ms:.2} ms/batch, {:.3} ms/token, {:.1} GFLOP/s → {:.2}× vs host \
         ({} expert-proj uploads over {} runs — residency cache)",
        dev_ms / batch as f64,
        flop_tok * batch as f64 / (dev_ms / 1e3) / 1e9,
        host_ms / dev_ms,
        moe.uploads,
        iters + 1,
    );

    let err = host
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mag = host.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
    eprintln!(
        "  parity host vs {dev_s}: rel_err {:e} ({})",
        err / mag,
        if err / mag < 1e-2 { "OK" } else { "MISMATCH" }
    );
}
