// RLX — versatile ML compiler + runtime. GPLv3.
//! **(f) `PagedGroupedMoe` as the decode-time MoE engine across REAL layers.**
//! Threads a hidden state through several real DeepSeek-V4-Flash MoE sublayers —
//! per layer: `ffn_norm` RMSNorm → real `ffn.gate` (+bias) routing → the GPU paged
//! grouped-MoE on that layer's active experts → residual. ONE `PagedGroupedMoe`
//! graph is reused across all layers; its (layer, expert)-keyed residency cache
//! pages each layer's active experts in/out. This is the exact per-layer MoE step
//! `V4Decoder` would delegate to; attention/HC/shared-expert are omitted here (they
//! live in the full decoder) so this isolates + exercises the paged MoE engine on
//! genuine weights.
//!
//!   dsv4_paged_moe_stack --ckpt <dir> --layers 3-10 --batch 4 --device metal
//!   (build --features metal)

use anyhow::{Context, Result};
use rlx_ir::quant::QuantScheme;
use rlx_models_core::standard_decoder::{
    DeepseekSpec, PackedExpertSource, PagedGroupedMoe, RopeScaling, dsv4_ref_expert_key,
    paged_moe_route,
};
use rlx_models_core::weight_loader::{MlxLoader, MlxPackedLinear, WeightLoader};
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
fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ckpt = flag(&a, "--ckpt")
        .unwrap_or_else(|| "/Volumes/FOUR/DeepSeek/DeepSeek-V4-Flash-0731-MXFP4-MLX".into());
    let batch = parse(&a, "--batch", 4);
    let dev_s = flag(&a, "--device").unwrap_or_else(|| "metal".into());
    let (l0, l1) = flag(&a, "--layers")
        .and_then(|s| {
            let (a, b) = s.split_once('-')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((3usize, 10usize));
    let layers: Vec<usize> = (l0..=l1).collect();

    let (h, inter, n, top_k, gs) = (4096usize, 2048usize, 256usize, 6usize, 32usize);
    let eps = 1e-6f32;
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
        rms_norm_eps: eps,
    };

    eprintln!(
        "[dsv4-paged-moe-stack] ckpt={ckpt}\n  layers={l0}..{l1} batch={batch} device={dev_s} (attention/HC/shared OMITTED — MoE engine only)"
    );
    let mut loader = MlxLoader::open_lazy(&ckpt).context("open checkpoint")?;

    // Pre-load every layer's ffn_norm + router (weight+bias) BEFORE borrowing the
    // loader for expert paging.
    struct LayerP {
        norm: Vec<f32>,
        router: Vec<f32>,
        bias: Vec<f32>,
    }
    let mut lp: Vec<LayerP> = Vec::new();
    for &il in &layers {
        let (norm, _) = loader.take(&format!("layers.{il}.ffn_norm.weight"))?;
        let (rw, rs) = loader.take(&format!("layers.{il}.ffn.gate.weight"))?;
        let router = if rs == vec![h, n] {
            let mut t = vec![0f32; n * h];
            for i in 0..h {
                for e in 0..n {
                    t[e * h + i] = rw[i * n + e];
                }
            }
            t
        } else {
            rw
        };
        let bias = loader
            .take(&format!("layers.{il}.ffn.gate.bias"))
            .map(|(b, _)| b)
            .unwrap_or_default();
        lp.push(LayerP { norm, router, bias });
    }

    // One reused paged MoE graph; residency pages each layer's experts in/out.
    let a_cap = (batch * top_k).min(n);
    let mut moe = PagedGroupedMoe::new(
        dev(&dev_s),
        a_cap,
        batch * top_k,
        h,
        inter,
        gs,
        spec.swiglu_limit,
        scheme,
    );
    let mut src = PagedMlx {
        loader: &mut loader,
    };

    // Random initial hidden state [B, h].
    let mut hid: Vec<f32> = (0..batch * h).map(|i| rnd(i + 1) * 0.5).collect();
    let mut total_ms = 0f64;
    let mut all_finite = true;
    for (li, &il) in layers.iter().enumerate() {
        // ffn_norm RMSNorm per token.
        let mut hn = vec![0f32; batch * h];
        for b in 0..batch {
            let row = &hid[b * h..b * h + h];
            let r = rms(row) + eps;
            for i in 0..h {
                hn[b * h + i] = row[i] / r * lp[li].norm[i];
            }
        }
        // Real routing (score + bias correction for selection).
        let ebias = (!lp[li].bias.is_empty()).then_some(lp[li].bias.as_slice());
        let routes: Vec<Vec<(usize, f32)>> = (0..batch)
            .map(|b| {
                let (top, w) =
                    paged_moe_route(&spec, &hn[b * h..b * h + h], &lp[li].router, ebias, None);
                top.into_iter().zip(w).collect()
            })
            .collect();
        let active: std::collections::HashSet<usize> =
            routes.iter().flatten().map(|&(e, _)| e).collect();
        let t = Instant::now();
        let moe_out = moe.forward(il, &hn, batch, &routes, &mut src)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        total_ms += ms;
        // Residual.
        for i in 0..batch * h {
            hid[i] += moe_out[i];
        }
        let fin = moe_out.iter().all(|x| x.is_finite()) && hid.iter().all(|x| x.is_finite());
        all_finite &= fin;
        eprintln!(
            "  layer {il:2}: active={:2} moe {ms:6.1} ms, finite={fin}, moe_rms={:.4}, hid_rms={:.4}",
            active.len(),
            rms(&moe_out),
            rms(&hid)
        );
    }
    eprintln!(
        "  → {} layers, {batch} tokens, total MoE {total_ms:.1} ms ({:.1} ms/layer-batch), \
         all_finite={all_finite}, expert uploads={}",
        layers.len(),
        total_ms / layers.len() as f64,
        moe.uploads
    );
    Ok(())
}

fn dev(s: &str) -> Device {
    match s {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cpu" => Device::Cpu,
        _ => Device::Cpu,
    }
}

/// Packed expert source straight off the lazy checkpoint (real Expert Paging).
struct PagedMlx<'a> {
    loader: &'a mut dyn WeightLoader,
}
impl PackedExpertSource for PagedMlx<'_> {
    fn fetch_packed(&mut self, il: usize, e: usize, proj: &str) -> Result<MlxPackedLinear> {
        self.loader
            .take_packed_mlx(&dsv4_ref_expert_key(il, e, proj))?
            .with_context(|| format!("expert {il}.{e}.{proj} not MLX-packed"))
    }
    fn prewarm(&mut self, experts: &[(usize, usize)]) {
        if std::env::var("RLX_NO_PREWARM").is_ok() {
            return;
        }
        let keys: Vec<String> = experts
            .iter()
            .flat_map(|&(il, e)| {
                ["gate_proj", "up_proj", "down_proj"]
                    .into_iter()
                    .map(move |p| dsv4_ref_expert_key(il, e, p))
            })
            .collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        self.loader.prewarm(&refs);
    }
}
