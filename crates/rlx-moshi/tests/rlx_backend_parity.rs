// Cross-backend parity: run the LM bucketed temporal-decode graph on every
// available accelerator backend (Metal, MLX, wgpu/Gpu, Vulkan, CoreML/Ane) and
// compare the logits against the CPU reference. A small synthetic model (f32) so
// any divergence is a backend bug, not precision. Unavailable backends are skipped.

use rlx_moshi::config::{PositionalEmbedding, TransformerConfig};
use rlx_moshi::rlx_lm::{HeliumDims, temporal_decode_bucketed_rlx};
use rlx_runtime::{Device, is_available};
use std::collections::HashMap;

fn fill(map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let seed: u32 = key
        .bytes()
        .fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
    let mut s = seed | 1;
    let mut data = Vec::with_capacity(n);
    let is_norm = key.ends_with(".alpha");
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let u = (s >> 8) as f32 / (1u32 << 24) as f32;
        data.push(if is_norm {
            0.9 + 0.2 * u
        } else {
            (u - 0.5) * 0.04
        });
    }
    map.insert(key.to_string(), (data, shape.to_vec()));
}

fn cfg() -> TransformerConfig {
    TransformerConfig {
        d_model: 64,
        num_heads: 4,
        num_layers: 2,
        dim_feedforward: 256,
        causal: true,
        norm_first: true,
        context: 32,
        max_period: 10_000,
        positional_embedding: PositionalEmbedding::Rope,
        kv_repeat: 1,
    }
}

fn weights(t: &TransformerConfig, vocab: usize) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let d = t.d_model;
    let h = t.swiglu_hidden();
    let mut w = HashMap::new();
    fill(&mut w, "text_emb.weight", &[vocab, d]);
    fill(&mut w, "text_linear.weight", &[vocab, d]);
    fill(&mut w, "out_norm.alpha", &[d]);
    for li in 0..t.num_layers {
        let p = format!("transformer.layers.{li}");
        fill(&mut w, &format!("{p}.norm1.alpha"), &[d]);
        fill(&mut w, &format!("{p}.norm2.alpha"), &[d]);
        fill(
            &mut w,
            &format!("{p}.self_attn.in_proj_weight"),
            &[3 * d, d],
        );
        fill(&mut w, &format!("{p}.self_attn.out_proj.weight"), &[d, d]);
        fill(&mut w, &format!("{p}.gating.linear_in.weight"), &[2 * h, d]);
        fill(&mut w, &format!("{p}.gating.linear_out.weight"), &[d, h]);
    }
    w
}

fn cosine(a: &[f32], b: &[f32]) -> (f64, f32) {
    let (mut dot, mut na, mut nb, mut maxd) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
        maxd = maxd.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
        )
        .0
}

#[test]
fn lm_logits_parity_across_backends() {
    let t = cfg();
    let vocab = 10;
    let w = weights(&t, vocab);
    let dims = HeliumDims::from_cfg(&t, vocab);
    let d = t.d_model;
    let emb = w["text_emb.weight"].0[5 * d..6 * d].to_vec();

    let run = |dev: Device| temporal_decode_bucketed_rlx(&dims, &w, &emb, &[], 0, 8, dev);
    let cpu = run(Device::Cpu).expect("cpu reference").0;
    assert!(cpu.iter().all(|x| x.is_finite()));

    let backends = [
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
        (Device::Gpu, "wgpu/Gpu"),
        (Device::Vulkan, "Vulkan"),
        (Device::Ane, "CoreML/ANE"),
    ];

    let mut tested = 0;
    for (dev, name) in backends {
        if !is_available(dev) {
            eprintln!("{name}: skipped (not available / feature off)");
            continue;
        }
        let logits = match run(dev) {
            Ok((l, _, _)) => l,
            Err(e) => panic!("{name}: run failed: {e:#}"),
        };
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "{name}: non-finite logits"
        );
        let (cos, maxd) = cosine(&logits, &cpu);
        eprintln!(
            "{name}: cosine={cos:.6} max|Δ|={maxd:.2e} argmax {} vs cpu {}",
            argmax(&logits),
            argmax(&cpu)
        );
        assert!(
            cos > 0.999,
            "{name}: logits diverge from CPU (cosine {cos})"
        );
        assert_eq!(argmax(&logits), argmax(&cpu), "{name}: argmax mismatch");
        tested += 1;
    }
    eprintln!("backend parity: CPU + {tested} accelerator backend(s) agree");
}
