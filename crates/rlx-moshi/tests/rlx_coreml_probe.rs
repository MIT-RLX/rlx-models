// Isolate which LM graph construct breaks CoreML's MIL ML Program. Runs each
// variant on CoreML and reports ok/fail so we can pin the unsupported op.

use rlx_moshi::config::{PositionalEmbedding, TransformerConfig};
use rlx_moshi::rlx_lm::{
    HeliumDims, temporal_decode_bucketed_rlx, temporal_decode_step_rlx, temporal_logits_rlx,
};
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

#[test]
fn coreml_lm_variant_probe() {
    if !is_available(Device::Ane) {
        eprintln!("skip: CoreML/ANE not available");
        return;
    }
    let t = cfg();
    let vocab = 10;
    let w = weights(&t, vocab);
    let dims = HeliumDims::from_cfg(&t, vocab);
    let d = t.d_model;
    let emb1 = w["text_emb.weight"].0[5 * d..6 * d].to_vec();
    let emb2 = {
        let mut e = emb1.clone();
        e.extend_from_slice(&w["text_emb.weight"].0[2 * d..3 * d]);
        e
    };
    let dev = Device::Ane;

    let report = |name: &str, r: Result<(), String>| match r {
        Ok(()) => eprintln!("  [ok]   {name}"),
        Err(e) => eprintln!("  [FAIL] {name}: {e}"),
    };

    report(
        "prefill seq=1 (rmsnorm+mm+rope+attn_kind(causal)+swiglu)",
        {
            temporal_logits_rlx(&dims, &w, &emb1, 1, dev)
                .map(|_| ())
                .map_err(|e| format!("{e:#}"))
        },
    );
    report("prefill seq=2 (multi-key causal attn)", {
        temporal_logits_rlx(&dims, &w, &emb2, 2, dev)
            .map(|_| ())
            .map_err(|e| format!("{e:#}"))
    });
    report("decode past=0 (causal, no KV concat)", {
        temporal_decode_step_rlx(&dims, &w, &emb1, &[], 0, dev)
            .map(|_| ())
            .map_err(|e| format!("{e:#}"))
    });
    // Causal decode WITH KV concat (past=1) — isolates concat from custom mask.
    report("decode past=1 (causal + KV concat)", {
        match temporal_decode_step_rlx(&dims, &w, &emb1, &[], 0, dev) {
            Ok((_, _, kv)) => temporal_decode_step_rlx(&dims, &w, &emb1, &kv, 1, dev)
                .map(|_| ())
                .map_err(|e| format!("{e:#}")),
            Err(e) => Err(format!("step0: {e:#}")),
        }
    });
    report("decode bucketed past=0 upper=8 (custom mask, KV concat)", {
        temporal_decode_bucketed_rlx(&dims, &w, &emb1, &[], 0, 8, dev)
            .map(|_| ())
            .map_err(|e| format!("{e:#}"))
    });
}
