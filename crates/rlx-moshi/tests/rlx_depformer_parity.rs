// Parity: native RLX DepFormer (`depformer_sample_rlx`) vs the eager
// `DepFormer::sample`, at small synthetic dims. Greedy sampling means matching
// logits ⇒ matching tokens, so token equality across all slices validates the
// per-slice graphs + KV inheritance + linear_in/emb/linear_out conditioning.

use ndarray::{Array1, Array2};
use rlx_moshi::config::{DepFormerConfig, PositionalEmbedding, TransformerConfig};
use rlx_moshi::depformer::DepFormer;
use rlx_moshi::nn::{linear, rms_norm};
use rlx_moshi::rlx_lm::{DepDims, depformer_forced_logits_rlx, depformer_run_slice};
use rlx_moshi::sampling::LogitsProcessor;
use rlx_runtime::Device;
use std::collections::HashMap;

const D_MAIN: usize = 64;
const TEXT_VOCAB: usize = 10;
const AUDIO_VOCAB: usize = 8;

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

fn small_dep_cfg() -> DepFormerConfig {
    DepFormerConfig {
        num_slices: 4,
        transformer: TransformerConfig {
            d_model: 48,
            num_heads: 4,
            num_layers: 2,
            dim_feedforward: 48 * 4, // → swiglu_hidden = 132
            causal: true,
            norm_first: true,
            context: 4,
            max_period: 10_000,
            positional_embedding: PositionalEmbedding::None,
            kv_repeat: 1,
        },
    }
}

fn synth_weights(cfg: &DepFormerConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let d = cfg.transformer.d_model;
    let h = cfg.transformer.swiglu_hidden();
    let mut w = HashMap::new();
    for si in 0..cfg.num_slices {
        let pre = format!("depformer.{si}");
        let in_vs = if si == 0 { TEXT_VOCAB } else { AUDIO_VOCAB };
        fill(&mut w, &format!("{pre}.emb.weight"), &[in_vs, d]);
        fill(&mut w, &format!("{pre}.linear_in.weight"), &[d, D_MAIN]);
        fill(
            &mut w,
            &format!("{pre}.linear_out.weight"),
            &[AUDIO_VOCAB, d],
        );
        for li in 0..cfg.transformer.num_layers {
            let p = format!("{pre}.transformer.layers.{li}");
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
    }
    w
}

#[test]
fn depformer_all_slices_parity() {
    let cfg = small_dep_cfg();
    let weights = synth_weights(&cfg);

    // Synthetic temporal hidden state.
    let hidden_vec: Vec<f32> = (0..D_MAIN)
        .map(|i| ((i as f32 * 0.37).sin()) * 0.1)
        .collect();
    let hidden = Array1::from_vec(hidden_vec.clone());
    let text_token = 5u32;
    // Deterministic forced conditioning so eager and RLX feed identical tokens
    // into every slice (removes argmax-flip cascades from the comparison).
    let forced: Vec<u32> = vec![3, 5, 2, 6];

    let mut df =
        DepFormer::build(&cfg, TEXT_VOCAB, AUDIO_VOCAB, D_MAIN, &weights).expect("build dep");
    let dd = DepDims::from_cfg(&cfg, D_MAIN, AUDIO_VOCAB);

    // Per-slice logits with full KV inheritance — the real test.
    let eager = df
        .forced_logits(&hidden, Some(text_token), &forced, true)
        .expect("eager logits");
    let rlx = depformer_forced_logits_rlx(
        &dd,
        &weights,
        &hidden_vec,
        Some(text_token),
        &forced,
        true,
        Device::Cpu,
    )
    .expect("rlx logits");

    assert_eq!(eager.len(), rlx.len(), "slice count");
    let mut worst = 0.0f32;
    for (si, (e, r)) in eager.iter().zip(rlx.iter()).enumerate() {
        let (mut max_abs, mut dot, mut na, mut nb) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in e.iter().zip(r.iter()) {
            max_abs = max_abs.max((a - b).abs());
            dot += a as f64 * b as f64;
            na += (a as f64).powi(2);
            nb += (b as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        worst = worst.max(max_abs);
        eprintln!("slice {si}: max_abs={max_abs:.3e} cosine={cos:.8}");
        assert!(cos > 0.9999, "slice {si} cosine too low: {cos}");
        assert!(max_abs < 1e-3, "slice {si} max abs too high: {max_abs}");
    }
    eprintln!("depformer worst max_abs across slices = {worst:.3e}");

    // Regression guard for the reshape-view-output bug: slice-0's emitted K must
    // equal the hand-computed K projection (it read back as zeros before the fix).
    let mat = |k: &str| {
        let (d, s) = &weights[k];
        Array2::from_shape_vec((s[0], s[1]), d.clone()).unwrap()
    };
    let dmodel = cfg.transformer.d_model;
    let hidden2 = Array2::from_shape_vec((1, D_MAIN), hidden_vec.clone()).unwrap();
    let mut h = linear(hidden2.view(), &mat("depformer.0.linear_in.weight"));
    let emb0 = mat("depformer.0.emb.weight");
    for di in 0..dmodel {
        h[[0, di]] += emb0[[text_token as usize, di]];
    }
    let alpha = Array1::from_vec(
        weights["depformer.0.transformer.layers.0.norm1.alpha"]
            .0
            .clone(),
    );
    let n1 = rms_norm(h.view(), &alpha);
    let qkv = linear(
        n1.view(),
        &mat("depformer.0.transformer.layers.0.self_attn.in_proj_weight"),
    );
    let k_hand: Vec<f32> = (0..dmodel).map(|i| qkv[[0, dmodel + i]]).collect();
    let (_lg, kv0) = depformer_run_slice(
        &dd,
        &weights,
        &hidden_vec,
        0,
        Some(text_token),
        &[],
        Device::Cpu,
    )
    .unwrap();
    let k_err = k_hand
        .iter()
        .zip(kv0[0].0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        k_err < 1e-5,
        "slice-0 emitted K wrong (reshape-view-output regression): {k_err}"
    );

    // Sanity: greedy sampling runs end-to-end and returns num_slices tokens.
    let mut lp = LogitsProcessor::new(0.0, 0, 0);
    let toks = df
        .sample(&hidden, Some(text_token), &[], &mut lp)
        .expect("sample");
    assert_eq!(toks.len(), cfg.num_slices);
}
