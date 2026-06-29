// Parity: native RLX temporal-transformer graph vs the eager `LmModel`, at small
// synthetic dims (same code path as the 7B — validates the graph math/conventions
// without loading 28 GB of dequantized weights).

use rlx_moshi::config::{LmConfig, PositionalEmbedding, TransformerConfig};
use rlx_moshi::lm::LmModel;
use rlx_moshi::rlx_lm::{
    HeliumDims, temporal_decode_bucketed_rlx, temporal_decode_step_rlx, temporal_logits_rlx,
};
use rlx_runtime::Device;
use std::collections::HashMap;

/// Deterministic pseudo-random small weights, seeded by key + index.
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
        let u = (s >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
        data.push(if is_norm {
            0.9 + 0.2 * u
        } else {
            (u - 0.5) * 0.04
        });
    }
    map.insert(key.to_string(), (data, shape.to_vec()));
}

fn small_cfg() -> LmConfig {
    LmConfig {
        transformer: TransformerConfig {
            d_model: 64,
            num_heads: 4,
            num_layers: 2,
            dim_feedforward: 256, // 4*d_model → swiglu_hidden = 176
            causal: true,
            norm_first: true,
            context: 32,
            max_period: 10_000,
            positional_embedding: PositionalEmbedding::Rope,
            kv_repeat: 1,
        },
        depformer: None,
        text_in_vocab_size: 10,
        text_out_vocab_size: 10,
        audio_vocab_size: 8,
        audio_codebooks: 2,
    }
}

fn synth_weights(cfg: &LmConfig) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let d = cfg.transformer.d_model;
    let h = cfg.transformer.swiglu_hidden();
    let mut w = HashMap::new();
    fill(&mut w, "text_emb.weight", &[cfg.text_in_vocab_size, d]);
    for i in 0..cfg.audio_codebooks {
        fill(
            &mut w,
            &format!("emb.{i}.weight"),
            &[cfg.audio_vocab_size, d],
        );
    }
    fill(&mut w, "text_linear.weight", &[cfg.text_out_vocab_size, d]);
    fill(&mut w, "out_norm.alpha", &[d]);
    for li in 0..cfg.transformer.num_layers {
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
fn temporal_single_token_parity() {
    let cfg = small_cfg();
    let weights = synth_weights(&cfg);

    // Eager reference: step 0 with a text token, no audio.
    let mut eager = LmModel::open(cfg.clone(), weights.clone()).expect("open eager");
    eager.reset_state();
    let text_token = 5u32;
    let audio: Vec<Option<u32>> = vec![None; cfg.audio_codebooks];
    let (eager_logits, _hidden) = eager
        .forward_step(Some(text_token), &audio)
        .expect("eager step");
    let eager_logits = eager_logits.to_vec();

    // RLX graph: feed the same summed embedding (= text_emb row, audio all None).
    let d = cfg.transformer.d_model;
    let emb_row =
        &weights["text_emb.weight"].0[text_token as usize * d..(text_token as usize + 1) * d];
    let dims = HeliumDims::from_cfg(&cfg.transformer, cfg.text_out_vocab_size);
    let rlx_logits =
        temporal_logits_rlx(&dims, &weights, emb_row, 1, Device::Cpu).expect("rlx temporal");

    assert_eq!(rlx_logits.len(), eager_logits.len());
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let (mut na, mut nb) = (0.0f64, 0.0f64);
    for (i, (&r, &e)) in rlx_logits.iter().zip(eager_logits.iter()).enumerate() {
        max_abs = max_abs.max((r - e).abs());
        dot += r as f64 * e as f64;
        na += (r as f64).powi(2);
        nb += (e as f64).powi(2);
        if i < 10 {
            eprintln!("  logit[{i}] rlx={r:+.5} eager={e:+.5} d={:+.2e}", r - e);
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let arg_r = argmax(&rlx_logits);
    let arg_e = argmax(&eager_logits);
    eprintln!("max_abs_diff={max_abs:.3e}  cosine={cos:.8}  argmax rlx={arg_r} eager={arg_e}");
    assert!(cos > 0.9999, "cosine too low: {cos}");
    assert!(max_abs < 1e-3, "max abs diff too high: {max_abs}");
    assert_eq!(arg_r, arg_e, "argmax mismatch");
}

// seq=2 exercises RoPE at position 1 + causal softmax (pos 1 attends to 0 and 1).
#[test]
fn temporal_two_token_rope_attention_parity() {
    let cfg = small_cfg();
    let weights = synth_weights(&cfg);
    let d = cfg.transformer.d_model;
    let v = cfg.text_out_vocab_size;
    let audio: Vec<Option<u32>> = vec![None; cfg.audio_codebooks];
    let toks = [5u32, 2u32];

    // Eager: two streaming steps; keep the step-1 logits (attends over both tokens).
    let mut eager = LmModel::open(cfg.clone(), weights.clone()).expect("open eager");
    eager.reset_state();
    let mut eager_logits = Vec::new();
    for &t in &toks {
        let (lg, _) = eager.forward_step(Some(t), &audio).expect("eager step");
        eager_logits = lg.to_vec();
    }

    // RLX: seq=2 prefill over [emb(tok0), emb(tok1)]; compare position 1.
    let mut emb = Vec::new();
    for &t in &toks {
        emb.extend_from_slice(&weights["text_emb.weight"].0[t as usize * d..(t as usize + 1) * d]);
    }
    let dims = HeliumDims::from_cfg(&cfg.transformer, v);
    let rlx_all = temporal_logits_rlx(&dims, &weights, &emb, 2, Device::Cpu).expect("rlx temporal");
    let rlx_logits = &rlx_all[v..2 * v];

    let mut max_abs = 0.0f32;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&r, &e) in rlx_logits.iter().zip(eager_logits.iter()) {
        max_abs = max_abs.max((r - e).abs());
        dot += r as f64 * e as f64;
        na += (r as f64).powi(2);
        nb += (e as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!(
        "seq2 pos1: max_abs_diff={max_abs:.3e} cosine={cos:.8} argmax rlx={} eager={}",
        argmax(rlx_logits),
        argmax(&eager_logits)
    );
    assert!(cos > 0.9999, "cosine too low: {cos}");
    assert!(max_abs < 1e-3, "max abs diff too high: {max_abs}");
    assert_eq!(argmax(rlx_logits), argmax(&eager_logits), "argmax mismatch");
}

// Streaming KV-decode must equal full-sequence prefill at every position.
#[test]
fn temporal_decode_matches_prefill() {
    let cfg = small_cfg();
    let weights = synth_weights(&cfg);
    let d = cfg.transformer.d_model;
    let v = cfg.text_out_vocab_size;
    let dims = HeliumDims::from_cfg(&cfg.transformer, v);
    let toks = [5u32, 2u32, 7u32];
    let emb_row =
        |t: u32| weights["text_emb.weight"].0[t as usize * d..(t as usize + 1) * d].to_vec();

    // Prefill reference over the full sequence.
    let mut emb = Vec::new();
    for &t in &toks {
        emb.extend_from_slice(&emb_row(t));
    }
    let prefill =
        temporal_logits_rlx(&dims, &weights, &emb, toks.len(), Device::Cpu).expect("prefill");

    // Decode step-by-step with a growing KV cache.
    let mut past_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for (pos, &t) in toks.iter().enumerate() {
        let row = emb_row(t);
        let (logits, _hidden, new_kv) =
            temporal_decode_step_rlx(&dims, &weights, &row, &past_kv, pos, Device::Cpu)
                .expect("decode");
        past_kv = new_kv;
        let pref = &prefill[pos * v..(pos + 1) * v];
        let mut max_abs = 0.0f32;
        for (a, b) in logits.iter().zip(pref.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        eprintln!(
            "decode pos {pos}: max_abs={max_abs:.3e} argmax dec={} pre={}",
            argmax(&logits),
            argmax(pref)
        );
        assert!(
            max_abs < 1e-3,
            "decode/prefill mismatch at pos {pos}: {max_abs}"
        );
        assert_eq!(
            argmax(&logits),
            argmax(pref),
            "argmax mismatch at pos {pos}"
        );
    }
}

// Bucketed Custom-mask decode (padded KV) must equal the Causal decode path.
#[test]
fn temporal_bucketed_matches_causal() {
    let cfg = small_cfg();
    let weights = synth_weights(&cfg);
    let d = cfg.transformer.d_model;
    let v = cfg.text_out_vocab_size;
    let dims = HeliumDims::from_cfg(&cfg.transformer, v);
    let toks = [5u32, 2u32, 7u32];
    let emb_row =
        |t: u32| weights["text_emb.weight"].0[t as usize * d..(t as usize + 1) * d].to_vec();
    let upper = 8; // single fixed bucket > max past (2)

    // Reference: Causal decode path.
    let mut causal_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let mut causal_logits: Vec<Vec<f32>> = Vec::new();
    for (pos, &t) in toks.iter().enumerate() {
        let (lg, _h, kv) =
            temporal_decode_step_rlx(&dims, &weights, &emb_row(t), &causal_kv, pos, Device::Cpu)
                .unwrap();
        causal_kv = kv;
        causal_logits.push(lg);
    }

    // Bucketed Custom-mask path, real KV accumulated from single-token emits.
    let mut buck_kv: Vec<(Vec<f32>, Vec<f32>)> = (0..dims.n_layers)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    for (pos, &t) in toks.iter().enumerate() {
        let (lg, _h, new_kv) = temporal_decode_bucketed_rlx(
            &dims,
            &weights,
            &emb_row(t),
            &buck_kv,
            pos,
            upper,
            Device::Cpu,
        )
        .unwrap();
        for (li, (k, vv)) in new_kv.iter().enumerate() {
            buck_kv[li].0.extend_from_slice(k);
            buck_kv[li].1.extend_from_slice(vv);
        }
        let m = lg
            .iter()
            .zip(causal_logits[pos].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("bucketed pos {pos}: max_abs vs causal={m:.3e}");
        assert!(m < 1e-3, "bucketed/causal mismatch at pos {pos}: {m}");
        assert_eq!(
            argmax(&lg),
            argmax(&causal_logits[pos]),
            "argmax mismatch at pos {pos}"
        );
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}
