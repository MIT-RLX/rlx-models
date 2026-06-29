// End-to-end wiring parity: the native-RLX `RlxGenerateState` (temporal decode +
// persistent KV cache + DepFormer) vs the eager `GenerateState`, driven by an
// identical fixed text sequence over several 12.5 Hz frames. Greedy sampling, so
// matching logits ⇒ matching tokens. Validates the frame loop, KV persistence,
// embedding summing, and acoustic-delay bookkeeping (per-op math is validated
// separately in rlx_temporal_parity / rlx_depformer_parity).

use rlx_moshi::config::{
    DepFormerConfig, GenerateConfig, LmConfig, PositionalEmbedding, TransformerConfig,
};
use rlx_moshi::generate::GenerateState;
use rlx_moshi::lm::LmModel;
use rlx_moshi::rlx_gen::{RlxGenerateState, RlxLm};
use rlx_moshi::sampling::LogitsProcessor;
use rlx_runtime::Device;
use std::collections::HashMap;

const TEXT_VOCAB: usize = 10;
const AUDIO_VOCAB: usize = 8;
const D_MAIN: usize = 64;
const D_DEP: usize = 48;
const N_CODEBOOKS: usize = 2;

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

fn lm_cfg() -> LmConfig {
    LmConfig {
        transformer: TransformerConfig {
            d_model: D_MAIN,
            num_heads: 4,
            num_layers: 2,
            dim_feedforward: D_MAIN * 4,
            causal: true,
            norm_first: true,
            context: 64,
            max_period: 10_000,
            positional_embedding: PositionalEmbedding::Rope,
            kv_repeat: 1,
        },
        depformer: Some(DepFormerConfig {
            num_slices: N_CODEBOOKS,
            transformer: TransformerConfig {
                d_model: D_DEP,
                num_heads: 4,
                num_layers: 2,
                dim_feedforward: D_DEP * 4,
                causal: true,
                norm_first: true,
                context: N_CODEBOOKS,
                max_period: 10_000,
                positional_embedding: PositionalEmbedding::None,
                kv_repeat: 1,
            },
        }),
        text_in_vocab_size: TEXT_VOCAB,
        text_out_vocab_size: TEXT_VOCAB,
        audio_vocab_size: AUDIO_VOCAB,
        audio_codebooks: N_CODEBOOKS,
    }
}

fn gen_cfg() -> GenerateConfig {
    GenerateConfig {
        generated_audio_codebooks: N_CODEBOOKS,
        input_audio_codebooks: 0,
        audio_vocab_size: AUDIO_VOCAB,
        acoustic_delay: 1,
        text_pad_token: 3,
        text_eop_token: 0,
        text_start_token: 9,
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
    // DepFormer
    let dep = cfg.depformer.as_ref().unwrap();
    let dd = dep.transformer.d_model;
    let dh = dep.transformer.swiglu_hidden();
    for si in 0..dep.num_slices {
        let pre = format!("depformer.{si}");
        let in_vs = if si == 0 {
            cfg.text_in_vocab_size
        } else {
            cfg.audio_vocab_size
        };
        fill(&mut w, &format!("{pre}.emb.weight"), &[in_vs, dd]);
        fill(&mut w, &format!("{pre}.linear_in.weight"), &[dd, d]);
        fill(
            &mut w,
            &format!("{pre}.linear_out.weight"),
            &[cfg.audio_vocab_size, dd],
        );
        for li in 0..dep.transformer.num_layers {
            let p = format!("{pre}.transformer.layers.{li}");
            fill(&mut w, &format!("{p}.norm1.alpha"), &[dd]);
            fill(&mut w, &format!("{p}.norm2.alpha"), &[dd]);
            fill(
                &mut w,
                &format!("{p}.self_attn.in_proj_weight"),
                &[3 * dd, dd],
            );
            fill(&mut w, &format!("{p}.self_attn.out_proj.weight"), &[dd, dd]);
            fill(
                &mut w,
                &format!("{p}.gating.linear_in.weight"),
                &[2 * dh, dd],
            );
            fill(&mut w, &format!("{p}.gating.linear_out.weight"), &[dd, dh]);
        }
    }
    w
}

fn greedy() -> LogitsProcessor {
    LogitsProcessor::new(0.0, 0, 0)
}

#[test]
fn rlx_gen_matches_eager_gen() {
    let cfg = lm_cfg();
    let gcfg = gen_cfg();
    let weights = synth_weights(&cfg);
    let max_steps = 6;

    let mut eager_lm = LmModel::open(cfg.clone(), weights.clone()).expect("eager lm");
    let mut eager_gs = GenerateState::new(max_steps, greedy(), greedy(), gcfg.clone());

    let mut rlx_lm =
        RlxLm::from_weights(cfg.clone(), weights.clone(), Device::Cpu).expect("rlx lm");
    let mut rlx_gs = RlxGenerateState::new(max_steps, greedy(), greedy(), gcfg.clone());

    let text_seq = [5u32, 2, 7, 1, 4, 6];
    for (step, &tt) in text_seq.iter().enumerate() {
        let et = eager_gs.step(&mut eager_lm, tt, &[]).expect("eager step");
        let rt = rlx_gs.step(&mut rlx_lm, tt, &[]).expect("rlx step");
        assert_eq!(
            et, rt,
            "sampled text mismatch at step {step}: eager={et} rlx={rt}"
        );

        let ef = eager_gs.last_audio_frame();
        let rf = rlx_gs.last_audio_frame();
        assert_eq!(ef, rf, "audio frame mismatch at step {step}");
        eprintln!("step {step}: text={rt} audio_frame={rf:?}");
    }

    assert_eq!(
        eager_gs.text_tokens(),
        rlx_gs.text_tokens(),
        "text token streams differ"
    );
    eprintln!("RLX gen state matches eager over {} steps", text_seq.len());
}
