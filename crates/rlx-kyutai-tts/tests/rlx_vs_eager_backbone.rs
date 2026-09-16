//! The RLX temporal backbone must agree with the eager reference.
//!
//! **This is the check the existing "parity" suite does not make.**
//! `rlx_backend_parity.rs` compares `temporal_decode_bucketed_rlx` on CPU against *itself*
//! (`assert_logits_match_cpu(label, &cpu, &cpu)`) and then against the same graph on other
//! devices. That is cross-*device* self-consistency: it passes just as happily when the graph
//! is uniformly wrong, which is what shipped — `KyutaiTtsBackend::open` defaults to the RLX
//! backbone, and end-to-end it speaks fluent English that ignores the script entirely
//! ("I can't do this." on repeat) while `RLX_KYUTAI_TTS_EAGER=1` says the prompt.
//!
//! The two `forward_step` implementations take the same arguments and return the same
//! `(sampled_text, hidden)`, so one step from a shared initial state is directly comparable.
//! `hidden` is the backbone output — upstream of the text head — so a mismatch localizes the
//! defect to the transformer stack rather than the projection.

use anyhow::Result;
use ndarray::Array1;
use rlx_kyutai_tts::config::KyutaiTtsConfig;
use rlx_kyutai_tts::model::KyutaiTtsModel;
use rlx_kyutai_tts::rlx_model::RlxKyutaiTtsModel;
use rlx_kyutai_tts::sampling::StreamSampler;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};

fn model_dir() -> Option<PathBuf> {
    let dir = rlx_kyutai_tts::download::default_kyutai_tts_dir();
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(PathBuf::from)
        .unwrap_or(dir);
    dir.join(rlx_kyutai_tts::download::TTS_WEIGHTS_FILE)
        .is_file()
        .then_some(dir)
}

fn cosine(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// One decode step through both backbones from a freshly reset state.
fn step_both(dir: &Path) -> Result<(Array1<f32>, Array1<f32>)> {
    let cfg = KyutaiTtsConfig::v1_6b_en_fr();
    let max_upper = cfg.context;

    let mut eager = KyutaiTtsModel::open(dir, cfg.clone(), Device::Cpu)?;
    let mut rlx = RlxKyutaiTtsModel::open(dir, cfg.clone(), Device::Cpu, max_upper)?;

    // Identical starting conditions: no speaker (cross-attention sees the unconditioned
    // context), same CFG scale, same reset state.
    eager.reset_state();
    rlx.reset_state();
    eager.set_generation_conditions(2.0, None)?;
    rlx.set_generation_conditions(2.0, None)?;

    // The very first step of a real generation: BOS-ish text token, all audio codebooks at the
    // LM's "no token yet" value.
    let text_token = cfg.text_card as u32;
    let audio_delayed = vec![eager.lm_zero_token(); cfg.n_q];

    // Temperature 0 → argmax, so the sampler cannot mask a real divergence with RNG.
    let mut s_eager = StreamSampler::new(0, 0.0, 0.0);
    let mut s_rlx = StreamSampler::new(0, 0.0, 0.0);

    let (_, h_eager) = eager.forward_step(text_token, &audio_delayed, &mut s_eager)?;
    let (_, h_rlx) = rlx.forward_step(text_token, &audio_delayed, &mut s_rlx)?;
    Ok((h_eager, h_rlx))
}

#[test]
fn rlx_backbone_hidden_matches_eager_reference() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: set RLX_KYUTAI_TTS_DIR (or `just fetch-kyutai-tts`)");
        return;
    };
    let (h_eager, h_rlx) = step_both(&dir).expect("one decode step through both backbones");

    assert_eq!(
        h_eager.len(),
        h_rlx.len(),
        "backbone hidden width differs: eager {} vs rlx {}",
        h_eager.len(),
        h_rlx.len()
    );

    let cos = cosine(&h_eager, &h_rlx);
    let max_abs = h_eager
        .iter()
        .zip(&h_rlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Both run on CPU f32 through the same weights, so this is a "same computation" check, not
    // a tolerance-tuning exercise — anything below ~0.999 means the stacks genuinely differ.
    assert!(
        cos > 0.999,
        "RLX backbone diverges from the eager reference: cosine {cos:.6}, max|Δ| {max_abs:.4}\n\
         eager[..8] = {:?}\n  rlx[..8] = {:?}",
        &h_eager.as_slice().unwrap()[..8.min(h_eager.len())],
        &h_rlx.as_slice().unwrap()[..8.min(h_rlx.len())],
    );
}
