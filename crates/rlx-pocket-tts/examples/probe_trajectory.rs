// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.
//
//! Track how the AR latent trajectory evolves over time. If every step produces
//! a near-identical latent (or near-identical backbone output), the model is
//! "stuck" instead of generating phoneme-varying content.

use anyhow::Result;
use ndarray::Array2;
use rlx_pocket_tts::TtsModel;
use rlx_pocket_tts::tokenizer::prepare_text_prompt;

fn l2(a: &Array2<f32>) -> f32 {
    a.iter().map(|v| v * v).sum::<f32>().sqrt()
}
fn cos(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (l2(a) * l2(b)).max(1e-12)
}

fn main() -> Result<()> {
    let text =
        std::env::var("POCKET_TTS_TEXT").unwrap_or_else(|_| "The cat sat on the mat.".to_string());
    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    let voice_path = rlx_pocket_tts::download::fetch_voice("alba")?;
    let model = TtsModel::open(&assets.weights, &assets.tokenizer)?;
    let voice = model.load_voice(&voice_path)?;

    // Push voice + text into a fresh cache.
    let mut kv = model.flow_lm.transformer.make_cache();
    let _ = model
        .flow_lm
        .transformer
        .forward(voice.conditioning.clone(), &mut kv);
    let (prepped, _) = prepare_text_prompt(&text, true);
    let toks = model.tokenizer.encode(&prepped)?;
    let text_emb = model.flow_lm.embed_tokens(&toks);
    let _ = model.flow_lm.transformer.forward(text_emb, &mut kv);
    eprintln!(
        "text: {prepped:?}  tokens={}  post-text offset={}",
        toks.len(),
        kv.offset
    );

    // AR-decode up to 30 frames and dump per-step c + latent vectors.
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(0);
    let normal = Normal::new(0.0_f32, 0.7_f32.sqrt()).unwrap();

    let mut latents: Vec<Array2<f32>> = Vec::new();
    let mut backbones: Vec<Array2<f32>> = Vec::new();
    let mut eos_logits: Vec<f32> = Vec::new();

    let max_steps = 30usize;
    let ldim = model.flow_lm.ldim();
    let d = model.flow_lm.d_model();
    for step in 0..max_steps {
        let prev = match latents.last() {
            Some(p) => p.clone(),
            None => Array2::<f32>::from_elem((1, ldim), f32::NAN),
        };
        let step_in = model.flow_lm.project_latent(&prev);
        let out = model.flow_lm.transformer.forward(step_in, &mut kv);
        let last: Vec<f32> = (0..d).map(|j| out[[out.shape()[0] - 1, j]]).collect();
        let normed = model
            .flow_lm
            .out_norm(Array2::from_shape_vec((1, d), last)?);
        let eos = model.flow_lm.eos_logit(&normed);
        eos_logits.push(eos);

        // Fresh noise per step, but same RNG sequence.
        let mut x0 = Array2::<f32>::zeros((1, ldim));
        for v in x0.iter_mut() {
            *v = normal.sample(&mut rng);
        }
        let flow = model.flow_lm.flow_net.forward(&normed, 0.0, 1.0, &x0);
        let mut latent = x0.clone();
        for (a, f) in latent.iter_mut().zip(flow.iter()) {
            *a += *f;
        }

        backbones.push(normed.clone());
        latents.push(latent);

        if step < 8 || step % 4 == 0 {
            println!(
                "step {step:>2}  backbone.l2={:.3}  latent.l2={:.3}  eos_logit={:+.3}  latent[0..4]={:?}",
                l2(backbones.last().unwrap()),
                l2(latents.last().unwrap()),
                eos,
                &latents
                    .last()
                    .unwrap()
                    .row(0)
                    .iter()
                    .take(4)
                    .collect::<Vec<_>>(),
            );
        }
    }

    println!();
    println!("cosine between consecutive backbone outputs (1.0 = stuck, low = diverging):");
    for i in 1..backbones.len().min(15) {
        println!(
            "  step {:>2} vs {:>2}: cos(backbone)={:+.4}  cos(latent)={:+.4}",
            i - 1,
            i,
            cos(&backbones[i - 1], &backbones[i]),
            cos(&latents[i - 1], &latents[i]),
        );
    }
    let n = backbones.len();
    println!();
    println!("first vs last:");
    println!(
        "  step  0 vs {:>2}: cos(backbone)={:+.4}  cos(latent)={:+.4}",
        n - 1,
        cos(&backbones[0], &backbones[n - 1]),
        cos(&latents[0], &latents[n - 1])
    );
    Ok(())
}
