// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.
//
//! Probe the flow head: with identical noise initial state, vary `c` (the
//! backbone conditioning) and report how much the produced latent moves. If
//! latents barely respond to `c`, the flow head isn't conditioning correctly.

use anyhow::Result;
use ndarray::Array2;
use rlx_pocket_tts::TtsModel;
use rlx_pocket_tts::tokenizer::prepare_text_prompt;

fn run_backbone(model: &TtsModel, voice_cond: &Array2<f32>, text: &str) -> Result<Array2<f32>> {
    let mut kv = model.flow_lm.transformer.make_cache();
    let _ = model
        .flow_lm
        .transformer
        .forward(voice_cond.clone(), &mut kv);
    let (prepped, _) = prepare_text_prompt(text, true);
    let toks = model.tokenizer.encode(&prepped)?;
    let text_emb = model.flow_lm.embed_tokens(&toks);
    let _ = model.flow_lm.transformer.forward(text_emb, &mut kv);

    let nan_row = Array2::<f32>::from_elem((1, model.flow_lm.ldim()), f32::NAN);
    let step_in = model.flow_lm.project_latent(&nan_row);
    let out = model.flow_lm.transformer.forward(step_in, &mut kv);
    let d = model.flow_lm.d_model();
    let last_row: Vec<f32> = (0..d).map(|j| out[[out.shape()[0] - 1, j]]).collect();
    let last_2d = Array2::from_shape_vec((1, d), last_row)?;
    Ok(model.flow_lm.out_norm(last_2d))
}

fn flow_step(model: &TtsModel, c: &Array2<f32>, fixed_x0: &Array2<f32>) -> Array2<f32> {
    // 1-step Euler at s=0, t=1.
    let flow = model.flow_lm.flow_net.forward(c, 0.0, 1.0, fixed_x0);
    let mut out = fixed_x0.clone();
    for i in 0..flow.shape()[0] {
        for j in 0..flow.shape()[1] {
            out[[i, j]] += flow[[i, j]];
        }
    }
    out
}

fn l2(a: &Array2<f32>) -> f32 {
    a.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn cosine(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (l2(a) * l2(b)).max(1e-12)
}

fn main() -> Result<()> {
    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    let voice_path = rlx_pocket_tts::download::fetch_voice("alba")?;
    let model = TtsModel::open(&assets.weights, &assets.tokenizer)?;
    let voice = model.load_voice(&voice_path)?;

    // FIXED noise — same for every test.
    let ldim = model.flow_lm.ldim();
    let std = (0.7_f32).sqrt();
    let mut x0 = Array2::<f32>::zeros((1, ldim));
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0_f32, std).unwrap();
    for v in x0.iter_mut() {
        *v = normal.sample(&mut rng);
    }
    println!("fixed x0  l2={:.3}", l2(&x0));
    println!();

    let prompts = [
        "Hello world.",
        "The cat sat on the mat.",
        "Goodbye forever.",
        "Hello world.",
    ];
    let mut latents: Vec<(String, Array2<f32>)> = Vec::new();
    let mut conds: Vec<Array2<f32>> = Vec::new();

    for p in &prompts {
        let c = run_backbone(&model, &voice.conditioning, p)?;
        let latent = flow_step(&model, &c, &x0);
        eprintln!(
            "{p:<32}  c.l2={:.3}  latent.l2={:.3}  Δ_from_x0.l2={:.3}  first 8 latent = {:?}",
            l2(&c),
            l2(&latent),
            {
                let mut d = latent.clone();
                for (a, b) in d.iter_mut().zip(x0.iter()) {
                    *a -= *b;
                }
                l2(&d)
            },
            latent.row(0).iter().take(8).collect::<Vec<_>>()
        );
        latents.push((p.to_string(), latent));
        conds.push(c);
    }

    println!();
    println!("cosine similarity between latents (same noise, different c):");
    for i in 0..latents.len() {
        for j in (i + 1)..latents.len() {
            let c = cosine(&latents[i].1, &latents[j].1);
            let dc = cosine(&conds[i], &conds[j]);
            println!(
                "  {:<24}  vs  {:<24}  →  cos(latent)={:+.4}  cos(c)={:+.4}",
                latents[i].0, latents[j].0, c, dc
            );
        }
    }
    Ok(())
}
