// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.
//
//! Check whether the FlowLM backbone's prediction-position output differs
//! when the text prompt differs. If two very different prompts produce
//! near-identical last-position vectors, text isn't reaching the flow head.

use anyhow::Result;
use ndarray::Array2;
use rlx_pocket_tts::TtsModel;
use rlx_pocket_tts::tokenizer::prepare_text_prompt;

fn run_once(model: &TtsModel, voice_cond: &Array2<f32>, text: &str) -> Result<Vec<f32>> {
    let mut kv = model.flow_lm.transformer.make_cache();
    let _ = model
        .flow_lm
        .transformer
        .forward(voice_cond.clone(), &mut kv);

    let (prepped, _) = prepare_text_prompt(text, true);
    let toks = model.tokenizer.encode(&prepped)?;
    let text_emb = model.flow_lm.embed_tokens(&toks);
    let _ = model.flow_lm.transformer.forward(text_emb, &mut kv);

    // First AR step: feed a single NaN row (→ bos_emb), take the prediction vector.
    let nan_row = Array2::<f32>::from_elem((1, model.flow_lm.ldim()), f32::NAN);
    let step_in = model.flow_lm.project_latent(&nan_row);
    let out = model.flow_lm.transformer.forward(step_in, &mut kv);
    let last: Vec<f32> = out.row(out.shape()[0] - 1).to_vec();
    let normed = model
        .flow_lm
        .out_norm(Array2::from_shape_vec((1, last.len()), last)?);
    Ok(normed.row(0).to_vec())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn l2(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() -> Result<()> {
    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    let voice_path = rlx_pocket_tts::download::fetch_voice("alba")?;
    let model = TtsModel::open(&assets.weights, &assets.tokenizer)?;
    let voice = model.load_voice(&voice_path)?;
    let voice_cond = voice.conditioning.clone();

    let prompts = [
        "Hello world.",
        "The cat sat on the mat.",
        "Goodbye forever.",
        "Hello world.", // same as first — should match exactly
    ];

    let mut outs: Vec<(String, Vec<f32>)> = Vec::new();
    for p in &prompts {
        let v = run_once(&model, &voice_cond, p)?;
        eprintln!("{p:<32}  l2={:.3}  first 8 = {:?}", l2(&v), &v[..8]);
        outs.push((p.to_string(), v));
    }

    println!();
    println!("cosine similarity between prediction vectors:");
    for i in 0..outs.len() {
        for j in (i + 1)..outs.len() {
            let c = cosine(&outs[i].1, &outs[j].1);
            println!(
                "  {:<24}  vs  {:<24}  →  cos = {:+.4}",
                outs[i].0, outs[j].0, c
            );
        }
    }
    Ok(())
}
