// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.
//
//! Diagnostic probe — dumps tokenization, embedding stats, and the post-text
//! KV cache size/norm so we can see where text conditioning might be getting
//! dropped between the conditioner and the AR latent loop.

use anyhow::Result;
use rlx_pocket_tts::TtsModel;
use rlx_pocket_tts::tokenizer::prepare_text_prompt;

fn main() -> Result<()> {
    let text = std::env::var("POCKET_TTS_TEXT").unwrap_or_else(|_| "Hello world.".to_string());
    let voice_name = std::env::var("POCKET_TTS_VOICE").unwrap_or_else(|_| "alba".to_string());

    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    let voice_path = rlx_pocket_tts::download::fetch_voice(&voice_name)?;
    let model = TtsModel::open(&assets.weights, &assets.tokenizer)?;
    let voice = model.load_voice(&voice_path)?;
    let d = model.flow_lm.d_model();
    let ldim = model.flow_lm.ldim();

    println!("vocab size           : {}", model.tokenizer.vocab_size());
    println!("voice frames         : {}", voice.num_frames());
    println!("voice embed dim      : {} (d_model={d})", voice.embed_dim());
    println!("ldim (latent)        : {ldim}");
    println!();

    let (prepped, guess) = prepare_text_prompt(&text, true);
    println!("raw text             : {text:?}");
    println!("prepped text         : {prepped:?}  frames_after_eos_guess={guess}");

    let toks = model.tokenizer.encode(&prepped)?;
    println!("tokens               : {:?}", toks);
    let max_tok = toks.iter().copied().max().unwrap_or(0);
    println!("max token id         : {max_tok}");

    let text_emb = model.flow_lm.embed_tokens(&toks);
    println!("text_emb shape       : {:?}", text_emb.shape());
    for (i, tok) in toks.iter().enumerate() {
        let row = text_emb.row(i);
        let l2 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mean = row.iter().sum::<f32>() / d as f32;
        println!("  tok {tok:>4} (#{i}): l2={l2:.3}  mean={mean:+.4}");
    }
    println!();

    // Forward voice through the cache; report cache state.
    let mut kv = model.flow_lm.transformer.make_cache();
    let voice_emb: ndarray::Array2<f32> = voice.conditioning.clone();
    let _ = model.flow_lm.transformer.forward(voice_emb, &mut kv);
    println!(
        "post-voice  offset={}  layer0.k_rows={}",
        kv.offset,
        kv.layers[0].k.shape()[0]
    );

    // Forward text through the same cache; report cache state again.
    let text_emb2 = model.flow_lm.embed_tokens(&toks);
    let _ = model.flow_lm.transformer.forward(text_emb2, &mut kv);
    println!(
        "post-text   offset={}  layer0.k_rows={}",
        kv.offset,
        kv.layers[0].k.shape()[0]
    );

    // Stats on the text K rows (positions 125..125+T_text) vs the voice K rows (0..125).
    let voice_t = voice.num_frames();
    let t_text = toks.len();
    let head_dim = model.flow_lm.transformer.num_layers(); // sentinel; recompute below
    let _ = head_dim;
    let l0 = &kv.layers[0];
    let (kv_t, num_heads, head_dim) = l0.k.dim();
    println!("kv shape             : T={kv_t}  H={num_heads}  D={head_dim}");

    let voice_norm: f32 =
        l0.k.slice(ndarray::s![0..voice_t, .., ..])
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            / (voice_t * num_heads * head_dim) as f32;
    let text_norm: f32 =
        l0.k.slice(ndarray::s![voice_t..voice_t + t_text, .., ..])
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            / (t_text * num_heads * head_dim) as f32;
    println!("layer0 K avg |v| per slot : voice={voice_norm:.4}   text={text_norm:.4}");

    // Sample the first 3 text positions of K (one head), one value each.
    for i in 0..3.min(t_text) {
        let slot = voice_t + i;
        println!(
            "  K[{slot}, head=0, dim=0..4] = {:?}",
            l0.k.slice(ndarray::s![slot, 0, 0..4]).to_vec()
        );
    }
    Ok(())
}
