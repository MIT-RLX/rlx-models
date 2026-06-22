// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.
//
//! Sanity-check that critical Mimi-decoder weights are loaded with sensible
//! magnitudes (catches silently-missing or zeroed tensors).

use anyhow::Result;
use rlx_pocket_tts::weights::WeightFile;

fn stats(name: &str, wf: &WeightFile) {
    match wf.get_dyn(name) {
        Ok(t) => {
            let flat: Vec<f32> = t.iter().copied().collect();
            let l2 = flat.iter().map(|v| v * v).sum::<f32>().sqrt();
            let mean = flat.iter().sum::<f32>() / (flat.len() as f32);
            let n = flat.len();
            let absmax = flat.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            println!(
                "  {name:<70} shape={:?}  N={n:>7}  l2={l2:>9.3}  mean={mean:+.5}  |max|={absmax:.3}",
                t.shape()
            );
        }
        Err(_) => println!("  {name:<70} MISSING"),
    }
}

fn main() -> Result<()> {
    let assets = rlx_pocket_tts::download::fetch_default_assets()?;
    let wf = WeightFile::open(&assets.weights)?;

    println!("Mimi quantizer + frame-rate resample:");
    stats("mimi.quantizer.output_proj.weight", &wf);
    stats("mimi.upsample.convtr.convtr.weight", &wf);
    stats("mimi.downsample.conv.conv.weight", &wf);
    println!();
    println!("Mimi decoder_transformer layer 0:");
    stats(
        "mimi.decoder_transformer.transformer.layers.0.self_attn.in_proj.weight",
        &wf,
    );
    stats(
        "mimi.decoder_transformer.transformer.layers.0.self_attn.out_proj.weight",
        &wf,
    );
    stats(
        "mimi.decoder_transformer.transformer.layers.0.norm1.weight",
        &wf,
    );
    stats(
        "mimi.decoder_transformer.transformer.layers.0.norm1.bias",
        &wf,
    );
    stats(
        "mimi.decoder_transformer.transformer.layers.0.linear1.weight",
        &wf,
    );
    stats(
        "mimi.decoder_transformer.transformer.layers.0.linear2.weight",
        &wf,
    );
    stats(
        "mimi.decoder_transformer.transformer.layers.0.layer_scale_1.scale",
        &wf,
    );
    stats(
        "mimi.decoder_transformer.transformer.layers.0.layer_scale_2.scale",
        &wf,
    );
    println!();
    println!("Mimi SEANet decoder (entry/exit + a resnet block):");
    stats("mimi.decoder.model.0.conv.weight", &wf);
    stats("mimi.decoder.model.2.convtr.weight", &wf);
    stats("mimi.decoder.model.3.block.1.conv.weight", &wf);
    stats("mimi.decoder.model.11.conv.weight", &wf);
    println!();
    println!("FlowLM emb stats:");
    stats("flow_lm.emb_mean", &wf);
    stats("flow_lm.emb_std", &wf);
    stats("flow_lm.bos_emb", &wf);
    Ok(())
}
