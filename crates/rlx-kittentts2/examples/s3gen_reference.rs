// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! S3Gen parity against `scripts/kittentts2_reference.py`, stage by stage:
//! encoder `mu`, both meanflow velocities, the mel, HiFT f0, the deterministic
//! sine source, and the waveform.
//!
//! ```text
//! cargo run -p rlx-kittentts2 --release --example s3gen_reference -- \
//!     weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors \
//!     weights/tts/kitten-tts-2/reference [cpu|metal|…]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_kittentts2::s3gen::{S3Gen, SourceNoise, hift};
use rlx_kittentts2::{S3Conditioning, parse_device, voices::read_npz};

fn report(name: &str, got: &[f32], want: &[f32]) {
    let n = got.len().min(want.len());
    let (mut d, mut m, mut dot, mut ga, mut wa) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (a, b) = (got[i] as f64, want[i] as f64);
        d = d.max((a - b).abs());
        m = m.max(b.abs());
        dot += a * b;
        ga += a * a;
        wa += b * b;
    }
    println!(
        "{name:>12}: len {} vs {}, max|Δ| {d:.3e} (max|ref| {m:.3}), cos {:.8}",
        got.len(),
        want.len(),
        dot / (ga.sqrt() * wa.sqrt()).max(1e-30)
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let s3 = PathBuf::from(args.first().context("s3gen safetensors")?);
    let refdir = PathBuf::from(args.get(1).context("reference dir")?);
    let device = parse_device(args.get(2).map_or("cpu", String::as_str))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refdir.join("reference.json"))?)?;
    let mut z = read_npz(&refdir.join("reference.npz"))?;
    let mut take = |k: &str| z.remove(k).with_context(|| k.to_string())?.into_f32();
    let prompt_token: Vec<u32> = take("cond_prompt_token")?
        .iter()
        .map(|&v| v as u32)
        .collect();
    let prompt_feat = take("cond_prompt_feat")?;
    let cond = S3Conditioning {
        prompt_frames: prompt_feat.len() / 80,
        prompt_token,
        prompt_feat,
        xvector: take("cond_embedding")?,
    };
    let tokens: Vec<u32> = meta["s3gen_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let t0 = Instant::now();
    let g = S3Gen::open(&s3, device)?;
    eprintln!("load {:.2}s", t0.elapsed().as_secs_f32());
    report("spks", &g.speaker(&cond.xvector)?, &take("flow_spks")?);

    let noise = take("flow_noise")?;
    let t0 = Instant::now();
    let tr = g.flow_with_noise(&tokens, &cond, &noise)?;
    eprintln!("flow {:.2}s (incl. compile)", t0.elapsed().as_secs_f32());
    report("mu", &tr.mu, &take("flow_mu")?);
    report("v0", &tr.velocities[0], &take("flow_v0")?);
    report("v1", &tr.velocities[1], &take("flow_v1")?);
    let ref_mel = take("flow_mel")?;
    report("mel", &tr.mel, &ref_mel);

    let t0 = Instant::now();
    let f0 = g.f0(&ref_mel)?;
    report("f0", &f0, &take("hift_f0")?);
    let src = hift::sine_source(g.weights(), &f0, SourceNoise::Off)?;
    let ref_src = take("hift_source")?;
    report("source", &src, &ref_src);
    let wav = g.decode_with_source(&ref_mel, &ref_src)?;
    eprintln!("hift {:.2}s (incl. compile)", t0.elapsed().as_secs_f32());
    report("wav", &wav, &take("hift_wav")?);

    // Warm timing: the same lengths again hit the compiled graphs.
    let t0 = Instant::now();
    let tr = g.flow_with_noise(&tokens, &cond, &noise)?;
    let _ = g.vocode(&tr.mel, SourceNoise::Off)?;
    eprintln!("warm flow+vocode {:.2}s", t0.elapsed().as_secs_f32());

    // A new length recompiles every graph: what one fresh utterance pays.
    let shorter = &tokens[..tokens.len() - 5];
    let noise2 = &noise[..80 * 2 * (cond.prompt_token.len() + shorter.len())];
    let t0 = Instant::now();
    let tr = g.flow_with_noise(shorter, &cond, noise2)?;
    let t1 = Instant::now();
    let _ = g.vocode(&tr.mel, SourceNoise::Off)?;
    eprintln!(
        "new length: flow {:.2}s, vocode {:.2}s",
        (t1 - t0).as_secs_f32(),
        t1.elapsed().as_secs_f32()
    );
    Ok(())
}
