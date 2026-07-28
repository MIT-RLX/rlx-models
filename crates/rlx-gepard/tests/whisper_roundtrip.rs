// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: Apache-2.0

//! Whisper round-trip for Gepard: synthesise a sentence and verify speech.
//!
//! Needs `weights/tts/gepard`, `nano_dec_1.89kbps.safetensors`, and Whisper Tiny
//! (`.cache/whisper-tiny`). Run via `just gepard-whisper`.

use std::path::{Path, PathBuf};

fn gepard_bundle() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/gepard");
    if !p.join("model.safetensors").is_file() || !p.join("tokenizer.json").is_file() {
        return None;
    }
    if !p.join("nano_dec_1.89kbps.safetensors").is_file() {
        return None;
    }
    Some(p)
}

fn whisper_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-tiny"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/whisper"),
        PathBuf::from(".cache/whisper-tiny"),
    ];
    candidates
        .into_iter()
        .find(|d| d.join("model.safetensors").is_file() && d.join("tokenizer.json").is_file())
}

fn whisper_coverage(hyp: &str, want: &[&str]) -> usize {
    let lower = hyp.to_ascii_lowercase();
    want.iter().filter(|w| lower.contains(*w)).count()
}

fn synthesize_and_whisper(
    bundle: &Path,
    wd: &Path,
    text: &str,
    device: &str,
) -> (String, usize, usize) {
    use rlx_gepard::{GepardSynthesizer, InferOpts, default_seed_for_text};
    use rlx_runtime::Device;
    let want = fox_words();
    // CPU compiled AR in cargo tests — MLX AR + test harness often SIGKILLs;
    // CLI `just gepard-demo DEVICE=mlx` validates on-device AR.
    let audio = {
        let synth = GepardSynthesizer::open_with_compiled(bundle, Device::Cpu, true)
            .expect("open gepard compiled")
            .with_opts(InferOpts {
                seed: default_seed_for_text(text),
                ..Default::default()
            });
        let _ = device; // NanoCodec device exercised via CLI / backend_matrix
        synth.synthesize(text, "").expect("gepard synthesis")
    };
    assert!(!audio.is_empty());
    assert!(audio.iter().any(|v| v.abs() > 1e-3), "near-silent audio");
    let pcm16 = resample_linear(&audio, 22_050, 16_000);
    use rlx_whisper::WhisperRunner;
    let mut session = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .build()
        .expect("build whisper session");
    let transcript = session
        .transcribe_greedy(&pcm16)
        .expect("whisper transcription");
    let hits = whisper_coverage(&transcript, &want);
    (transcript, hits, want.len())
}

fn fox_words() -> Vec<&'static str> {
    vec!["quick", "brown", "fox", "jumps", "lazy", "dog"]
}

fn long_words() -> Vec<&'static str> {
    vec![
        "quick", "brown", "fox", "jumps", "lazy", "dog", "courage", "kindness", "matter", "people",
        "hard", "times", "help", "each", "other",
    ]
}

const FOX: &str = "The quick brown fox jumps over the lazy dog.";

const LONG: &str = "The quick brown fox jumps over the lazy dog. Courage and kindness matter more than cleverness alone when people face hard times together and choose to help each other without waiting for perfect conditions.";

#[test]
fn test_gepard_whisper_fox() {
    let bundle = match gepard_bundle() {
        Some(p) => p,
        None => {
            eprintln!("skipping: missing gepard bundle or nano_dec weights");
            return;
        }
    };
    let wd = match whisper_dir() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Whisper weights");
            return;
        }
    };
    let (transcript, hits, n) = synthesize_and_whisper(&bundle, &wd, FOX, "metal");
    eprintln!("[gepard whisper fox] {hits}/{n} {transcript:?}");
    assert_eq!(hits, n, "fox: expected {n}/{n} words in {transcript:?}");
}

#[test]
fn test_gepard_compiled_cpu_prefill_parity() {
    use rlx_gepard::backbone::{BackboneWeights, GepardKvCache, backbone_prefill};
    use rlx_gepard::compiled_session::GepardCompiledSession;
    use rlx_gepard::config::GepardConfig;
    use rlx_gepard::tokenizer::GepardTokenizer;
    use rlx_runtime::Device;

    let bundle = match gepard_bundle() {
        Some(p) => p,
        None => {
            eprintln!("skipping: missing gepard bundle");
            return;
        }
    };
    let cfg = GepardConfig::from_path(&bundle).expect("config");
    let tok = GepardTokenizer::load(&bundle, &cfg).expect("tokenizer");
    let bytes = std::fs::read(bundle.join("model.safetensors")).expect("read");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse");
    let backbone = BackboneWeights::load(&st, &cfg.backbone).expect("backbone");
    let ids = tok.build_prompt_ids(FOX).expect("tokenize");
    let embeds = backbone.embed_tokens(&ids);
    let hidden = cfg.hidden_size();
    let mut kv = GepardKvCache::new(cfg.backbone.num_hidden_layers);
    let all_h = backbone_prefill(&embeds, ids.len(), &backbone, &mut kv);
    let eager_sos = &all_h[(ids.len() - 1) * hidden..ids.len() * hidden];

    let mut session = GepardCompiledSession::new(Device::Cpu, &cfg, &bundle).expect("compiled");
    let (compiled_sos, _) = session.prefill_hidden(&embeds, ids.len()).expect("prefill");
    let cos = cosine_f32(eager_sos, &compiled_sos);
    eprintln!("[gepard compiled parity] prefill SOS cosine={cos:.6}");
    assert!(cos > 0.99, "eager vs compiled prefill hidden cosine {cos}");
}

#[test]
fn test_eager_vs_compiled_decode_step() {
    use rlx_gepard::backbone::{
        BackboneWeights, GepardKvCache, backbone_decode_step, backbone_prefill,
    };
    use rlx_gepard::compiled_session::GepardCompiledSession;
    use rlx_gepard::config::GepardConfig;
    use rlx_gepard::synthesis::{embed_audio_frame, sample_all_heads_temp};
    use rlx_gepard::tokenizer::GepardTokenizer;
    use rlx_gepard::weights::GepardOverlay;
    use rlx_runtime::Device;

    let bundle = match gepard_bundle() {
        Some(p) => p,
        None => return,
    };
    let cfg = GepardConfig::from_path(&bundle).expect("config");
    let tok = GepardTokenizer::load(&bundle, &cfg).expect("tokenizer");
    let bytes = std::fs::read(bundle.join("model.safetensors")).expect("read");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse");
    let backbone = BackboneWeights::load(&st, &cfg.backbone).expect("backbone");
    let overlay = GepardOverlay::load(&st, cfg.num_audio_heads()).expect("overlay");
    let ids = tok.build_prompt_ids(FOX).expect("tokenize");
    let embeds = backbone.embed_tokens(&ids);
    let hidden = cfg.hidden_size();

    let mut kv = GepardKvCache::new(cfg.backbone.num_hidden_layers);
    let all_h = backbone_prefill(&embeds, ids.len(), &backbone, &mut kv);
    let sos = &all_h[(ids.len() - 1) * hidden..ids.len() * hidden];

    fastrand::seed(54);
    let vocabs = cfg.codec.channel_vocabs();
    let first = sample_all_heads_temp(sos, &overlay, &vocabs, 0.4, 1.0, None, 32);
    let frame_emb = embed_audio_frame(&first, &overlay, cfg.audio_embed_dim, hidden);

    let eager_h = backbone_decode_step(&frame_emb, &backbone, &mut kv);

    let mut session = GepardCompiledSession::new(Device::Cpu, &cfg, &bundle).expect("compiled");
    let (_sos_c, mut cache) = session.prefill_hidden(&embeds, ids.len()).expect("prefill");
    let compiled_h = session
        .decode_hidden(&mut cache, &frame_emb)
        .expect("decode");

    let cos = cosine_f32(&eager_h, &compiled_h);
    eprintln!("[gepard compiled parity] decode step-1 cosine={cos:.6}");
    assert!(cos > 0.99, "eager vs compiled decode hidden cosine {cos}");

    // Multi-step: bucketed KV must keep the new row at `upper` (not the pad).
    let mut eager_h = eager_h;
    let mut min_cos = cos;
    for step in 2..=12 {
        fastrand::seed(54 + step as u64);
        let codes = sample_all_heads_temp(&eager_h, &overlay, &vocabs, 0.4, 1.0, None, 32);
        let frame_emb = embed_audio_frame(&codes, &overlay, cfg.audio_embed_dim, hidden);
        eager_h = backbone_decode_step(&frame_emb, &backbone, &mut kv);
        let compiled_h = session
            .decode_hidden(&mut cache, &frame_emb)
            .expect("decode");
        let c = cosine_f32(&eager_h, &compiled_h);
        min_cos = min_cos.min(c);
    }
    eprintln!("[gepard compiled parity] decode steps 1..12 min cosine={min_cos:.6}");
    assert!(
        min_cos > 0.99,
        "eager vs compiled multi-step decode cosine {min_cos}"
    );
}

#[test]
fn test_compiled_cpu_metal_decode_parity() {
    use rlx_gepard::backbone::{BackboneWeights, GepardKvCache, backbone_prefill};
    use rlx_gepard::compiled_session::GepardCompiledSession;
    use rlx_gepard::config::GepardConfig;
    use rlx_gepard::synthesis::{embed_audio_frame, sample_all_heads_temp};
    use rlx_gepard::tokenizer::GepardTokenizer;
    use rlx_gepard::weights::GepardOverlay;
    use rlx_runtime::{Device, is_available};

    if !is_available(Device::Metal) {
        eprintln!("skipping: Metal not available");
        return;
    }
    let bundle = match gepard_bundle() {
        Some(p) => p,
        None => return,
    };
    let cfg = GepardConfig::from_path(&bundle).expect("config");
    let tok = GepardTokenizer::load(&bundle, &cfg).expect("tokenizer");
    let bytes = std::fs::read(bundle.join("model.safetensors")).expect("read");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse");
    let backbone = BackboneWeights::load(&st, &cfg.backbone).expect("backbone");
    let overlay = GepardOverlay::load(&st, cfg.num_audio_heads()).expect("overlay");
    let ids = tok.build_prompt_ids(FOX).expect("tokenize");
    let embeds = backbone.embed_tokens(&ids);
    let hidden = cfg.hidden_size();

    let mut kv = GepardKvCache::new(cfg.backbone.num_hidden_layers);
    let all_h = backbone_prefill(&embeds, ids.len(), &backbone, &mut kv);
    let sos = &all_h[(ids.len() - 1) * hidden..ids.len() * hidden];
    fastrand::seed(54);
    let first = sample_all_heads_temp(
        sos,
        &overlay,
        &cfg.codec.channel_vocabs(),
        0.4,
        1.0,
        None,
        32,
    );
    let frame_emb = embed_audio_frame(&first, &overlay, cfg.audio_embed_dim, hidden);

    let cpu_h = {
        let mut cpu = GepardCompiledSession::new(Device::Cpu, &cfg, &bundle).expect("cpu");
        let (_s, mut cpu_cache) = cpu.prefill_hidden(&embeds, ids.len()).expect("cpu prefill");
        cpu.decode_hidden(&mut cpu_cache, &frame_emb)
            .expect("cpu decode")
    };
    let metal_h = {
        let mut metal = GepardCompiledSession::new(Device::Metal, &cfg, &bundle).expect("metal");
        let (_s, mut metal_cache) = metal
            .prefill_hidden(&embeds, ids.len())
            .expect("metal prefill");
        metal
            .decode_hidden(&mut metal_cache, &frame_emb)
            .expect("metal decode")
    };

    let cos = cosine_f32(&cpu_h, &metal_h);
    eprintln!("[gepard compiled parity] cpu vs metal-routed decode cosine={cos:.6}");
    assert!(
        cos > 0.99,
        "cpu vs metal-routed (MLX) decode hidden cosine {cos}"
    );
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

#[test]
fn test_gepard_whisper_long() {
    let bundle = match gepard_bundle() {
        Some(p) => p,
        None => {
            eprintln!("skipping: missing gepard bundle");
            return;
        }
    };
    let wd = match whisper_dir() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no Whisper weights");
            return;
        }
    };
    use rlx_gepard::{GepardSynthesizer, InferOpts};
    use rlx_runtime::Device;
    let want = long_words();
    let audio = {
        let synth = GepardSynthesizer::open_with_compiled(&bundle, Device::Cpu, true)
            .expect("open cpu compiled")
            .with_opts(InferOpts {
                max_frames: 2000,
                seed: 4,
                ..Default::default()
            });
        synth.synthesize(LONG, "").expect("synth long")
    };
    assert!(audio.len() > 22_050 * 5, "long audio too short");
    let pcm16 = resample_linear(&audio, 22_050, 16_000);
    use rlx_whisper::WhisperRunner;
    let mut session = WhisperRunner::builder()
        .weights(wd.join("model.safetensors"))
        .config_path(wd.join("config.json"))
        .tokenizer_path(wd.join("tokenizer.json"))
        .build()
        .expect("whisper");
    let transcript = session.transcribe_greedy(&pcm16).expect("transcribe");
    let hits = whisper_coverage(&transcript, &want);
    eprintln!(
        "[gepard whisper long] {hits}/{n} {transcript:?}",
        n = want.len()
    );
    assert_eq!(
        hits,
        want.len(),
        "long paragraph: expected all words in {transcript:?}"
    );
}

fn resample_linear(pcm: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || pcm.is_empty() {
        return pcm.to_vec();
    }
    let n_out = ((pcm.len() as u64) * u64::from(dst_sr) / u64::from(src_sr)) as usize;
    let mut out = Vec::with_capacity(n_out);
    let scale = src_sr as f64 / dst_sr as f64;
    for i in 0..n_out {
        let src = i as f64 * scale;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(pcm.len() - 1);
        let t = (src - i0 as f64) as f32;
        out.push(pcm[i0] * (1.0 - t) + pcm[i1] * t);
    }
    out
}
