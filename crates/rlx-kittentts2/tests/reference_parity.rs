// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Parity against the upstream PyTorch pipeline, stage by stage.
//!
//! Needs the weights and a reference dump; skips (with a note) without them:
//!
//! ```text
//! just fetch-kittentts2
//! .venv-kitten2/bin/python scripts/kittentts2_reference.py \
//!     --repo weights/tts/kitten-tts-2 \
//!     --s3gen weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors \
//!     --out weights/tts/kitten-tts-2/reference
//! cargo test -p rlx-kittentts2 --release --test reference_parity
//! ```
//!
//! `RLX_KITTENTTS2_DIR`, `RLX_KITTENTTS2_S3GEN`, `RLX_KITTENTTS2_REF` override
//! the locations; `RLX_KITTENTTS2_TEST_DEVICE` picks the device (default cpu).
//!
//! Tolerances sit between measured float noise and the defect each stage
//! guards against: the S3Gen stages agree to ≤ 4e-5 absolute (CPU and Metal),
//! where a dropped bias or a one-frame shift moves them by ≥ 1e-2; the LM's
//! logits agree to 1.5e-4 of a 71.5 range, and every greedy token must match.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rlx_kittentts2::s3gen::{S3Gen, SourceNoise, hift};
use rlx_kittentts2::voices::{NpyArray, read_npz};
use rlx_kittentts2::{Kitten2Config, Rng, S3Conditioning, SamplerOpts, SpeechLm, parse_device};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn path_or(key: &str, default: &str) -> PathBuf {
    rlx_ir::env::var_result(key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join(default))
}

fn device() -> rlx_kittentts2::Device {
    parse_device(
        &rlx_ir::env::var_result("RLX_KITTENTTS2_TEST_DEVICE").unwrap_or_else(|_| "cpu".into()),
    )
    .expect("device")
}

struct Reference {
    meta: serde_json::Value,
    arrays: BTreeMap<String, NpyArray>,
}

impl Reference {
    fn load() -> Option<Self> {
        let dir = path_or("RLX_KITTENTTS2_REF", "weights/tts/kitten-tts-2/reference");
        let npz = dir.join("reference.npz");
        if !npz.is_file() {
            eprintln!(
                "skip: no reference dump at {} (see scripts/kittentts2_reference.py)",
                dir.display()
            );
            return None;
        }
        Some(Self {
            meta: serde_json::from_str(
                &std::fs::read_to_string(dir.join("reference.json")).unwrap(),
            )
            .unwrap(),
            arrays: read_npz(&npz).unwrap(),
        })
    }

    fn f32(&self, k: &str) -> Vec<f32> {
        self.arrays
            .get(k)
            .cloned()
            .unwrap_or_else(|| panic!("{k}"))
            .into_f32()
            .unwrap()
    }

    fn ids(&self, k: &str) -> Vec<u32> {
        self.meta[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn s3gen_matches_the_reference_stage_by_stage() {
    let Some(r) = Reference::load() else { return };
    let s3 = path_or(
        "RLX_KITTENTTS2_S3GEN",
        "weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors",
    );
    if !s3.is_file() {
        eprintln!("skip: no {}", s3.display());
        return;
    }
    let g = S3Gen::open(&s3, device()).unwrap();
    let feat = r.f32("cond_prompt_feat");
    let cond = S3Conditioning {
        prompt_frames: feat.len() / 80,
        prompt_token: r
            .f32("cond_prompt_token")
            .iter()
            .map(|&v| v as u32)
            .collect(),
        prompt_feat: feat,
        xvector: r.f32("cond_embedding"),
    };
    let tr = g
        .flow_with_noise(&r.ids("s3gen_tokens"), &cond, &r.f32("flow_noise"))
        .unwrap();
    for (name, got, want, tol) in [
        ("mu", &tr.mu, r.f32("flow_mu"), 1e-5),
        ("v0", &tr.velocities[0], r.f32("flow_v0"), 1e-4),
        ("v1", &tr.velocities[1], r.f32("flow_v1"), 1e-4),
        ("mel", &tr.mel, r.f32("flow_mel"), 1e-4),
    ] {
        let d = max_abs_diff(got, &want);
        assert!(d < tol, "{name}: max |Δ| {d:e} ≥ {tol:e}");
    }
    let mel = r.f32("flow_mel");
    let f0 = g.f0(&mel).unwrap();
    assert!(max_abs_diff(&f0, &r.f32("hift_f0")) < 1e-5);
    let src = hift::sine_source(g.weights(), &f0, SourceNoise::Off).unwrap();
    assert!(max_abs_diff(&src, &r.f32("hift_source")) < 1e-4);
    let wav = g.decode_with_source(&mel, &r.f32("hift_source")).unwrap();
    let d = max_abs_diff(&wav, &r.f32("hift_wav"));
    assert!(d < 2e-4, "wav: max |Δ| {d:e}");
    let unused = g.weights().untouched();
    assert!(unused.is_empty(), "S3Gen tensors no stage read: {unused:?}");
}

#[test]
fn speech_lm_matches_logits_and_greedy_tokens() {
    let Some(r) = Reference::load() else { return };
    let repo = path_or("RLX_KITTENTTS2_DIR", "weights/tts/kitten-tts-2");
    let cfg = Kitten2Config::from_dir(&repo).unwrap();
    let ternary = repo.join(cfg.lm_packed.as_deref().unwrap());
    if !ternary.is_file() {
        eprintln!("skip: no {}", ternary.display());
        return;
    }
    let mut lm = SpeechLm::open(&repo.join(&cfg.lm_dir), &ternary, device()).unwrap();
    let spk = r.f32("speaker_embedding");
    let proj = lm.speaker_projection().project(&spk).unwrap();
    assert!(max_abs_diff(&proj, &r.f32("speaker_projected")) < 1e-5);

    let prompt = r.ids("prompt_ids");
    let logits = lm.prefill_logits(&prompt, &spk).unwrap();
    let d = max_abs_diff(&logits, &r.f32("first_logits"));
    assert!(d < 1e-3, "first logits: max |Δ| {d:e}");

    let want = r.ids("greedy_tokens");
    let mut o = SamplerOpts::from_preset(
        cfg.preset("stable").unwrap(),
        cfg.generation.repetition_window,
    );
    (o.temperature, o.top_k, o.top_p, o.greedy) = (1.0, 0, 1.0, true);
    let (got, _) = lm
        .generate(
            &prompt,
            &spk,
            &cfg.token_map,
            &o,
            want.len(),
            &mut Rng::new(0),
            |_| {},
        )
        .unwrap();
    assert_eq!(got, want);
}

/// The distilled student decoder (`decoders/student_w4.pt`): one BitNet
/// meanflow step from the reference's noise and conditioning, then HiFT on the
/// deterministic source (`scripts/kittentts2_student_reference.py`).
#[test]
fn student_decoder_matches_the_reference() {
    let dir = path_or(
        "RLX_KITTENTTS2_STUDENT_REF",
        "weights/tts/kitten-tts-2/student_reference",
    );
    let repo = path_or("RLX_KITTENTTS2_DIR", "weights/tts/kitten-tts-2");
    let s3 = path_or(
        "RLX_KITTENTTS2_S3GEN",
        "weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors",
    );
    let student = repo.join("decoders/student_w4.pt");
    let npz = dir.join("student_reference.npz");
    if !(npz.is_file() && student.is_file() && s3.is_file()) {
        eprintln!(
            "skip: need {}, {} and {}",
            npz.display(),
            student.display(),
            s3.display()
        );
        return;
    }
    let mut a = read_npz(&npz).unwrap();
    let mut take = |k: &str| {
        a.remove(k)
            .unwrap_or_else(|| panic!("{k}"))
            .into_f32()
            .unwrap()
    };
    let feat = take("prompt_feat");
    let cond = S3Conditioning {
        prompt_frames: feat.len() / 80,
        prompt_token: take("prompt_token").iter().map(|&v| v as u32).collect(),
        prompt_feat: feat,
        xvector: take("embedding"),
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("student_reference.json")).unwrap())
            .unwrap();
    let tokens: Vec<u32> = meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let noise = take("noise");
    let stats = |got: &[f32], want: &[f32]| {
        let max = max_abs_diff(got, want);
        let mean = got
            .iter()
            .zip(want)
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / want.len() as f32;
        let dot: f32 = got.iter().zip(want).map(|(x, y)| x * y).sum();
        let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        (max, mean, dot / (n(got) * n(want)))
    };

    // The network itself, with the int8 activation fake-quant switched off on
    // both sides: exact to float noise (measured 7.6e-6).
    rlx_ir::env::set("RLX_KITTENTTS2_STUDENT_NO_ACT_QUANT", "1");
    let exact = S3Gen::open_student(&student, &s3, device()).unwrap();
    rlx_ir::env::remove("RLX_KITTENTTS2_STUDENT_NO_ACT_QUANT");
    let tr = exact.flow_with_noise(&tokens, &cond, &noise).unwrap();
    let (max, _, _) = stats(&tr.mel, &take("mel_no_act_quant"));
    assert!(
        max < 1e-4,
        "student mel without fake-quant: max |Δ| {max:e}"
    );
    drop(exact);

    // As shipped, with fake-quant: codes sitting on a round-half boundary flip
    // under ulp-level input differences, and 474 layers carry that forward
    // (measured max 0.14, mean 0.025, cosine 0.99997 — a missing layer or a
    // wrong scale lands far outside these bounds).
    let g = S3Gen::open_student(&student, &s3, device()).unwrap();
    let tr = g.flow_with_noise(&tokens, &cond, &noise).unwrap();
    let want = take("mel");
    let (max, mean, cos) = stats(&tr.mel, &want);
    eprintln!("student mel: max |Δ| {max:e}, mean |Δ| {mean:e}, cos {cos}");
    assert!(
        mean < 0.06 && cos > 0.9998,
        "student mel: mean |Δ| {mean:e}, cos {cos}"
    );
    let source = take("source");
    let wav = g.decode_with_source(&want, &source).unwrap();
    let d = max_abs_diff(&wav, &take("wav"));
    assert!(d < 2e-4, "student wav |Δ| {d:e}");
    let f0 = g.f0(&want).unwrap();
    let src = hift::sine_source(g.weights(), &f0, SourceNoise::Off).unwrap();
    let d = max_abs_diff(&src, &source);
    assert!(d < 1e-4, "student source |Δ| {d:e}");
    let unused = g.weights().untouched();
    assert!(
        unused.is_empty(),
        "student tensors no stage read: {unused:?}"
    );
    drop(g);

    // The student's prompt, derived from the voice's clip like upstream's
    // `reference_conditioning` (≤ 4 s, ≤ 112 tokens).
    let spk = repo.join("speaker/model.safetensors");
    let enc = rlx_kittentts2::reference::ReferenceEncoder::open(&s3, &spk, device()).unwrap();
    let cfg = Kitten2Config::from_dir(&repo).unwrap();
    let lib = rlx_kittentts2::VoiceLibrary::open(
        &repo.join(&cfg.voices),
        &repo.join(cfg.s3gen_voices_path().unwrap()),
    )
    .unwrap();
    let (pcm, sr) = rlx_core::asr_bench::read_wav_mono(
        &lib.reference_wav(meta["voice"].as_str().unwrap()).unwrap(),
    )
    .unwrap();
    let mine = enc
        .decoder_conditioning(
            &pcm,
            sr,
            rlx_kittentts2::s3gen::student::STUDENT_REF_SECONDS,
            Some(rlx_kittentts2::s3gen::student::STUDENT_MAX_PROMPT_TOKENS),
        )
        .unwrap();
    assert_eq!(mine.prompt_frames, cond.prompt_frames);
    let same = mine
        .prompt_token
        .iter()
        .zip(&cond.prompt_token)
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        same as f32 >= 0.9 * cond.prompt_token.len() as f32,
        "prompt tokens {same}/{}",
        cond.prompt_token.len()
    );
    let d = max_abs_diff(&mine.prompt_feat, &cond.prompt_feat);
    assert!(d < 0.05, "prompt mel |Δ| {d:e}");
    let dot: f32 = mine
        .xvector
        .iter()
        .zip(&cond.xvector)
        .map(|(a, b)| a * b)
        .sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (n(&mine.xvector) * n(&cond.xvector));
    eprintln!(
        "student prompt: tokens {same}/{}, mel |Δ| {d:e}, x-vector cos {cos}",
        cond.prompt_token.len()
    );
    assert!(cos > 0.99, "x-vector cosine {cos}");
}
