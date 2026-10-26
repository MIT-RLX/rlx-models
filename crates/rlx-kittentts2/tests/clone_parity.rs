// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Voice-cloning front end vs `scripts/kittentts2_clone_reference.py`.
//!
//! Each stage is fed the exact input the reference consumed, so the DSP and
//! the networks are checked independently of the resamplers. The clip is a
//! built-in voice's own reference, so the end-to-end case can also be scored
//! against the artifacts the repository ships for it.
//!
//! `RLX_KITTENTTS2_CLONE_REF` (default `weights/tts/kitten-tts-2/clone_reference`),
//! `RLX_KITTENTTS2_DIR`, `RLX_KITTENTTS2_S3GEN`, `RLX_KITTENTTS2_TEST_DEVICE`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rlx_kittentts2::reference::{ReferenceEncoder, dsp, tokenizer};
use rlx_kittentts2::voices::{NpyArray, read_npz};
use rlx_kittentts2::{Kitten2Config, VoiceLibrary, parse_device};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn path_or(key: &str, default: &str) -> PathBuf {
    rlx_ir::env::var_result(key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join(default))
}

struct Fixture {
    a: BTreeMap<String, NpyArray>,
    enc: ReferenceEncoder,
    repo: PathBuf,
}

impl Fixture {
    fn load() -> Option<Self> {
        let dir = path_or(
            "RLX_KITTENTTS2_CLONE_REF",
            "weights/tts/kitten-tts-2/clone_reference",
        );
        let npz = dir.join("clone_reference.npz");
        let repo = path_or("RLX_KITTENTTS2_DIR", "weights/tts/kitten-tts-2");
        let s3 = path_or(
            "RLX_KITTENTTS2_S3GEN",
            "weights/tts/chatterbox-turbo/s3gen_meanflow.safetensors",
        );
        let spk = repo.join("speaker/model.safetensors");
        if !(npz.is_file() && s3.is_file() && spk.is_file()) {
            eprintln!(
                "skip: need {}, {} and {}",
                npz.display(),
                s3.display(),
                spk.display()
            );
            return None;
        }
        let device = parse_device(
            &rlx_ir::env::var_result("RLX_KITTENTTS2_TEST_DEVICE").unwrap_or_else(|_| "cpu".into()),
        )
        .unwrap();
        Some(Self {
            a: read_npz(&npz).unwrap(),
            enc: ReferenceEncoder::open(&s3, &spk, device).unwrap(),
            repo,
        })
    }

    fn f32(&self, k: &str) -> Vec<f32> {
        self.a
            .get(k)
            .cloned()
            .unwrap_or_else(|| panic!("{k}"))
            .into_f32()
            .unwrap()
    }

    fn u32(&self, k: &str) -> Vec<u32> {
        self.a
            .get(k)
            .cloned()
            .unwrap_or_else(|| panic!("{k}"))
            .into_u32()
            .unwrap()
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn agreement(a: &[u32], b: &[u32]) -> f32 {
    let n = a.len().max(b.len()).max(1);
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f32 / n as f32
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    d / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt())
}

#[test]
fn features_match_the_upstream_extractors() {
    let Some(f) = Fixture::load() else { return };
    let (frames, mel) = dsp::tokenizer_log_mel(&f.f32("lm_wav16"));
    assert_eq!(frames * tokenizer::N_MELS, f.f32("tok_mel").len());
    let d = max_abs(&mel, &f.f32("tok_mel"));
    assert!(d < 1e-4, "tokenizer log-mel |Δ| {d:e}");

    let (_, pf) = dsp::s3gen_mel(&f.f32("s3_wav24"));
    let d = max_abs(&pf, &f.f32("prompt_feat"));
    assert!(d < 1e-3, "S3Gen mel |Δ| {d:e}");

    let (frames, mut fb) = dsp::kaldi_fbank(&f.f32("s3_wav16"));
    dsp::mean_normalize(&mut fb, frames, 80);
    let d = max_abs(&fb, &f.f32("campp_fbank"));
    assert!(d < 1e-3, "Kaldi fbank |Δ| {d:e}");
}

#[test]
fn encoders_match_on_identical_inputs() {
    let Some(f) = Fixture::load() else { return };
    let mel = f.f32("tok_mel");
    let hidden = f
        .enc
        .tokenizer_hidden(&mel, mel.len() / tokenizer::N_MELS)
        .unwrap();
    let d = max_abs(&hidden, &f.f32("tok_hidden"));
    assert!(d < 1e-3, "S3 tokenizer hidden |Δ| {d:e}");
    let codes = f.enc.quantize(&f.f32("tok_hidden")).unwrap();
    assert_eq!(
        codes,
        f.u32("lm_reference_tokens"),
        "FSQ on the reference hidden"
    );

    let raw = f.enc.speaker_raw(&f.f32("spk_wav16")).unwrap();
    let want = f.f32("speaker_raw");
    let d = max_abs(&raw, &want);
    assert!(
        d < 1e-3 * want.iter().fold(0f32, |m, v| m.max(v.abs())),
        "pyannote |Δ| {d:e}"
    );

    let x = f.enc.xvector(&f.f32("s3_wav16")).unwrap();
    let c = cosine(&x, &f.f32("xvector"));
    assert!(c > 0.9999, "CAM++ cosine {c}");
}

#[test]
fn cloned_voice_reproduces_the_shipped_artifacts() {
    let Some(f) = Fixture::load() else { return };
    let cfg = Kitten2Config::from_dir(&f.repo).unwrap();
    let lib = VoiceLibrary::open(
        &f.repo.join(&cfg.voices),
        &f.repo.join(cfg.s3gen_voices_path().unwrap()),
    )
    .unwrap();
    let shipped = lib.load("Bruno").unwrap();
    let wav = lib.reference_wav("Bruno").unwrap();
    let v = f.enc.voice_from_wav(&wav, &shipped.transcript).unwrap();
    let lm = agreement(&v.reference_tokens, &shipped.reference_tokens);
    let s3 = agreement(&v.s3.prompt_token, &shipped.s3.prompt_token);
    let spk = cosine(&v.speaker_embedding, &shipped.speaker_embedding);
    let xv = cosine(&v.s3.xvector, &shipped.s3.xvector);
    eprintln!("LM tokens {lm:.3}, S3 tokens {s3:.3}, speaker cos {spk:.4}, x-vector cos {xv:.4}");
    assert!(lm > 0.9 && s3 > 0.9, "token agreement LM {lm}, S3 {s3}");
    assert!(
        spk > 0.98 && xv > 0.99,
        "speaker cos {spk}, x-vector cos {xv}"
    );
    assert_eq!(v.s3.prompt_frames, shipped.s3.prompt_frames);
}

#[test]
fn long_clips_tokenize_through_overlapping_windows() {
    let Some(f) = Fixture::load() else { return };
    if !f.a.contains_key("long_tokens") {
        eprintln!("skip: reference dump predates the long-clip case");
        return;
    }
    let got = f.enc.tokenize(&f.f32("long_wav16")).unwrap();
    let want = f.u32("long_tokens");
    assert_eq!(got.len(), want.len(), "merged length");
    let a = agreement(&got, &want);
    assert!(a > 0.98, "long-clip token agreement {a}");
}
