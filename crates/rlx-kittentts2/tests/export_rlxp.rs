// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! An exported KittenTTS 2 (`weights/tts/kitten-tts-2`, S3Gen from
//! `chatterbox-turbo` beside it), opened by rlx alone: the speech LM's
//! tokens for the default voice's prompt and speaker row as
//! `SpeechLm::generate` samples them, and text to speech as `KittenTts2`
//! speaks it (greedy: the normalizer, the prompt, the LM, S3Gen).
#![cfg(feature = "export")]

use std::path::PathBuf;

use rlx_core::export::ExportOptions;
use rlx_kittentts2::config::Kitten2Config;
use rlx_kittentts2::export::export_rlxp;
use rlx_kittentts2::lm::SpeechLm;
use rlx_kittentts2::sampling::{Rng, SamplerOpts};
use rlx_kittentts2::text;
use rlx_kittentts2::voices::VoiceLibrary;
use rlx_runtime::Device;
use serde_json::json;

#[test]
fn an_exported_kittentts2_speaks_as_the_runner_does() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/kitten-tts-2");
    if !repo.join("config.json").is_file() {
        eprintln!("skip: no KittenTTS 2 repository");
        return;
    }
    rlx_text::install_model_tokenizer();
    let cfg = Kitten2Config::from_dir(&repo).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("kittentts2.rlxp");
    let opts = ExportOptions {
        uppers: vec![1024],
        ..ExportOptions::default()
    };
    eprintln!("{}", export_rlxp(&repo, &out, &opts).unwrap());
    let mut m = rlx_runtime::model::open_source(out.to_str().unwrap(), json!({})).unwrap();

    // The default voice's prompt for one sentence.
    let lm_dir = repo.join(&cfg.lm_dir);
    let tok = tokenizers::Tokenizer::from_file(lm_dir.join("tokenizer.json")).unwrap();
    let encode = |s: &str| tok.encode(s, false).unwrap().get_ids().to_vec();
    let voices = VoiceLibrary::open(
        &repo.join(&cfg.voices),
        &repo.join(cfg.s3gen_voices_path().unwrap()),
    )
    .unwrap();
    let voice = voices.load(&cfg.default_voice).unwrap();
    let tm = &cfg.token_map;
    let prefix = text::build_reference_prefix(
        tm,
        &encode(voice.transcript.trim()),
        &voice.reference_tokens,
    );
    let prompt = text::build_generation_prompt(
        tm,
        &encode("Good morning, the tea is ready."),
        Some(&prefix),
        None,
    );
    let preset = cfg.preset(&cfg.default_preset).unwrap();
    let sampler = SamplerOpts::from_preset(preset, cfg.generation.repetition_window);
    let (seed, new) = (7u64, 30usize);

    let ternary = repo.join(cfg.lm_packed.as_deref().unwrap());
    let mut lm = SpeechLm::open(&lm_dir, &ternary, Device::Cpu).unwrap();
    let (want, _) = lm
        .generate(
            &prompt,
            &voice.speaker_embedding,
            tm,
            &sampler,
            new,
            &mut Rng::new(seed),
            |_| {},
        )
        .unwrap();
    let got = m
        .run(
            &[],
            &json!({"entry": "speak", "ids": prompt, "speaker": voice.speaker_embedding,
                    "seed": seed, "max_new_tokens": new,
                    "temperature": sampler.temperature, "top_k": sampler.top_k,
                    "top_p": sampler.top_p, "min_p": sampler.min_p}),
        )
        .unwrap();
    let got: Vec<u32> = got[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
    eprintln!(
        "{} prompt tokens; {same} of {} generated the runner's ({} from the package)",
        prompt.len(),
        want.len(),
        got.len()
    );
    assert!(same >= 10, "{same} the same");
    drop(lm);

    // Text to speech, greedy: the runner's normalizer, prompt, LM, S3Gen.
    let s3 = rlx_kittentts2::pipeline::resolve_s3gen(&repo).unwrap();
    let mut tts = rlx_kittentts2::pipeline::KittenTts2::open(&repo, &s3, Device::Cpu).unwrap();
    let text = "It costs $12.50, and the tea is ready.";
    let request = rlx_kittentts2::pipeline::GenerateOpts {
        max_new_tokens: 60,
        seed,
        greedy: true,
        ..Default::default()
    };
    let (want, reports) = tts.generate(text, &request, |_| {}).unwrap();
    let want_codes: Vec<u32> = reports.iter().flat_map(|r| r.audio.clone()).collect();
    let got_codes = m
        .run(
            &[],
            &json!({"entry": "speak", "text": text, "max_new_tokens": 60, "seed": seed,
                    "greedy": true}),
        )
        .unwrap();
    let got_codes: Vec<u32> = got_codes[0]
        .data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
        .collect();
    let same = got_codes
        .iter()
        .zip(&want_codes)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "codec tokens: {same} of {} the runner's ({} from the package)",
        want_codes.len(),
        got_codes.len()
    );
    // Greedy, until the runner's packed LM and the package's (its weights
    // the same, dequantized) tip a near tie.
    assert!(same >= 30, "{same} codec tokens the same");
    // S3Gen alone, on the runner's tokens: the flow, then the vocoder.
    let voice = voices.load(&cfg.default_voice).unwrap();
    let mut tokens = want_codes.clone();
    tokens.extend([4299; 3]);
    let l = 2 * (voice.s3.prompt_token.len() + tokens.len());
    let noise: Vec<f32> = {
        let mut r = Rng::new(seed ^ 0x5EED_F10E);
        (0..80 * l).map(|_| r.normal()).collect()
    };
    let trace = tts
        .s3gen()
        .flow_with_noise(&tokens, &voice.s3, &noise)
        .unwrap();
    let flow = m
        .run(
            &[],
            &json!({"entry": "speak", "stage": "flow", "codes": want_codes, "seed": seed}),
        )
        .unwrap();
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    for (k, want) in [&trace.mu, &trace.mel].into_iter().enumerate() {
        let got = f32s(&flow[k].data);
        let d = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "flow {}: {} vs {}, max |d| {d}",
            flow[k].name,
            got.len(),
            want.len()
        );
        assert_eq!(got.len(), want.len());
        assert!(d < 1e-3, "flow {}: max |d| {d}", flow[k].name);
    }
    let want_s3 = tts
        .s3gen()
        .synthesize(
            &want_codes,
            &voice.s3,
            &std::cell::RefCell::new(Rng::new(seed ^ 0x5EED_F10E)),
        )
        .unwrap()
        .unwrap();
    let got_s3 = m
        .synthesize("-", &json!({"codes": want_codes, "seed": seed}))
        .unwrap();
    let cos_of = |a: &[f32], b: &[f32]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x * y)).sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(x * x)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b))
    };
    let s3_cos = cos_of(&got_s3.samples, &want_s3);
    eprintln!(
        "S3Gen on the runner's tokens: {} vs {} samples, cosine {s3_cos}",
        got_s3.samples.len(),
        want_s3.len()
    );
    assert_eq!(got_s3.samples.len(), want_s3.len());
    assert!(s3_cos > 0.9999, "S3Gen cosine {s3_cos}");
    // The whole path: speech as long as the runner's (its tokens past the
    // near tie its own).
    let got = m
        .synthesize(
            text,
            &json!({"max_new_tokens": 60, "seed": seed, "greedy": true}),
        )
        .unwrap();
    assert_eq!(got.sample_rate, cfg.sample_rate);
    assert_eq!(got.samples.len(), want.len());
    let rms = (got.samples.iter().map(|x| f64::from(x * x)).sum::<f64>()
        / got.samples.len() as f64)
        .sqrt();
    eprintln!(
        "{} samples, rms {rms}, cosine {}",
        want.len(),
        cos_of(&got.samples, &want)
    );
    assert!(rms > 0.01, "silent: rms {rms}");
}
