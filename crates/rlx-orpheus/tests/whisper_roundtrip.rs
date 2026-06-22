// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Orpheus SNAC → Whisper (fast, default) and optional LM e2e (slow, env-gated).

mod support;

use rlx_orpheus::{
    GenerationConfig, SAMPLE_RATE, SnacBackend, SnacLoadOptions, decode_orpheus_codes,
};
use support::{
    assert_audible, load_golden_codes, load_orpheus, orpheus_gguf_path, snac_decoder_path,
    synth_device, transcribe_pcm_24k, transcript_covers_reference, whisper_asr_dir,
};

const E2E_TEXT: &str = "Hi.";
const E2E_VOICE: &str = "tara";
/// Minimum SNAC frames (×7 tokens) for a short utterance without long LM runs.
const E2E_MAX_TOKENS: u32 = 56;

/// Fast default (~15–45s): golden SNAC codes → decode → Whisper. No 3B LM.
#[test]
fn golden_codec_intelligible_via_whisper() {
    let Some(snac_path) = snac_decoder_path() else {
        eprintln!("skip: set ORPHEUS_SNAC_PATH or run `just fetch-orpheus-snac`");
        return;
    };
    let Some(whisper_dir) = whisper_asr_dir() else {
        eprintln!("skip: run `just fetch-whisper`");
        return;
    };

    let codes = load_golden_codes();
    let snac = SnacBackend::open(&snac_path, SnacLoadOptions::default()).expect("SNAC load");
    let samples = decode_orpheus_codes(&snac, &codes).expect("decode golden codes");
    eprintln!(
        "golden {} codes -> {} samples ({:.2}s)",
        codes.len(),
        samples.len(),
        samples.len() as f64 / SAMPLE_RATE as f64
    );
    assert_audible(&samples, 2_000);

    let transcript = transcribe_pcm_24k(&samples, &whisper_dir);
    eprintln!("whisper: {transcript}");

    assert!(
        !transcript.trim().is_empty(),
        "Whisper returned empty transcript for SNAC decode output"
    );
}

/// Full LM → SNAC → Whisper. Opt-in: `ORPHEUS_WHISPER_E2E=1` + `--ignored`.
#[test]
#[ignore = "slow: 3B LM synthesis — set ORPHEUS_WHISPER_E2E=1"]
#[cfg(feature = "llama")]
fn roundtrip_text_via_whisper_e2e() {
    if std::env::var("ORPHEUS_WHISPER_E2E").ok().as_deref() != Some("1") {
        eprintln!("skip e2e: set ORPHEUS_WHISPER_E2E=1");
        return;
    }

    let Some(gguf) = orpheus_gguf_path() else {
        eprintln!("skip: run `just fetch-orpheus`");
        return;
    };
    let Some(snac_path) = snac_decoder_path() else {
        eprintln!("skip: run `just fetch-orpheus-snac`");
        return;
    };
    let Some(whisper_dir) = whisper_asr_dir() else {
        eprintln!("skip: run `just fetch-whisper`");
        return;
    };

    let device = synth_device();
    let mut tts = load_orpheus(&gguf, &snac_path, device).expect("load Orpheus");
    tts.config = GenerationConfig {
        max_new_tokens: E2E_MAX_TOKENS,
        ..GenerationConfig::default()
    };

    let out = tts
        .synthesize(E2E_TEXT, Some(E2E_VOICE))
        .expect("synthesize");
    eprintln!(
        "e2e {} codes -> {} samples ({:.2}s)",
        out.code_count,
        out.samples.len(),
        out.samples.len() as f64 / out.sample_rate as f64
    );
    assert!(
        out.code_count >= 28,
        "expected at least 4 SNAC frames (28 codes), got {}",
        out.code_count
    );
    assert_audible(&out.samples, SAMPLE_RATE as usize / 4);

    let transcript = transcribe_pcm_24k(&out.samples, &whisper_dir);
    eprintln!("reference: {E2E_TEXT}");
    eprintln!("whisper:   {transcript}");

    assert!(
        transcript_covers_reference(E2E_TEXT, &transcript, 0.5),
        "Whisper missed reference.\nref: {E2E_TEXT}\ngot: {transcript}"
    );

    if std::env::var("ORPHEUS_WHISPER_REGEN_FIXTURE")
        .ok()
        .as_deref()
        == Some("1")
    {
        write_golden_fixture(&out.codes);
    }
}

fn write_golden_fixture(codes: &[i32]) {
    use support::golden_fixture_path;
    let path = golden_fixture_path();
    let mut body = format!("{}\n", codes.len());
    for (i, c) in codes.iter().enumerate() {
        if i > 0 {
            body.push(' ');
        }
        body.push_str(&c.to_string());
    }
    body.push('\n');
    std::fs::write(&path, body).expect("write golden fixture");
    eprintln!("updated golden fixture: {}", path.display());
}
