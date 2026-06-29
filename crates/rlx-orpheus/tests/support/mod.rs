// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Shared helpers for Orpheus integration tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rlx_orpheus::{
    BackboneLoadOptions, GenerationConfig, OrpheusTts, SAMPLE_RATE, SAMPLES_PER_FRAME,
    VoiceCloneReference, lm_kv_decode_supported, synth_device_for_tests,
};
use rlx_runtime::Device;
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const MIN_AUDIBLE_PEAK: f32 = 0.01;

pub struct SynthBenchResult {
    pub wall: Duration,
    pub code_count: usize,
    pub sample_count: usize,
    pub transcript: Option<String>,
}

pub struct BenchRow {
    pub label: String,
    pub device: Device,
    pub wall_ms: f64,
    pub audio_s: f64,
    pub rtf: f64,
    pub codes: usize,
    pub whisper_ok: bool,
    pub transcript: String,
}

impl BenchRow {
    pub fn print_header() {
        eprintln!(
            "{:<12} {:>8} {:>8} {:>6} {:>6} {:>6}  transcript",
            "device", "wall_ms", "audio_s", "rtf", "codes", "whisp"
        );
    }

    pub fn print(&self) {
        eprintln!(
            "{:<12} {:>8.0} {:>8.2} {:>6.2} {:>6} {:>6}  {}",
            self.label,
            self.wall_ms,
            self.audio_s,
            self.rtf,
            self.codes,
            if self.whisper_ok { "ok" } else { "FAIL" },
            self.transcript.trim()
        );
    }
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn repo_cache() -> PathBuf {
    repo_root().join(".cache")
}

pub fn orpheus_gguf_path() -> Option<PathBuf> {
    std::env::var("ORPHEUS_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            for name in [
                "orpheus-3b-0.1-ft-Q4_K_M.gguf",
                "orpheus-3b-0.1-ft-Q8_0.gguf",
            ] {
                let p = PathBuf::from(format!("/tmp/rlx-weights/orpheus/{name}"));
                if p.is_file() {
                    return Some(p);
                }
            }
            None
        })
}

pub fn snac_decoder_path() -> Option<PathBuf> {
    std::env::var("ORPHEUS_SNAC_PATH")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = PathBuf::from("/tmp/rlx-weights/snac/snac_24khz_decoder.safetensors");
            p.is_file().then_some(p)
        })
}

pub fn orpheus_pretrained_gguf_path() -> Option<PathBuf> {
    std::env::var("ORPHEUS_PRETRAINED_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

pub fn voice_clone_ref_path() -> Option<PathBuf> {
    std::env::var("ORPHEUS_CLONE_REF_JSON")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| {
            let p = PathBuf::from("/tmp/jfk_orpheus_ref.json");
            p.is_file().then_some(p)
        })
}

pub fn bench_text() -> String {
    std::env::var("ORPHEUS_BENCH_TEXT").unwrap_or_else(|_| "The weather is nice today.".into())
}

/// Minimum SNAC frames for `text` (one frame per ~word, at least four frames).
pub fn bench_min_frames(text: &str) -> usize {
    let words = normalize_words(text).len().max(1);
    (words + 1).max(4)
}

pub fn bench_min_codes(text: &str) -> usize {
    bench_min_frames(text) * 7
}

pub fn bench_min_audio_seconds(text: &str) -> f64 {
    bench_min_frames(text) as f64 * SAMPLES_PER_FRAME as f64 / SAMPLE_RATE as f64
}

/// Assert SNAC code count and PCM length match a short sentence utterance.
pub fn assert_synthesis_length(text: &str, code_count: usize, sample_count: usize) {
    let min_frames = bench_min_frames(text);
    let min_codes = min_frames * 7;
    assert!(
        code_count >= min_codes,
        "expected >= {min_codes} codes ({min_frames} frames) for {text:?}, got {code_count}"
    );
    let frames = code_count / 7;
    assert!(
        frames >= min_frames,
        "expected >= {min_frames} SNAC frames for {text:?}, got {frames} ({code_count} codes)"
    );
    let min_samples = min_frames * SAMPLES_PER_FRAME;
    let max_samples = (frames + 1) * SAMPLES_PER_FRAME;
    assert!(
        sample_count >= min_samples * 3 / 4,
        "audio too short for {text:?}: {sample_count} samples ({:.2}s), expected >= {:.2}s",
        sample_count as f64 / SAMPLE_RATE as f64,
        bench_min_audio_seconds(text),
    );
    assert!(
        sample_count <= max_samples,
        "audio too long for {frames} frames: {sample_count} samples (max {max_samples})"
    );
}

pub fn bench_voice() -> String {
    std::env::var("ORPHEUS_BENCH_VOICE").unwrap_or_else(|_| "tara".into())
}

/// Whisper intelligibility benches sample by default (upstream Orpheus mode).
/// Set `ORPHEUS_BENCH_GREEDY=1` to force greedy argmax (degenerate for this LM).
pub fn bench_force_greedy() -> bool {
    matches!(
        std::env::var("ORPHEUS_BENCH_GREEDY").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

pub fn bench_max_tokens() -> u32 {
    std::env::var("ORPHEUS_BENCH_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| bench_max_tokens_for(&bench_text()))
}

/// LM step budget scaled to utterance length (~2 SNAC frames per content word).
pub fn bench_max_tokens_for(text: &str) -> u32 {
    let words = normalize_words(text).len().max(1);
    ((words * 14 + 28).min(512)) as u32
}

pub fn bench_warmup_iters() -> u32 {
    std::env::var("ORPHEUS_BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

pub fn bench_measure_iters() -> u32 {
    std::env::var("ORPHEUS_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

pub fn whisper_word_coverage_min() -> f32 {
    std::env::var("ORPHEUS_WHISPER_MIN_COVERAGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.45)
}

pub fn require_weights() -> Option<(PathBuf, PathBuf)> {
    let gguf = orpheus_gguf_path()?;
    let snac = snac_decoder_path()?;
    Some((gguf, snac))
}

pub fn whisper_runner(dir: &Path) -> WhisperRunner {
    WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper runner")
}

pub fn transcribe_pcm_24k(pcm_24k: &[f32], whisper_dir: &Path) -> String {
    let mut pcm_16k = resample_linear(pcm_24k, SAMPLE_RATE, WHISPER_RATE as u32);
    let min_len = WHISPER_RATE / 2;
    if pcm_16k.len() < min_len {
        pcm_16k.resize(min_len, 0.0);
    }
    whisper_runner(whisper_dir)
        .transcribe_greedy(&pcm_16k)
        .expect("whisper transcribe")
}

pub fn bench_named_voice(
    gguf: &Path,
    snac: &Path,
    device: Device,
    text: &str,
    voice: &str,
    whisper_dir: Option<&Path>,
) -> anyhow::Result<SynthBenchResult> {
    let mut tts = load_orpheus(gguf, snac, device)?;
    // Sampling (upstream Orpheus defaults: temp 0.6 / top_p 0.8 / rep 1.3, seed 42).
    // Greedy argmax is degenerate for this sampling-trained TTS LM and yields
    // non-speech audio; the deterministic cross-backend comparison lives in the
    // separate codes-parity test (`greedy_codes_on_device`).
    tts.config = GenerationConfig {
        max_new_tokens: bench_max_tokens(),
        greedy: bench_force_greedy(),
        ..GenerationConfig::default()
    };
    for _ in 0..bench_warmup_iters() {
        let _ = tts.synthesize(text, Some(voice))?;
    }
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..bench_measure_iters() {
        last = Some(tts.synthesize(text, Some(voice))?);
    }
    let wall = t0.elapsed();
    let out = last.expect("measure iter");
    assert_audible(&out.samples, SAMPLE_RATE as usize / 8);
    assert_synthesis_length(text, out.code_count, out.samples.len());
    let transcript = whisper_dir.map(|dir| transcribe_pcm_24k(&out.samples, dir));
    Ok(SynthBenchResult {
        wall,
        code_count: out.code_count,
        sample_count: out.samples.len(),
        transcript,
    })
}

pub fn bench_voice_clone(
    gguf: &Path,
    snac: &Path,
    device: Device,
    reference: &VoiceCloneReference,
    target_text: &str,
    whisper_dir: Option<&Path>,
) -> anyhow::Result<SynthBenchResult> {
    let mut tts = load_orpheus(gguf, snac, device)?;
    tts.config = GenerationConfig {
        max_new_tokens: bench_max_tokens(),
        greedy: bench_force_greedy(),
        ..GenerationConfig::default()
    };
    for _ in 0..bench_warmup_iters() {
        let _ =
            tts.synthesize_voice_clone(&reference.transcript, &reference.token_ids, target_text)?;
    }
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..bench_measure_iters() {
        last = Some(tts.synthesize_voice_clone(
            &reference.transcript,
            &reference.token_ids,
            target_text,
        )?);
    }
    let wall = t0.elapsed();
    let out = last.expect("measure iter");
    assert_audible(&out.samples, SAMPLE_RATE as usize / 8);
    let transcript = whisper_dir.map(|dir| transcribe_pcm_24k(&out.samples, dir));
    Ok(SynthBenchResult {
        wall,
        code_count: out.code_count,
        sample_count: out.samples.len(),
        transcript,
    })
}

pub fn synth_bench_to_row(
    label: &str,
    device: Device,
    result: &SynthBenchResult,
    reference_text: &str,
) -> BenchRow {
    let audio_s = result.sample_count as f64 / SAMPLE_RATE as f64;
    let wall_ms = result.wall.as_secs_f64() * 1000.0;
    let rtf = if audio_s > 0.0 {
        result.wall.as_secs_f64() / audio_s
    } else {
        0.0
    };
    let transcript = result.transcript.clone().unwrap_or_default();
    let whisper_ok = result.transcript.as_ref().is_none_or(|t| {
        !t.trim().is_empty()
            && transcript_covers_reference(reference_text, t, whisper_word_coverage_min())
    });
    BenchRow {
        label: label.to_string(),
        device,
        wall_ms,
        audio_s,
        rtf,
        codes: result.code_count,
        whisper_ok,
        transcript,
    }
}

pub fn device_label(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Gpu => "wgpu",
        Device::Vulkan => "vulkan",
        _ => "other",
    }
}

pub fn synth_device() -> Device {
    synth_device_for_tests()
}

pub fn parse_bench_devices(csv: &str) -> Vec<Device> {
    if csv.eq_ignore_ascii_case("all") {
        let mut out = vec![Device::Cpu];
        for d in [
            Device::Metal,
            Device::Mlx,
            Device::Cuda,
            Device::Rocm,
            Device::Gpu,
            Device::Vulkan,
        ] {
            if rlx_runtime::is_available(d)
                && !out.contains(&d)
                && (d == Device::Cpu || lm_kv_decode_supported(d))
            {
                out.push(d);
            }
        }
        return out;
    }
    csv.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(s).expect("parse --devices entry"))
        .collect()
}

pub fn load_orpheus(gguf: &Path, snac: &Path, device: Device) -> anyhow::Result<OrpheusTts> {
    let opts = if std::env::var("ORPHEUS_BENCH_FOR_TTS").ok().as_deref() == Some("1") {
        BackboneLoadOptions::for_tts(device)
    } else {
        BackboneLoadOptions::synthesis()
    };
    OrpheusTts::load_on_with_device(gguf, snac, device, opts)
}

/// Backbone options for cross-backend **codes** parity tests.
///
/// Default: same dynamic decode path as CPU [`BackboneLoadOptions::synthesis`] on every
/// device so GPU kernels are compared apples-to-apples. Set `ORPHEUS_FOR_TTS_PARITY=1`
/// to exercise production [`BackboneLoadOptions::for_tts`] (bucket decode on Metal/CUDA).
pub fn parity_backbone_opts(device: Device) -> BackboneLoadOptions {
    if std::env::var("ORPHEUS_FOR_TTS_PARITY").ok().as_deref() == Some("1") {
        BackboneLoadOptions::for_tts(device)
    } else {
        BackboneLoadOptions::synthesis()
    }
}

/// Greedy SNAC codes for parity (CPU synthesis reference vs per-backend `for_tts`).
pub fn greedy_codes_on_device(
    gguf: &Path,
    device: Device,
    text: &str,
    voice: &str,
    max_tokens: u32,
) -> anyhow::Result<Vec<i32>> {
    use rlx_orpheus::{BackboneModel, DEFAULT_N_CTX, tokens::build_prompt_ids};

    let prompt = build_prompt_ids(gguf, &format!("{voice}: {text}"))?;
    let backbone =
        BackboneModel::load_on_with(gguf, DEFAULT_N_CTX, device, parity_backbone_opts(device))?;
    let cfg = GenerationConfig {
        max_new_tokens: max_tokens,
        greedy: true,
        repetition_penalty: 1.0,
        ..GenerationConfig::default()
    };
    backbone.generate_codes_from_prompt(&prompt, &cfg)
}

pub fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

pub fn assert_audible(audio: &[f32], min_samples: usize) {
    assert!(
        audio.len() >= min_samples,
        "expected at least {min_samples} samples at {SAMPLE_RATE} Hz, got {}",
        audio.len()
    );
    let peak = peak_amplitude(audio);
    assert!(
        peak >= MIN_AUDIBLE_PEAK,
        "expected audible waveform (peak >= {MIN_AUDIBLE_PEAK}), got peak={peak:.2e}"
    );
}

pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64).max(1) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * from_hz as f64 / to_hz as f64;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn normalize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

pub fn transcript_covers_reference(reference: &str, transcript: &str, min_ratio: f32) -> bool {
    let reference_words = normalize_words(reference);
    if reference_words.is_empty() {
        return false;
    }
    let heard = normalize_words(transcript);
    let hits = reference_words
        .iter()
        .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
        .count();
    hits as f32 / reference_words.len() as f32 >= min_ratio
}

pub fn whisper_asr_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("RLX_WHISPER_DIR") {
        return whisper_dir_if_ready(PathBuf::from(dir));
    }
    let cache = repo_cache();
    for name in [
        "whisper-base.en",
        "whisper-small.en",
        "whisper-tiny.en",
        "whisper-tiny",
    ] {
        if let Some(dir) = whisper_dir_if_ready(cache.join(name)) {
            return Some(dir);
        }
    }
    None
}

fn whisper_dir_if_ready(dir: PathBuf) -> Option<PathBuf> {
    if dir.join("model.safetensors").is_file() && dir.join("tokenizer.json").is_file() {
        Some(dir)
    } else {
        None
    }
}

pub fn golden_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/orpheus_hi_codes.txt")
}

pub fn load_golden_codes() -> Vec<i32> {
    let path = golden_fixture_path();
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let count: usize = lines
        .next()
        .unwrap_or_else(|| panic!("missing count line in {}", path.display()))
        .trim()
        .parse()
        .expect("code count");
    let nums: Vec<i32> = lines
        .flat_map(|line| line.split_whitespace())
        .map(|s| s.parse().expect("code integer"))
        .collect();
    assert_eq!(
        nums.len(),
        count,
        "fixture {} lists {count} codes but parsed {}",
        path.display(),
        nums.len()
    );
    nums
}
