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

//! Speculative-decoding walkthrough — baseline vs speculative generation
//! (batch + progressive streaming), with end-to-end timing, acceptance stats,
//! and Whisper round-trip validation for quality parity.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3-tts \
//!     --features "apple-silicon speculative-decode dev-whisper-validate" \
//!     --example speculative_walkthrough -- \
//!     --ref-wav assets/jfk/jfk_voice_clone.wav \
//!     --out-dir /tmp/spec_jfk \
//!     --k 4
//!
//! The eager talker backend is required (no GPU rollback yet) — pick a
//! device for which `is_eager()` returns true (CPU on the shipped
//! backends).

use anyhow::{Context, Result};
use rlx_qwen3_tts::talker::speculative::{
    ShiftedHistoryDraft, SpecRunStats, TopKDraft, TrivialDraft,
};
use rlx_qwen3_tts::{StreamConfig, StreamControl, StreamEvent, StreamStats, VoiceClone};
use rlx_runtime::{Device, is_available};
use std::path::{Path, PathBuf};
use std::time::Instant;

enum DraftChoice {
    Trivial,
    Shifted,
    TopK,
    EarlyExit(usize),
    Learned { dir: PathBuf, n_layers: usize },
}

impl std::str::FromStr for DraftChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "trivial" => Ok(Self::Trivial),
            "shifted" | "shifted-history" => Ok(Self::Shifted),
            "topk" | "top-k" => Ok(Self::TopK),
            s if s.starts_with("ee") || s.starts_with("early-exit") => {
                let n_str = s.split(':').nth(1).unwrap_or("4");
                let n: usize = n_str
                    .parse()
                    .map_err(|_| format!("early-exit needs integer layer count, got {s:?}"))?;
                Ok(Self::EarlyExit(n))
            }
            s if s.starts_with("learned:") => {
                let mut parts = s.splitn(3, ':');
                let _ = parts.next();
                let dir = parts.next().ok_or("learned: needs <dir>[:<n_layers>]")?;
                let n_str = parts.next().unwrap_or("4");
                let n: usize = n_str
                    .parse()
                    .map_err(|_| format!("learned needs integer layer count, got {s:?}"))?;
                Ok(Self::Learned {
                    dir: PathBuf::from(dir),
                    n_layers: n,
                })
            }
            other => Err(format!(
                "unknown draft {other:?} (try: trivial, shifted, topk, ee:N, learned:DIR:N)"
            )),
        }
    }
}

const DEFAULT_TARGET: &str =
    "Ask not what your country can do for you, ask what you can do for your country.";

const SAMPLE_RATE: u32 = 24_000;

struct Args {
    model_dir: PathBuf,
    ref_wav: PathBuf,
    out_dir: PathBuf,
    target: String,
    k: usize,
    frames_per_chunk: usize,
    draft_choice: DraftChoice,
}

fn parse_args() -> Args {
    let mut model_dir = PathBuf::from(".cache/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base");
    let mut ref_wav = PathBuf::from("assets/jfk/jfk_voice_clone.wav");
    let mut out_dir = PathBuf::from("/tmp/spec_jfk");
    let mut target = DEFAULT_TARGET.to_string();
    let mut k = 4usize;
    let mut frames_per_chunk = 16usize;
    let mut draft_choice = DraftChoice::Shifted;
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model-dir" => {
                model_dir = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--ref-wav" => {
                ref_wav = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(&raw[i + 1]);
                i += 2;
            }
            "--target" => {
                target = raw[i + 1].clone();
                i += 2;
            }
            "--k" => {
                k = raw[i + 1].parse().context("--k expects integer").unwrap();
                i += 2;
            }
            "--frames-per-chunk" => {
                frames_per_chunk = raw[i + 1]
                    .parse()
                    .context("--frames-per-chunk expects integer")
                    .unwrap();
                i += 2;
            }
            "--draft" => {
                draft_choice = raw[i + 1].parse().expect("--draft");
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: speculative_walkthrough \
                    [--model-dir DIR] [--ref-wav WAV] [--out-dir DIR] \
                    [--target TEXT] [--k N] [--frames-per-chunk N] \
                    [--draft trivial|shifted|topk|ee:N|learned:DIR:N]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg {other:?}");
                std::process::exit(2);
            }
        }
    }
    Args {
        model_dir,
        ref_wav,
        out_dir,
        target,
        k,
        frames_per_chunk,
        draft_choice,
    }
}

fn draft_label(choice: &DraftChoice) -> String {
    match choice {
        DraftChoice::Trivial => "trivial".into(),
        DraftChoice::Shifted => "shifted".into(),
        DraftChoice::TopK => "topk".into(),
        DraftChoice::EarlyExit(n) => format!("ee:{n}"),
        DraftChoice::Learned { dir, n_layers } => format!("learned:{}:{n_layers}", dir.display()),
    }
}

fn pick_device() -> Device {
    Device::Cpu
}

const ONE_SEC_SAMPLES: usize = 24_000;
const REALTIME_STRETCH: f64 = 1.2;
const REALTIME_BUDGET: f64 = 3.0;

const REALTIME_TARGET: &str = "Count one two three four five six seven eight nine ten.";

/// Warm Metal session → stream with [`StreamConfig::realtime_second`].
/// Reports whether 1 s of PCM lands within 1.2 s wall (post-open, after warm-up).
fn metal_realtime_second_check(model_dir: &Path, ref_wav: &Path) -> Result<()> {
    if !is_available(Device::Metal) {
        println!("\nMetal realtime check skipped (Metal not available).");
        return Ok(());
    }
    if std::env::var("VECLIB_MAXIMUM_THREADS").is_err() {
        unsafe {
            std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
        }
    }
    println!("\n┌─ Metal realtime_second check (warm session) ───────────────────");
    println!("│ target: {REALTIME_TARGET:?}");
    let config = StreamConfig::realtime_second();
    let mut tts = VoiceClone::open(model_dir, Device::Metal)?;
    let reference = tts.extract_reference(ref_wav)?;

    let _ = tts.generate(&reference, REALTIME_TARGET)?;

    let t0 = Instant::now();
    let mut to_1s = None;
    let mut samples = 0usize;
    let stats = tts.generate_stream(&reference, REALTIME_TARGET, config, |evt| {
        if let StreamEvent::Pcm(chunk) = evt {
            samples += chunk.samples.len();
            if samples >= ONE_SEC_SAMPLES && to_1s.is_none() {
                to_1s = Some(t0.elapsed().as_secs_f64());
            }
        }
        StreamControl::Continue
    })?;
    let to_1s = to_1s.unwrap_or(stats.wall_secs);
    println!(
        "│ measured   to_1s={to_1s:.2}s  ttfa={:.2}s  wall={:.2}s  audio={:.2}s",
        stats.time_to_first_audio_secs, stats.wall_secs, stats.audio_secs
    );
    let pass = to_1s <= REALTIME_BUDGET;
    let stretch = to_1s <= REALTIME_STRETCH;
    println!(
        "│ => {} {:.2}s ≤ {REALTIME_BUDGET:.1}s budget",
        if pass { "PASS" } else { "FAIL" },
        to_1s
    );
    println!(
        "│    {} {:.2}s ≤ {REALTIME_STRETCH:.1}s stretch goal",
        if stretch { "PASS" } else { "FAIL" },
        to_1s
    );
    println!("└────────────────────────────────────────────────────────────────");
    if std::env::var("RLX_QWEN3_TTS_REALTIME_ASSERT")
        .ok()
        .as_deref()
        == Some("1")
    {
        anyhow::ensure!(
            pass,
            "Metal realtime_second took {to_1s:.2}s (budget {REALTIME_BUDGET:.1}s)"
        );
    }
    Ok(())
}

struct RunRow {
    label: String,
    wall_secs: f64,
    audio_secs: f64,
    rtf: f64,
    ttfa_secs: Option<f64>,
}

fn rtf(wall: f64, audio: f64) -> f64 {
    wall / audio.max(1e-6)
}

fn stream_to_pcm<F>(mut run: F) -> Result<(Vec<f32>, StreamStats)>
where
    F: FnMut(&mut dyn FnMut(StreamEvent) -> StreamControl) -> Result<StreamStats>,
{
    let mut pcm = Vec::<f32>::new();
    let stats = run(&mut |evt| {
        if let StreamEvent::Pcm(chunk) = evt {
            pcm.extend_from_slice(&chunk.samples);
        }
        StreamControl::Continue
    })?;
    Ok((pcm, stats))
}

fn generate_spec_batch(
    tts: &mut VoiceClone,
    reference: &rlx_qwen3_tts::SpeakerReference,
    target: &str,
    k: usize,
    draft_choice: &DraftChoice,
) -> Result<(Vec<f32>, SpecRunStats)> {
    match draft_choice {
        DraftChoice::Trivial => tts.generate_speculative(reference, target, &mut TrivialDraft, k),
        DraftChoice::Shifted => {
            tts.generate_speculative(reference, target, &mut ShiftedHistoryDraft, k)
        }
        DraftChoice::TopK => {
            tts.generate_speculative(reference, target, &mut TopKDraft::default(), k)
        }
        DraftChoice::EarlyExit(n) => tts.generate_speculative_early_exit(reference, target, *n, k),
        DraftChoice::Learned { dir, n_layers } => {
            let talker_cfg = tts.talker_config().clone();
            let mut draft = rlx_qwen3_tts::talker::learned_draft::LearnedDraft::open(
                dir,
                &talker_cfg,
                *n_layers,
            )?;
            tts.generate_speculative_learned_draft(reference, target, &mut draft, k)
        }
    }
}

fn spec_stream_to_pcm<F>(mut run: F) -> Result<(Vec<f32>, StreamStats, SpecRunStats)>
where
    F: FnMut(&mut dyn FnMut(StreamEvent) -> StreamControl) -> Result<(StreamStats, SpecRunStats)>,
{
    let mut pcm = Vec::<f32>::new();
    let (stream_stats, spec_stats) = run(&mut |evt| {
        if let StreamEvent::Pcm(chunk) = evt {
            pcm.extend_from_slice(&chunk.samples);
        }
        StreamControl::Continue
    })?;
    Ok((pcm, stream_stats, spec_stats))
}

fn generate_spec_stream(
    tts: &mut VoiceClone,
    reference: &rlx_qwen3_tts::SpeakerReference,
    target: &str,
    k: usize,
    draft_choice: &DraftChoice,
    stream_cfg: StreamConfig,
) -> Result<(Vec<f32>, StreamStats, SpecRunStats)> {
    match draft_choice {
        DraftChoice::Trivial => {
            let mut draft = TrivialDraft;
            spec_stream_to_pcm(|on_event| {
                tts.generate_speculative_stream(
                    reference,
                    target,
                    &mut draft,
                    k,
                    stream_cfg.clone(),
                    on_event,
                )
            })
        }
        DraftChoice::Shifted => {
            let mut draft = ShiftedHistoryDraft;
            spec_stream_to_pcm(|on_event| {
                tts.generate_speculative_stream(
                    reference,
                    target,
                    &mut draft,
                    k,
                    stream_cfg.clone(),
                    on_event,
                )
            })
        }
        DraftChoice::TopK => {
            let mut draft = TopKDraft::default();
            spec_stream_to_pcm(|on_event| {
                tts.generate_speculative_stream(
                    reference,
                    target,
                    &mut draft,
                    k,
                    stream_cfg.clone(),
                    on_event,
                )
            })
        }
        DraftChoice::EarlyExit(n) => spec_stream_to_pcm(|on_event| {
            tts.generate_speculative_stream_early_exit(
                reference,
                target,
                *n,
                k,
                stream_cfg.clone(),
                on_event,
            )
        }),
        DraftChoice::Learned { dir, n_layers } => {
            let talker_cfg = tts.talker_config().clone();
            let mut draft = rlx_qwen3_tts::talker::learned_draft::LearnedDraft::open(
                dir,
                &talker_cfg,
                *n_layers,
            )?;
            spec_stream_to_pcm(|on_event| {
                tts.generate_speculative_stream_learned_draft(
                    reference,
                    target,
                    &mut draft,
                    k,
                    stream_cfg.clone(),
                    on_event,
                )
            })
        }
    }
}

#[cfg(feature = "dev-whisper-validate")]
mod whisper {
    use super::*;
    use rlx_whisper::SAMPLE_RATE as WHISPER_RATE;
    use rlx_whisper::WhisperRunner;

    const MIN_PEAK: f32 = 1e-4;
    const TARGET_PEAK: f32 = 0.95;

    fn repo_cache() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache")
    }

    pub fn whisper_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
            let p = PathBuf::from(d);
            if p.join("model.safetensors").is_file() {
                return Some(p);
            }
        }
        for name in ["whisper-base.en", "whisper-small.en", "whisper-tiny.en"] {
            let p = repo_cache().join(name);
            if p.join("model.safetensors").is_file() && p.join("tokenizer.json").is_file() {
                return Some(p);
            }
        }
        None
    }

    fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
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
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }

    pub fn transcript_covers_reference(reference: &str, transcript: &str, min_ratio: f32) -> bool {
        let reference_words: Vec<_> = normalize_words(reference)
            .into_iter()
            .filter(|w| w.len() >= 3)
            .collect();
        if reference_words.is_empty() {
            let lower = transcript.to_lowercase();
            return reference
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .all(|w| lower.contains(w));
        }
        let heard = normalize_words(transcript);
        let hits = reference_words
            .iter()
            .filter(|w| heard.iter().any(|h| h == *w || h.contains(w.as_str())))
            .count();
        hits as f32 / reference_words.len() as f32 >= min_ratio
    }

    fn peak(pcm: &[f32]) -> f32 {
        pcm.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    }

    fn scale_to_peak(pcm: &[f32], target: f32) -> Vec<f32> {
        let p = peak(pcm);
        if p < MIN_PEAK {
            return pcm.to_vec();
        }
        let g = target / p;
        pcm.iter().map(|v| v * g).collect()
    }

    fn whisper_runner(dir: &Path) -> Result<WhisperRunner> {
        WhisperRunner::builder()
            .weights(dir.join("model.safetensors"))
            .config_path(dir.join("config.json"))
            .tokenizer_path(dir.join("tokenizer.json"))
            .device(Device::Cpu)
            .language("en")
            .build()
    }

    pub fn transcribe(pcm_24k: &[f32], whisper_dir: &Path) -> Result<String> {
        let scaled = scale_to_peak(pcm_24k, TARGET_PEAK);
        let pcm_16k = resample_linear(&scaled, SAMPLE_RATE, WHISPER_RATE as u32);
        anyhow::ensure!(
            pcm_16k.len() >= WHISPER_RATE / 2,
            "resampled audio too short for Whisper"
        );
        let mut whisper = whisper_runner(whisper_dir)?;
        whisper.transcribe_greedy(&pcm_16k)
    }
}

#[cfg(not(feature = "dev-whisper-validate"))]
mod whisper {
    use super::*;

    pub fn whisper_dir() -> Option<PathBuf> {
        None
    }

    pub fn transcript_covers_reference(
        _reference: &str,
        _transcript: &str,
        _min_ratio: f32,
    ) -> bool {
        true
    }

    pub fn transcribe(_pcm_24k: &[f32], _whisper_dir: &Path) -> Result<String> {
        Ok(String::new())
    }
}

fn main() -> Result<()> {
    let args = parse_args();
    std::fs::create_dir_all(&args.out_dir)?;
    let device = pick_device();
    unsafe {
        std::env::set_var("RLX_QWEN3_TTS_TALKER_EAGER", "1");
    }

    let draft_label = draft_label(&args.draft_choice);

    println!("┌─ Speculative-decode walkthrough ──────────────────────────────");
    println!("│ model:    {}", args.model_dir.display());
    println!("│ ref WAV:  {}", args.ref_wav.display());
    println!("│ target:   {:?}", args.target);
    println!("│ K drafts: {}", args.k);
    println!("│ stream:   progressive({})", args.frames_per_chunk);
    println!("│ draft:    {draft_label}");
    println!("│ device:   {device:?}");
    println!("│ out dir:  {}", args.out_dir.display());
    println!("└───────────────────────────────────────────────────────────────\n");

    let t_open = Instant::now();
    let mut tts = VoiceClone::open(&args.model_dir, device)?;
    println!("opened model in {:.2}s\n", t_open.elapsed().as_secs_f64());

    let reference = tts.extract_reference(&args.ref_wav)?;
    let stream_cfg = StreamConfig::progressive(args.frames_per_chunk).with_chunk_samples(8_000);

    let mut rows: Vec<RunRow> = Vec::new();

    // ---- Baseline (batch) ----
    println!("[baseline batch] generating…");
    let t_base = Instant::now();
    let pcm_base = tts.generate(&reference, &args.target)?;
    let secs_base = t_base.elapsed().as_secs_f64();
    let audio_base = pcm_base.len() as f64 / SAMPLE_RATE as f64;
    let out_base = args.out_dir.join("baseline.wav");
    rlx_qwen3_tts::runner::write_wav_mono(&out_base, &pcm_base, SAMPLE_RATE)?;
    println!(
        "[baseline batch] {:.2}s wall, {:.2}s audio → RTF {:.3}",
        secs_base,
        audio_base,
        rtf(secs_base, audio_base)
    );
    rows.push(RunRow {
        label: "baseline batch".into(),
        wall_secs: secs_base,
        audio_secs: audio_base,
        rtf: rtf(secs_base, audio_base),
        ttfa_secs: None,
    });

    // ---- Speculative (batch) ----
    println!("[spec batch K={} draft={draft_label}] generating…", args.k);
    let t_spec = Instant::now();
    let (pcm_spec, spec_batch_stats) = generate_spec_batch(
        &mut tts,
        &reference,
        &args.target,
        args.k,
        &args.draft_choice,
    )?;
    let secs_spec = t_spec.elapsed().as_secs_f64();
    let audio_spec = pcm_spec.len() as f64 / SAMPLE_RATE as f64;
    let out_spec = args
        .out_dir
        .join(format!("spec_batch_k{}_{draft_label}.wav", args.k));
    rlx_qwen3_tts::runner::write_wav_mono(&out_spec, &pcm_spec, SAMPLE_RATE)?;
    println!(
        "[spec batch K={}] {:.2}s wall, {:.2}s audio → RTF {:.3} (accept {:.1}%, {:.2} tok/verify)",
        args.k,
        secs_spec,
        audio_spec,
        rtf(secs_spec, audio_spec),
        spec_batch_stats.acceptance_rate() * 100.0,
        spec_batch_stats.tokens_per_verify()
    );
    rows.push(RunRow {
        label: format!("spec batch K={}", args.k),
        wall_secs: secs_spec,
        audio_secs: audio_spec,
        rtf: rtf(secs_spec, audio_spec),
        ttfa_secs: None,
    });

    // ---- Baseline (progressive stream) ----
    println!(
        "[baseline stream progressive({})] generating…",
        args.frames_per_chunk
    );
    let t_stream_base = Instant::now();
    let (pcm_stream_base, stream_stats_base) = stream_to_pcm(|on_event| {
        tts.generate_stream(&reference, &args.target, stream_cfg.clone(), on_event)
    })?;
    let secs_stream_base = t_stream_base.elapsed().as_secs_f64();
    let audio_stream_base = pcm_stream_base.len() as f64 / SAMPLE_RATE as f64;
    let out_stream_base = args.out_dir.join("baseline_stream.wav");
    rlx_qwen3_tts::runner::write_wav_mono(&out_stream_base, &pcm_stream_base, SAMPLE_RATE)?;
    println!(
        "[baseline stream] {:.2}s wall, {:.2}s audio → RTF {:.3}, TTFA {:.2}s",
        secs_stream_base,
        audio_stream_base,
        rtf(secs_stream_base, audio_stream_base),
        stream_stats_base.time_to_first_audio_secs
    );
    rows.push(RunRow {
        label: "baseline stream".into(),
        wall_secs: secs_stream_base,
        audio_secs: audio_stream_base,
        rtf: rtf(secs_stream_base, audio_stream_base),
        ttfa_secs: Some(stream_stats_base.time_to_first_audio_secs),
    });

    // ---- Speculative (progressive stream) ----
    println!("[spec stream K={} draft={draft_label}] generating…", args.k);
    let t_stream_spec = Instant::now();
    let (pcm_stream_spec, stream_stats_spec, spec_stream_stats) = generate_spec_stream(
        &mut tts,
        &reference,
        &args.target,
        args.k,
        &args.draft_choice,
        stream_cfg,
    )?;
    let secs_stream_spec = t_stream_spec.elapsed().as_secs_f64();
    let audio_stream_spec = pcm_stream_spec.len() as f64 / SAMPLE_RATE as f64;
    let out_stream_spec = args
        .out_dir
        .join(format!("spec_stream_k{}_{draft_label}.wav", args.k));
    rlx_qwen3_tts::runner::write_wav_mono(&out_stream_spec, &pcm_stream_spec, SAMPLE_RATE)?;
    println!(
        "[spec stream K={}] {:.2}s wall, {:.2}s audio → RTF {:.3}, TTFA {:.2}s (accept {:.1}%, {:.2} tok/verify)",
        args.k,
        secs_stream_spec,
        audio_stream_spec,
        rtf(secs_stream_spec, audio_stream_spec),
        stream_stats_spec.time_to_first_audio_secs,
        spec_stream_stats.acceptance_rate() * 100.0,
        spec_stream_stats.tokens_per_verify()
    );
    rows.push(RunRow {
        label: format!("spec stream K={}", args.k),
        wall_secs: secs_stream_spec,
        audio_secs: audio_stream_spec,
        rtf: rtf(secs_stream_spec, audio_stream_spec),
        ttfa_secs: Some(stream_stats_spec.time_to_first_audio_secs),
    });

    // ---- Summary ----
    println!("\n┌─ Summary ──────────────────────────────────────────────────────");
    println!(
        "{:<22} {:>7} {:>7} {:>7} {:>7}",
        "Mode", "Wall", "Audio", "RTF", "TTFA"
    );
    for row in &rows {
        println!(
            "{:<22} {:>6.2}s {:>6.2}s {:>6.3} {:>6}",
            row.label,
            row.wall_secs,
            row.audio_secs,
            row.rtf,
            row.ttfa_secs
                .map(|t| format!("{t:.2}s"))
                .unwrap_or_else(|| "—".into())
        );
    }
    let speedup_batch = secs_base / secs_spec.max(1e-6);
    let speedup_stream = secs_stream_base / secs_stream_spec.max(1e-6);
    let ttfa_gain = stream_stats_base.time_to_first_audio_secs
        / stream_stats_spec.time_to_first_audio_secs.max(1e-6);
    println!("│");
    println!(
        "│ batch speedup        {:.2}x (accept {:.1}%, {:.2} tok/verify)",
        speedup_batch,
        spec_batch_stats.acceptance_rate() * 100.0,
        spec_batch_stats.tokens_per_verify()
    );
    println!(
        "│ stream speedup       {:.2}x, TTFA gain {:.2}x (accept {:.1}%)",
        speedup_stream,
        ttfa_gain,
        spec_stream_stats.acceptance_rate() * 100.0
    );
    println!("└────────────────────────────────────────────────────────────────");

    // ---- Whisper validation (dev feature only) ----
    #[cfg(feature = "dev-whisper-validate")]
    {
        if let Some(whisper_dir) = whisper::whisper_dir() {
            println!("\n┌─ Whisper validation (reference coverage ≥ 50%) ───────────────");
            let min_recall = 0.5f32;
            let validations = [
                ("baseline batch", &pcm_base),
                ("spec batch", &pcm_spec),
                ("baseline stream", &pcm_stream_base),
                ("spec stream", &pcm_stream_spec),
            ];
            let mut all_ok = true;
            for (label, pcm) in validations {
                let transcript = whisper::transcribe(pcm, &whisper_dir)?;
                let ok =
                    whisper::transcript_covers_reference(&args.target, &transcript, min_recall);
                println!("│ {label:<18} {ok}  whisper: {transcript:?}");
                all_ok &= ok;
            }
            println!("└────────────────────────────────────────────────────────────────");
            anyhow::ensure!(
                all_ok,
                "Whisper validation failed — at least one output did not cover the target text"
            );
            println!("\n✓ All outputs passed Whisper round-trip validation.");
        } else {
            println!("\nWhisper validation skipped (no weights). Run `just fetch-whisper-base`.");
            println!("  {}", out_base.display());
            println!("  {}", out_spec.display());
            println!("  {}", out_stream_base.display());
            println!("  {}", out_stream_spec.display());
        }
    }
    #[cfg(not(feature = "dev-whisper-validate"))]
    {
        println!("\nWhisper validation disabled (rebuild with `--features dev-whisper-validate`).");
    }

    metal_realtime_second_check(&args.model_dir, &args.ref_wav)?;

    Ok(())
}
