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

//! Build JFK manifest + `train_raw.jsonl` for Qwen3-TTS fine-tune.
//!
//! Default **`reference`** mode slices the public-domain inaugural transcript by clip
//! time (6 s segments, 24 kHz). Optional **`whisper`** / **`hybrid`** use RLX Whisper.

use rlx_whisper::runner::WhisperRunnerBuilder;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TranscriptMode {
    Reference,
    Whisper,
    Hybrid,
}

#[derive(Serialize)]
struct ManifestFile {
    path: String,
    duration_sec: f32,
    transcript: Option<String>,
}

#[derive(Serialize)]
struct Manifest {
    sample_rate: u32,
    files: Vec<ManifestFile>,
}

fn main() -> anyhow::Result<()> {
    let mut wav_dir = PathBuf::from(".cache/qwen3-tts/jfk/wavs");
    let mut manifest_out = PathBuf::from(".cache/qwen3-tts/jfk/manifest.json");
    let mut train_jsonl = PathBuf::from(".cache/qwen3-tts/jfk/train_raw.jsonl");
    let mut ref_wav = PathBuf::from(".cache/qwen3-tts/jfk/wavs/jfk_0000.wav");
    let mut reference_file = PathBuf::from("scripts/qwen3_tts_jfk_reference.txt");
    let mut segment_sec = 6.0f32;
    let mut mode = TranscriptMode::Reference;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--wav-dir" => wav_dir = PathBuf::from(it.next().expect("--wav-dir")),
            "--manifest" => manifest_out = PathBuf::from(it.next().expect("--manifest")),
            "--train-jsonl" => train_jsonl = PathBuf::from(it.next().expect("--train-jsonl")),
            "--ref-wav" => ref_wav = PathBuf::from(it.next().expect("--ref-wav")),
            "--reference-file" => {
                reference_file = PathBuf::from(it.next().expect("--reference-file"))
            }
            "--segment-sec" => {
                segment_sec = it.next().expect("--segment-sec").parse()?;
            }
            "--transcript-mode" => {
                mode = match it.next().expect("--transcript-mode").as_str() {
                    "reference" => TranscriptMode::Reference,
                    "whisper" => TranscriptMode::Whisper,
                    "hybrid" => TranscriptMode::Hybrid,
                    other => anyhow::bail!(
                        "unknown --transcript-mode {other} (reference|whisper|hybrid)"
                    ),
                };
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
    }

    if let Ok(v) = std::env::var("JFK_TRANSCRIPT_MODE") {
        mode = match v.to_ascii_lowercase().as_str() {
            "reference" => TranscriptMode::Reference,
            "whisper" => TranscriptMode::Whisper,
            "hybrid" => TranscriptMode::Hybrid,
            other => anyhow::bail!("unknown JFK_TRANSCRIPT_MODE={other}"),
        };
    }
    if let Ok(v) = std::env::var("SEGMENT_SEC") {
        segment_sec = v.parse()?;
    }

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&wav_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    paths.sort_by_key(|a| clip_index(a));

    let max_clips = std::env::var("JFK_MAX_CLIPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if max_clips > 0 && paths.len() > max_clips {
        paths.truncate(max_clips);
    }

    let reference_words = if mode != TranscriptMode::Whisper {
        Some(load_reference_words(&reference_file)?)
    } else {
        None
    };

    let total_duration = total_clip_duration(&paths, segment_sec)?;
    if let Some(words) = &reference_words {
        eprintln!(
            "[jfk-manifest] reference: {} words, {:.1}s audio, {} clips, {:.1}s/clip",
            words.len(),
            total_duration,
            paths.len(),
            segment_sec
        );
    }

    let mut runner = if mode != TranscriptMode::Reference {
        open_whisper_runner()?
    } else {
        None
    };

    let ref_abs = ref_wav.canonicalize().unwrap_or(ref_wav);
    let mut manifest_files = Vec::new();
    let mut jsonl_lines = Vec::new();

    let num_clips = paths.len();
    for path in &paths {
        let idx = clip_index(path);
        let duration_24k = wav_duration_sec(path)?;

        let reference = reference_words
            .as_ref()
            .map(|words| reference_transcript_for_clip(words, idx, num_clips));

        let whisper_text = if let Some(r) = &mut runner {
            let pcm = load_wav_for_whisper(path)?;
            let text = r.transcribe_greedy(&pcm)?;
            let t = text.trim().to_string();
            if t.is_empty() { None } else { Some(t) }
        } else {
            None
        };

        let transcript = match mode {
            TranscriptMode::Reference => Some(reference.unwrap_or_default()),
            TranscriptMode::Whisper => whisper_text,
            TranscriptMode::Hybrid => {
                pick_hybrid(reference.filter(|s| !s.is_empty()), whisper_text)
            }
        };

        let rel = format!("wavs/{}", path.file_name().unwrap().to_string_lossy());
        manifest_files.push(ManifestFile {
            path: rel.clone(),
            duration_sec: duration_24k,
            transcript: transcript.clone(),
        });

        let audio_abs = path.canonicalize().unwrap_or(path.clone());
        let text = transcript.unwrap_or_default();
        let line = serde_json::json!({
            "audio": audio_abs.to_string_lossy(),
            "text": text,
            "ref_audio": ref_abs.to_string_lossy(),
        });
        jsonl_lines.push(line.to_string());
    }

    if let Some(parent) = manifest_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let manifest = Manifest {
        sample_rate: 24_000,
        files: manifest_files,
    };
    std::fs::write(&manifest_out, serde_json::to_string_pretty(&manifest)?)?;
    std::fs::write(&train_jsonl, jsonl_lines.join("\n") + "\n")?;

    eprintln!(
        "manifest {} clips (mode {:?}), jsonl {} lines",
        manifest.files.len(),
        mode,
        jsonl_lines.len()
    );
    Ok(())
}

fn clip_index(path: &Path) -> usize {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("jfk_"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn load_reference_words(path: &Path) -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let normalized: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(normalized.split_whitespace().map(str::to_string).collect())
}

fn total_clip_duration(paths: &[PathBuf], segment_sec: f32) -> anyhow::Result<f32> {
    if paths.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0f32;
    for (i, p) in paths.iter().enumerate() {
        let d = wav_duration_sec(p)?;
        if i + 1 == paths.len() && d < segment_sec * 0.99 {
            total += d;
        } else {
            total += segment_sec;
        }
    }
    Ok(total)
}

fn wav_duration_sec(path: &Path) -> anyhow::Result<f32> {
    let spec = hound::WavReader::open(path)?.spec();
    let n = hound::WavReader::open(path)?.len();
    Ok(n as f32 / spec.sample_rate as f32)
}

/// Even word partition across fixed-length clips (no duplicate words at boundaries).
fn reference_transcript_for_clip(words: &[String], clip_index: usize, num_clips: usize) -> String {
    if words.is_empty() || num_clips == 0 {
        return String::new();
    }
    let n = words.len();
    let start_w = clip_index * n / num_clips;
    let end_w = (clip_index + 1) * n / num_clips;
    let end_w = end_w.max(start_w.saturating_add(1)).min(n);
    words[start_w..end_w].join(" ")
}

fn pick_hybrid(reference: Option<String>, whisper: Option<String>) -> Option<String> {
    match (reference, whisper) {
        (Some(r), Some(w)) if transcript_quality(&r) >= transcript_quality(&w) => Some(r),
        (Some(r), Some(w)) if w.len() > r.len() / 2 => Some(w),
        (r, w) => r.or(w),
    }
}

/// Prefer readable English over punctuation runs / non-Latin noise.
fn transcript_quality(text: &str) -> i32 {
    let t = text.trim();
    if t.is_empty() {
        return 0;
    }
    let letters = t.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let words = t.split_whitespace().count();
    let punct = t.chars().filter(|c| *c == '.' || *c == ',').count();
    (letters as i32) + (words as i32) * 3 - (punct as i32)
}

fn open_whisper_runner() -> anyhow::Result<Option<WhisperRunner>> {
    use rlx_runtime::Device;

    let weights = std::env::var("RLX_WHISPER_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("model.safetensors").is_file())
        .or_else(|| {
            for name in ["whisper-base.en", "whisper-small.en", "whisper-tiny"] {
                let p = PathBuf::from(format!(".cache/{name}"));
                if p.join("model.safetensors").is_file() {
                    return Some(p);
                }
            }
            None
        });

    let Some(dir) = weights else {
        eprintln!("skip whisper: set RLX_WHISPER_DIR or run just fetch-whisper-base");
        return Ok(None);
    };
    let weights_file = dir.join("model.safetensors");
    let device = if rlx_runtime::is_available(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };
    eprintln!("[jfk-manifest] whisper {} on {device:?}", dir.display());
    Ok(Some(
        WhisperRunnerBuilder::default()
            .weights(&weights_file)
            .device(device)
            .build()?,
    ))
}

type WhisperRunner = rlx_whisper::runner::WhisperRunner;

fn load_wav_for_whisper(path: &Path) -> anyhow::Result<Vec<f32>> {
    if let Some(pcm) = resample_ffmpeg_16k(path)? {
        return Ok(pcm);
    }
    let mut pcm = rlx_whisper::load_wav_mono_f32(path)?;
    let spec = hound::WavReader::open(path)?.spec();
    if spec.sample_rate != 16_000 {
        pcm = resample_linear(&pcm, spec.sample_rate, 16_000);
    }
    Ok(pcm)
}

fn resample_ffmpeg_16k(path: &Path) -> anyhow::Result<Option<Vec<f32>>> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return Ok(None);
    }
    let out = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path.to_str().unwrap_or(""),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ])
        .output();
    let out = match out {
        Ok(o) if o.status.success() => o,
        _ => return Ok(None),
    };
    let bytes = out.stdout;
    if bytes.len() % 4 != 0 {
        return Ok(None);
    }
    let pcm: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Some(pcm))
}

fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * to_hz as u64 / from_hz as u64) as usize;
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len {
        let src = (i as f64 * from_hz as f64 / to_hz as f64) as usize;
        out.push(samples[src.min(samples.len() - 1)]);
    }
    out
}
