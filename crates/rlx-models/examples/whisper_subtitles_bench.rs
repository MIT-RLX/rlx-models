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

// Whisper subtitles pipeline bench — ASR + segment timestamps + word alignment + VAD + diarize.
//
// Stages reported: `asr_ms` (encode + cross + prefill + decode), `align_ms` (DTW),
// `diar_ms` (speaker clustering).
//
// ```bash
// just fetch-whisper fetch-whisper-bench
// just bench-whisper-subtitles -- --device metal --modes timestamps+dtw
// just bench-whisper-subtitles-all-backends -- --modes timestamps+dtw
// ```
//
// Flags: `--region-batch N`, `--no-parallel-align`, `--all-backends`, `--parallel-backends`.

use anyhow::Context;
use rlx_cli::parse_device;
use rlx_models::diarize::{DiarizeConfig, DiarizeSession};
use rlx_models::whisper::{
    WhisperRunner, WhisperTranscript, WordAlignMode, assign_speakers, jfk_wav_path,
    load_wav_mono_f32,
};
use rlx_runtime::{Device, is_available};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn default_model_dir() -> PathBuf {
    env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".cache/whisper-tiny"))
}

fn default_wav() -> PathBuf {
    jfk_wav_path()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchMode {
    Plain,
    Timestamps,
    TimestampsDtw,
    TimestampsDtwSilero,
    TimestampsDtwDiarize,
}

impl BenchMode {
    fn all() -> &'static [Self] {
        &[
            Self::Plain,
            Self::Timestamps,
            Self::TimestampsDtw,
            Self::TimestampsDtwSilero,
            Self::TimestampsDtwDiarize,
        ]
    }

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Timestamps => "timestamps",
            Self::TimestampsDtw => "timestamps+dtw",
            Self::TimestampsDtwSilero => "timestamps+dtw+silero",
            Self::TimestampsDtwDiarize => "timestamps+dtw+diarize",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "plain" => Some(Self::Plain),
            "timestamps" => Some(Self::Timestamps),
            "timestamps+dtw" | "dtw" => Some(Self::TimestampsDtw),
            "timestamps+dtw+silero" | "silero" => Some(Self::TimestampsDtwSilero),
            "timestamps+dtw+diarize" | "diarize" => Some(Self::TimestampsDtwDiarize),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct StageMs {
    asr: f64,
    align: f64,
    diarize: f64,
}

#[derive(Debug, Clone)]
struct BenchRow {
    mode: BenchMode,
    stages: StageMs,
    segments: usize,
    words: usize,
    text_len: usize,
}

impl BenchRow {
    fn total_ms(&self) -> f64 {
        self.stages.asr + self.stages.align + self.stages.diarize
    }
}

fn transcript_stats(t: &WhisperTranscript) -> (usize, usize, usize) {
    let segments = t.segments.len();
    let words: usize = t.segments.iter().map(|s| s.words.len()).sum();
    let text_len = t.plain_text().len();
    (segments, words, text_len)
}

fn build_runner(
    model_dir: &Path,
    device: Device,
    timestamps: bool,
    pcm: &[f32],
    region_batch: usize,
    parallel_align: bool,
) -> anyhow::Result<WhisperRunner> {
    let weights = model_dir.join("model.safetensors");
    anyhow::ensure!(weights.is_file(), "missing {}", weights.display());
    let mut builder = WhisperRunner::builder()
        .weights(&weights)
        .config_path(model_dir.join("config.json"))
        .tokenizer_path(model_dir.join("tokenizer.json"))
        .device(device)
        .language("en")
        .mel_frames_for_pcm(pcm)
        .parallel_align(parallel_align);
    if region_batch > 0 {
        builder = builder.max_region_batch(region_batch);
    }
    if timestamps {
        builder = builder.timestamps(true);
    }
    builder.build()
}

fn run_mode(runner: &mut WhisperRunner, pcm: &[f32], mode: BenchMode) -> anyhow::Result<BenchRow> {
    let mut stages = StageMs::default();
    let (segments, words, text_len);

    match mode {
        BenchMode::Plain => {
            let t = Instant::now();
            let text = runner.transcribe_greedy(pcm)?;
            stages.asr = t.elapsed().as_secs_f64() * 1000.0;
            segments = 0;
            words = 0;
            text_len = text.len();
        }
        BenchMode::Timestamps => {
            let t = Instant::now();
            let transcript = runner.transcribe_structured(pcm, 1, 0.0)?;
            stages.asr = t.elapsed().as_secs_f64() * 1000.0;
            (segments, words, text_len) = transcript_stats(&transcript);
        }
        BenchMode::TimestampsDtw => {
            let t = Instant::now();
            let mut transcript = runner.transcribe_structured(pcm, 1, 0.0)?;
            stages.asr = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            runner.apply_word_alignment(pcm, &mut transcript, WordAlignMode::Dtw)?;
            stages.align = t.elapsed().as_secs_f64() * 1000.0;
            (segments, words, text_len) = transcript_stats(&transcript);
        }
        BenchMode::TimestampsDtwSilero => {
            let t = Instant::now();
            let mut transcript = runner.transcribe_structured_silero(pcm)?;
            stages.asr = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            runner.apply_word_alignment(pcm, &mut transcript, WordAlignMode::Dtw)?;
            stages.align = t.elapsed().as_secs_f64() * 1000.0;
            (segments, words, text_len) = transcript_stats(&transcript);
        }
        BenchMode::TimestampsDtwDiarize => {
            let t = Instant::now();
            let mut transcript = runner.transcribe_structured(pcm, 1, 0.0)?;
            stages.asr = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            runner.apply_word_alignment(pcm, &mut transcript, WordAlignMode::Dtw)?;
            stages.align = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let mut diar = DiarizeSession::new(DiarizeConfig::default());
            assign_speakers(&mut diar, pcm, &mut transcript)?;
            stages.diarize = t.elapsed().as_secs_f64() * 1000.0;
            (segments, words, text_len) = transcript_stats(&transcript);
        }
    }

    Ok(BenchRow {
        mode,
        stages,
        segments,
        words,
        text_len,
    })
}

fn mean_ms(rows: &[BenchRow], pick: fn(&StageMs) -> f64) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().map(|r| pick(&r.stages)).sum::<f64>() / rows.len() as f64
}

fn print_header(
    device: Device,
    wav: &Path,
    pcm_len: usize,
    model_dir: &Path,
    warmup: usize,
    runs: usize,
) {
    let duration = pcm_len as f64 / 16_000.0;
    println!(
        "whisper_subtitles_bench device={device:?} model={} wav={} duration={duration:.2}s warmup={warmup} runs={runs}",
        model_dir.display(),
        wav.display(),
    );
    println!(
        "{:<28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6}",
        "mode", "asr_ms", "align_ms", "diar_ms", "total_ms", "rtf", "segs", "words", "chars"
    );
}

fn print_row(row: &BenchRow, pcm_len: usize) {
    let duration_ms = pcm_len as f64 / 16_000.0 * 1000.0;
    let total = row.total_ms();
    let rtf = total / duration_ms;
    println!(
        "{:<28} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.3} {:>6} {:>6} {:>6}",
        row.mode.name(),
        row.stages.asr,
        row.stages.align,
        row.stages.diarize,
        total,
        rtf,
        row.segments,
        row.words,
        row.text_len,
    );
}

fn print_summary(mode: BenchMode, rows: &[BenchRow], pcm_len: usize) {
    if rows.len() <= 1 {
        return;
    }
    let duration_ms = pcm_len as f64 / 16_000.0 * 1000.0;
    let mean_total = rows.iter().map(BenchRow::total_ms).sum::<f64>() / rows.len() as f64;
    let mean_asr = mean_ms(rows, |s| s.asr);
    let mean_align = mean_ms(rows, |s| s.align);
    let mean_diar = mean_ms(rows, |s| s.diarize);
    println!(
        "  mean({}) asr={mean_asr:.1} align={mean_align:.1} diar={mean_diar:.1} total={mean_total:.1} rtf={:.3}",
        mode.name(),
        mean_total / duration_ms,
    );
}

#[allow(clippy::vec_init_then_push)]
fn all_backend_devices() -> Vec<Device> {
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    out.push(Device::Cuda);
    #[cfg(feature = "metal")]
    out.push(Device::Metal);
    #[cfg(feature = "mlx")]
    out.push(Device::Mlx);
    #[cfg(feature = "rocm")]
    out.push(Device::Rocm);
    #[cfg(feature = "gpu")]
    out.push(Device::Gpu);
    #[cfg(feature = "vulkan")]
    out.push(Device::Vulkan);
    out.push(Device::Cpu);
    out
}

fn bench_device(
    model_dir: &Path,
    pcm: &[f32],
    wav: &Path,
    device: Device,
    modes: &[BenchMode],
    warmup: usize,
    runs: usize,
    region_batch: usize,
    parallel_align: bool,
) -> anyhow::Result<()> {
    if !is_available(device) {
        eprintln!("skip: {device:?} not available");
        return Ok(());
    }

    print_header(device, wav, pcm.len(), model_dir, warmup, runs);

    for &mode in modes {
        let timestamps = mode != BenchMode::Plain;
        for _ in 0..warmup {
            let mut runner = build_runner(
                model_dir,
                device,
                timestamps,
                pcm,
                region_batch,
                parallel_align,
            )?;
            let _ = run_mode(&mut runner, pcm, mode)?;
        }

        let mut rows = Vec::with_capacity(runs);
        let mut runner = build_runner(
            model_dir,
            device,
            timestamps,
            pcm,
            region_batch,
            parallel_align,
        )?;
        for _ in 0..runs {
            rows.push(run_mode(&mut runner, pcm, mode)?);
        }

        for row in &rows {
            print_row(row, pcm.len());
        }
        print_summary(mode, &rows, pcm.len());
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = env::args().skip(1).filter(|a| a != "--").collect();
    let model_dir = match args.first() {
        Some(p) if !p.starts_with('-') => PathBuf::from(args.remove(0)),
        _ => default_model_dir(),
    };
    let mut device = Device::Cpu;
    let mut wav = default_wav();
    let mut warmup = 1usize;
    let mut runs = 1usize;
    let mut all_backends = false;
    let mut parallel_backends = false;
    let mut region_batch = 0usize;
    let mut parallel_align = true;
    let mut modes: Vec<BenchMode> = BenchMode::all().to_vec();

    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--device" => device = parse_device(&it.next().context("--device")?)?,
            other if other.starts_with("--device=") => {
                device = parse_device(other.trim_start_matches("--device="))?;
            }
            "--wav" => wav = PathBuf::from(it.next().context("--wav")?),
            "--warmup" => warmup = it.next().context("value")?.parse()?,
            "--runs" => runs = it.next().context("value")?.parse()?,
            "--all-backends" => all_backends = true,
            "--parallel-backends" => parallel_backends = true,
            "--region-batch" => {
                region_batch = it.next().context("value")?.parse()?;
            }
            "--no-parallel-align" => parallel_align = false,
            "--modes" => {
                modes.clear();
                let list = it.next().context("--modes")?;
                for part in list.split(',') {
                    let m = BenchMode::parse(part.trim())
                        .ok_or_else(|| anyhow::anyhow!("unknown mode {part:?}"))?;
                    modes.push(m);
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "whisper_subtitles_bench [MODEL_DIR] [--device NAME] [--wav PATH] \
                     [--warmup N] [--runs N] [--all-backends] [--parallel-backends] \
                     [--region-batch N] [--no-parallel-align] \
                     [--modes plain,timestamps,timestamps+dtw,timestamps+dtw+silero,timestamps+dtw+diarize]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag: {other}"),
        }
    }

    anyhow::ensure!(
        model_dir.is_dir(),
        "model dir not found: {}",
        model_dir.display()
    );

    let pcm = if wav.is_file() {
        load_wav_mono_f32(&wav)?
    } else {
        eprintln!("wav not found ({}); using 10 s silence", wav.display());
        vec![0.0f32; 16_000 * 10]
    };

    if all_backends {
        let devices: Vec<Device> = all_backend_devices()
            .into_iter()
            .filter(|&d| is_available(d))
            .collect();
        if parallel_backends && devices.len() > 1 {
            use std::sync::mpsc;
            let (tx, rx) = mpsc::channel();
            std::thread::scope(|s| {
                for d in devices {
                    let tx = tx.clone();
                    let model_dir = model_dir.clone();
                    let wav = wav.clone();
                    let pcm = pcm.clone();
                    let modes = modes.clone();
                    s.spawn(move || {
                        let mut buf = Vec::new();
                        let r = (|| {
                            print_header(d, &wav, pcm.len(), &model_dir, warmup, runs);
                            for &mode in &modes {
                                let timestamps = mode != BenchMode::Plain;
                                for _ in 0..warmup {
                                    let mut runner = build_runner(
                                        &model_dir,
                                        d,
                                        timestamps,
                                        &pcm,
                                        region_batch,
                                        parallel_align,
                                    )?;
                                    let _ = run_mode(&mut runner, &pcm, mode)?;
                                }
                                let mut runner = build_runner(
                                    &model_dir,
                                    d,
                                    timestamps,
                                    &pcm,
                                    region_batch,
                                    parallel_align,
                                )?;
                                let mut rows = Vec::with_capacity(runs);
                                for _ in 0..runs {
                                    rows.push(run_mode(&mut runner, &pcm, mode)?);
                                }
                                for row in &rows {
                                    buf.push((mode, row.clone()));
                                }
                                print_summary(mode, &rows, pcm.len());
                            }
                            Ok::<_, anyhow::Error>(())
                        })();
                        let _ = tx.send((d, r, buf));
                    });
                }
                drop(tx);
            });
            let mut collected: Vec<_> = rx.iter().collect();
            collected.sort_by_key(|(d, _, _)| format!("{d:?}"));
            for (d, r, rows) in collected {
                println!("--- {d:?} ---");
                if let Err(e) = r {
                    eprintln!("bench {d:?} failed: {e:#}");
                    continue;
                }
                for (mode, row) in rows {
                    print_row(&row, pcm.len());
                    let _ = mode;
                }
                println!();
            }
        } else {
            for d in devices {
                println!("--- {d:?} ---");
                bench_device(
                    &model_dir,
                    &pcm,
                    &wav,
                    d,
                    &modes,
                    warmup,
                    runs,
                    region_batch,
                    parallel_align,
                )?;
                println!();
            }
        }
    } else {
        bench_device(
            &model_dir,
            &pcm,
            &wav,
            device,
            &modes,
            warmup,
            runs,
            region_batch,
            parallel_align,
        )?;
    }
    Ok(())
}
