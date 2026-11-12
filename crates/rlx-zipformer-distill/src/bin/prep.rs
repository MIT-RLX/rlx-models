// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Distillation targets: LibriSpeech FLAC → fbank (`rlx-kroko-core`) → the
//! teacher's encoder output on `K`-chunk windows, each from a zero state as
//! the student sees them. Every core runs a share of the files.
//!
//! ```text
//! zipformer-distill-prep --rkr .cache/zipformer-m/m-int8.rkr \
//!     --audio LibriSpeech/train-clean-100 \
//!     --hours 10 --chunks 8 --out .cache/zipformer-distill-10h
//! ```
//! FLAC is decoded in Rust (`symphonia`). The subset is every n-th file (n
//! from `--hours`); `--offset N` starts at the N-th file instead of the first,
//! so a second run with the same `--hours` and `--offset 1` picks a disjoint
//! subset.
//!
//! `--contexts --teacher DIR --data DIR` adds `contexts.u16` to an existing
//! set: the teacher's greedy decoding of each stored target, for training on
//! its token distributions (no audio or encoder pass needed).

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result, anyhow, ensure};
use rlx_kroko_core as core;
use rlx_zipformer_distill::config::{CHUNK_SHIFT, N_MELS, OUT_FRAMES, StudentConfig};
use rlx_zipformer_distill::data;
use rlx_zipformer_distill::init::Teacher;
use rlx_zipformer_distill::joiner::Joiner;

fn flac_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            flac_files(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "flac" || x == "wav") {
            out.push(p);
        }
    }
    Ok(())
}

/// 16 kHz mono FLAC (or WAV) → f32 PCM in [-1, 1].
fn decode(path: &Path) -> Result<Vec<f32>> {
    if path.extension().is_some_and(|x| x == "wav") {
        let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(path)?;
        ensure!(rate == 16_000, "{}: {rate} Hz", path.display());
        return Ok(pcm);
    }
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("flac");
    let mut format = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probe {}", path.display()))?
        .format;
    let track = format.default_track().context("no audio track")?.clone();
    ensure!(
        track.codec_params.sample_rate == Some(16_000),
        "{}: not 16 kHz",
        path.display()
    );
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut out = Vec::new();
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track.id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .with_context(|| format!("decode {}", path.display()))?;
        let spec = *decoded.spec();
        let ch = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        out.extend(
            buf.samples()
                .chunks(ch)
                .map(|f| f.iter().sum::<f32>() / ch as f32),
        );
    }
    ensure!(!out.is_empty(), "{}: no audio", path.display());
    Ok(out)
}

fn aligned(raw: &[u8]) -> Vec<u32> {
    raw.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

/// Greedy-decode every record's target with the teacher's predictor and
/// joiner; write the per-frame contexts beside the records.
fn contexts(teacher_dir: &Path, dir: &Path) -> Result<()> {
    let teacher = Teacher::open(teacher_dir)?;
    let joiner = Joiner::open(&teacher, teacher_dir)?;
    let recs = data::Records::open(dir)?;
    let t0 = std::time::Instant::now();
    let n_threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = recs.n.div_ceil(n_threads);
    let parts: Vec<(Vec<[u16; 2]>, usize)> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let (recs, joiner) = (&recs, &joiner);
                sc.spawn(move || {
                    let (mut out, mut tokens) = (Vec::new(), 0);
                    let (mut f, mut target) = (Vec::new(), Vec::new());
                    for i in t * per..((t + 1) * per).min(recs.n) {
                        f.clear();
                        target.clear();
                        recs.read_into(i, &mut f, &mut target);
                        let c = joiner.contexts(&target);
                        tokens += c.windows(2).filter(|w| w[0] != w[1]).count();
                        out.extend(c);
                    }
                    (out, tokens)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("contexts thread"))
            .collect()
    });
    let tokens: usize = parts.iter().map(|p| p.1).sum();
    let all: Vec<[u16; 2]> = parts.into_iter().flat_map(|p| p.0).collect();
    data::write_contexts(dir, &all)?;
    // The first records' token paths as text, to read against the audio.
    let pieces: Vec<String> = std::fs::read_to_string(teacher_dir.join("tokens.txt"))?
        .lines()
        .map(|l| l.rsplit_once(' ').map_or(l, |p| p.0).to_string())
        .collect();
    let f = data::frames_len(recs.k);
    for rec in all.chunks(f).take(3) {
        let text: String = rec
            .windows(2)
            .filter(|w| w[0] != w[1])
            .filter_map(|w| pieces.get(w[1][1] as usize))
            .map(|p| p.replace('\u{2581}', " "))
            .collect();
        println!("  {}", text.trim());
    }
    println!(
        "{} records, {} frames, {tokens} tokens emitted, in {:.0} s → {}",
        recs.n,
        all.len(),
        t0.elapsed().as_secs_f64(),
        dir.join("contexts.u16").display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
    };
    if args.iter().any(|a| a == "--contexts") {
        let teacher_dir = PathBuf::from(get("--teacher").context("--teacher")?);
        let data = PathBuf::from(get("--data").context("--data")?);
        return contexts(&teacher_dir, &data);
    }
    let raw = std::fs::read(get("--rkr").context("--rkr")?)?;
    let audio = PathBuf::from(get("--audio").context("--audio")?);
    let out = PathBuf::from(get("--out").context("--out")?);
    let hours: f64 = get("--hours")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(10.0);
    let k: usize = get("--chunks").map(|v| v.parse()).transpose()?.unwrap_or(8);
    let offset: usize = get("--offset").map(|v| v.parse()).transpose()?.unwrap_or(0);

    let mut files = Vec::new();
    flac_files(&audio, &mut files)?;
    files.sort();
    // Spread the subset over speakers: every n-th file.
    let est_files = (hours * 3600.0 / 12.7).ceil() as usize; // ~12.7 s per utterance
    let step = (files.len() / est_files.max(1)).max(1);
    let files: Vec<PathBuf> = files.into_iter().skip(offset).step_by(step).collect();
    eprintln!(
        "{} files (every {step}th from #{offset}), target {hours} h",
        files.len()
    );

    let words = aligned(&raw);
    let raw_len = raw.len();
    let n_threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (tx, rx) = mpsc::sync_channel::<(Vec<f32>, Vec<f32>, f64)>(64);
    let seg = StudentConfig::segment_frames(k);
    let budget = hours * 3600.0;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done_s = std::sync::Mutex::new(0f64);
    let mut writer = data::Writer::create(&out, k)?;
    let t0 = std::time::Instant::now();
    let (_, secs) = std::thread::scope(|sc| -> Result<(usize, f64)> {
        for _ in 0..n_threads {
            let tx = tx.clone();
            let (files, words, next, done_s) = (&files, &words, &next, &done_s);
            sc.spawn(move || -> Result<()> {
                // SAFETY: u32 storage viewed as bytes.
                let bytes =
                    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), raw_len) };
                let model = core::Model::parse(bytes).map_err(|e| anyhow!("{e}"))?;
                let need = core::scratch_len(&model, seg * 160);
                let (mut fs, mut is, mut page) = (
                    vec![0f32; need.f32s],
                    vec![0i8; need.i8s],
                    vec![0u32; need.page_words],
                );
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= files.len() || *done_s.lock().unwrap() >= budget {
                        return Ok(());
                    }
                    let pcm = decode(&files[i])?;
                    *done_s.lock().unwrap() += pcm.len() as f64 / 16_000.0;
                    let mut fb = vec![0f32; core::fbank_frames(pcm.len()) * N_MELS];
                    let frames = core::fbank_features(&model, pcm.as_slice(), &mut fb);
                    let mut start = 0;
                    while start + seg <= frames {
                        let feats = fb[start * N_MELS..(start + seg) * N_MELS].to_vec();
                        let mut target = Vec::with_capacity(k * OUT_FRAMES * 512);
                        core::encode_features(
                            &model,
                            &mut core::SlicePager(bytes),
                            &feats,
                            seg,
                            &mut fs,
                            &mut is,
                            &mut page,
                            &mut |enc, _| target.extend_from_slice(enc),
                        )
                        .map_err(|e| anyhow!("{e}"))?;
                        target.truncate(k * OUT_FRAMES * 512);
                        if tx.send((feats, target, k as f64 * 0.32)).is_err() {
                            return Ok(());
                        }
                        start += k * CHUNK_SHIFT;
                    }
                }
            });
        }
        drop(tx);
        let (mut n, mut secs) = (0usize, 0f64);
        for (feats, target, s) in rx {
            writer.push(&feats, &target)?;
            n += 1;
            secs += s;
            if n.is_multiple_of(500) {
                eprintln!(
                    "{n} segments, {:.2} h of segments, {:.0} s",
                    secs / 3600.0,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        Ok((n, secs))
    })?;
    let n = writer.finish(
        &out,
        serde_json::json!({"segment_hours": secs / 3600.0, "source": audio.display().to_string(), "offset": offset, "every": step}),
    )?;
    println!(
        "{n} segments ({:.2} h) in {:.0} s → {}",
        secs / 3600.0,
        t0.elapsed().as_secs_f64(),
        out.display()
    );
    Ok(())
}
