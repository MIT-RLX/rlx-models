// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Build a distillation set from public datasets: features in, teacher
//! probabilities out.
//!
//! Quantisation-aware training needs no labels here — the target is whatever
//! the f32 teacher says — so this only has to cover the input distribution.
//! Two datasets do that between them:
//!
//! * **LibriSpeech `clean/validation`** (`openslr/librispeech_asr`) — read
//!   speech, the positive class.
//! * **ESC-50** (`ashraq/esc50`) — 50 classes of environmental sound, the
//!   negative class that matters: rain, engines, animals, machinery. Training a
//!   VAD against silence alone teaches it nothing.
//!
//! Both are non-gated parquet. Audio is decoded with symphonia, resampled to
//! 16 kHz, and augmented with gain and additive-noise variants.
//!
//! ```text
//! cargo run -p rlx-ten-vad --release --example dump_distill_set -- --clips 400
//! ```

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use rlx_ten_vad_core::frontend::{Frontend, pre_emphasis};
use rlx_ten_vad_core::net::Net;
use rlx_ten_vad_core::{CONTEXT_FRAMES, FEATURE_LEN, HOP_SIZE, weights::embedded_net};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Datasets this size do not belong in the repo checkout.
fn cache_dir() -> PathBuf {
    let four = Path::new("/Volumes/FOUR/datasets");
    if four.parent().is_some_and(Path::exists) {
        return four.to_path_buf();
    }
    std::env::var("HF_HOME").map_or_else(
        |_| std::env::temp_dir().join("hf-datasets"),
        |h| PathBuf::from(h).join("datasets"),
    )
}

fn fetch(repo: &str, file: &str) -> anyhow::Result<PathBuf> {
    let api = ApiBuilder::new().with_cache_dir(cache_dir()).build()?;
    let r = api.repo(Repo::with_revision(
        repo.to_string(),
        RepoType::Dataset,
        "main".to_string(),
    ));
    eprintln!("  fetching {repo}/{file} …");
    Ok(r.get(file)?)
}

/// Pull the raw encoded bytes out of a HuggingFace `audio` column, which is a
/// struct of `{ bytes, path }`.
fn audio_blobs(parquet: &Path, limit: usize) -> anyhow::Result<Vec<Vec<u8>>> {
    let reader = SerializedFileReader::new(File::open(parquet)?)?;
    let mut out = Vec::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        for (name, field) in row.get_column_iter() {
            if name != "audio" {
                continue;
            }
            if let Field::Group(g) = field {
                for (sub, val) in g.get_column_iter() {
                    if let (true, Field::Bytes(b)) = (sub == "bytes", val) {
                        out.push(b.data().to_vec());
                    }
                }
            }
        }
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

/// Decode any container symphonia knows, downmix to mono, resample to 16 kHz.
fn decode_16k(bytes: &[u8]) -> Option<Vec<i16>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes.to_vec())),
        Default::default(),
    );
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?.clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;
    let rate = track.codec_params.sample_rate? as f64;
    let channels = track.codec_params.channels.map_or(1, |c| c.count());

    let mut mono: Vec<f32> = Vec::new();
    while let Ok(packet) = format.next_packet() {
        let Ok(decoded) = decoder.decode(&packet) else {
            continue;
        };
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if mono.is_empty() {
        return None;
    }

    // Linear interpolation. Any artefacts become part of the input
    // distribution rather than label noise — the teacher labels what it is given.
    let ratio = rate / 16_000.0;
    let out_len = (mono.len() as f64 / ratio) as usize;
    Some(
        (0..out_len)
            .map(|i| {
                let x = i as f64 * ratio;
                let (a, f) = (x.floor() as usize, x.fract() as f32);
                let s0 = mono[a.min(mono.len() - 1)];
                let s1 = mono[(a + 1).min(mono.len() - 1)];
                ((s0 + (s1 - s0) * f) * 32767.0).clamp(-32768.0, 32767.0) as i16
            })
            .collect(),
    )
}

/// Deterministic LCG — reproducible augmentation without a dependency.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

fn frames_of(pcm: &[i16], feats: &mut Vec<f32>, probs: &mut Vec<f32>) {
    let mut fe = Frontend::new(rlx_ten_vad_core::weights::embedded());
    let mut net = Net::new(embedded_net());
    let mut prev = 0.0f32;
    let mut emph = vec![0.0f32; HOP_SIZE];
    for hop in pcm.chunks_exact(HOP_SIZE) {
        let raw: Vec<f32> = hop.iter().map(|&s| f32::from(s)).collect();
        pre_emphasis(&raw, &mut prev, &mut emph);
        fe.push(&raw, &emph);
        let ctx = fe.context();
        feats.extend_from_slice(&ctx[(CONTEXT_FRAMES - 1) * FEATURE_LEN..]);
        probs.push(net.forward(ctx));
    }
}

fn main() -> anyhow::Result<()> {
    let mut clips = 300usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--clips" {
            clips = args.next().and_then(|v| v.parse().ok()).unwrap_or(clips);
        }
    }

    let sources: [(&str, &str, usize); 2] = [
        (
            "openslr/librispeech_asr",
            "clean/validation/0000.parquet",
            clips,
        ),
        (
            "ashraq/esc50",
            "data/train-00000-of-00002-2f1ab7b824ec751f.parquet",
            clips / 2,
        ),
    ];

    let (mut feats, mut probs) = (Vec::new(), Vec::new());
    let mut rng = Rng(0x5eed);
    let mut decoded_total = 0usize;

    for (repo, file, want) in sources {
        let path = fetch(repo, file)?;
        let blobs = audio_blobs(&path, want)?;
        let mut decoded = 0usize;
        for b in &blobs {
            let Some(pcm) = decode_16k(b) else { continue };
            if pcm.len() < HOP_SIZE * 4 {
                continue;
            }
            decoded += 1;
            frames_of(&pcm, &mut feats, &mut probs);
            // Level and SNR variation, so the VAD sees more than one operating point.
            let gain: Vec<i16> = pcm
                .iter()
                .map(|&s| (f32::from(s) * 0.3).clamp(-32768.0, 32767.0) as i16)
                .collect();
            frames_of(&gain, &mut feats, &mut probs);
            let noisy: Vec<i16> = pcm
                .iter()
                .map(|&s| (f32::from(s) + rng.next_f32() * 600.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            frames_of(&noisy, &mut feats, &mut probs);
        }
        eprintln!("  {repo}: {decoded} clips decoded of {} rows", blobs.len());
        decoded_total += decoded;
    }

    // Near-silence, which neither dataset contains much of.
    for &amp in &[3.0f32, 40.0] {
        let n: Vec<i16> = (0..16_000 * 5)
            .map(|_| (rng.next_f32() * amp) as i16)
            .collect();
        frames_of(&n, &mut feats, &mut probs);
    }

    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/distill");
    std::fs::create_dir_all(&out)?;
    std::fs::write(
        out.join("features.f32"),
        feats
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    std::fs::write(
        out.join("teacher.f32"),
        probs
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;

    let speech = probs.iter().filter(|&&p| p >= 0.5).count();
    println!(
        "{decoded_total} clips -> {} frames ({:.0} s), {speech} speech / {} non-speech",
        probs.len(),
        probs.len() as f32 * 0.016,
        probs.len() - speech
    );
    println!("wrote {}", out.display());
    Ok(())
}
