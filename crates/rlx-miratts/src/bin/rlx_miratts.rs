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

use std::path::PathBuf;

use rlx_miratts::{MiraConfig, MiraTts};
use rlx_runtime::{Device, parse_device};

/// Write a mono f32 waveform as a 16-bit PCM WAV.
fn write_wav(path: &str, samples: &[f32], sr: u32) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

fn read_wav_mono(path: &PathBuf) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect(),
        hound::SampleFormat::Int => r
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect(),
    };
    let interleaved = samples?;
    let mono = if ch <= 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, sr))
}

fn resample_linear(pcm: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || pcm.is_empty() {
        return pcm.to_vec();
    }
    let n_out = ((pcm.len() as u64 * to as u64) / from as u64).max(1) as usize;
    let mut out = vec![0f32; n_out];
    let scale = from as f64 / to as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let src = i as f64 * scale;
        let j = src.floor() as usize;
        let frac = (src - j as f64) as f32;
        let a = pcm[j.min(pcm.len() - 1)];
        let b = pcm[(j + 1).min(pcm.len() - 1)];
        *o = a + (b - a) * frac;
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dir = PathBuf::from("weights/tts/miratts");
    let mut text = "The quick brown fox jumps over the lazy dog.".to_string();
    let mut out = PathBuf::from("miratts_out.wav");
    let mut ref_wav: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" if i + 1 < args.len() => {
                dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--text" if i + 1 < args.len() => {
                text = args[i + 1].clone();
                i += 2;
            }
            "--out" | "-o" if i + 1 < args.len() => {
                out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--ref-wav" if i + 1 < args.len() => {
                ref_wav = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--device" if i + 1 < args.len() => {
                device = parse_device(&args[i + 1])?;
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "rlx-miratts --dir DIR [--text TEXT] [--ref-wav WAV] [--device cpu] [-o out.wav]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let cfg = MiraConfig::load(&dir).unwrap_or_default();
    println!("MiraTTS — Qwen2-0.5B + FastBiCodec (native s_encoder + detokenizer) on {device:?}");
    println!(
        "config: hidden={} layers={} heads={} kv={} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size
    );
    let mut tts = MiraTts::load(&dir, device)?;
    let wav = if let Some(ref_path) = ref_wav {
        let (pcm, sr) = read_wav_mono(&ref_path)?;
        let pcm16 = resample_linear(&pcm, sr, tts.sample_rate());
        println!(
            "ref: {} ({} samples @ {} → 16 kHz)",
            ref_path.display(),
            pcm.len(),
            sr
        );
        tts.synthesize_with_ref(&text, &pcm16, 0)?
    } else {
        // No ref: zeros globals (unconditioned) — prefer --ref-wav for cloning.
        let global = vec![0u32; 32];
        tts.synthesize(&text, &global, &global, 0)?
    };
    write_wav(
        out.to_str().unwrap_or("miratts_out.wav"),
        &wav,
        tts.sample_rate(),
    )?;
    println!(
        "wrote {} ({:.2}s, {} samples)",
        out.display(),
        wav.len() as f32 / tts.sample_rate() as f32,
        wav.len()
    );
    Ok(())
}
