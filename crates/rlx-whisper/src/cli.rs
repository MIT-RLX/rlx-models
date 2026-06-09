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

// RLX CLI
use crate::WhisperRunner;
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut language: Option<String> = None;
    let mut translate = false;
    let mut beam_size = 0usize;
    let mut use_f16 = false;
    let mut use_vad = false;
    let mut max_region_batch = 0usize;
    let mut encoder_attn_chunk = 0usize;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--tokenizer" => tokenizer = Some(req(args, &mut i)?.into()),
            "--wav" => wav = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--language" | "--lang" => language = Some(req(args, &mut i)?),
            "--translate" => {
                translate = true;
                i += 1;
            }
            "--beam-size" => {
                beam_size = req(args, &mut i)?.parse().context("--beam-size: usize")?;
            }
            "--f16" => {
                use_f16 = true;
                i += 1;
            }
            "--vad" => {
                use_vad = true;
                i += 1;
            }
            "--max-region-batch" => {
                max_region_batch = req(args, &mut i)?
                    .parse()
                    .context("--max-region-batch: usize")?;
            }
            "--encoder-attn-chunk" => {
                encoder_attn_chunk = req(args, &mut i)?
                    .parse()
                    .context("--encoder-attn-chunk: usize")?;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-whisper — OpenAI Whisper ASR\n\
                     Flags: --weights PATH [--config PATH] [--tokenizer PATH] [--wav PATH]\n\
                       [--device cpu|metal|cuda|…] [--lang en] [--translate] [--beam-size N]\n\
                       [--f16] [--vad] [--max-region-batch N] [--encoder-attn-chunk N] [--dry]"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_standard_device("whisper", &device)?;

    eprintln!("[rlx-whisper] weights={weights:?} device={device:?} wav={wav:?} lang={language:?}");
    let mut builder = WhisperRunner::builder().weights(&weights).device(device);
    if let Some(cfg) = config {
        builder = builder.config_path(cfg);
    }
    if let Some(tok) = tokenizer {
        builder = builder.tokenizer_path(tok);
    }
    if let Some(lang) = language {
        builder = builder.language(lang);
    }
    if translate {
        builder = builder.translate(true);
    }
    if beam_size > 0 {
        builder = builder.beam_size(beam_size);
    }
    if use_f16 {
        builder = builder.use_f16_compute(true);
    }
    if use_vad {
        builder = builder.vad_config(crate::vad::VadConfig::default());
    }
    if max_region_batch > 0 {
        builder = builder.max_region_batch(max_region_batch);
    }
    if encoder_attn_chunk > 0 {
        builder = builder.encoder_attn_chunk(encoder_attn_chunk);
    }
    let mut runner = builder.build()?;
    let cfg = runner.config().clone();
    eprintln!(
        "[rlx-whisper] compiled encoder — d_model={} enc_layers={} dec_layers={} mel={}",
        cfg.d_model, cfg.encoder_layers, cfg.decoder_layers, cfg.num_mel_bins,
    );

    if dry {
        eprintln!("[rlx-whisper] --dry set; skipping forward pass");
        return Ok(());
    }

    #[cfg(feature = "tokenizer")]
    if let Some(wav_path) = wav {
        let t0 = std::time::Instant::now();
        let pcm = crate::audio::load_wav_mono_f32(&wav_path)?;
        let text = if use_vad {
            runner.transcribe_with_vad(&pcm)?
        } else if beam_size > 1 {
            runner.transcribe_beam(&pcm)?
        } else {
            runner.transcribe_greedy(&pcm)?
        };
        eprintln!("[rlx-whisper] transcribed in {:?}:\n{text}", t0.elapsed());
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    let hidden = if let Some(wav_path) = wav {
        runner.encode_wav(&wav_path)?
    } else {
        let sr = crate::audio::SAMPLE_RATE;
        let pcm: Vec<f32> = (0..sr)
            .map(|i| (440.0 * 2.0 * std::f32::consts::PI * i as f32 / sr as f32).sin() * 0.2)
            .collect();
        runner.encode_pcm(&pcm)?
    };
    let dt = t0.elapsed();
    let norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!(
        "[rlx-whisper] encoder out in {dt:?} — len={} ||h||₂={norm:.3}",
        hidden.len()
    );

    #[cfg(not(feature = "tokenizer"))]
    if wav.is_some() {
        bail!("rebuild with --features tokenizer to transcribe (--wav)");
    }

    Ok(())
}
