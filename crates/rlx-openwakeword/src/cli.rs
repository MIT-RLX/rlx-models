// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::{Result, bail};
use rlx_wake::{
    SAMPLE_RATE_16K, WakeConfig, WakeEngine, bind_streaming_device, load_wav_mono_f32, parse_device_list,
    peak_score, resample_linear, score_wav,
};
use std::path::PathBuf;

use crate::{OpenWakeWordEngine, OpenWakeWordWeights};

pub fn run(args: &[String]) -> Result<()> {
    let mut wav: Option<PathBuf> = None;
    let mut weights_dir: Option<PathBuf> = None;
    let mut device_s = "cpu".to_string();
    let mut threshold = 0.5f32;
    let mut keyword = "wake".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--wav" => {
                i += 1;
                wav = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--wav needs path"))?,
                ));
            }
            "--weights" => {
                i += 1;
                weights_dir = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--weights needs dir"))?,
                ));
            }
            "--device" => {
                i += 1;
                device_s = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--device needs name"))?
                    .clone();
            }
            "--threshold" => {
                i += 1;
                threshold = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--threshold needs value"))?
                    .parse()?;
            }
            "--keyword" => {
                i += 1;
                keyword = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--keyword needs name"))?
                    .clone();
            }
            "--" => {}
            "-h" | "--help" => {
                println!(
                    "rlx-openwakeword --wav PATH [--weights DIR] [--device cpu|metal|…|all] [--threshold 0.5] [--keyword NAME]"
                );
                return Ok(());
            }
            other => bail!("unknown arg {other}"),
        }
        i += 1;
    }
    let wav = wav.ok_or_else(|| anyhow::anyhow!("--wav is required"))?;
    let devices = parse_device_list(&device_s)?;
    let weights = if let Some(dir) = weights_dir {
        OpenWakeWordWeights::load_dir(&dir, &keyword)?
    } else {
        OpenWakeWordWeights::stub(&keyword)
    };
    let cfg = WakeConfig {
        threshold,
        keyword: keyword.clone(),
        ..WakeConfig::default()
    };
    let (sr, pcm) = load_wav_mono_f32(&wav)?;
    let pcm = if sr != SAMPLE_RATE_16K {
        resample_linear(&pcm, sr, SAMPLE_RATE_16K)
    } else {
        pcm
    };
    for device in devices {
        let (_, label) = bind_streaming_device(device)?;
        let mut eng =
            OpenWakeWordEngine::new(weights.clone(), cfg.clone()).with_device_label(label);
        let steps = score_wav(&mut eng, &pcm)?;
        let peak = peak_score(&steps);
        let fires: Vec<_> = steps.iter().filter(|s| s.fired).collect();
        println!(
            "keyword={} device={} peak={:.4} fires={} steps={}",
            eng.keyword(),
            eng.device_label(),
            peak,
            fires.len(),
            steps.len()
        );
        for f in &fires {
            println!("  fire t_ms={:.1} score={:.4}", f.t_ms, f.score);
        }
    }
    Ok(())
}
