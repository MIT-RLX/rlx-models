// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::{Result, bail};
use std::path::PathBuf;

use crate::bundle::{WakewordBundle, stub_bundle, validate_hop_ms};
use crate::session::WakeEvent;
use rlx_wake::{
    SAMPLE_RATE_16K, bind_streaming_device, load_wav_mono_f32, parse_device_list, resample_linear,
};

pub fn run(args: &[String]) -> Result<()> {
    let mut wav: Option<PathBuf> = None;
    let mut bundle: Option<PathBuf> = None;
    let mut device_s = "cpu".to_string();
    let mut hop_ms = 40u32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--" => {}
            "--wav" => {
                i += 1;
                wav = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--wav needs path"))?,
                ));
            }
            "--bundle" => {
                i += 1;
                bundle = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--bundle needs dir"))?,
                ));
            }
            "--device" => {
                i += 1;
                device_s = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--device needs name"))?
                    .clone();
            }
            "--hop-ms" => {
                i += 1;
                hop_ms = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--hop-ms needs value"))?
                    .parse()?;
            }
            "-h" | "--help" => {
                println!(
                    "rlx-wakeword --wav PATH [--bundle DIR] [--hop-ms 40] [--device cpu|metal|…|all]"
                );
                return Ok(());
            }
            other => bail!("unknown arg {other}"),
        }
        i += 1;
    }
    let wav = wav.ok_or_else(|| anyhow::anyhow!("--wav is required"))?;
    let _ = validate_hop_ms(hop_ms)?;
    let devices = parse_device_list(&device_s)?;

    let mut loaded = if let Some(dir) = bundle {
        WakewordBundle::load_dir(&dir)?
    } else {
        stub_bundle("wake", hop_ms)
    };
    loaded.config.hop_samples = validate_hop_ms(hop_ms)?;

    let (sr, pcm) = load_wav_mono_f32(&wav)?;
    let pcm = if sr != SAMPLE_RATE_16K {
        resample_linear(&pcm, sr, SAMPLE_RATE_16K)
    } else {
        pcm
    };

    for device in devices {
        let (_, label) = bind_streaming_device(device)?;
        let mut sess = loaded.open_session()?.with_device_label(label);
        let events = sess.push(&pcm);
        let cands: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WakeEvent::Candidate { .. }))
            .collect();
        println!(
            "device={} hop_ms={} events={} candidates={}",
            label,
            hop_ms,
            events.len(),
            cands.len()
        );
        for e in &cands {
            if let WakeEvent::Candidate {
                phrase_id,
                score,
                t_ms,
                latency_ms,
                speaker_id,
                speaker_score,
            } = e
            {
                print!(
                    "  candidate phrase={phrase_id} score={score:.4} t_ms={t_ms:.1} latency_ms={latency_ms:.1}"
                );
                if let (Some(sid), Some(ss)) = (speaker_id, speaker_score) {
                    print!(" speaker={sid} speaker_score={ss:.3}");
                }
                println!();
            }
        }
    }
    Ok(())
}
