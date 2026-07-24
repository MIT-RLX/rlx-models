// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train openWakeWord phrase head in RLX (frozen embedding).
//!
//! ```text
//! rlx-openwakeword-train --pos DIR --neg DIR --keyword "hey rlx" \
//!   --out-dir crates/rlx-openwakeword/weights --device all
//! ```

use anyhow::{Result, bail};
use rlx_openwakeword::embedding::EmbeddingWeights;
use rlx_openwakeword::phrase::PhraseWeights;
use rlx_openwakeword::train_phrase_head;
use rlx_wake::train::{SgdConfig, load_pos_neg_dirs, synth_pos_neg_dataset};
use rlx_wake::{bench_device_label, bind_streaming_device, parse_device_list};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "rlx-openwakeword-train --pos DIR --neg DIR --keyword NAME --out-dir DIR \\\n\
               [--epochs N] [--device cpu|metal|…|all|apple-silicon]\n\
             or: rlx-openwakeword-train --synth --keyword NAME --out-dir DIR [--device all]"
        );
        return Ok(());
    }
    let keyword = flag(&args, "--keyword").unwrap_or_else(|| "wake".into());
    let out_dir = PathBuf::from(
        flag(&args, "--out-dir").ok_or_else(|| anyhow::anyhow!("--out-dir required"))?,
    );
    let epochs: usize = flag(&args, "--epochs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let device_spec = flag(&args, "--device").unwrap_or_else(|| "cpu".into());
    let devices = parse_device_list(&device_spec)?;
    for d in &devices {
        let (_, label) = bind_streaming_device(*d)?;
        eprintln!("[rlx-openwakeword-train] device={label}");
    }
    let clips = if args.iter().any(|a| a == "--synth") {
        synth_pos_neg_dataset(8, 8, 2.0)
    } else {
        let pos = PathBuf::from(flag(&args, "--pos").ok_or_else(|| anyhow::anyhow!("--pos"))?);
        let neg = PathBuf::from(flag(&args, "--neg").ok_or_else(|| anyhow::anyhow!("--neg"))?);
        load_pos_neg_dirs(&pos, &neg)?
    };
    if clips.is_empty() {
        bail!("empty dataset");
    }

    let embed = EmbeddingWeights::stub(32);
    let mut phrase = PhraseWeights::stub(&keyword);
    let sgd = SgdConfig {
        lr: 5e-3,
        epochs,
        log_every: 5,
        weight_decay: 1e-4,
    };
    let report = train_phrase_head(&mut phrase, &embed, &clips, &sgd)?;
    std::fs::create_dir_all(&out_dir)?;
    embed.save(&out_dir.join("embedding.safetensors"))?;
    phrase.save(&out_dir.join("phrase.safetensors"))?;
    eprintln!(
        "saved phrase head under {}  devices={}  loss {:.4}->{:.4} acc={:.1}%",
        out_dir.display(),
        devices
            .iter()
            .map(|d| bench_device_label(*d))
            .collect::<Vec<_>>()
            .join(","),
        report.initial_loss,
        report.final_loss,
        report.train_acc * 100.0
    );
    Ok(())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}
