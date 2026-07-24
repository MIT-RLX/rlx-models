// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train custom wake words **in RLX only**.
//!
//! ```text
//! rlx-wake-train cnn --pos DIR --neg DIR --keyword "hey rlx" --out model.safetensors
//! rlx-wake-train mlp     --pos DIR --neg DIR --keyword "hey rlx" --out mlp.safetensors
//! rlx-wake-train synth   --out-dir .cache/wake_train/demo   # write pos/neg WAVs
//! ```

use anyhow::{Result, bail};
use rlx_wake::cnn::{WakeCnnConfig, WakeCnnWeights};
use rlx_wake::train::dataset::{load_pos_neg_dirs, synth_pos_neg_dataset, write_synth_corpus};
use rlx_wake::train::mlp::{MlpConfig, MlpWeights, clips_to_mel_features, train_mlp};
use rlx_wake::train::sgd::SgdConfig;
use rlx_wake::train::cnn::{CnnTrainConfig, train_wake_cnn};
use rlx_wake::weights_io::save_f32_map;
use rlx_wake::{bench_device_label, bind_streaming_device, parse_device_list};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_help();
        return Ok(());
    }
    match args[0].as_str() {
        "cnn" => cmd_cnn(&args[1..]),
        "mlp" => cmd_mlp(&args[1..]),
        "synth" => cmd_synth(&args[1..]),
        other => bail!("unknown subcommand {other}"),
    }
}

fn print_help() {
    println!(
        "\
rlx-wake-train — custom wake-word training in RLX (no PyTorch / ONNX train)

  cnn --pos DIR --neg DIR --keyword NAME --out FILE [--epochs N] [--lr F] [--device DEV]
  mlp     --pos DIR --neg DIR --keyword NAME --out FILE [--epochs N] [--device DEV]
  synth   --out-dir DIR [--n-pos N] [--n-neg N]

--device accepts cpu|metal|mlx|cuda|…|all|apple-silicon (comma list).
Numerics use rlx-cpu BLAS; --device validates each RLX backend is available.
positives/ and negatives/ must contain 16 kHz (or resamplable) mono WAVs.
"
    );
}

fn parse_pair<'a>(args: &'a [String], flag: &str) -> Result<&'a str> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--" {
            i += 1;
            continue;
        }
        if args[i] == flag {
            return Ok(args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))?);
        }
        i += 1;
    }
    bail!("missing {flag}");
}

fn parse_opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--" {
            i += 1;
            continue;
        }
        if args[i] == flag {
            return args.get(i + 1).map(|s| s.as_str());
        }
        i += 1;
    }
    None
}

fn parse_opt_usize(args: &[String], flag: &str, default: usize) -> usize {
    parse_opt(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_opt_f32(args: &[String], flag: &str, default: f32) -> f32 {
    parse_opt(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn bind_devices(args: &[String]) -> Result<Vec<rlx_runtime::Device>> {
    let spec = parse_opt(args, "--device").unwrap_or("cpu");
    let devices = parse_device_list(spec)?;
    for d in &devices {
        let _ = bind_streaming_device(*d)?;
    }
    Ok(devices)
}

fn load_clips(args: &[String]) -> Result<Vec<rlx_wake::train::LabeledClip>> {
    if args.iter().any(|a| a == "--synth") {
        return Ok(synth_pos_neg_dataset(8, 8, 1.2));
    }
    let pos = PathBuf::from(parse_pair(args, "--pos")?);
    let neg = PathBuf::from(parse_pair(args, "--neg")?);
    load_pos_neg_dirs(&pos, &neg)
}

fn cmd_cnn(args: &[String]) -> Result<()> {
    let keyword = parse_pair(args, "--keyword").unwrap_or("wake");
    let out = PathBuf::from(parse_pair(args, "--out")?);
    let epochs = parse_opt_usize(args, "--epochs", 50);
    let lr = parse_opt_f32(args, "--lr", 1e-2);
    let devices = bind_devices(args)?;
    let clips = load_clips(args)?;
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    let mut cfg = CnnTrainConfig {
        keyword: keyword.into(),
        ..CnnTrainConfig::default()
    };
    cfg.sgd = SgdConfig {
        lr,
        epochs,
        log_every: 5,
        weight_decay: 1e-4,
    };
    // Bind every requested backend slot, then train once (CPU BLAS numerics).
    for d in &devices {
        let (_, label) = bind_streaming_device(*d)?;
        eprintln!("[rlx-wake-train cnn] device={label} keyword={}", cfg.keyword);
    }
    let report = train_wake_cnn(&mut w, &clips, &cfg);
    w.save(&out)?;
    eprintln!(
        "saved {}  devices={}  loss {:.4}->{:.4}  train_acc={:.1}%  improved={}",
        out.display(),
        devices
            .iter()
            .map(|d| bench_device_label(*d))
            .collect::<Vec<_>>()
            .join(","),
        report.initial_loss,
        report.final_loss,
        report.train_acc * 100.0,
        report.improved()
    );
    Ok(())
}

fn cmd_mlp(args: &[String]) -> Result<()> {
    let keyword = parse_pair(args, "--keyword").unwrap_or("wake");
    let out = PathBuf::from(parse_pair(args, "--out")?);
    let epochs = parse_opt_usize(args, "--epochs", 80);
    let devices = bind_devices(args)?;
    for d in &devices {
        let (_, label) = bind_streaming_device(*d)?;
        eprintln!("[rlx-wake-train mlp] device={label}");
    }
    let clips = load_clips(args)?;
    let n_mels = 32;
    let feats = clips_to_mel_features(&clips, n_mels);
    let mut w = MlpWeights::new(
        MlpConfig {
            in_dim: n_mels,
            hidden: 64,
        },
        0x51A7,
    );
    let sgd = SgdConfig {
        lr: 1e-2,
        epochs,
        log_every: 10,
        weight_decay: 1e-4,
    };
    let report = train_mlp(&mut w, &feats, &sgd, keyword);
    let mut map = HashMap::new();
    map.insert("mlp.in_dim".into(), vec![w.cfg.in_dim as f32]);
    map.insert("mlp.hidden".into(), vec![w.cfg.hidden as f32]);
    map.insert("mlp.fc1.weight".into(), w.fc1_w);
    map.insert("mlp.fc1.bias".into(), w.fc1_b);
    map.insert("mlp.fc2.weight".into(), w.fc2_w);
    map.insert("mlp.fc2.bias".into(), w.fc2_b);
    save_f32_map(&out, &map)?;
    eprintln!(
        "saved {}  devices={}  loss {:.4}->{:.4}  train_acc={:.1}%",
        out.display(),
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

fn cmd_synth(args: &[String]) -> Result<()> {
    let out = PathBuf::from(parse_pair(args, "--out-dir")?);
    let n_pos = parse_opt_usize(args, "--n-pos", 8);
    let n_neg = parse_opt_usize(args, "--n-neg", 8);
    write_synth_corpus(&out, n_pos, n_neg)?;
    eprintln!("wrote synth corpus under {}", out.display());
    Ok(())
}
