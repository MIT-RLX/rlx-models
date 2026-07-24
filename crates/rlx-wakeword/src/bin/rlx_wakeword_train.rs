// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train one or many wake phrases in RLX → bundle (optional ternary + `.rlxw` pack).
//!
//! ```text
//! # Synth N words (word0..wordN-1), ternarize FC
//! rlx-wakeword-train --out-dir DIR --synth-n 4 --epochs 20 --ternary
//!
//! # Explicit phrases (dirs)
//! rlx-wakeword-train --out-dir DIR \
//!   --phrase hey_rlx=data/hey_rlx/pos:data/hey_rlx/neg \
//!   --phrase assist=data/assist/pos:data/assist/neg
//!
//! # Auto-discover phrases-dir/<id>/{positives,negatives}/
//! rlx-wakeword-train --out-dir DIR --phrases-dir data/phrases --epochs 40
//! ```

use anyhow::{Result, bail};
use rlx_wake::{TernaryOpts, bind_streaming_device, parse_device_list};
use rlx_wakeword::bundle::pack_rlxw;
use rlx_wakeword::train::{
    PhraseTrainSpec, TrainOpts, parse_phrase_arg, specs_from_phrases_dir, train_phrases,
    train_synth_n, validate_hop_ms,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return Ok(());
    }
    let out_dir =
        PathBuf::from(flag(&args, "--out-dir").ok_or_else(|| anyhow::anyhow!("--out-dir required"))?);
    let mut opts = TrainOpts::default();
    opts.epochs = flag(&args, "--epochs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(opts.epochs);
    opts.lr = flag(&args, "--lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(opts.lr);
    opts.threshold = flag(&args, "--threshold")
        .and_then(|s| s.parse().ok())
        .unwrap_or(opts.threshold);
    if args.iter().any(|a| a == "--ternary") {
        let layers = flag(&args, "--ternary-layers").unwrap_or_else(|| "fc".into());
        let mut topts = TernaryOpts::parse(&layers).ok_or_else(|| {
            anyhow::anyhow!("--ternary-layers must be fc|all|conv (got {layers})")
        })?;
        if let Some(kf) = flag(&args, "--ternary-keep").and_then(|s| s.parse().ok()) {
            topts.keep_frac = kf;
        }
        opts.ternary = Some(topts);
    }
    let hop_ms: u32 = flag(&args, "--hop-ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    opts.hop_samples = validate_hop_ms(hop_ms)?;

    let device_spec = flag(&args, "--device").unwrap_or_else(|| "cpu".into());
    for d in parse_device_list(&device_spec)? {
        let (_, label) = bind_streaming_device(d)?;
        eprintln!("[rlx-wakeword-train] device={label}");
    }

    let specs = collect_specs(&args)?;
    let bundle = if let Some(n) = flag(&args, "--synth-n").and_then(|s| s.parse().ok()) {
        train_synth_n(n, &opts, Some(&out_dir))?
    } else if specs.is_empty() && args.iter().any(|a| a == "--synth") {
        let id = flag(&args, "--phrase").unwrap_or_else(|| "wake".into());
        // bare --phrase id with --synth
        let id = if id.contains('=') {
            parse_phrase_arg(&id)?.0
        } else {
            id
        };
        train_phrases(&[PhraseTrainSpec::synth(id)], &opts, Some(&out_dir))?
    } else {
        if specs.is_empty() {
            bail!("provide --synth-n N, --synth, --phrase ID=POS:NEG (repeatable), or --phrases-dir DIR");
        }
        train_phrases(&specs, &opts, Some(&out_dir))?
    };

    if let Some(pack) = flag(&args, "--pack") {
        pack_rlxw(&out_dir, &PathBuf::from(&pack))?;
        eprintln!("packed {pack}");
    }
    eprintln!(
        "saved bundle {}  phrases={}",
        out_dir.display(),
        bundle
            .config
            .phrases
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}

fn collect_specs(args: &[String]) -> Result<Vec<PhraseTrainSpec>> {
    if let Some(dir) = flag(args, "--phrases-dir") {
        return specs_from_phrases_dir(PathBuf::from(dir).as_path());
    }
    let mut specs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--phrase" {
            let v = args
                .get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("--phrase needs value"))?;
            let (id, pos, neg) = parse_phrase_arg(v)?;
            if let (Some(pos), Some(neg)) = (pos, neg) {
                specs.push(PhraseTrainSpec::from_dirs(id, pos, neg));
            } else if args.iter().any(|a| a == "--synth") {
                specs.push(PhraseTrainSpec::synth(id));
            } else if let (Some(pos), Some(neg)) = (flag(args, "--pos"), flag(args, "--neg")) {
                // legacy single-phrase: --phrase ID --pos --neg
                specs.push(PhraseTrainSpec::from_dirs(id, pos, neg));
            } else if specs.is_empty() {
                // defer empty; main handles --synth alone
            } else {
                bail!("--phrase {id} needs ID=POS:NEG or --synth");
            }
            i += 2;
            continue;
        }
        i += 1;
    }
    // legacy: --pos/--neg without --phrase=
    if specs.is_empty() {
        if let (Some(pos), Some(neg)) = (flag(args, "--pos"), flag(args, "--neg")) {
            let id = flag(args, "--phrase").unwrap_or_else(|| "wake".into());
            let id = parse_phrase_arg(&id).map(|(i, _, _)| i).unwrap_or(id);
            specs.push(PhraseTrainSpec::from_dirs(id, pos, neg));
        }
    }
    Ok(specs)
}

fn print_help() {
    println!(
        "\
rlx-wakeword-train — multi-phrase wake training in RLX

  --out-dir DIR              output bundle (manifest.json + phrase_*.safetensors)
  --synth-n N                train N synthetic phrases word0..wordN-1
  --synth                    synth data for --phrase ID (no dirs)
  --phrase ID=POS:NEG        repeatable; POS/NEG are WAV directories
  --phrase ID                with --synth
  --phrases-dir DIR          auto: DIR/<id>/positives + negatives
  --epochs N  --lr F  --threshold F  --hop-ms 20|32|40|80
  --ternary                      after SGD: exact {{-1,0,+1}} weights (bake TQ2 / fused kernels)
  --ternary-layers fc|all|conv   which tensors (default fc)
  --ternary-keep F               fraction of |w| kept as ±1 (default ~0.333)
  --device cpu|all  --pack FILE.rlxw
"
    );
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}
