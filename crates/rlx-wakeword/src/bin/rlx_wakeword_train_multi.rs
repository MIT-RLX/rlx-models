// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train several wake phrases as ONE network (one output per phrase) and write a bundle that
//! `WakewordBundle::load_dir` runs as a single streaming forward per hop.
//!
//!   rlx-wakeword-train-multi --phrase hey_peachy=POS_DIR[:POS_DIR…] --phrase bye_peachy=… \
//!     --neg NEG_DIR [--neg …] --out-dir OUT [--window-ms 1200] [--model lite|mid|full] \
//!     [--device cpu|metal] [--epochs 100] [--batch 1024] [--lr 4e-3] [--threshold 0.5] \
//!     [--mine-threshold 0.2] [--mine-ratio 2] [--mine-every 10] [--gain-db 12] [--pool-chunks 4] [--mel-cache DIR]
//!
//! A file under a phrase's directories is that phrase's positive (copies count), and is dropped
//! from the negatives if a negative directory holds it too; each negative file is used once
//! (symlinked training sets often repeat them). Mel
//! features are cached (default `~/.cache/rlx-wake/mel`, `--mel-cache none` to skip), so a
//! second run over the same clips starts training in seconds.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_wake::train::multi::{
    MultiClip, MultiTrainConfig, init_multi, load_mels_cached, train_multi,
};
use rlx_wake::{WakeCnnConfig, parse_device_list};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flags(args: &[String], name: &str) -> Vec<String> {
    args.windows(2)
        .filter(|w| w[0] == name)
        .map(|w| w[1].clone())
        .collect()
}

fn wavs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")))
        .collect();
    v.sort();
    Ok(v)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let phrases: Vec<(String, Vec<PathBuf>)> = flags(&args, "--phrase")
        .iter()
        .map(|s| {
            let (id, dirs) = s
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--phrase ID=DIR[:DIR…], got {s}"))?;
            Ok((id.to_string(), dirs.split(':').map(PathBuf::from).collect()))
        })
        .collect::<Result<_>>()?;
    if phrases.is_empty() {
        bail!("need --phrase ID=DIR (one per phrase)");
    }
    let out =
        PathBuf::from(flag(&args, "--out-dir").ok_or_else(|| anyhow::anyhow!("need --out-dir"))?);
    let parse =
        |n: &str, d: f64| -> f64 { flag(&args, n).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let window_ms = parse("--window-ms", 1200.0) as f32;
    let model = flag(&args, "--model").unwrap_or_else(|| "mid".into());
    let mut cfg = WakeCnnConfig::named(&model)
        .ok_or_else(|| anyhow::anyhow!("--model lite|mid|full, got {model}"))?;
    // Order-aware pooling: the window's last layer averaged in this many stretches, in order.
    cfg.pool_chunks = parse("--pool-chunks", 4.0) as usize;
    let device = *parse_device_list(&flag(&args, "--device").unwrap_or_else(|| "cpu".into()))?
        .first()
        .unwrap_or(&rlx_wake::Device::Cpu);
    let threshold = parse("--threshold", 0.5) as f32;
    let tc = MultiTrainConfig {
        epochs: parse("--epochs", 100.0) as usize,
        batch: parse("--batch", 1024.0) as usize,
        lr: parse("--lr", 4e-3) as f32,
        window_frames: (window_ms / 10.0).round() as usize,
        mine_threshold: parse("--mine-threshold", 0.2) as f32,
        mine_ratio: parse("--mine-ratio", 2.0) as f32,
        mine_every: parse("--mine-every", 10.0) as usize,
        gain_db: parse("--gain-db", 12.0) as f32,
        ..MultiTrainConfig::default()
    };
    let cache = match flag(&args, "--mel-cache").as_deref() {
        Some("none") => None,
        Some(d) => Some(PathBuf::from(d)),
        None => std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/rlx-wake/mel")),
    };

    // Clips: each phrase's, then negatives that are nobody's positive.
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut classes: Vec<Option<usize>> = Vec::new();
    let mut taken: HashSet<PathBuf> = HashSet::new();
    for (c, (_, dirs)) in phrases.iter().enumerate() {
        for d in dirs {
            for p in wavs(d)? {
                taken.insert(std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone()));
                paths.push(p);
                classes.push(Some(c));
            }
        }
    }
    // Negatives once each (training sets link the same file under several names and dirs;
    // mining, not copies, gives the hard ones their weight).
    let mut dropped = 0;
    for d in flags(&args, "--neg") {
        for p in wavs(Path::new(&d))? {
            if !taken.insert(std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone())) {
                dropped += 1;
                continue;
            }
            paths.push(p);
            classes.push(None);
        }
    }
    let counts: Vec<String> = phrases
        .iter()
        .enumerate()
        .map(|(c, (id, _))| format!("{id} {}", classes.iter().filter(|&&k| k == Some(c)).count()))
        .collect();
    eprintln!(
        "[rlx-wakeword-train-multi] {} clips: {}, negatives {} ({dropped} duplicates or positives, dropped); window {window_ms} ms, {model}, {device:?}",
        paths.len(),
        counts.join(", "),
        classes.iter().filter(|c| c.is_none()).count()
    );
    let t = std::time::Instant::now();
    let mels = load_mels_cached(&paths, cache.as_deref())?;
    eprintln!(
        "[rlx-wakeword-train-multi] mel features in {:.1?} (cache {})",
        t.elapsed(),
        cache
            .as_ref()
            .map_or("off".into(), |c| c.display().to_string())
    );
    let clips: Vec<MultiClip> = mels
        .into_iter()
        .zip(classes)
        .map(|(mel, class)| MultiClip { mel, class })
        .collect();

    let mut w = init_multi(cfg, phrases.len(), 0x5EED);
    let r = train_multi(&mut w, &clips, &tc, device)?;
    for ((id, _), (found, total)) in phrases.iter().zip(&r.recall) {
        eprintln!("  {id}: {found}/{total} training positives at 0.5");
    }
    eprintln!(
        "  loss {:.4} -> {:.4}, negatives firing {}/{}, {:.2} s/epoch, optimizer {}",
        r.initial_loss,
        r.final_loss,
        r.negatives_firing.0,
        r.negatives_firing.1,
        r.secs_per_epoch,
        if r.resident {
            "on the device"
        } else {
            "host-chained"
        }
    );

    std::fs::create_dir_all(&out)?;
    w.save(&out.join("wake.safetensors"))?;
    let man = serde_json::json!({
        "hop_ms": 40,
        "context_ms": window_ms,
        "min_hops": 3,
        "phrases": phrases.iter().enumerate().map(|(i, (id, _))| serde_json::json!({
            "id": id, "threshold": threshold, "weights": "wake.safetensors", "output": i
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&man)? + "\n",
    )?;
    eprintln!("[rlx-wakeword-train-multi] wrote {}", out.display());
    Ok(())
}
