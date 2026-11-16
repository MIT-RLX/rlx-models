// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train several wake phrases as ONE network (one output per phrase) and write a bundle that
//! `WakewordBundle::load_dir` runs as a single streaming forward per hop.
//!
//!   rlx-wakeword-train-multi --phrase hey=POS_DIR[:POS_DIR…] --phrase bye=… \
//!     --neg NEG_DIR [--neg …] --out-dir OUT [--window-ms 1200] [--model lite|mid|full] \
//!     [--device cpu|metal] [--epochs 100] [--batch 1024] [--lr 4e-3] [--threshold 0.5] \
//!     [--mine-threshold 0.2] [--mine-ratio 2] [--mine-every 10] [--gain-db 12] [--pool-chunks 4] [--mel-cache DIR]
//!     [--noise DIR [--noise …]] [--noise-p 0.5] [--noise-snr 0,25] [--mask-p 0.3] [--pos-tail-ms 200]
//!
//! Architecture: `--res-blocks N` (dilated residual blocks), `--attn` (attention pooling),
//! `--pcen` (PCEN front end), `--dual` (log-mel and PCEN side by side), `--bn` (batch
//! normalization while training, folded on export). Distillation: `--teacher FILE:OUT[,OUT…]`
//! (repeatable; a trained network whose scores join those outputs' targets, at
//! `--teacher-weight`, default 0.5; `--teacher-window-ms`: the window the teachers were trained
//! on, when the student's is longer, e.g. a `--causal` one).
//! `--save-pack FILE` writes the loaded clips and noise as one file (`--epochs 0`: and stops);
//! `--pack FILE` trains from such a file instead of `--phrase` / `--neg` / `--noise`.
//!
//! `--pos-tail-ms`: positives' windows also end up to this long after the clip, so a phrase
//! keeps its output up for the hops a detection needs (`min_hops`), not one instant.
//!
//! `--noise`: recordings (or generated noise) whose stretches are mixed into training windows
//! on the fly, a fresh draw each time (`--noise-p` of the windows, at `--noise-snr` dB under
//! positives, 10 dB lower still for negatives). `--mask-p`: SpecAugment-style masking of a mel
//! band and a short stretch of time in that share of windows.
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
    MultiClip, MultiTrainConfig, Teacher, init_multi, load_mels_cached, train_multi_taught,
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
    let pack = flag(&args, "--pack");
    if phrases.is_empty() && pack.is_none() {
        bail!("need --phrase ID=DIR (one per phrase), or --pack FILE");
    }
    let out =
        PathBuf::from(flag(&args, "--out-dir").ok_or_else(|| anyhow::anyhow!("need --out-dir"))?);
    let parse =
        |n: &str, d: f64| -> f64 { flag(&args, n).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let has = |n: &str| args.iter().any(|a| a == n);
    let window_ms = parse("--window-ms", 1200.0) as f32;
    let model = flag(&args, "--model").unwrap_or_else(|| "mid".into());
    let mut cfg = WakeCnnConfig::named(&model)
        .ok_or_else(|| anyhow::anyhow!("--model lite|mid|full, got {model}"))?;
    // Order-aware pooling: the window's last layer averaged in this many stretches, in order.
    cfg.pool_chunks = parse("--pool-chunks", 4.0) as usize;
    cfg.res_blocks = parse("--res-blocks", 0.0) as usize;
    cfg.attn = has("--attn");
    cfg.pcen = has("--pcen");
    cfg.dual = has("--dual");
    cfg.causal = has("--causal");
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
        noise_p: parse("--noise-p", 0.5) as f32,
        noise_snr_db: match flag(&args, "--noise-snr") {
            Some(s) => {
                let (a, b) = s
                    .split_once(',')
                    .ok_or_else(|| anyhow::anyhow!("--noise-snr LO,HI dB, got {s}"))?;
                (a.trim().parse()?, b.trim().parse()?)
            }
            None => (0.0, 25.0),
        },
        mask_p: parse("--mask-p", 0.3) as f32,
        pos_tail: (parse("--pos-tail-ms", 200.0) / 10.0).round() as usize,
        bn: has("--bn"),
        teacher_weight: parse("--teacher-weight", 0.5) as f32,
        ..MultiTrainConfig::default()
    };
    let cache = match flag(&args, "--mel-cache").as_deref() {
        Some("none") => None,
        Some(d) => Some(PathBuf::from(d)),
        None => std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/rlx-wake/mel")),
    };

    let (ids, clips, noise) = match &pack {
        Some(p) => {
            let t = std::time::Instant::now();
            let r = read_pack(Path::new(p))?;
            eprintln!(
                "[rlx-wakeword-train-multi] {p}: {} clips, {} noise clips, phrases {:?} ({:.1?})",
                r.1.len(),
                r.2.len(),
                r.0,
                t.elapsed()
            );
            r
        }
        None => load_dirs(&args, &phrases, cache.as_deref(), &tc)?,
    };
    let counts: Vec<String> = ids
        .iter()
        .enumerate()
        .map(|(c, id)| {
            format!(
                "{id} {}",
                clips.iter().filter(|k| k.class == Some(c)).count()
            )
        })
        .collect();
    eprintln!(
        "[rlx-wakeword-train-multi] {} clips: {}, negatives {}; window {window_ms} ms, {model}, {device:?}",
        clips.len(),
        counts.join(", "),
        clips.iter().filter(|c| c.class.is_none()).count()
    );
    if let Some(p) = flag(&args, "--save-pack") {
        write_pack(Path::new(&p), &ids, &clips, &noise)?;
        eprintln!("[rlx-wakeword-train-multi] wrote {p}");
        if tc.epochs == 0 {
            return Ok(());
        }
    }
    if !noise.is_empty() {
        eprintln!(
            "[rlx-wakeword-train-multi] {} noise clips mixed into {:.0}% of windows at {:?} dB SNR; masking {:.0}%",
            noise.len(),
            tc.noise_p * 100.0,
            tc.noise_snr_db,
            tc.mask_p * 100.0
        );
    }

    // Teachers: `FILE:OUT,OUT…` (weights trained before; which outputs they teach).
    let teachers: Vec<Teacher> = flags(&args, "--teacher")
        .iter()
        .map(|s| {
            let (file, outs) = s
                .rsplit_once(':')
                .ok_or_else(|| anyhow::anyhow!("--teacher FILE:OUT[,OUT…], got {s}"))?;
            let weights = rlx_wake::WakeCnnWeights::load(Path::new(file))
                .with_context(|| format!("load teacher {file}"))?;
            let outputs = outs.split(',').map(|o| o.trim().parse()).collect::<Result<Vec<usize>, _>>()?;
            eprintln!("[rlx-wakeword-train-multi] teacher {file} for outputs {outputs:?} (weight {})", tc.teacher_weight);
            let window_frames = flag(&args, "--teacher-window-ms")
                .and_then(|v| v.parse::<f32>().ok())
                .map(|ms| (ms / 10.0).round() as usize);
            Ok(Teacher { weights, outputs, window_frames })
        })
        .collect::<Result<_>>()?;

    let mut w = init_multi(cfg, ids.len(), 0x5EED);
    let r = train_multi_taught(&mut w, &clips, &noise, &teachers, &tc, device)?;
    for (id, (found, total)) in ids.iter().zip(&r.recall) {
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
        "phrases": ids.iter().enumerate().map(|(i, id)| serde_json::json!({
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

/// Clips from `--phrase` / `--neg` directories and noise from `--noise` ones, mel features
/// through the cache: (phrase ids, clips, noise).
fn load_dirs(
    args: &[String],
    phrases: &[(String, Vec<PathBuf>)],
    cache: Option<&Path>,
    _tc: &MultiTrainConfig,
) -> Result<(Vec<String>, Vec<MultiClip>, Vec<Vec<f32>>)> {
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
    for d in flags(args, "--neg") {
        for p in wavs(Path::new(&d))? {
            if !taken.insert(std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone())) {
                dropped += 1;
                continue;
            }
            paths.push(p);
            classes.push(None);
        }
    }
    eprintln!("[rlx-wakeword-train-multi] {dropped} negatives dropped (duplicates or positives)");
    let t = std::time::Instant::now();
    let mels = load_mels_cached(&paths, cache)?;
    eprintln!(
        "[rlx-wakeword-train-multi] mel features in {:.1?} (cache {})",
        t.elapsed(),
        cache.map_or("off".into(), |c| c.display().to_string())
    );
    let clips: Vec<MultiClip> = mels
        .into_iter()
        .zip(classes)
        .map(|(mel, class)| MultiClip { mel, class })
        .collect();
    let noise_paths: Vec<PathBuf> = flags(args, "--noise")
        .iter()
        .map(|d| wavs(Path::new(d)))
        .collect::<Result<Vec<_>>>()?
        .concat();
    let noise = load_mels_cached(&noise_paths, cache)?;
    Ok((
        phrases.iter().map(|(id, _)| id.clone()).collect(),
        clips,
        noise,
    ))
}

const PACK_MAGIC: &[u8; 8] = b"RLXWPK1\n";

/// Everything a training run reads, in one file to copy to another machine: the magic, a JSON
/// header's length (u64 LE) and the header (phrase ids, each clip's class and length, each
/// noise clip's length, in floats), then every clip's mel features and the noise's (f32 LE).
fn write_pack(path: &Path, ids: &[String], clips: &[MultiClip], noise: &[Vec<f32>]) -> Result<()> {
    use std::io::Write;
    let header = serde_json::json!({
        "ids": ids,
        "classes": clips.iter().map(|c| c.class.map_or(-1, |k| k as i64)).collect::<Vec<_>>(),
        "lens": clips.iter().map(|c| c.mel.len()).collect::<Vec<_>>(),
        "noise_lens": noise.iter().map(Vec::len).collect::<Vec<_>>(),
    })
    .to_string();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(PACK_MAGIC)?;
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for m in clips.iter().map(|c| &c.mel).chain(noise) {
        for v in m {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

fn read_pack(path: &Path) -> Result<(Vec<String>, Vec<MultiClip>, Vec<Vec<f32>>)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 16 || &bytes[..8] != PACK_MAGIC {
        bail!("{}: not a training pack", path.display());
    }
    let hl = u64::from_le_bytes(bytes[8..16].try_into()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[16..16 + hl])?;
    let nums = |k: &str| -> Result<Vec<i64>> {
        header[k]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("pack header: no {k}"))?
            .iter()
            .map(|v| {
                v.as_i64()
                    .ok_or_else(|| anyhow::anyhow!("pack header: bad {k}"))
            })
            .collect()
    };
    let ids: Vec<String> = serde_json::from_value(header["ids"].clone())?;
    let (classes, lens, noise_lens) = (nums("classes")?, nums("lens")?, nums("noise_lens")?);
    let mut at = 16 + hl;
    let mut take = |n: usize| -> Result<Vec<f32>> {
        let end = at + n * 4;
        if end > bytes.len() {
            bail!("pack truncated");
        }
        let v = bytes[at..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        at = end;
        Ok(v)
    };
    let clips = classes
        .iter()
        .zip(&lens)
        .map(|(&c, &n)| {
            Ok(MultiClip {
                mel: take(n as usize)?,
                class: (c >= 0).then_some(c as usize),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let noise = noise_lens
        .iter()
        .map(|&n| take(n as usize))
        .collect::<Result<Vec<_>>>()?;
    Ok((ids, clips, noise))
}
