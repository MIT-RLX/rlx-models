// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Several wake phrases in one network, trained on an RLX device with the optimizer on the
//! device too.
//!
//! The model is [`crate::cnn::WakeCnnWeights`] with `K` outputs (the last layer's width), one
//! sigmoid per phrase: a clip of phrase `c` is a positive for output `c` and a negative for the
//! others, everything else a negative for all. One network instead of one per phrase: one
//! forward per hop at run time, one weight file, and the phrases learn from each other's clips.
//!
//! Faster than [`super::graph_cnn`] where it was slow:
//! - the AdamW update is part of the compiled step ([`rlx_runtime::train::ResidentTrainer`]):
//!   weights and moments stay in device buffers, only the loss comes back;
//! - the next batch is cut on every core while the device runs the current one;
//! - mel features come from a cache on disk ([`load_mels_cached`]): repeat runs skip decoding
//!   and FFTs.
//!
//! Sampling is [`super::graph_cnn`]'s: positives' windows end at the phrase's end (jittered),
//! negatives' anywhere in the clip, plus partial-phrase windows of positives (negatives) and,
//! from `mine_after`, the negative windows the network currently scores highest on any output.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::train::{AdamSpec, ResidentTrainer, TrainableParam};
use rlx_runtime::{Device, Session};

use crate::cnn::{WakeCnnConfig, WakeCnnWeights};
use crate::train::dataset::clip_mel_frames;

/// A clip's mel frames (frame-major, `n_mels` each) and its phrase, if any.
#[derive(Debug, Clone)]
pub struct MultiClip {
    pub mel: Vec<f32>,
    pub class: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MultiTrainConfig {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub weight_decay: f32,
    /// Window, mel frames (10 ms each).
    pub window_frames: usize,
    /// Partial-phrase windows (negatives) per positive per epoch, and how far before the
    /// phrase's end they stop (frames).
    pub partial_ratio: f32,
    pub partial_gap: usize,
    /// Background frames kept after a clip, so windows can end past the phrase.
    pub tail_frames: usize,
    pub mine_every: usize,
    pub mine_after: usize,
    pub mine_stride: usize,
    pub mine_threshold: f32,
    pub mine_ratio: f32,
    pub log_every: usize,
}

impl Default for MultiTrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch: 1024,
            lr: 4e-3,
            weight_decay: 1e-4,
            window_frames: 120,
            partial_ratio: 1.0,
            partial_gap: 50,
            tail_frames: 60,
            mine_every: 10,
            mine_after: 20,
            mine_stride: 8,
            mine_threshold: 0.2,
            mine_ratio: 0.5,
            log_every: 5,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MultiReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    /// The optimizer ran on the device (not the host-chained fallback).
    pub resident: bool,
    pub secs_per_epoch: f64,
    /// Per phrase: training positives detected at 0.5, of how many.
    pub recall: Vec<(usize, usize)>,
    /// Training negatives (end of clip) firing any output at 0.5, of how many.
    pub negatives_firing: (usize, usize),
}

/// A positive's window ends within this many frames of the clip's end.
const POS_JITTER: usize = 3;

fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

const PARAMS: [&str; 10] = ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "conv3_w", "conv3_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"];

/// The CNN over `x [B, T, n_mels]` (frames as stored: a batch is plain copies; the transpose
/// to channel-major is the device's): logits `[B, K]` and the parameter nodes. The 1-D
/// convolutions are `conv2d` over `[B, C, 1, T]` with `1 × k` kernels, in the engine's weight
/// layout. (Rewriting them as GEMMs over shifted views was slower on Metal at these shapes:
/// tall, skinny products plus the slicing.)
fn body(g: &mut Graph, c: &WakeCnnConfig, n_out: usize, b: usize, t: usize) -> (NodeId, Vec<NodeId>) {
    let (k, pad) = (c.k, c.k / 2);
    let x = g.input("x", sh(&[b, t, c.n_mels]));
    let x = g.transpose_(x, vec![0, 2, 1]);
    let x = g.reshape_(x, vec![b as i64, c.n_mels as i64, 1, t as i64]);
    let p: Vec<NodeId> = vec![
        g.param("conv1_w", sh(&[c.c1, c.n_mels, 1, k])),
        g.param("conv1_b", sh(&[1, c.c1, 1, 1])),
        g.param("conv2_w", sh(&[c.c2, c.c1, 1, k])),
        g.param("conv2_b", sh(&[1, c.c2, 1, 1])),
        g.param("conv3_w", sh(&[c.c3, c.c2, 1, k])),
        g.param("conv3_b", sh(&[1, c.c3, 1, 1])),
        g.param("fc1_w", sh(&[c.c3, c.hidden])),
        g.param("fc1_b", sh(&[1, c.hidden])),
        g.param("fc2_w", sh(&[c.hidden, n_out])),
        g.param("fc2_b", sh(&[1, n_out])),
    ];
    let conv = |g: &mut Graph, x, w, bias, stride| -> NodeId {
        let y = g.conv2d(x, w, [1, k], [1, stride], [0, pad], [1, 1], 1);
        let y = g.add(y, bias);
        g.relu(y)
    };
    let h = conv(g, x, p[0], p[1], 1);
    let h = conv(g, h, p[2], p[3], 2);
    let h = conv(g, h, p[4], p[5], 2);
    let pooled = g.mean(h, vec![2, 3], false); // [B, c3]
    let h = g.mm(pooled, p[6]);
    let h = g.add(h, p[7]);
    let h = g.relu(h);
    let z = g.mm(h, p[8]);
    let z = g.add(z, p[9]);
    (z, p)
}

fn build_probs(c: &WakeCnnConfig, n_out: usize, b: usize, t: usize) -> Graph {
    let mut g = Graph::new("wake_multi_probs");
    let (z, _) = body(&mut g, c, n_out, b, t);
    let p = g.sigmoid(z);
    g.set_outputs(vec![p]);
    g
}

/// Mean BCE-with-logits over `[B, K]` against `y [B, K]`.
fn build_loss(c: &WakeCnnConfig, n_out: usize, b: usize, t: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("wake_multi_loss");
    let (z, p) = body(&mut g, c, n_out, b, t);
    let y = g.input("y", sh(&[b, n_out]));
    // max(z, 0) − z·y + ln(1 + e^(−|z|)).
    let pos = g.relu(z);
    let zy = g.mul(z, y);
    let a = g.abs(z);
    let na = g.neg(a);
    let e = g.exp(na);
    let one = g.constant(1.0, DType::F32);
    let e1 = g.add(e, one);
    let soft = g.log(e1);
    let l = g.sub(pos, zy);
    let l = g.add(l, soft);
    let loss = g.mean(l, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    (g, p)
}

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

fn to_graph(w: &WakeCnnWeights) -> HashMap<String, Vec<f32>> {
    let c = &w.cfg;
    let k = w.fc2_b.len();
    let v = [
        w.conv1_w.clone(),
        w.conv1_b.clone(),
        w.conv2_w.clone(),
        w.conv2_b.clone(),
        w.conv3_w.clone(),
        w.conv3_b.clone(),
        transpose(&w.fc1_w, c.hidden, c.c3),
        w.fc1_b.clone(),
        transpose(&w.fc2_w, k, c.hidden),
        w.fc2_b.clone(),
    ];
    PARAMS.iter().map(|s| s.to_string()).zip(v).collect()
}

fn from_graph(w: &mut WakeCnnWeights, p: &HashMap<String, Vec<f32>>) {
    let (c3, hidden, k) = (w.cfg.c3, w.cfg.hidden, w.fc2_b.len());
    w.conv1_w.clone_from(&p["conv1_w"]);
    w.conv1_b.clone_from(&p["conv1_b"]);
    w.conv2_w.clone_from(&p["conv2_w"]);
    w.conv2_b.clone_from(&p["conv2_b"]);
    w.conv3_w.clone_from(&p["conv3_w"]);
    w.conv3_b.clone_from(&p["conv3_b"]);
    w.fc1_w = transpose(&p["fc1_w"], c3, hidden);
    w.fc1_b.clone_from(&p["fc1_b"]);
    w.fc2_w = transpose(&p["fc2_w"], hidden, k);
    w.fc2_b.clone_from(&p["fc2_b"]);
}

/// `he_init` with `n_out` outputs.
pub fn init_multi(cfg: WakeCnnConfig, n_out: usize, seed: u64) -> WakeCnnWeights {
    let mut w = WakeCnnWeights::he_init(cfg, seed);
    let h = w.cfg.hidden;
    let base = WakeCnnWeights::he_init(w.cfg.clone(), seed ^ 0x5151).fc1_w;
    let a = (6.0 / h as f32).sqrt();
    w.fc2_w = (0..n_out * h).map(|i| base[i % base.len()].signum() * a * ((i * 2_654_435_761 % 1000) as f32 / 1000.0)).collect();
    w.fc2_b = vec![0.0; n_out];
    w
}

/// The `t` frames ending at frame `end` (exclusive) of a frame-major clip, frame-major into
/// `dst` (zero frames in front if the clip is short): one contiguous copy.
fn window_into(mel: &[f32], n_mels: usize, end: usize, t: usize, dst: &mut [f32]) {
    let end = end.min(mel.len() / n_mels);
    let take = end.min(t);
    let gap = (t - take) * n_mels;
    dst[..gap].fill(0.0);
    dst[gap..].copy_from_slice(&mel[(end - take) * n_mels..end * n_mels]);
}

/// `(clip, window end, class)` per batch row into `x [B, n_mels·t]` and `y [B, K]`, on every core.
fn fill_batch(items: &[(usize, usize, Option<usize>)], mels: &[Vec<f32>], n_mels: usize, t: usize, n_out: usize, x: &mut [f32], y: &mut [f32]) {
    let per = items.len().div_ceil(8).max(16);
    std::thread::scope(|s| {
        for ((it, xc), yc) in items.chunks(per).zip(x.chunks_mut(per * n_mels * t)).zip(y.chunks_mut(per * n_out)) {
            s.spawn(move || {
                for (j, &(i, e, c)) in it.iter().enumerate() {
                    window_into(&mels[i], n_mels, e, t, &mut xc[j * n_mels * t..(j + 1) * n_mels * t]);
                    let row = &mut yc[j * n_out..(j + 1) * n_out];
                    row.fill(0.0);
                    if let Some(c) = c {
                        row[c] = 1.0;
                    }
                }
            });
        }
    });
}

/// Mel features of WAV files, from `cache` when there (keyed by path, size, modification time
/// and the mel front end), else decoded, computed (on every core) and stored there.
pub fn load_mels_cached(paths: &[PathBuf], cache: Option<&Path>) -> Result<Vec<Vec<f32>>> {
    if let Some(c) = cache {
        std::fs::create_dir_all(c).with_context(|| format!("create {}", c.display()))?;
    }
    let key = |p: &Path| -> Option<PathBuf> {
        let c = cache?;
        let real = std::fs::canonicalize(p).ok()?;
        let md = std::fs::metadata(&real).ok()?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (real, md.len(), md.modified().ok(), "mel-v1-16k-32").hash(&mut h);
        Some(c.join(format!("{:016x}.mel", h.finish())))
    };
    let one = |p: &PathBuf| -> Result<Vec<f32>> {
        let k = key(p);
        if let Some(k) = &k
            && let Ok(b) = std::fs::read(k)
        {
            return Ok(b.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect());
        }
        let (sr, pcm) = crate::load_wav_mono_f32(p).with_context(|| format!("read {}", p.display()))?;
        let pcm = crate::resample_linear(&pcm, sr, crate::SAMPLE_RATE_16K);
        let mel = clip_mel_frames(&pcm);
        if let Some(k) = &k {
            let bytes: Vec<u8> = mel.iter().flat_map(|v| v.to_le_bytes()).collect();
            let tmp = k.with_extension("tmp");
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, k);
            }
        }
        Ok(mel)
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = paths.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let parts: Vec<_> = paths.chunks(per).map(|part| s.spawn(move || part.iter().map(one).collect::<Result<Vec<_>>>())).collect();
        let mut out = Vec::with_capacity(paths.len());
        for h in parts {
            out.extend(h.join().expect("mel thread")?);
        }
        Ok(out)
    })
}

/// Train `w` (`K` = `w.fc2_b.len()` outputs) on `clips` on `device`.
pub fn train_multi(w: &mut WakeCnnWeights, clips: &[MultiClip], cfg: &MultiTrainConfig, device: Device) -> Result<MultiReport> {
    let n_mels = w.cfg.n_mels;
    let n_out = w.fc2_b.len();
    let t = cfg.window_frames.max(8);
    let t0 = Instant::now();
    // Each clip between `t` frames of its own background (its opening, repeated if short) and
    // `tail_frames` more: windows can end from a window before its speech to well after it.
    let lead = t;
    let mels: Vec<Vec<f32>> = clips
        .iter()
        .map(|c| {
            let m = &c.mel;
            let opening = &m[..(cfg.tail_frames.max(lead) * n_mels).min(m.len())];
            let mut ext = Vec::with_capacity(m.len() + (lead + cfg.tail_frames) * n_mels);
            while ext.len() < lead * n_mels && !opening.is_empty() {
                let need = lead * n_mels - ext.len();
                ext.extend_from_slice(&opening[..need.min(opening.len())]);
            }
            ext.extend_from_slice(m);
            ext.extend_from_slice(&opening[..(cfg.tail_frames * n_mels).min(opening.len())]);
            ext
        })
        .collect();
    let frames: Vec<usize> = clips.iter().map(|c| c.mel.len() / n_mels + lead).collect();
    let ext: Vec<usize> = mels.iter().map(|m| m.len() / n_mels).collect();
    let n = clips.len();
    let positives: Vec<usize> = (0..n).filter(|&i| clips[i].class.is_some() && frames[i] >= t + cfg.partial_gap).collect();
    let negatives: Vec<usize> = (0..n).filter(|&i| clips[i].class.is_none()).collect();
    let n_partial = if positives.is_empty() { 0 } else { (positives.len() as f32 * cfg.partial_ratio) as usize };
    // Balance the phrases: each phrase's positives repeated up to the largest phrase's count.
    let per_class: Vec<usize> = (0..n_out).map(|c| (0..n).filter(|&i| clips[i].class == Some(c)).count()).collect();
    let largest = per_class.iter().copied().max().unwrap_or(0);
    let extra: Vec<usize> = (0..n)
        .filter_map(|i| clips[i].class.map(|c| (i, c)))
        .flat_map(|(i, c)| std::iter::repeat_n(i, largest / per_class[c].max(1) - 1))
        .collect();
    let n = n + extra.len();
    let n_clips = clips.len();
    let b = cfg.batch.max(1);

    let (fwd, nodes) = build_loss(&w.cfg, n_out, b, t);
    let wrt: Vec<TrainableParam> = PARAMS.iter().zip(&nodes).map(|(name, &node)| TrainableParam { name: name.to_string(), node }).collect();
    let spec = AdamSpec { lr: cfg.lr, weight_decay: cfg.weight_decay, ..AdamSpec::default() };
    let mut trainer = ResidentTrainer::new(&fwd, &wrt, &to_graph(w), &spec, device).map_err(|e| anyhow::anyhow!("fused step: {e}"))?;
    let mut probe = Session::new(device).compile(build_probs(&w.cfg, n_out, b, t));
    let resident = trainer.is_resident();
    if cfg.log_every > 0 {
        eprintln!(
            "[rlx-wake multi/{device:?}] {n} clips, {} outputs, window {t}: setup {:.1?}, optimizer {}",
            n_out,
            t0.elapsed(),
            if resident { "on the device" } else { "host-chained (no device handles)" }
        );
    }

    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };
    // Score windows `(clip, end)` with the current weights: probabilities `[len, K]`.
    let score = |probe: &mut rlx_runtime::CompiledGraph, params: &HashMap<String, Vec<f32>>, wins: &[(usize, usize)]| -> Vec<f32> {
        for (name, p) in params {
            probe.set_param(name, p);
        }
        let mut out = Vec::with_capacity(wins.len() * n_out);
        let mut px = vec![0.0f32; b * n_mels * t];
        for chunk in wins.chunks(b) {
            let items: Vec<(usize, usize, Option<usize>)> = chunk.iter().map(|&(i, e)| (i, e, None)).collect();
            let mut dummy = vec![0.0f32; b * n_out];
            px.fill(0.0);
            fill_batch(&items, &mels, n_mels, t, n_out, &mut px, &mut dummy);
            let o = probe.run(&[("x", &px)]);
            out.extend_from_slice(&o[0][..chunk.len() * n_out]);
        }
        out
    };

    let mut hard: Vec<(usize, usize)> = Vec::new();
    let (mut xb, mut yb) = (vec![0.0f32; b * n_mels * t], vec![0.0f32; b * n_out]);
    let (mut xn, mut yn) = (xb.clone(), yb.clone());
    let (mut initial, mut last) = (0.0f32, 0.0f32);
    let t_train = Instant::now();
    for epoch in 0..cfg.epochs {
        if cfg.mine_every > 0 && epoch >= cfg.mine_after && (epoch - cfg.mine_after).is_multiple_of(cfg.mine_every) {
            let tm = Instant::now();
            let wins: Vec<(usize, usize)> = negatives.iter().flat_map(|&i| (t..=ext[i].max(t)).step_by(cfg.mine_stride.max(1)).map(move |e| (i, e))).collect();
            let probs = score(&mut probe, &trainer.params(), &wins);
            // Each window counts against the output it fires most; each output keeps its own
            // worst ones, up to `mine_ratio` × its (balanced) positives.
            let mut by_out: Vec<Vec<(f32, usize, usize)>> = vec![Vec::new(); n_out];
            for (j, &(i, e)) in wins.iter().enumerate() {
                let row = &probs[j * n_out..(j + 1) * n_out];
                let (k, m) = row.iter().copied().enumerate().fold((0, 0.0f32), |a, (k, v)| if v > a.1 { (k, v) } else { a });
                if m > cfg.mine_threshold {
                    by_out[k].push((m, i, e));
                }
            }
            let cap = (largest.max(1) as f32 * cfg.mine_ratio) as usize;
            let found: usize = by_out.iter().map(Vec::len).sum();
            hard = by_out
                .into_iter()
                .flat_map(|mut v| {
                    v.sort_by(|a, b| b.0.total_cmp(&a.0));
                    v.into_iter().take(cap).map(|(_, i, e)| (i, e))
                })
                .collect();
            if cfg.log_every > 0 {
                eprintln!("[rlx-wake multi] epoch={epoch} mined {found} of {} negative windows above {}: training on {} ({:.1?})", wins.len(), cfg.mine_threshold, hard.len(), tm.elapsed());
            }
        }
        let total = n + n_partial + hard.len();
        let mut order: Vec<usize> = (0..total).collect();
        for i in (1..total).rev() {
            order.swap(i, next() % (i + 1));
        }
        let steps = total.div_ceil(b);
        let pick = |s: usize, next: &mut dyn FnMut() -> usize| -> Vec<(usize, usize, Option<usize>)> {
            (0..b)
                .map(|j| {
                    let item = order[(s * b + j) % total];
                    if item < n {
                        let item = if item < n_clips { item } else { extra[item - n_clips] };
                        let f = frames[item];
                        match clips[item].class {
                            Some(c) => (item, f - next() % (POS_JITTER + 1).min(f.max(1)), Some(c)),
                            None if ext[item] > t => (item, t + next() % (ext[item] - t + 1), None),
                            None => (item, f, None),
                        }
                    } else if item < n + n_partial {
                        let i = positives[next() % positives.len()];
                        (i, t + next() % (frames[i] - cfg.partial_gap - t + 1), None)
                    } else {
                        let (i, e) = hard[item - n - n_partial];
                        (i, e, None)
                    }
                })
                .collect()
        };
        let first = pick(0, &mut next);
        fill_batch(&first, &mels, n_mels, t, n_out, &mut xb, &mut yb);
        let mut sum = 0.0f32;
        let (mut in_step, mut waited) = (0.0f64, 0.0f64);
        for s in 0..steps {
            // Cut the next batch while the device runs this one.
            let upcoming = (s + 1 < steps).then(|| pick(s + 1, &mut next));
            let ts = Instant::now();
            let loss = std::thread::scope(|sc| {
                if let Some(items) = &upcoming {
                    let (xn, yn, mels) = (&mut xn, &mut yn, &mels);
                    sc.spawn(move || fill_batch(items, mels, n_mels, t, n_out, xn, yn));
                }
                let r = trainer.step(&[("x", &xb), ("y", &yb)]);
                in_step += ts.elapsed().as_secs_f64();
                r
            });
            waited += ts.elapsed().as_secs_f64();
            let l = loss.first().and_then(|v| v.first()).copied().unwrap_or(f32::NAN);
            sum += l;
            std::mem::swap(&mut xb, &mut xn);
            std::mem::swap(&mut yb, &mut yn);
        }
        let mean = sum / steps.max(1) as f32;
        if epoch == 0 {
            initial = mean;
        }
        last = mean;
        if cfg.log_every > 0 && (epoch % cfg.log_every == 0 || epoch + 1 == cfg.epochs) {
            eprintln!(
                "[rlx-wake multi/{device:?}] epoch={epoch} loss={mean:.4} ({:.2}s/epoch; this one {steps} steps, {:.1} ms device step, {:.1} ms waiting for the next batch)",
                t_train.elapsed().as_secs_f64() / (epoch + 1) as f64,
                in_step * 1e3 / steps.max(1) as f64,
                (waited - in_step).max(0.0) * 1e3 / steps.max(1) as f64
            );
        }
    }
    let secs_per_epoch = t_train.elapsed().as_secs_f64() / cfg.epochs.max(1) as f64;
    let params = trainer.params();
    from_graph(w, &params);

    // Training-set check on the device: positives at their end, negatives at theirs.
    let wins: Vec<(usize, usize)> = (0..n_clips).map(|i| (i, frames[i])).collect();
    let probs = score(&mut probe, &params, &wins);
    let mut recall = vec![(0usize, 0usize); n_out];
    let mut neg = (0usize, 0usize);
    for i in 0..n_clips {
        let p = &probs[i * n_out..(i + 1) * n_out];
        match clips[i].class {
            Some(c) => {
                recall[c].1 += 1;
                recall[c].0 += usize::from(p[c] >= 0.5);
            }
            None => {
                neg.1 += 1;
                neg.0 += usize::from(p.iter().any(|&v| v >= 0.5));
            }
        }
    }
    Ok(MultiReport { initial_loss: initial, final_loss: last, resident, secs_per_epoch, recall, negatives_firing: neg })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The training graph scores a window exactly as the streaming engine on the robot does
    /// (same weights after the layout conversions, K outputs).
    #[test]
    fn graph_matches_the_engine() {
        let cfg = WakeCnnConfig::lite();
        let w = init_multi(cfg.clone(), 3, 11);
        let (b, t) = (2usize, 40usize);
        let n_mels = cfg.n_mels;
        let mut s = 3u32;
        let frames: Vec<f32> = (0..t * n_mels)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect();
        let mut x = frames.clone();
        x.extend_from_slice(&frames);
        let mut probe = Session::new(Device::Cpu).compile(build_probs(&cfg, 3, b, t));
        for (name, v) in to_graph(&w) {
            probe.set_param(&name, &v);
        }
        let got = probe.run(&[("x", &x)]).remove(0);
        let core = rlx_wakeword_core::WakeCnnWeights::from_parts(
            rlx_wakeword_core::WakeCnnConfig { n_mels, c1: cfg.c1, c2: cfg.c2, c3: cfg.c3, k: cfg.k, hidden: cfg.hidden },
            w.conv1_w.clone(), w.conv1_b.clone(), w.conv2_w.clone(), w.conv2_b.clone(), w.conv3_w.clone(), w.conv3_b.clone(),
            w.fc1_w.clone(), w.fc1_b.clone(), w.fc2_w.clone(), w.fc2_b.clone(),
        );
        let mut engine = rlx_wakeword_core::WakeCnn::new(core).with_window_frames(t);
        let want = engine.push_mel_frames_all(&frames).unwrap().to_vec();
        for (j, (g, e)) in got[..3].iter().zip(&want).enumerate() {
            assert!((g - e).abs() < 1e-4, "output {j}: graph {g}, engine {e}");
        }
        assert_eq!(&got[..3], &got[3..6], "both batch rows are the same window");
        // And the export layout round-trips.
        let mut back = w.clone();
        from_graph(&mut back, &to_graph(&w));
        assert_eq!(back.conv2_w, w.conv2_w);
        assert_eq!(back.fc2_w, w.fc2_w);
    }

    /// Two "phrases" (tones in different mel bands) and noise: the network learns to tell
    /// them apart, each output firing for its own phrase only.
    #[test]
    fn learns_two_phrases_on_cpu() {
        let cfg = WakeCnnConfig::lite();
        let n_mels = cfg.n_mels;
        let mut clips = Vec::new();
        let mut s = 7u32;
        let mut noise = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        };
        for i in 0..120 {
            let class = match i % 3 {
                0 => Some(0),
                1 => Some(1),
                _ => None,
            };
            let frames = 80;
            let mut mel = vec![0.0f32; frames * n_mels];
            for f in 0..frames {
                for m in 0..n_mels {
                    mel[f * n_mels + m] = noise() * 0.5;
                }
                if f >= 50
                    && let Some(c) = class
                {
                    let band = if c == 0 { 4 } else { 20 };
                    for m in band..band + 4 {
                        mel[f * n_mels + m] += 3.0;
                    }
                }
            }
            clips.push(MultiClip { mel, class });
        }
        let mut w = init_multi(cfg, 2, 3);
        let tc = MultiTrainConfig { epochs: 40, batch: 32, window_frames: 40, partial_ratio: 0.0, mine_every: 0, log_every: 0, lr: 3e-3, ..Default::default() };
        let r = train_multi(&mut w, &clips, &tc, Device::Cpu).unwrap();
        assert!(r.final_loss < r.initial_loss * 0.5, "{r:?}");
        for (found, total) in &r.recall {
            assert!(*found as f32 >= 0.9 * *total as f32, "{r:?}");
        }
        assert!(r.negatives_firing.0 as f32 <= 0.1 * r.negatives_firing.1 as f32, "{r:?}");
    }
}
