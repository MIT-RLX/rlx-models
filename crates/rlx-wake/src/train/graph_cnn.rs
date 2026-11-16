// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Train `WakeCnn` on an RLX device (Metal, MLX, wgpu, CUDA, … or CPU): the CNN and a
//! numerically stable binary cross-entropy are an RLX graph over a minibatch, differentiated by
//! `rlx-autodiff` and compiled once for the device; AdamW updates the parameters on the host.
//!
//! Same model, features and window as [`super::cnn::train_wake_cnn`] (the per-clip SGD
//! reference), so the weights load into the streaming engine unchanged: the 1-D convolutions are
//! `conv2d` over `[B, n_mels, 1, T]` with `1 × k` kernels, the fully connected layers are
//! matmuls whose weights are transposed back to the engine's `[out, in]` layout on export.

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::{Device, Session};

use crate::cnn::WakeCnnWeights;
use crate::train::cnn::CnnTrainConfig;
use crate::train::dataset::{LabeledClip, clip_mel_frames};
use crate::train::report::TrainReport;

/// Minibatch training options.
#[derive(Debug, Clone)]
pub struct GraphTrainConfig {
    /// Clips per step (the last partial batch of an epoch wraps around).
    pub batch: usize,
    /// Streaming scores a window every hop, anywhere in the audio, so train that way: each
    /// epoch a negative clip contributes a window ending at a random frame (not only its end),
    /// a positive one a window ending within [`POS_JITTER`] frames of its end (the phrase just
    /// complete), and `partial_ratio` × the positives windows of positive clips ending at least
    /// `partial_gap` frames early (the phrase not yet complete: "no").
    pub shift_negatives: bool,
    pub partial_ratio: f32,
    pub partial_gap: usize,
    /// Frames of each clip's own background (its first frames) appended after it, so that a
    /// negative's window can also end well after its speech: what streaming scores for a
    /// second after any phrase (a "hey" phrase must not read as a "bye" one 0.4 s later).
    pub tail_frames: usize,
    /// Hard-negative mining: from epoch `mine_after`, every `mine_every` epochs (0: never),
    /// score every negative window `mine_stride` frames apart with the current model (forward
    /// only, on the device) and add those above `mine_threshold`, worst first, up to
    /// `mine_ratio` × the positives, to the epoch's items. The false triggers that are left
    /// after a plain run are a few near misses at a few offsets; this finds and repeats them.
    pub mine_every: usize,
    pub mine_after: usize,
    pub mine_stride: usize,
    pub mine_threshold: f32,
    pub mine_ratio: f32,
    /// AdamW: learning rate, betas, epsilon; weight decay is `cnn.sgd.weight_decay`.
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for GraphTrainConfig {
    fn default() -> Self {
        Self {
            batch: 128,
            shift_negatives: true,
            partial_ratio: 1.0,
            partial_gap: 50,
            tail_frames: 60,
            mine_every: 10,
            mine_after: 20,
            mine_stride: 8,
            mine_threshold: 0.3,
            mine_ratio: 1.0,
            lr: 2e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// A positive's window ends within this many frames of the clip's end (where the dataset puts
/// the phrase's end, 50-250 ms before it).
pub const POS_JITTER: usize = 3;

fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// The parameters in graph layout, in the order of the graph's gradients.
const PARAMS: [&str; 10] = [
    "conv1_w", "conv1_b", "conv2_w", "conv2_b", "conv3_w", "conv3_b", "fc1_w", "fc1_b", "fc2_w",
    "fc2_b",
];

/// The CNN over a batch `x [B, n_mels, 1, T]`: logits `[B, 1]` and the parameter nodes.
fn body(g: &mut Graph, w: &WakeCnnWeights, b: usize, t: usize) -> (NodeId, Vec<NodeId>) {
    let c = &w.cfg;
    let (k, pad) = (c.k, c.k / 2);
    let x = g.input("x", sh(&[b, c.n_mels, 1, t]));
    let p: Vec<NodeId> = vec![
        g.param("conv1_w", sh(&[c.c1, c.n_mels, 1, k])),
        g.param("conv1_b", sh(&[1, c.c1, 1, 1])),
        g.param("conv2_w", sh(&[c.c2, c.c1, 1, k])),
        g.param("conv2_b", sh(&[1, c.c2, 1, 1])),
        g.param("conv3_w", sh(&[c.c3, c.c2, 1, k])),
        g.param("conv3_b", sh(&[1, c.c3, 1, 1])),
        g.param("fc1_w", sh(&[c.c3, c.hidden])),
        g.param("fc1_b", sh(&[1, c.hidden])),
        g.param("fc2_w", sh(&[c.hidden, 1])),
        g.param("fc2_b", sh(&[1, 1])),
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
    let z = g.add(z, p[9]); // logits [B, 1]
    (z, p)
}

/// Probabilities `[B, 1]` (for mining).
fn build_probs(w: &WakeCnnWeights, b: usize, t: usize) -> Graph {
    let mut g = Graph::new("wake_cnn_probs");
    let (z, _) = body(&mut g, w, b, t);
    let p = g.sigmoid(z);
    g.set_outputs(vec![p]);
    g
}

/// Forward + mean BCE-with-logits over a batch: `x [B, n_mels, 1, T]`, `y [B, 1]`.
fn build_loss(w: &WakeCnnWeights, b: usize, t: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("wake_cnn_loss");
    let (z, p) = body(&mut g, w, b, t);
    let y = g.input("y", sh(&[b, 1]));
    // BCE with logits, stable: max(z, 0) − z·y + ln(1 + e^(−|z|)).
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

/// `[out, in]` (engine) ↔ `[in, out]` (graph).
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; m.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

fn to_graph(w: &WakeCnnWeights) -> Vec<Vec<f32>> {
    let c = &w.cfg;
    vec![
        w.conv1_w.clone(),
        w.conv1_b.clone(),
        w.conv2_w.clone(),
        w.conv2_b.clone(),
        w.conv3_w.clone(),
        w.conv3_b.clone(),
        transpose(&w.fc1_w, c.hidden, c.c3),
        w.fc1_b.clone(),
        transpose(&w.fc2_w, 1, c.hidden),
        w.fc2_b.clone(),
    ]
}

fn from_graph(w: &mut WakeCnnWeights, p: &[Vec<f32>]) {
    let (c3, hidden) = (w.cfg.c3, w.cfg.hidden);
    w.conv1_w.clone_from(&p[0]);
    w.conv1_b.clone_from(&p[1]);
    w.conv2_w.clone_from(&p[2]);
    w.conv2_b.clone_from(&p[3]);
    w.conv3_w.clone_from(&p[4]);
    w.conv3_b.clone_from(&p[5]);
    w.fc1_w = transpose(&p[6], c3, hidden);
    w.fc1_b.clone_from(&p[7]);
    w.fc2_w = transpose(&p[8], hidden, 1);
    w.fc2_b.clone_from(&p[9]);
}

/// The `t` mel frames ending at frame `end` (exclusive) of a frame-major `[frames, n_mels]`
/// clip, channel-major `[n_mels, t]` into `dst` (zero-padded in front if the clip is short).
fn window_into(mel: &[f32], n_mels: usize, end: usize, t: usize, dst: &mut [f32]) {
    let end = end.min(mel.len() / n_mels);
    let take = end.min(t);
    dst.fill(0.0);
    for i in 0..take {
        let src = (end - take + i) * n_mels;
        let ti = t - take + i;
        for m in 0..n_mels {
            dst[m * t + ti] = mel[src + m];
        }
    }
}

/// A clip's last `t` mel frames, channel-major `[n_mels, t]`.
#[cfg(test)]
fn features(clip: &LabeledClip, n_mels: usize, t: usize) -> Vec<f32> {
    let mel = clip_mel_frames(&clip.pcm);
    let mut x = vec![0.0; n_mels * t];
    window_into(&mel, n_mels, mel.len() / n_mels, t, &mut x);
    x
}

/// Train on `device`. `cnn` supplies the window, epochs, weight decay, logging and keyword.
pub fn train_wake_cnn_device(
    weights: &mut WakeCnnWeights,
    clips: &[LabeledClip],
    cnn: &CnnTrainConfig,
    opt: &GraphTrainConfig,
    device: Device,
) -> TrainReport {
    let n_mels = weights.cfg.n_mels;
    let t = cnn.window_frames.max(8);
    let t0 = std::time::Instant::now();
    // Mel features on every core (the per-clip FFTs dominate a run otherwise), whole clips:
    // windows are cut from them per epoch.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = clips.len().div_ceil(threads).max(1);
    let mels: Vec<Vec<f32>> = std::thread::scope(|s| {
        let parts: Vec<_> = clips
            .chunks(per)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|c| clip_mel_frames(&c.pcm))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        parts
            .into_iter()
            .flat_map(|h| h.join().expect("feature thread"))
            .collect()
    });
    let frames: Vec<usize> = mels.iter().map(|m| m.len() / n_mels).collect();
    // Each clip between `lead` frames of its own background (its opening frames, repeated if
    // short) and `tail_frames` more: windows can then end anywhere from a window before the
    // clip's speech to well after it, whatever the window length (streaming sees all of them).
    let lead = t;
    let mels: Vec<Vec<f32>> = mels
        .into_iter()
        .map(|m| {
            let opening = &m[..(opt.tail_frames.max(lead) * n_mels).min(m.len())];
            let mut ext = Vec::with_capacity(m.len() + (lead + opt.tail_frames) * n_mels);
            while ext.len() < lead * n_mels && !opening.is_empty() {
                let need = lead * n_mels - ext.len();
                ext.extend_from_slice(&opening[..need.min(opening.len())]);
            }
            ext.extend_from_slice(&m);
            ext.extend_from_slice(&opening[..(opt.tail_frames * n_mels).min(opening.len())]);
            ext
        })
        .collect();
    // Clip frame `f` is now at `lead + f`.
    let frames: Vec<usize> = frames.iter().map(|f| f + lead).collect();
    let ext: Vec<usize> = mels.iter().map(|m| m.len() / n_mels).collect();
    let t_feats = t0.elapsed();
    let labels: Vec<f32> = clips.iter().map(|c| c.label).collect();
    let positives: Vec<usize> = (0..clips.len())
        .filter(|&i| labels[i] >= 0.5 && frames[i] >= t + opt.partial_gap)
        .collect();
    debug_assert!(
        frames.iter().all(|&f| f >= t),
        "the lead makes every clip at least a window"
    );
    let n_partial = if opt.partial_ratio > 0.0 {
        (positives.len() as f32 * opt.partial_ratio) as usize
    } else {
        0
    };
    let n = clips.len();
    let n_partial = if positives.is_empty() { 0 } else { n_partial };
    let mut total = n + n_partial;
    let negatives: Vec<usize> = (0..n).filter(|&i| labels[i] < 0.5).collect();
    // Mined hard negatives: (clip, window end); items `n + n_partial..total`.
    let mut hard: Vec<(usize, usize)> = Vec::new();
    let mut probe = (opt.mine_every > 0)
        .then(|| Session::new(device).compile(build_probs(weights, opt.batch.max(1), t)));
    let b = opt.batch.clamp(1, total.max(1));

    let (fwd, wrt) = build_loss(weights, b, t);
    let bwd = rlx_autodiff::grad_with_loss(&fwd, &wrt);
    let mut sess = Session::new(device).compile(bwd);
    if cnn.sgd.log_every > 0 {
        eprintln!(
            "[rlx-wake-train cnn/{device:?}] {n} clips (+{} partial-phrase windows/epoch): features {t_feats:.1?}, compile {:.1?}",
            total - n,
            t0.elapsed() - t_feats
        );
    }
    let t_train = std::time::Instant::now();
    // Eval: the same loss graph, forward only, for accuracy at the end.
    let mut params = to_graph(weights);
    let mut m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
    let mut v = m.clone();
    let (mut b1t, mut b2t) = (1.0f32, 1.0f32);
    let mut steps_per_epoch = total.div_ceil(b);
    // Items `0..n` are the clips; `n..n + n_partial` partial-phrase windows of positives;
    // the rest mined hard negatives.
    let mut order: Vec<usize> = (0..total).collect();
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };
    let mut xb = vec![0.0f32; b * n_mels * t];
    let mut yb = vec![0.0f32; b];
    let (mut initial, mut final_loss) = (0.0f32, 0.0f32);
    for epoch in 0..cnn.sgd.epochs {
        if let Some(probe) = probe.as_mut().filter(|_| {
            epoch >= opt.mine_after && (epoch - opt.mine_after).is_multiple_of(opt.mine_every)
        }) {
            let tm = std::time::Instant::now();
            for (name, p) in PARAMS.iter().zip(&params) {
                probe.set_param(name, p);
            }
            let pb = opt.batch.max(1);
            let windows: Vec<(usize, usize)> = negatives
                .iter()
                .flat_map(|&i| {
                    (t..=ext[i].max(t))
                        .step_by(opt.mine_stride.max(1))
                        .map(move |e| (i, e))
                })
                .collect();
            let mut scored: Vec<(f32, usize, usize)> = Vec::new();
            let mut px = vec![0.0f32; pb * n_mels * t];
            for chunk in windows.chunks(pb) {
                for (j, &(i, e)) in chunk.iter().enumerate() {
                    window_into(
                        &mels[i],
                        n_mels,
                        e,
                        t,
                        &mut px[j * n_mels * t..(j + 1) * n_mels * t],
                    );
                }
                let out = probe.run(&[("x", &px)]);
                for (j, &(i, e)) in chunk.iter().enumerate() {
                    if out[0][j] > opt.mine_threshold {
                        scored.push((out[0][j], i, e));
                    }
                }
            }
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            let cap = (positives.len().max(1) as f32 * opt.mine_ratio) as usize;
            let found = scored.len();
            hard = scored
                .into_iter()
                .take(cap)
                .map(|(_, i, e)| (i, e))
                .collect();
            total = n + n_partial + hard.len();
            order = (0..total).collect();
            steps_per_epoch = total.div_ceil(b);
            if cnn.sgd.log_every > 0 {
                eprintln!(
                    "[rlx-wake-train cnn/{device:?}] epoch={epoch} mined {found} of {} negative windows above {}: training on {} ({:.1?})",
                    windows.len(),
                    opt.mine_threshold,
                    hard.len(),
                    tm.elapsed()
                );
            }
        }
        for i in (1..total).rev() {
            order.swap(i, next() % (i + 1));
        }
        let mut sum = 0.0f32;
        for s in 0..steps_per_epoch {
            for j in 0..b {
                let item = order[(s * b + j) % total];
                let (idx, end, label) = if item < n {
                    let f = frames[item];
                    let end = if labels[item] >= 0.5 {
                        f - next() % (POS_JITTER + 1).min(f.max(1))
                    } else if opt.shift_negatives && ext[item] > t {
                        t + next() % (ext[item] - t + 1)
                    } else {
                        f
                    };
                    (item, end, labels[item])
                } else if item < n + n_partial {
                    // A positive, cut before its phrase is complete.
                    let i = positives[next() % positives.len()];
                    (i, t + next() % (frames[i] - opt.partial_gap - t + 1), 0.0)
                } else {
                    let (i, e) = hard[item - n - n_partial];
                    (i, e, 0.0)
                };
                window_into(
                    &mels[idx],
                    n_mels,
                    end,
                    t,
                    &mut xb[j * n_mels * t..(j + 1) * n_mels * t],
                );
                yb[j] = label;
            }
            for (name, p) in PARAMS.iter().zip(&params) {
                sess.set_param(name, p);
            }
            let out = sess.run(&[("d_output", &[1.0]), ("x", &xb), ("y", &yb)]);
            sum += out[0][0];
            // AdamW.
            b1t *= opt.beta1;
            b2t *= opt.beta2;
            let lr_t = opt.lr * (1.0 - b2t).sqrt() / (1.0 - b1t);
            for (pi, grad) in out[1..].iter().enumerate() {
                let (p, m, v) = (&mut params[pi], &mut m[pi], &mut v[pi]);
                for i in 0..p.len() {
                    let gi = grad[i];
                    m[i] = opt.beta1 * m[i] + (1.0 - opt.beta1) * gi;
                    v[i] = opt.beta2 * v[i] + (1.0 - opt.beta2) * gi * gi;
                    p[i] -= lr_t * m[i] / (v[i].sqrt() + opt.eps)
                        + opt.lr * cnn.sgd.weight_decay * p[i];
                }
            }
        }
        let mean = sum / steps_per_epoch.max(1) as f32;
        if epoch == 0 {
            initial = mean;
        }
        final_loss = mean;
        if cnn.sgd.log_every > 0 && epoch % cnn.sgd.log_every == 0 {
            eprintln!(
                "[rlx-wake-train cnn/{device:?}] epoch={epoch} loss={mean:.4} keyword={} ({:.1?}/epoch)",
                cnn.keyword,
                t_train.elapsed() / (epoch as u32 + 1)
            );
        }
    }
    from_graph(weights, &params);

    // Training accuracy with the host engine (the reference forward), on the features above.
    let w: &WakeCnnWeights = weights;
    let feats: Vec<Vec<f32>> = mels
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut x = vec![0.0; n_mels * t];
            window_into(m, n_mels, frames[i], t, &mut x);
            x
        })
        .collect();
    let correct: usize = std::thread::scope(|s| {
        let parts: Vec<_> = feats
            .chunks(per)
            .zip(labels.chunks(per))
            .map(|(xs, ys)| {
                s.spawn(move || {
                    xs.iter()
                        .zip(ys)
                        .filter(|(x, y)| {
                            // Back to frame-major for the reference forward.
                            let mut frames = vec![0.0f32; n_mels * t];
                            for m in 0..n_mels {
                                for ti in 0..t {
                                    frames[ti * n_mels + m] = x[m * t + ti];
                                }
                            }
                            (crate::train::cnn::predict(w, &frames, t) >= 0.5) == (**y >= 0.5)
                        })
                        .count()
                })
            })
            .collect();
        parts
            .into_iter()
            .map(|h| h.join().expect("eval thread"))
            .sum()
    });
    TrainReport {
        epochs: cnn.sgd.epochs,
        final_loss,
        initial_loss: initial,
        train_acc: correct as f32 / n.max(1) as f32,
        keyword: cnn.keyword.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::WakeCnnConfig;
    use crate::train::synth_pos_neg_dataset;

    /// The graph's loss matches the per-clip reference forward on CPU.
    #[test]
    fn graph_loss_matches_the_reference_forward() {
        let clips = synth_pos_neg_dataset(1, 1, 1.2);
        let w = WakeCnnWeights::he_init(WakeCnnConfig::lite(), 3);
        // A window the 1.2 s clip fills (118 frames): shorter clips are zero-padded here but
        // not by the reference (training clips are 2 s, longer than any window).
        let t = 100;
        let (fwd, _) = build_loss(&w, 1, t);
        let mut sess = Session::new(Device::Cpu).compile(fwd);
        for (name, p) in PARAMS.iter().zip(to_graph(&w)) {
            sess.set_param(name, &p);
        }
        for c in &clips {
            let x = features(c, w.cfg.n_mels, t);
            let out = sess.run(&[("x", &x), ("y", &[c.label])]);
            let p =
                crate::train::cnn::predict(&w, &clip_mel_frames(&c.pcm), t).clamp(1e-6, 1.0 - 1e-6);
            let want = -(c.label * p.ln() + (1.0 - c.label) * (1.0 - p).ln());
            assert!(
                (out[0][0] - want).abs() < 1e-3 * (1.0 + want),
                "graph {} vs reference {want}",
                out[0][0]
            );
        }
    }

    #[test]
    fn learns_the_synthetic_task_on_cpu() {
        let clips = synth_pos_neg_dataset(16, 16, 1.2);
        let mut w = WakeCnnWeights::he_init(WakeCnnConfig::lite(), 1);
        let mut cnn = CnnTrainConfig {
            window_frames: 120,
            ..CnnTrainConfig::default()
        };
        cnn.sgd.epochs = 30;
        cnn.sgd.log_every = 0;
        // The toy clips are signal from end to end (no background to pad with, no phrase to
        // cut): plain end-of-clip windows.
        let opt = GraphTrainConfig {
            batch: 8,
            shift_negatives: false,
            partial_ratio: 0.0,
            tail_frames: 0,
            mine_every: 0,
            ..Default::default()
        };
        let r = train_wake_cnn_device(&mut w, &clips, &cnn, &opt, Device::Cpu);
        assert!(r.final_loss < r.initial_loss * 0.5, "{r:?}");
        assert!(r.train_acc >= 0.9, "{r:?}");
    }
}
