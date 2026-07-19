//! Phase-0b: a real (small) extractive TUI cleaner, trained on Metal.
//!
//! Trains on a train split, measures generalization on held-out val (overall +
//! per-layout F1), then demonstrates cleaning: predict per-char content/chrome,
//! drop the chrome, reconstruct the clean text.
//!
//! Model: char embedding + learned positional + N pre-norm bidirectional
//! fused-attention blocks (rms_norm) + GeLU FFN blocks + a per-token
//! content/chrome head (MSE to {0,1} with a padding mask).
//!
//! Wiring notes on the rlx-tensor DSL:
//!  - a leading-dim broadcast `[B,L,D] + [1,L,D]` NaNs; positional is a matmul
//!    (`pos_onehot[B*L,L] @ pe[L,D]`).
//!  - normalization backward NaNs when activation variance is tiny (small
//!    init); with healthy init + a final norm it's stable and unlocks depth.
//!  - depth without normalization blows the residual stream up.
//!
//! Run: `cargo run -q -p rlx-termclean --bin rlx-termclean-train-tagger --features train`

use std::collections::BTreeMap;
use std::collections::HashMap;

use rlx_tensor::{
    Adam, Device, Func, GraphScope, LrSchedule, MaskKind, Optimizer, Tensor, is_available, shape,
};

use rlx_termclean::{Rng, Sample, generate};

const L: usize = 96;
const B: usize = 32;
const NH: usize = 4;
const DH: usize = 16;
const D: usize = NH * DH; // 64
const FF: usize = 4 * D;
// 1 residual block, no norm — the robust ceiling. Manual RMSNorm (rsqrt/mean/mul
// from primitives) DOES train (2-layer synth reached 0.927) but the rsqrt
// backward is fragile on longer seqs (NaN'd at epoch 0 on the L=200 multi task);
// the fused layer_norm/rms_norm backward NaNs outright.
const NLAYERS: usize = 1;

fn param_names() -> Vec<String> {
    let mut p = vec!["we".to_string(), "pe".to_string()];
    for i in 0..NLAYERS {
        for base in ["wq", "wk", "wv", "w1", "b1", "w2", "b2"] {
            p.push(format!("{base}{i}"));
        }
    }
    p.push("wo".to_string());
    p.push("bo".to_string());
    p
}

#[derive(Clone, Copy)]
struct Dims {
    v: usize,
}

fn w(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 31 + seed * 17) % 23) as f32) - 11.0) * scale)
        .collect()
}

fn init_params(v: usize) -> Vec<(String, Vec<f32>)> {
    let mut p = vec![
        ("we".to_string(), w(v * D, 1, 0.05)),
        ("pe".to_string(), w(L * D, 8, 0.03)),
    ];
    for i in 0..NLAYERS {
        let s = i * 11;
        p.push((format!("wq{i}"), w(D * D, 2 + s, 0.08)));
        p.push((format!("wk{i}"), w(D * D, 3 + s, 0.08)));
        p.push((format!("wv{i}"), w(D * D, 4 + s, 0.08)));
        p.push((format!("w1{i}"), w(D * FF, 6 + s, 0.06)));
        p.push((format!("b1{i}"), vec![0.0; FF]));
        p.push((format!("w2{i}"), w(FF * D, 7 + s, 0.03)));
        p.push((format!("b2{i}"), vec![0.0; D]));
    }
    // tiny head init -> logits ~0 at start (well-conditioned)
    p.push(("wo".to_string(), w(D, 5, 0.005)));
    p.push(("bo".to_string(), vec![0.0]));
    p
}

fn forward(s: &mut GraphScope, dm: Dims, with_loss: bool) -> Tensor {
    let v = dm.v;
    let bl = (B * L) as i64;
    let heads = |t: Tensor| t.reshape(vec![B as i64, L as i64, NH as i64, DH as i64]);

    let x = s.input("x", shape![B, L, v]);
    let pos = s.input("pos", shape![B * L, L]);
    let we = s.param("we", shape![v, D]);
    let pe = s.param("pe", shape![L, D]);

    let emb = x.reshape(vec![bl, v as i64]).matmul(&we);
    let mut h = &emb + &pos.matmul(&pe); // [B*L, D]

    for i in 0..NLAYERS {
        // bidirectional attention block (residual)
        let wq = s.param(format!("wq{i}"), shape![D, D]);
        let wk = s.param(format!("wk{i}"), shape![D, D]);
        let wv = s.param(format!("wv{i}"), shape![D, D]);
        // tanh-bounded Q/K caps scores at ±sqrt(dh) so the softmax can't
        // overflow to NaN in this unnormalized net (smooth, not a clamp).
        let q = heads(h.matmul(&wq).tanh());
        let k = heads(h.matmul(&wk).tanh());
        let vv = heads(h.matmul(&wv));
        let attn = q
            .attention(&k, &vv, NH, DH, MaskKind::None)
            .reshape(vec![bl, D as i64]);
        h = &h + &attn;

        // GeLU FFN block (residual)
        let w1 = s.param(format!("w1{i}"), shape![D, FF]);
        let b1 = s.param(format!("b1{i}"), shape![FF]);
        let w2 = s.param(format!("w2{i}"), shape![FF, D]);
        let b2 = s.param(format!("b2{i}"), shape![D]);
        let ff = &(&h.matmul(&w1) + &b1).gelu().matmul(&w2) + &b2;
        h = &h + &ff;
    }

    let wo = s.param("wo", shape![D, 1]);
    let bo = s.param("bo", shape![1]);
    let logit = &h.matmul(&wo) + &bo; // [B*L, 1]
    if with_loss {
        let y = s.input("y", shape![B * L, 1]);
        let m = s.input("m", shape![B * L, 1]);
        let diff = &logit - &y;
        (&m * &(&diff * &diff)).mean([0, 1], false)
    } else {
        logit
    }
}

fn build(dm: Dims, params: &[(String, Vec<f32>)], with_loss: bool) -> Func {
    let mut f = Func::new("tagger", move |s| forward(s, dm, with_loss));
    for (n, data) in params {
        f = f.with_param(n.clone(), data.clone());
    }
    f
}

fn train_step_on(
    model: &Func,
    dev: Device,
    opt: &mut dyn Optimizer,
    feed: &[(&str, &[f32])],
    names: &[&str],
) -> (Func, f32) {
    let out = model.value_and_grad(names).run_on(dev, feed);
    let loss = out[0][0];
    let mut sumsq = 0f64;
    for g in &out[1..=names.len()] {
        for &x in g {
            sumsq += (x as f64) * (x as f64);
        }
    }
    let norm = sumsq.sqrt();
    let scale = if norm > 1.0 { (1.0 / norm) as f32 } else { 1.0 };
    let mut m = model.clone();
    for (i, name) in names.iter().enumerate() {
        let mut data = model.param_binding(name).unwrap().to_vec();
        let g: Vec<f32> = out[i + 1].iter().map(|x| x * scale).collect();
        opt.step(name, &[data.len()], &mut data, &g);
        m = m.with_param(*name, data);
    }
    opt.end_iteration();
    (m, loss)
}

fn gen_samples(n: usize, seed: u64) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    (0..n).map(|i| generate(&mut rng, i as u64)).collect()
}

fn build_vocab(train: &[Sample]) -> HashMap<char, usize> {
    let mut map = HashMap::new();
    let mut next = 1usize; // 0 = UNK/PAD
    for s in train {
        for c in s.input.chars().take(L) {
            map.entry(c).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
        }
    }
    map
}

fn encode(s: &Sample, map: &HashMap<char, usize>) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let inp: Vec<char> = s.input.chars().collect();
    let tags: Vec<char> = s.tags.chars().collect();
    let mut idx = vec![0usize; L];
    let mut lab = vec![0f32; L];
    let mut mask = vec![0f32; L];
    for i in 0..L.min(inp.len()) {
        idx[i] = *map.get(&inp[i]).unwrap_or(&0);
        lab[i] = if tags[i] == 'C' { 1.0 } else { 0.0 };
        mask[i] = 1.0;
    }
    (idx, lab, mask)
}

fn minibatch(
    enc: &[(Vec<usize>, Vec<f32>, Vec<f32>)],
    rows: &[usize],
    v: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut onehot = vec![0.0f32; B * L * v];
    let mut lab = vec![0.0f32; B * L];
    let mut mask = vec![0.0f32; B * L];
    for (bi, &r) in rows.iter().enumerate() {
        let (idx, l, m) = &enc[r];
        for li in 0..L {
            onehot[(bi * L + li) * v + idx[li]] = 1.0;
            lab[bi * L + li] = l[li];
            mask[bi * L + li] = m[li];
        }
    }
    (onehot, lab, mask)
}

fn predict_logits(
    dm: Dims,
    model: &Func,
    enc: &[(Vec<usize>, Vec<f32>, Vec<f32>)],
    rows: &[usize],
    pos: &[f32],
    names: &[&str],
    dev: Device,
) -> Vec<f32> {
    let mut pred = build(dm, &[], false);
    for name in names {
        pred = pred.with_param(*name, model.param_binding(name).unwrap().to_vec());
    }
    let (onehot, _, _) = minibatch(enc, rows, dm.v);
    pred.run_on(dev, &[("x", &onehot), ("pos", pos)]).remove(0)
}

fn f1_of(tp: u64, fp: u64, fnn: u64) -> f32 {
    let prec = tp as f32 / (tp + fp).max(1) as f32;
    let rec = tp as f32 / (tp + fnn).max(1) as f32;
    if prec + rec > 0.0 {
        2.0 * prec * rec / (prec + rec)
    } else {
        0.0
    }
}

fn evaluate(
    dm: Dims,
    model: &Func,
    val: &[(Vec<usize>, Vec<f32>, Vec<f32>)],
    val_s: &[Sample],
    pos: &[f32],
    names: &[&str],
    dev: Device,
) -> (f32, f32, Vec<(&'static str, f32, usize)>) {
    let (mut correct, mut total) = (0u64, 0u64);
    let (mut tp, mut fp, mut fnn) = (0u64, 0u64, 0u64);
    let mut per: BTreeMap<&'static str, (u64, u64, u64, usize)> = BTreeMap::new();
    for b in 0..val.len() / B {
        let rows: Vec<usize> = (b * B..b * B + B).collect();
        let logit = predict_logits(dm, model, val, &rows, pos, names, dev);
        for (bi, &r) in rows.iter().enumerate() {
            let (_, lab, m) = &val[r];
            let e = per.entry(val_s[r].kind).or_default();
            e.3 += 1;
            for li in 0..L {
                if m[li] < 0.5 {
                    continue;
                }
                total += 1;
                let p = logit[bi * L + li] > 0.5;
                let t = lab[li] > 0.5;
                if p == t {
                    correct += 1;
                }
                match (p, t) {
                    (true, true) => {
                        tp += 1;
                        e.0 += 1;
                    }
                    (true, false) => {
                        fp += 1;
                        e.1 += 1;
                    }
                    (false, true) => {
                        fnn += 1;
                        e.2 += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    let per_kind: Vec<(&'static str, f32, usize)> = per
        .into_iter()
        .map(|(k, (t, f, n, c))| (k, f1_of(t, f, n), c))
        .collect();
    (
        correct as f32 / total.max(1) as f32,
        f1_of(tp, fp, fnn),
        per_kind,
    )
}

fn main() {
    const TRAIN_N: usize = 1024;
    const VAL_N: usize = 192;
    const EPOCHS: usize = 50;

    let train_s = gen_samples(TRAIN_N, 100);
    let val_s = gen_samples(VAL_N, 999);
    let map = build_vocab(&train_s);
    let v = map.len() + 1;
    let dm = Dims { v };

    let train_enc: Vec<_> = train_s.iter().map(|s| encode(s, &map)).collect();
    let val_enc: Vec<_> = val_s.iter().map(|s| encode(s, &map)).collect();

    let dev = if is_available(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };
    let nb = TRAIN_N / B;
    let total_steps = EPOCHS * nb;
    println!(
        "tagger: device={dev:?} layers={NLAYERS}(no-norm) D={D} vocab={v} train={TRAIN_N} val={VAL_N} epochs={EPOCHS} steps={total_steps}"
    );

    let mut pos = vec![0f32; B * L * L];
    for bi in 0..B {
        for li in 0..L {
            pos[(bi * L + li) * L + li] = 1.0;
        }
    }

    let names_owned = param_names();
    let names: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();

    let mut model = build(dm, &init_params(v), true);
    let mut opt = Adam::new(0.004);
    let sched = LrSchedule::WarmupCosine {
        base: 0.004,
        min: 0.0003,
        warmup: total_steps / 20,
        total: total_steps,
    };
    let mut rng = Rng::new(1);
    let mut gstep = 0usize;

    for epoch in 0..EPOCHS {
        let mut order: Vec<usize> = (0..TRAIN_N).collect();
        rng.shuffle(&mut order);
        let mut ep_loss = 0.0f32;
        for b in 0..nb {
            opt.set_lr(sched.lr_at(gstep));
            gstep += 1;
            let rows = &order[b * B..b * B + B];
            let (onehot, lab, mask) = minibatch(&train_enc, rows, v);
            let feed: &[(&str, &[f32])] =
                &[("x", &onehot), ("y", &lab), ("m", &mask), ("pos", &pos)];
            let (next, loss) = train_step_on(&model, dev, &mut opt, feed, &names);
            model = next;
            ep_loss += loss;
        }
        if epoch == 0 || epoch == EPOCHS - 1 || (epoch + 1) % 6 == 0 {
            let (acc, f1, _) = evaluate(dm, &model, &val_enc, &val_s, &pos, &names, dev);
            println!(
                "epoch {epoch:>2}  train_loss {:.4}  val_acc {:.1}%  content_F1 {:.3}",
                ep_loss / nb as f32,
                100.0 * acc,
                f1
            );
        }
    }

    let (acc, f1, per_kind) = evaluate(dm, &model, &val_enc, &val_s, &pos, &names, dev);
    println!(
        "\nFINAL (held-out val): per-token acc {:.1}%, content-class F1 {:.3}",
        100.0 * acc,
        f1
    );
    println!("per-layout content F1:");
    for (kind, kf1, n) in &per_kind {
        println!("  {kind:<10} F1 {kf1:.3}  (n={n})");
    }

    let demo_i = val_s
        .iter()
        .position(|s| s.input.chars().count() <= L)
        .unwrap_or(0);
    let mut rows: Vec<usize> = (0..B).collect();
    rows[0] = demo_i;
    let logit = predict_logits(dm, &model, &val_enc, &rows, &pos, &names, dev);
    let s0 = &val_s[demo_i];
    let cleaned: String = s0
        .input
        .chars()
        .take(L)
        .enumerate()
        .filter(|(i, _)| logit[*i] > 0.5)
        .map(|(_, c)| c)
        .collect();
    let show = |s: &str| s.replace('\x1b', "⟨ESC⟩");
    println!(
        "\n=== cleaning demo (held-out val sample, kind={}) ===",
        s0.kind
    );
    println!("--- RAW INPUT ---\n{}", show(&s0.input));
    println!(
        "--- MODEL-CLEANED (kept predicted-content chars) ---\n{}",
        show(&cleaned)
    );
    println!("--- TRUE TARGET ---\n{}", s0.target);

    rlx_tensor::clear_cache();
}
