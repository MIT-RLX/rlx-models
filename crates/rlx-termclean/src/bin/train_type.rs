//! ML 5-way line-type classifier (JSON/code/text/file/UI) — head-to-head vs the
//! fast code classifier. Char embed + 1 attention block + mean-pool → 5-class
//! head, MSE-to-one-hot, keep-best. Data from `gen_typed_line`.
//!
//! Run: `cargo run -q -p rlx-termclean --bin rlx-termclean-train-type --features train`

use std::collections::HashMap;

use rlx_tensor::{
    AdamW, Device, Func, GraphScope, LrSchedule, MaskKind, Optimizer, Tensor, is_available, shape,
};

use rlx_termclean::Rng;
use rlx_termclean::typeclass::{CType, classify_type, gen_typed_line};

const B: usize = 32;
const L: usize = 96;
const NH: usize = 4;
const DH: usize = 16;
const D: usize = NH * DH;
const HID: usize = 64;
const NC: usize = 5;
const PARAMS: &[&str] = &["we", "pe", "wq", "wk", "wv", "w1", "b1", "w2", "b2"];

fn w(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 31 + seed * 17) % 23) as f32) - 11.0) * scale)
        .collect()
}

fn init_params(v: usize) -> Vec<(String, Vec<f32>)> {
    vec![
        ("we".into(), w(v * D, 1, 0.05)),
        ("pe".into(), w(L * D, 8, 0.03)),
        ("wq".into(), w(D * D, 2, 0.08)),
        ("wk".into(), w(D * D, 3, 0.08)),
        ("wv".into(), w(D * D, 4, 0.08)),
        ("w1".into(), w(D * HID, 6, 0.08)),
        ("b1".into(), vec![0.0; HID]),
        ("w2".into(), w(HID * NC, 7, 0.02)),
        ("b2".into(), vec![0.0; NC]),
    ]
}

fn forward(s: &mut GraphScope, v: usize, with_loss: bool) -> Tensor {
    let bl = (B * L) as i64;
    let heads = |t: Tensor| t.reshape(vec![B as i64, L as i64, NH as i64, DH as i64]);
    let x = s.input("x", shape![B, L, v]);
    let pos = s.input("pos", shape![B * L, L]);
    let we = s.param("we", shape![v, D]);
    let pe = s.param("pe", shape![L, D]);
    let h = &x.reshape(vec![bl, v as i64]).matmul(&we) + &pos.matmul(&pe);
    let wq = s.param("wq", shape![D, D]);
    let wk = s.param("wk", shape![D, D]);
    let wv = s.param("wv", shape![D, D]);
    // tanh-bounded Q/K: squashing to [-1,1] caps scores at ±sqrt(dh), so the
    // attention softmax cannot overflow no matter how large the weights grow.
    // A smooth saturating nonlinearity (differentiable everywhere) — not a clamp
    // — and its 1-tanh^2 backward avoids the reduction/broadcast NaN of RMSNorm.
    let q = heads(h.matmul(&wq).tanh());
    let k = heads(h.matmul(&wk).tanh());
    let attn = q
        .attention(&k, &heads(h.matmul(&wv)), NH, DH, MaskKind::None)
        .reshape(vec![bl, D as i64]);
    let h = &h + &attn;
    // mean-pool over the line -> one vector per sample [B, D]
    let pooled = h
        .reshape(vec![B as i64, L as i64, D as i64])
        .mean([1], false);
    let w1 = s.param("w1", shape![D, HID]);
    let b1 = s.param("b1", shape![HID]);
    let hid = (&pooled.matmul(&w1) + &b1).gelu();
    let w2 = s.param("w2", shape![HID, NC]);
    let b2 = s.param("b2", shape![NC]);
    let logits = &hid.matmul(&w2) + &b2; // [B, NC]
    if with_loss {
        let y = s.input("y", shape![B, NC]);
        let diff = &logits - &y;
        (&diff * &diff).mean([0, 1], false)
    } else {
        logits
    }
}

fn build(v: usize, params: &[(String, Vec<f32>)], with_loss: bool) -> Func {
    let mut f = Func::new("typecls", move |s| forward(s, v, with_loss));
    for (n, data) in params {
        f = f.with_param(n.clone(), data.clone());
    }
    f
}

fn train_step(
    model: &Func,
    dev: Device,
    opt: &mut dyn Optimizer,
    feed: &[(&str, &[f32])],
) -> (Func, f32) {
    let out = model.value_and_grad(PARAMS).run_on(dev, feed);
    let loss = out[0][0];
    let mut sumsq = 0f64;
    for g in &out[1..=PARAMS.len()] {
        for &x in g {
            sumsq += (x as f64) * (x as f64);
        }
    }
    let scale = {
        let nrm = sumsq.sqrt();
        if nrm > 0.5 { (0.5 / nrm) as f32 } else { 1.0 }
    };
    let mut m = model.clone();
    for (i, name) in PARAMS.iter().enumerate() {
        let mut data = model.param_binding(name).unwrap().to_vec();
        let g: Vec<f32> = out[i + 1].iter().map(|x| x * scale).collect();
        opt.step(name, &[data.len()], &mut data, &g);
        m = m.with_param(*name, data);
    }
    opt.end_iteration();
    (m, loss)
}

fn gen_split(n_per: usize, seed: u64) -> Vec<(String, usize)> {
    let mut rng = Rng::new(seed);
    let mut v = Vec::new();
    for &t in &CType::ALL {
        for _ in 0..n_per {
            v.push((gen_typed_line(&mut rng, t), t.idx()));
        }
    }
    // shuffle
    let mut r = seed ^ 0x9e37;
    for i in (1..v.len()).rev() {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        v.swap(i, (r % (i as u64 + 1)) as usize);
    }
    v
}

fn encode(line: &str, map: &HashMap<char, usize>) -> Vec<usize> {
    let mut idx = vec![0usize; L];
    for (i, c) in line.chars().take(L).enumerate() {
        idx[i] = *map.get(&c).unwrap_or(&0);
    }
    idx
}

fn minibatch(enc: &[(Vec<usize>, usize)], rows: &[usize], v: usize) -> (Vec<f32>, Vec<f32>) {
    let mut onehot = vec![0f32; B * L * v];
    let mut y = vec![0f32; B * NC];
    for (bi, &r) in rows.iter().enumerate() {
        let (idx, cls) = &enc[r];
        for li in 0..L {
            onehot[(bi * L + li) * v + idx[li]] = 1.0;
        }
        y[bi * NC + cls] = 1.0;
    }
    (onehot, y)
}

fn predict(
    model: &Func,
    enc: &[(Vec<usize>, usize)],
    rows: &[usize],
    pos: &[f32],
    v: usize,
    dev: Device,
) -> Vec<usize> {
    let mut pred = build(v, &[], false);
    for name in PARAMS {
        pred = pred.with_param(*name, model.param_binding(name).unwrap().to_vec());
    }
    let (onehot, _) = minibatch(enc, rows, v);
    let logits = pred.run_on(dev, &[("x", &onehot), ("pos", pos)]).remove(0);
    (0..B)
        .map(|bi| {
            (0..NC)
                .max_by(|&a, &c| {
                    logits[bi * NC + a]
                        .partial_cmp(&logits[bi * NC + c])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        })
        .collect()
}

fn eval(model: &Func, enc: &[(Vec<usize>, usize)], pos: &[f32], v: usize, dev: Device) -> f32 {
    let (mut correct, mut total) = (0u64, 0u64);
    for b in 0..enc.len() / B {
        let rows: Vec<usize> = (b * B..b * B + B).collect();
        let preds = predict(model, enc, &rows, pos, v, dev);
        for (bi, &r) in rows.iter().enumerate() {
            total += 1;
            if preds[bi] == enc[r].1 {
                correct += 1;
            }
        }
    }
    correct as f32 / total.max(1) as f32
}

fn main() {
    const NPER: usize = 12000; // 10x the data (60k train lines across 5 classes)
    const EPOCHS: usize = 15;
    let train = gen_split(NPER, 100);
    let val = gen_split(400, 999);

    let mut map = HashMap::new();
    let mut next = 1usize;
    for (line, _) in &train {
        for c in line.chars().take(L) {
            map.entry(c).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
        }
    }
    let v = map.len() + 1;
    let tr: Vec<(Vec<usize>, usize)> = train.iter().map(|(l, c)| (encode(l, &map), *c)).collect();
    let va: Vec<(Vec<usize>, usize)> = val.iter().map(|(l, c)| (encode(l, &map), *c)).collect();

    // fast-code baseline on the same val lines
    let (mut cc, mut ct) = (0u64, 0u64);
    for (l, c) in &val {
        ct += 1;
        if classify_type(l).idx() == *c {
            cc += 1;
        }
    }
    let code_acc = 100.0 * cc as f32 / ct as f32;

    let dev = if is_available(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };
    let nb = tr.len() / B;
    let steps = EPOCHS * nb;
    println!(
        "ML type classifier: device={dev:?} train={} val={} vocab={v} classes={NC}",
        tr.len(),
        va.len()
    );

    let mut pos = vec![0f32; B * L * L];
    for bi in 0..B {
        for li in 0..L {
            pos[(bi * L + li) * L + li] = 1.0;
        }
    }
    let mut model = build(v, &init_params(v), true);
    // The NaN is fixed structurally in forward() by tanh-bounding Q/K so the
    // attention softmax can't f32-overflow. AdamW weight decay 0.1 additionally
    // bounds the parameter norm (keeps the non-attention path off inf).
    let mut opt = AdamW::new(0.003).with_weight_decay(0.1);
    let sched = LrSchedule::WarmupCosine {
        base: 0.003,
        min: 0.0003,
        warmup: steps / 12,
        total: steps,
    };
    let mut rng = 0x1234u64;
    let mut nextu = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut g = 0usize;
    let mut best = model.clone();
    let mut best_acc = -1.0f32;

    for epoch in 0..EPOCHS {
        let mut order: Vec<usize> = (0..tr.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, (nextu() % (i as u64 + 1)) as usize);
        }
        let mut ep = 0.0f32;
        for b in 0..nb {
            opt.set_lr(sched.lr_at(g));
            g += 1;
            let rows = &order[b * B..b * B + B];
            let (onehot, y) = minibatch(&tr, rows, v);
            let (next, loss) = train_step(
                &model,
                dev,
                &mut opt,
                &[("x", &onehot), ("pos", &pos), ("y", &y)],
            );
            model = next;
            ep += loss;
        }
        let avg = ep / nb as f32;
        if !avg.is_finite() {
            println!("epoch {epoch}: NaN — stop, keep best ({best_acc:.1}%)");
            break;
        }
        if epoch == 0 || epoch == EPOCHS - 1 || (epoch + 1) % 5 == 0 {
            let a = eval(&model, &va, &pos, v, dev);
            println!(
                "epoch {epoch:>2}  loss {avg:.4}  ML val acc {:.1}%",
                100.0 * a
            );
            if a > best_acc {
                best_acc = a;
                best = model.clone();
            }
        }
    }

    let ml = 100.0 * eval(&best, &va, &pos, v, dev);
    println!("\n=== head-to-head (5-way type, richer data) ===");
    println!("  ML classifier  : {ml:.1}% (best)");
    println!("  fast code rule : {code_acc:.1}%");
    println!(
        "  winner: {}",
        if ml > code_acc {
            "ML"
        } else {
            "fast code (same rule-shaped lesson)"
        }
    );
    rlx_tensor::clear_cache();
}
