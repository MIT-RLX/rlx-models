//! Fine-tune the extractive tagger on REAL htop terminal captures (on Metal).
//!
//! Data: htop frames captured at widths 92/118 (train) and 150 (held-out val),
//! auto-labeled per-char by the rule cleaner (content = process-row data span +
//! column header; chrome = meters/gauges/header/footer/padding). This is the
//! capture -> auto-label -> train-on-Metal -> evaluate-on-unseen-width loop.
//!
//! Same architecture as `train_tagger` (1 bidirectional attention block, no
//! norm — the DSL's norm backward NaNs). Reads the label files produced by
//! scratchpad/htop_train/label_htop.py.
//!
//! Run: `cargo run -q -p rlx-termclean --bin rlx-termclean-train-htop --features train -- <data_dir>`

use std::collections::HashMap;

use rlx_tensor::{
    Adam, Device, Func, GraphScope, LrSchedule, MaskKind, Optimizer, Tensor, is_available, shape,
};

const L: usize = 160;
const B: usize = 32;
const NH: usize = 4;
const DH: usize = 16;
const D: usize = NH * DH;
const FF: usize = 4 * D;
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
    let mut h = &emb + &pos.matmul(&pe);
    for i in 0..NLAYERS {
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
        let w1 = s.param(format!("w1{i}"), shape![D, FF]);
        let b1 = s.param(format!("b1{i}"), shape![FF]);
        let w2 = s.param(format!("w2{i}"), shape![FF, D]);
        let b2 = s.param(format!("b2{i}"), shape![D]);
        let ff = &(&h.matmul(&w1) + &b1).gelu().matmul(&w2) + &b2;
        h = &h + &ff;
    }
    let wo = s.param("wo", shape![D, 1]);
    let bo = s.param("bo", shape![1]);
    let logit = &h.matmul(&wo) + &bo;
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
    let mut f = Func::new("htop", move |s| forward(s, dm, with_loss));
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
    let scale = {
        let n = sumsq.sqrt();
        if n > 1.0 { (1.0 / n) as f32 } else { 1.0 }
    };
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

// ---- data ----
type Ex = (Vec<usize>, Vec<f32>, Vec<f32>); // idx, label, mask

fn load_pairs(dir: &str, name: &str) -> Vec<(String, String)> {
    let inp = std::fs::read_to_string(format!("{dir}/{name}_input.txt")).expect("input file");
    let tg = std::fs::read_to_string(format!("{dir}/{name}_tags.txt")).expect("tags file");
    inp.lines()
        .zip(tg.lines())
        .filter(|(a, b)| a.chars().count() == b.chars().count() && !a.trim().is_empty())
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

fn build_vocab(pairs: &[(String, String)]) -> HashMap<char, usize> {
    let mut map = HashMap::new();
    let mut next = 1usize;
    for (inp, _) in pairs {
        for c in inp.chars().take(L) {
            map.entry(c).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
        }
    }
    map
}

fn encode(inp: &str, tags: &str, map: &HashMap<char, usize>) -> Ex {
    let ic: Vec<char> = inp.chars().collect();
    let tc: Vec<char> = tags.chars().collect();
    let mut idx = vec![0usize; L];
    let mut lab = vec![0f32; L];
    let mut mask = vec![0f32; L];
    for i in 0..L.min(ic.len()) {
        idx[i] = *map.get(&ic[i]).unwrap_or(&0);
        lab[i] = if tc[i] == 'C' { 1.0 } else { 0.0 };
        mask[i] = 1.0;
    }
    (idx, lab, mask)
}

fn minibatch(enc: &[Ex], rows: &[usize], v: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut onehot = vec![0f32; B * L * v];
    let mut lab = vec![0f32; B * L];
    let mut mask = vec![0f32; B * L];
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

fn predict(
    dm: Dims,
    model: &Func,
    enc: &[Ex],
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

fn f1(tp: u64, fp: u64, fnn: u64) -> f32 {
    let p = tp as f32 / (tp + fp).max(1) as f32;
    let r = tp as f32 / (tp + fnn).max(1) as f32;
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

fn evaluate(
    dm: Dims,
    model: &Func,
    val: &[Ex],
    pos: &[f32],
    names: &[&str],
    dev: Device,
) -> (f32, f32) {
    let (mut correct, mut total) = (0u64, 0u64);
    let (mut tp, mut fp, mut fnn) = (0u64, 0u64, 0u64);
    for b in 0..val.len() / B {
        let rows: Vec<usize> = (b * B..b * B + B).collect();
        let logit = predict(dm, model, val, &rows, pos, names, dev);
        for (bi, &r) in rows.iter().enumerate() {
            let (_, lab, m) = &val[r];
            for li in 0..L {
                if m[li] < 0.5 {
                    continue;
                }
                total += 1;
                let (p, t) = (logit[bi * L + li] > 0.5, lab[li] > 0.5);
                if p == t {
                    correct += 1;
                }
                match (p, t) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fnn += 1,
                    _ => {}
                }
            }
        }
    }
    (correct as f32 / total.max(1) as f32, f1(tp, fp, fnn))
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            if it.peek() == Some(&'[') {
                it.next();
            }
            while let Some(&n) = it.peek() {
                it.next();
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Clean a full htop frame: run the model per line, keep predicted-content chars.
fn clean_frame(
    dm: Dims,
    model: &Func,
    path: &str,
    map: &HashMap<char, usize>,
    pos: &[f32],
    names: &[&str],
    dev: Device,
) -> Vec<String> {
    let raw = std::fs::read_to_string(path).unwrap_or_default();
    let mut cleaned = Vec::new();
    for line in raw.lines() {
        let s = strip_ansi(line);
        if s.trim().is_empty() {
            continue;
        }
        let e = encode(&s, &"C".repeat(s.chars().count()), map); // labels unused for predict
        let mut rows = vec![0usize; B];
        // place this single example at row 0 by building a 1-example batch
        let batch = vec![e; 1];
        rows[0] = 0;
        let logit = predict(dm, model, &batch_fill(&batch), &rows, pos, names, dev);
        let kept: String = s
            .chars()
            .take(L)
            .enumerate()
            .filter(|(i, _)| logit[*i] > 0.5)
            .map(|(_, c)| c)
            .collect();
        if !kept.trim().is_empty() {
            cleaned.push(kept.trim_end().to_string());
        }
    }
    cleaned
}

/// Expand a 1-example slice to a B-row batch (row 0 = the example, rest copies).
fn batch_fill(one: &[Ex]) -> Vec<Ex> {
    let mut v = Vec::with_capacity(B);
    for _ in 0..B {
        v.push(one[0].clone());
    }
    v
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-Shared-rlx-models/47c4aeaa-f216-4fbb-9241-d0928201ee4a/scratchpad/htop_train".to_string()
    });
    const EPOCHS: usize = 45;

    let train_p = load_pairs(&dir, "train");
    let val_p = load_pairs(&dir, "val");
    let map = build_vocab(&train_p);
    let v = map.len() + 1;
    let dm = Dims { v };

    let train_enc: Vec<Ex> = train_p.iter().map(|(a, b)| encode(a, b, &map)).collect();
    let val_enc: Vec<Ex> = val_p.iter().map(|(a, b)| encode(a, b, &map)).collect();

    let dev = if is_available(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };
    let nb = train_enc.len() / B;
    let total_steps = EPOCHS * nb;
    println!(
        "htop-tagger: device={dev:?} train={} val={} (held-out width 150) vocab={v} L={L} B={B} epochs={EPOCHS} steps={total_steps}",
        train_enc.len(),
        val_enc.len()
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
        warmup: total_steps / 15,
        total: total_steps,
    };
    let mut rng = 0x1234u64;
    let mut nextu = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut gstep = 0usize;

    for epoch in 0..EPOCHS {
        let mut order: Vec<usize> = (0..train_enc.len()).collect();
        for i in (1..order.len()).rev() {
            let j = (nextu() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut ep = 0.0f32;
        for b in 0..nb {
            opt.set_lr(sched.lr_at(gstep));
            gstep += 1;
            let rows = &order[b * B..b * B + B];
            let (onehot, lab, mask) = minibatch(&train_enc, rows, v);
            let feed: &[(&str, &[f32])] =
                &[("x", &onehot), ("y", &lab), ("m", &mask), ("pos", &pos)];
            let (next, loss) = train_step(&model, dev, &mut opt, feed, &names);
            model = next;
            ep += loss;
        }
        if epoch == 0 || epoch == EPOCHS - 1 || (epoch + 1) % 5 == 0 {
            let (acc, f) = evaluate(dm, &model, &val_enc, &pos, &names, dev);
            println!(
                "epoch {epoch:>2}  train_loss {:.4}  val(width150) per-char acc {:.1}%  content_F1 {:.3}",
                ep / nb as f32,
                100.0 * acc,
                f
            );
        }
    }

    let (acc, f) = evaluate(dm, &model, &val_enc, &pos, &names, dev);
    println!(
        "\nFINAL on held-out width-150 htop: per-char acc {:.1}%, content-class F1 {:.3}",
        100.0 * acc,
        f
    );

    // Clean a full held-out frame end-to-end.
    let frame = format!("{dir}/f_150_1.txt");
    let cleaned = clean_frame(dm, &model, &frame, &map, &pos, &names, dev);
    println!("\n=== model-cleaned held-out htop frame (first 12 kept lines) ===");
    for l in cleaned.iter().take(12) {
        println!("  {}", l.chars().take(96).collect::<String>());
    }

    rlx_tensor::clear_cache();
}
