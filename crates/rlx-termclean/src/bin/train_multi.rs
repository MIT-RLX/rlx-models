//! Unified multi-tool extractive tagger — trained on REAL captures from 16 TUI
//! apps (htop/top/btop/less/vim/nano/cat/fzf/psaux/lsla/ipython/node/tig/
//! lazygit/ncdu/bat), auto-labeled per-char, trained on Metal, evaluated on a
//! held-out terminal width with per-app F1.
//!
//! One model, many tools — the thing per-app rules can't do. Loads the files
//! produced by scratchpad/multi/label_multi.py.
//!
//! Run: `cargo run -q -p rlx-termclean --bin rlx-termclean-train-multi --features train -- <data_dir>`

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use rlx_tensor::{
    AdamW, Device, Func, GraphScope, LrSchedule, MaskKind, Optimizer, Tensor, is_available, shape,
};

const L: usize = 200;
const B: usize = 32;
const NH: usize = 4;
const DH: usize = 16;
const D: usize = NH * DH;
const FF: usize = 4 * D;
// Ablation toggles — env-driven so ONE binary runs the whole study:
//   ABL_LAYERS=N  depth (default 3)      ABL_NOGATE  plain residuals (no ReZero)
//   ABL_NOTANH    unbounded Q/K          ABL_NOWD    weight decay 0
fn nlayers() -> usize {
    std::env::var("ABL_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}
fn no_gate() -> bool {
    std::env::var("ABL_NOGATE").is_ok()
}
fn no_tanh() -> bool {
    std::env::var("ABL_NOTANH").is_ok()
}
fn no_wd() -> bool {
    std::env::var("ABL_NOWD").is_ok()
}
// ReZero per-branch gains (init small) make norm-free DEPTH stable — the lever
// that was blocked (2 plain residual blocks blew up). Unlike the seq2seq (capped
// by alignment), this tagger classifies each char in place, so depth HELPS:
// Ablation: content-F1 1L 0.932 → 2L 0.954 → 3L 0.954 — DEPTH PLATEAUS AT 2, so
// default=2 (same F1 as 3, ~1.5x faster). ReZero gains −15.5pp / tanh −11.3pp+NaN
// if removed (both load-bearing); weight decay −0.5pp.
const GATE0: f32 = 0.1;

fn param_names() -> Vec<String> {
    let mut p = vec!["we".to_string(), "pe".to_string()];
    for i in 0..nlayers() {
        for base in ["wq", "wk", "wv", "w1", "b1", "w2", "b2"] {
            p.push(format!("{base}{i}"));
        }
        if !no_gate() {
            p.push(format!("ga{i}"));
            p.push(format!("gf{i}"));
        }
    }
    p.push("wo".to_string());
    p.push("bo".to_string());
    p.push("wvd".to_string());
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
    for i in 0..nlayers() {
        let s = i * 11;
        p.push((format!("wq{i}"), w(D * D, 2 + s, 0.08)));
        p.push((format!("wk{i}"), w(D * D, 3 + s, 0.08)));
        p.push((format!("wv{i}"), w(D * D, 4 + s, 0.08)));
        p.push((format!("w1{i}"), w(D * FF, 6 + s, 0.06)));
        p.push((format!("b1{i}"), vec![0.0; FF]));
        p.push((format!("w2{i}"), w(FF * D, 7 + s, 0.03)));
        p.push((format!("b2{i}"), vec![0.0; D]));
        if !no_gate() {
            p.push((format!("ga{i}"), vec![GATE0; D]));
            p.push((format!("gf{i}"), vec![GATE0; D]));
        }
    }
    p.push(("wo".to_string(), w(D, 5, 0.005)));
    p.push(("bo".to_string(), vec![0.0]));
    p.push(("wvd".to_string(), w(D, 25, 0.02))); // small init: feature learned in, not forced
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
    // vertical-consistency feature (per column) projected into the hidden — lets
    // the model use screen-level structure to resolve ACS/panel dividers.
    let vd = s.input("vd", shape![B * L, 1]);
    let wvd = s.param("wvd", shape![1, D]);
    let mut h = &(&emb + &pos.matmul(&pe)) + &vd.matmul(&wvd);
    for i in 0..nlayers() {
        let wq = s.param(format!("wq{i}"), shape![D, D]);
        let wk = s.param(format!("wk{i}"), shape![D, D]);
        let wv = s.param(format!("wv{i}"), shape![D, D]);
        // tanh-bounded Q/K caps scores at ±sqrt(dh) so the softmax can't overflow
        // to NaN (ABL_NOTANH removes it to demonstrate the NaN it prevents).
        let (qp, kp) = if no_tanh() {
            (h.matmul(&wq), h.matmul(&wk))
        } else {
            (h.matmul(&wq).tanh(), h.matmul(&wk).tanh())
        };
        let q = heads(qp);
        let k = heads(kp);
        let vv = heads(h.matmul(&wv));
        let attn = q
            .attention(&k, &vv, NH, DH, MaskKind::None)
            .reshape(vec![bl, D as i64]);
        // ReZero gate on the attention branch (ABL_NOGATE → plain residual).
        if no_gate() {
            h = &h + &attn;
        } else {
            let ga = s.param(format!("ga{i}"), shape![D]);
            h = &h + &(&attn * &ga);
        }
        let w1 = s.param(format!("w1{i}"), shape![D, FF]);
        let b1 = s.param(format!("b1{i}"), shape![FF]);
        let w2 = s.param(format!("w2{i}"), shape![FF, D]);
        let b2 = s.param(format!("b2{i}"), shape![D]);
        let ff = &(&h.matmul(&w1) + &b1).gelu().matmul(&w2) + &b2;
        if no_gate() {
            h = &h + &ff;
        } else {
            let gf = s.param(format!("gf{i}"), shape![D]);
            h = &h + &(&ff * &gf);
        }
    }
    let wo = s.param("wo", shape![D, 1]);
    let bo = s.param("bo", shape![1]);
    let logit = &h.matmul(&wo) + &bo;
    if with_loss {
        let y = s.input("y", shape![B * L, 1]);
        let m = s.input("m", shape![B * L, 1]);
        // BCE-with-logits (proper classification loss, better F1 than MSE-to-0/1),
        // numerically-stable form: relu(z) - z*y + log(1 + exp(-|z|)).
        let sp = (&(&logit.abs() * -1.0f32).exp() + 1.0f32).log();
        let bce = &(&logit.relu() - &(&logit * &y)) + &sp;
        (&m * &bce).mean([0, 1], false)
    } else {
        logit
    }
}

fn build(dm: Dims, params: &[(String, Vec<f32>)], with_loss: bool) -> Func {
    let mut f = Func::new("multi", move |s| forward(s, dm, with_loss));
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
        if n > 0.5 { (0.5 / n) as f32 } else { 1.0 }
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

type Ex = (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>); // idx, label, mask, vdiv

fn load(dir: &str, name: &str) -> (Vec<(String, String)>, Vec<String>) {
    let inp = std::fs::read_to_string(format!("{dir}/{name}_input.txt")).expect("input");
    let tg = std::fs::read_to_string(format!("{dir}/{name}_tags.txt")).expect("tags");
    let ap = std::fs::read_to_string(format!("{dir}/{name}_app.txt")).expect("app");
    let mut pairs = Vec::new();
    let mut apps = Vec::new();
    for ((a, b), c) in inp.lines().zip(tg.lines()).zip(ap.lines()) {
        if a.chars().count() == b.chars().count() && !a.trim().is_empty() {
            pairs.push((a.to_string(), b.to_string()));
            apps.push(c.to_string());
        }
    }
    (pairs, apps)
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

/// Per-column vertical-consistency feature: within each screen (consecutive
/// same-app lines), how dominant is the modal non-space char at each column. A
/// vertical rule/divider (│, |, or a VT100-ACS letter like x/q that survives as
/// ASCII) is ~1.0; varied content columns are low. Fed as a LEARNED input (not a
/// hard label rule) so the model can tell a vertical-line 'x' from an 'x' in a
/// word — the exact signal the panel/dashboard apps need.
fn compute_vdiv(pairs: &[(String, String)], apps: &[String]) -> Vec<Vec<f32>> {
    let n = pairs.len();
    let mut out = vec![vec![0f32; L]; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && apps[j] == apps[i] {
            j += 1;
        }
        let rows: Vec<Vec<char>> = (i..j)
            .map(|k| pairs[k].0.chars().take(L).collect())
            .collect();
        let mut cons = vec![0f32; L];
        for c in 0..L {
            let mut counts: HashMap<char, u32> = HashMap::new();
            let mut nonspace = 0u32;
            for r in &rows {
                if let Some(&ch) = r.get(c) {
                    if ch != ' ' {
                        *counts.entry(ch).or_insert(0) += 1;
                        nonspace += 1;
                    }
                }
            }
            if nonspace >= 3 {
                cons[c] = counts.values().copied().max().unwrap_or(0) as f32 / nonspace as f32;
            }
        }
        for (li, k) in (i..j).enumerate() {
            for c in 0..L.min(rows[li].len()) {
                if rows[li][c] != ' ' {
                    out[k][c] = cons[c];
                }
            }
        }
        i = j;
    }
    out
}

fn encode(inp: &str, tags: &str, vdiv: &[f32], map: &HashMap<char, usize>) -> Ex {
    let ic: Vec<char> = inp.chars().collect();
    let tc: Vec<char> = tags.chars().collect();
    let mut idx = vec![0usize; L];
    let mut lab = vec![0f32; L];
    let mut mask = vec![0f32; L];
    let mut vd = vec![0f32; L];
    for i in 0..L.min(ic.len()) {
        idx[i] = *map.get(&ic[i]).unwrap_or(&0);
        lab[i] = if tc[i] == 'C' { 1.0 } else { 0.0 };
        mask[i] = 1.0;
        vd[i] = vdiv.get(i).copied().unwrap_or(0.0);
    }
    (idx, lab, mask, vd)
}

fn minibatch(enc: &[Ex], rows: &[usize], v: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut onehot = vec![0f32; B * L * v];
    let mut lab = vec![0f32; B * L];
    let mut mask = vec![0f32; B * L];
    let mut vd = vec![0f32; B * L];
    for (bi, &r) in rows.iter().enumerate() {
        let (idx, l, m, vdv) = &enc[r];
        for li in 0..L {
            onehot[(bi * L + li) * v + idx[li]] = 1.0;
            lab[bi * L + li] = l[li];
            mask[bi * L + li] = m[li];
            vd[bi * L + li] = vdv[li];
        }
    }
    (onehot, lab, mask, vd)
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
    let (onehot, _, _, vd) = minibatch(enc, rows, dm.v);
    pred.run_on(dev, &[("x", &onehot), ("pos", pos), ("vd", &vd)])
        .remove(0)
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
    apps: &[String],
    pos: &[f32],
    names: &[&str],
    dev: Device,
) -> (f32, f32, BTreeMap<String, (u64, u64, u64, u64, u64)>) {
    let (mut correct, mut total) = (0u64, 0u64);
    let (mut tp, mut fp, mut fnn) = (0u64, 0u64, 0u64);
    let mut per: BTreeMap<String, (u64, u64, u64, u64, u64)> = BTreeMap::new(); // tp,fp,fn,correct,total
    for b in 0..val.len() / B {
        let rows: Vec<usize> = (b * B..b * B + B).collect();
        let logit = predict(dm, model, val, &rows, pos, names, dev);
        for (bi, &r) in rows.iter().enumerate() {
            let (_, lab, m, _) = &val[r];
            let e = per.entry(apps[r].clone()).or_default();
            for li in 0..L {
                if m[li] < 0.5 {
                    continue;
                }
                total += 1;
                e.4 += 1;
                let (p, t) = (logit[bi * L + li] > 0.0, lab[li] > 0.5); // BCE: threshold pre-sigmoid at 0
                if p == t {
                    correct += 1;
                    e.3 += 1;
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
    (correct as f32 / total.max(1) as f32, f1(tp, fp, fnn), per)
}

/// Persist the trained tagger as a self-describing bundle under `dir`: a
/// `manifest.json` (architecture, metrics, and a param table), `vocab.txt`
/// (id-to-codepoint), and `weights.f32` (raw little-endian f32 in param order).
/// The format is documented in `weights/README.md`.
fn save_weights(
    dir: &str,
    model: &Func,
    names: &[String],
    map: &HashMap<char, usize>,
    v: usize,
    f1: f32,
    acc: f32,
) {
    use std::io::Write;
    std::fs::create_dir_all(dir).expect("create weights dir");
    let mut wf = std::io::BufWriter::new(
        std::fs::File::create(format!("{dir}/weights.f32")).expect("weights.f32"),
    );
    let mut table = Vec::new();
    let mut off = 0usize;
    for name in names {
        let data = model.param_binding(name).unwrap().to_vec();
        for &x in &data {
            wf.write_all(&x.to_le_bytes()).unwrap();
        }
        table.push(format!(
            "{{\"name\":\"{name}\",\"offset\":{off},\"len\":{}}}",
            data.len()
        ));
        off += data.len();
    }
    wf.flush().unwrap();
    let mut inv = vec![0u32; v];
    for (&c, &i) in map {
        if i < v {
            inv[i] = c as u32;
        }
    }
    let vocab: String = (1..v)
        .map(|i| inv[i].to_string())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(format!("{dir}/vocab.txt"), vocab).expect("vocab.txt");
    let manifest = format!(
        "{{\n  \"model\": \"rlx-termclean-tagger\",\n  \"task\": \"per-char content/chrome tagging\",\n  \"arch\": {{\"seq_len\": {L}, \"d_model\": {D}, \"n_heads\": {NH}, \"head_dim\": {DH}, \"ff\": {FF}, \"layers\": {}, \"gate_init\": {GATE0}, \"vocab\": {v}, \"residual\": \"rezero-gain\", \"attn\": \"tanh-bounded-qk\", \"extra_feature\": \"vertical-consistency\"}},\n  \"loss\": \"bce_with_logits\",\n  \"decision\": \"logit > 0.0\",\n  \"metrics\": {{\"content_f1\": {f1:.4}, \"per_char_acc\": {acc:.4}}},\n  \"dtype\": \"f32_le\",\n  \"vocab_file\": \"vocab.txt\",\n  \"weights_file\": \"weights.f32\",\n  \"params\": [{}]\n}}\n",
        nlayers(),
        table.join(",")
    );
    std::fs::write(format!("{dir}/manifest.json"), manifest).expect("manifest.json");
    println!(
        "saved weights bundle → {dir}/ ({} params, {off} f32)",
        names.len()
    );
}

fn main() {
    // args: `[<data_dir>] [--save <weights_dir>]` in any order.
    let raw: Vec<String> = std::env::args().collect();
    let (mut save_dir, mut positional) = (None, None);
    let mut i = 1;
    while i < raw.len() {
        if raw[i] == "--save" {
            save_dir = raw.get(i + 1).cloned();
            i += 2;
        } else {
            positional.get_or_insert(raw[i].clone());
            i += 1;
        }
    }
    let dir = positional.unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-Shared-rlx-models/47c4aeaa-f216-4fbb-9241-d0928201ee4a/scratchpad/multi".to_string()
    });
    const EPOCHS: usize = 80;

    let (train_p, train_app) = load(&dir, "train");
    let (val_p, val_app) = load(&dir, "val");
    let map = build_vocab(&train_p);
    let v = map.len() + 1;
    let dm = Dims { v };

    let train_vd = compute_vdiv(&train_p, &train_app);
    let val_vd = compute_vdiv(&val_p, &val_app);
    let train_enc: Vec<Ex> = train_p
        .iter()
        .zip(&train_vd)
        .map(|((a, b), vd)| encode(a, b, vd, &map))
        .collect();
    let val_enc: Vec<Ex> = val_p
        .iter()
        .zip(&val_vd)
        .map(|((a, b), vd)| encode(a, b, vd, &map))
        .collect();

    // Group train indices by app for BALANCED sampling (small apps not drowned).
    let mut ag: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, a) in train_app.iter().enumerate() {
        ag.entry(a.as_str()).or_default().push(i);
    }
    let groups: Vec<Vec<usize>> = ag.into_values().collect();
    // sqrt-temperature app weights: lift small apps without starving big ones.
    let mut app_pick: Vec<usize> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        for _ in 0..((g.len() as f64).sqrt().round() as usize + 1) {
            app_pick.push(gi);
        }
    }

    let dev = if is_available(Device::Metal) {
        Device::Metal
    } else {
        Device::Cpu
    };
    let nb = train_enc.len() / B;
    let total_steps = EPOCHS * nb;
    println!(
        "multi-tagger: device={dev:?} apps=16 train={} val={} (held-out width 140) vocab={v} L={L} epochs={EPOCHS} steps={total_steps}",
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
    // The NaN is fixed structurally in forward() by tanh-bounding Q/K so the
    // attention softmax can't f32-overflow. AdamW weight decay 0.1 additionally
    // bounds the parameter norm (keeps the non-attention path off inf); keep-best
    // below is now just model selection, not a NaN crutch.
    let wd = if no_wd() { 0.0 } else { 0.1 };
    println!(
        "ABLATION cfg: layers={} rezero_gate={} tanh_qk={} weight_decay={wd}",
        nlayers(),
        !no_gate(),
        !no_tanh()
    );
    let mut opt = AdamW::new(0.0025).with_weight_decay(wd);
    let sched = LrSchedule::WarmupCosine {
        base: 0.0025,
        min: 0.0002,
        warmup: total_steps / 12,
        total: total_steps,
    };
    let mut rng = 0x2468u64;
    let mut nextu = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut gstep = 0usize;
    let mut best_model = model.clone();
    let mut best_f1 = -1.0f32;

    let t_train = Instant::now();
    for epoch in 0..EPOCHS {
        let mut ep = 0.0f32;
        for _ in 0..nb {
            opt.set_lr(sched.lr_at(gstep));
            gstep += 1;
            // per-app balanced batch: uniform app, then uniform example of that app
            let mut rows = Vec::with_capacity(B);
            for _ in 0..B {
                let gi = app_pick[(nextu() % app_pick.len() as u64) as usize];
                let g = &groups[gi];
                rows.push(g[(nextu() % g.len() as u64) as usize]);
            }
            let (onehot, lab, mask, vd) = minibatch(&train_enc, &rows, v);
            let feed: &[(&str, &[f32])] = &[
                ("x", &onehot),
                ("y", &lab),
                ("m", &mask),
                ("pos", &pos),
                ("vd", &vd),
            ];
            let (next, loss) = train_step(&model, dev, &mut opt, feed, &names);
            model = next;
            ep += loss;
        }
        let avg = ep / nb as f32;
        if !avg.is_finite() {
            println!(
                "epoch {epoch:>2}: non-finite loss — stopping, keeping best (F1 {best_f1:.3})"
            );
            break;
        }
        if epoch == 0 || epoch == EPOCHS - 1 || (epoch + 1) % 5 == 0 {
            let (acc, fscore, _) = evaluate(dm, &model, &val_enc, &val_app, &pos, &names, dev);
            println!(
                "epoch {epoch:>2}  train_loss {avg:.4}  val per-char acc {:.1}%  content_F1 {:.3}",
                100.0 * acc,
                fscore
            );
            if fscore > best_f1 {
                best_f1 = fscore;
                best_model = model.clone();
            }
        }
    }

    println!("best val content-F1 during training: {best_f1:.3}");
    let secs = t_train.elapsed().as_secs_f32();
    println!(
        "TRAIN speed: {secs:.1}s total, {:.2} ms/step ({} layers, {total_steps} steps)",
        1000.0 * secs / total_steps as f32,
        nlayers()
    );
    let t_inf = Instant::now();
    let (acc, fscore, per) = evaluate(dm, &best_model, &val_enc, &val_app, &pos, &names, dev);
    let inf = t_inf.elapsed().as_secs_f32();
    let chars: usize = val_enc.len() * L;
    println!(
        "INFER speed: {:.0} chars/s ({} val chars in {inf:.2}s)",
        chars as f32 / inf,
        chars
    );
    println!(
        "\nFINAL unified model on held-out width-140 (all 16 apps): per-char acc {:.1}%, content-F1 {:.3}",
        100.0 * acc,
        fscore
    );
    println!("per-app: acc / content-F1 (val chars)");
    for (app, (tp, fp, fnn, correct, total)) in &per {
        let a = *correct as f32 / (*total).max(1) as f32;
        println!(
            "  {app:<9} acc {:.1}%  F1 {:.3}  ({} chars)",
            100.0 * a,
            f1(*tp, *fp, *fnn),
            total
        );
    }

    if let Some(sd) = &save_dir {
        save_weights(sd, &best_model, &names_owned, &map, v, fscore, acc);
    }
    rlx_tensor::clear_cache();
}
