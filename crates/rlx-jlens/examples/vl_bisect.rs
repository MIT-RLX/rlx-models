//! Find the first Qwen2.5-VL layer whose output diverges from the reference.
//!
//! The VL trunk's real-weight forward is degenerate in rlx — the crate's own
//! GGUF CLI emits `<|im_end|>` on repeat — and CPU and Metal agree on the
//! garbage, which rules out a kernel and points at the model definition.
//!
//! This isolates the trunk from everything around it. The *reference's own*
//! embedding output is fed in as `prefill_hidden`, every layer exit is tapped,
//! and each is compared to the reference's hidden state at the same boundary.
//! Whichever layer first drops in cosine is the one that is wrong; layers
//! before it are proof that the loader, the norms and the mRoPE tables are all
//! fine up to that point.
//!
//! ```bash
//! python3 crates/rlx-jlens/scripts/qwen25vl_reference.py \
//!     --weights /Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct --out /tmp/vl_ref.safetensors
//! cargo run -p rlx-jlens --features qwen25-vl --release --example vl_bisect -- \
//!     --ref /tmp/vl_ref.safetensors
//! ```

#![cfg(feature = "qwen25-vl")]

use anyhow::{Context, Result, bail};
use rlx_jlens::model::LensModel;
use rlx_jlens::models::qwen25_vl::Qwen25VlLensModel;
use rlx_runtime::{Device, Session};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32
}

fn rms(a: &[f32]) -> f32 {
    (a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64).sqrt() as f32
}

fn main() -> Result<()> {
    let mut dir = "/Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct".to_string();
    let mut reference = String::new();
    let mut device = Device::Cpu;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String> {
            argv.get(i + 1)
                .cloned()
                .with_context(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--weights" => {
                dir = need(i)?;
                i += 2;
            }
            "--ref" => {
                reference = need(i)?;
                i += 2;
            }
            "--device" => {
                device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }
    if reference.is_empty() {
        bail!("pass --ref (run scripts/qwen25vl_reference.py first)");
    }

    let refs = rlx_core::load_weight_map(std::path::Path::new(&reference), &[])?;
    let get = |k: &str| -> Result<Vec<f32>> {
        refs.get(k)
            .map(|t| t.0.to_vec())
            .with_context(|| format!("{k} missing from the reference dump"))
    };

    let model = Qwen25VlLensModel::open(&dir)?;
    let n = model.n_layers();
    let d = model.d_model();
    let h0 = get("hidden.0")?;
    let seq = h0.len() / d;
    eprintln!("{n} layers, d {d}, seq {seq}, {device:?}");

    // Text tokens carry `[i, i, i, 0]` in every mRoPE section, which is what
    // plain 1-D positions reduce to — but pass them explicitly so this exercises
    // the same `with_sections` path an image prompt would.
    let sections: Vec<[usize; 4]> = (0..seq).map(|i| [i, i, i, 0]).collect();
    let model = model.with_sections(sections);

    let all: Vec<usize> = (0..n).collect();
    let stack = model.stack(&all, n - 1, 1, seq)?;

    let mut g = stack.tapped.graph().clone();
    // `outputs[0]` is the target residual; `outputs[1..]` are the layer exits.
    let outs_ids = g.outputs.clone();
    g.set_outputs(outs_ids.clone());
    let mut sess = Session::new(device).compile(g);
    for (name, data) in &stack.params {
        sess.set_param(name, data);
    }
    let mut feeds: Vec<(&str, &[f32])> = vec![(stack.token_input.as_str(), &h0[..])];
    for (name, data) in &stack.extra_feeds {
        feeds.push((name.as_str(), &data[..]));
    }
    let outs = sess.run(&feeds);

    println!(
        "\n{:<8}{:>12}{:>14}{:>14}",
        "layer", "cosine", "rlx rms", "ref rms"
    );
    let mut first_bad = None;
    for (slot, &layer) in all.iter().enumerate() {
        let got = &outs[slot + 1];
        let want = get(&format!("hidden.{}", layer + 1))?;
        let c = cosine(got, &want);
        if c < 0.99 && first_bad.is_none() {
            first_bad = Some(layer);
        }
        if layer < 4 || first_bad.is_some_and(|b| layer <= b + 2) || layer + 1 == n {
            println!(
                "{:<8}{:>12.6}{:>14.3}{:>14.3}",
                layer,
                c,
                rms(got),
                rms(&want)
            );
        }
    }
    // HF appends each hidden state *before* the layer that consumes it and then
    // appends `norm(last)` at the end, so `hidden.{n}` is the normed output, not
    // what block `n-1` emits. Comparing the two directly always shows a break at
    // the last layer. Run the model's own unembedding instead, which applies
    // that same norm, and check the logits.
    let unembed = model.unembed(seq)?;
    let mut ug = unembed.graph.clone();
    ug.set_outputs(vec![*ug.outputs.first().context("unembed has no output")?]);
    let mut usess = Session::new(device).compile(ug);
    for (name, data) in &unembed.params {
        usess.set_param(name, data);
    }
    let logits = usess.run(&[(unembed.residual_input.as_str(), &outs[0][..])]);
    let want = get("logits")?;
    let vocab = unembed.vocab;
    let last = &logits[0][(seq - 1) * vocab..seq * vocab];
    let want_last = &want[(seq - 1) * vocab..seq * vocab];
    println!("\nlogits cosine {:.6}", cosine(last, want_last));

    let top = |v: &[f32]| -> Vec<(usize, f32)> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx[..5].iter().map(|&i| (i, v[i])).collect()
    };
    println!("rlx top5 {:?}", top(last));
    println!("ref top5 {:?}", top(want_last));

    match first_bad {
        Some(l) if l + 1 == n => {
            println!("\nevery block matches; the last row is the norm convention above")
        }
        Some(l) => println!("\nfirst divergent layer: {l}"),
        None => println!("\nevery layer matches"),
    }
    Ok(())
}
