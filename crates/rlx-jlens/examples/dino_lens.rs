//! The Jacobian lens on a vision transformer.
//!
//! DINOv3 has no vocabulary, so there is nothing to "read out" in the sense an
//! LM readout means. What there *is* is a well-defined target: the
//! representation each token actually ends up with. So the lens question
//! becomes a trajectory —
//!
//! > transported forward by `J_l`, how close is token *t*'s layer-*l*
//! > representation to its own final representation?
//!
//! — measured as cosine after the model's final LayerNorm. That is the direct
//! analogue of "rank of the answer" in the text case, and it needs no
//! vocabulary and nothing chosen by hand.
//!
//! The control is the same cosine **without** the transport, which is the
//! logit-lens analogue: decoding a mid-stack representation as if the remaining
//! layers were the identity. The gap between the two columns is what `J_l`
//! buys, exactly as in the text readout.
//!
//! ```bash
//! cargo run -p rlx-jlens --features dinov3,metal --release --example dino_lens -- \
//!     --device metal --weights /Volumes/FOUR/hiphop/weights/dinov3 --every 4
//! ```
//!
//! Input is synthetic unless `--hidden <f32.bin>` is given: the patch embedding
//! is host-side preprocessing outside the trunk, so what this feeds is the
//! assembled `[CLS, registers, patches]` sequence directly.

#![cfg(feature = "dinov3")]

use anyhow::{Context, Result, bail};
use rlx_jlens::models::dinov3::Dinov3LensModel;
use rlx_jlens::{FitConfig, LensModel, StackLens};
use rlx_runtime::{Device, Session};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(f32::MIN_POSITIVE)
}

fn main() -> Result<()> {
    let mut dir = "/Volumes/FOUR/hiphop/weights/dinov3".to_string();
    let mut device = Device::Cpu;
    let mut every = 4usize;
    let mut dim_batch = 8usize;
    let mut hidden_path: Option<String> = None;
    let mut image_path: Option<String> = None;
    let mut skip_first = 0usize;
    let mut out_path: Option<String> = None;

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
            "--device" => {
                device = rlx_runtime::parse_device(&need(i)?)?;
                i += 2;
            }
            "--every" => {
                every = need(i)?.parse::<usize>()?.max(1);
                i += 2;
            }
            "--dim-batch" => {
                dim_batch = need(i)?.parse::<usize>()?.max(1);
                i += 2;
            }
            "--hidden" => {
                hidden_path = Some(need(i)?);
                i += 2;
            }
            "--image" => {
                image_path = Some(need(i)?);
                i += 2;
            }
            "--skip-first" => {
                skip_first = need(i)?.parse()?;
                i += 2;
            }
            "--out" => {
                out_path = Some(need(i)?);
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }

    let model = Dinov3LensModel::open(&dir)?.with_name("dinov3");
    let d = model.d_model();
    let seq = model.seq_len();
    let first_patch = model.first_patch();
    eprintln!(
        "dinov3: {} layers, d_model {d}, {seq} tokens (CLS + {} registers + {} patches) on {:?}",
        model.n_layers(),
        first_patch - 1,
        seq - first_patch,
        device
    );

    // One image's assembled token sequence. Synthetic unless a real one is
    // supplied — the trajectory shape is a property of the trunk, but a real
    // image is what makes the *values* meaningful.
    let one: Vec<f32> = if let Some(p) = &image_path {
        // Real image through the model's own preprocessing: resize to
        // `image_size`, ImageNet-normalize, patch-embed, prepend CLS+registers.
        let img = image::open(p)
            .with_context(|| format!("opening {p}"))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        eprintln!("image {p} ({w}x{h})");
        model.hidden_from_rgb(img.as_raw(), h as usize, w as usize)?
    } else {
        match &hidden_path {
            Some(p) => {
                let raw = std::fs::read(p).with_context(|| format!("reading {p}"))?;
                let v: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                if v.len() != seq * d {
                    bail!(
                        "{p} holds {} floats, need {} ([{seq}, {d}])",
                        v.len(),
                        seq * d
                    );
                }
                eprintln!("hidden from {p}");
                v
            }
            None => {
                eprintln!("hidden: synthetic (pass --hidden for a real image)");
                (0..seq * d)
                    .map(|i| {
                        let t = i / d;
                        let c = i % d;
                        ((t as f32 * 0.11 + c as f32 * 0.017).sin() + (c as f32 * 0.003).cos())
                            * 0.5
                    })
                    .collect()
            }
        }
    };

    let target = model.n_layers() - 1;
    let layers: Vec<usize> = (0..target).step_by(every).collect();

    // The lens's premise is that the residual stream is dominated by its
    // identity path, so `J ≈ I + (what the blocks did)`. That is checkable
    // without fitting anything: if ‖h‖ grows sharply with depth, the identity
    // path does not dominate and no average linear map can bridge the gap.
    if std::env::var("RLX_DINO_NORMS_ONLY").is_ok() {
        let all: Vec<usize> = (0..target).collect();
        let stack = model.stack(&all, target, 1, seq)?;
        let mut fwd = Session::new(device).compile(stack.tapped.graph().clone());
        for (name, data) in &stack.params {
            fwd.set_param(name, data);
        }
        let outs = fwd.run(&[(stack.token_input.as_str(), &one[..])]);
        let norm = |o: &[f32], t: usize| -> f32 {
            o[t * d..(t + 1) * d]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
        };
        println!("\nresidual norm by layer (‖h‖), CLS and mean patch");
        println!(
            "{:<7} {:<14} {:<14} vs layer 0",
            "layer", "CLS", "patch mean"
        );
        let base: f32 =
            (first_patch..seq).map(|t| norm(&outs[1], t)).sum::<f32>() / (seq - first_patch) as f32;
        for (slot, &layer) in all.iter().enumerate() {
            let pm: f32 = (first_patch..seq)
                .map(|t| norm(&outs[slot + 1], t))
                .sum::<f32>()
                / (seq - first_patch) as f32;
            if layer % 2 == 0 || layer == target - 1 {
                println!(
                    "{:<7} {:<14.2} {:<14.2} {:.1}x",
                    layer,
                    norm(&outs[slot + 1], 0),
                    pm,
                    pm / base
                );
            }
        }
        let fpm: f32 =
            (first_patch..seq).map(|t| norm(&outs[0], t)).sum::<f32>() / (seq - first_patch) as f32;
        println!(
            "{:<7} {:<14.2} {:<14.2} {:.1}x   <- target (layer {target} exit)",
            "final",
            norm(&outs[0], 0),
            fpm,
            fpm / base
        );
        return Ok(());
    }
    eprintln!("fitting J for {} layers (target {target}) …", layers.len());

    let start = std::time::Instant::now();
    let mut lens = StackLens::new(
        &model,
        &layers,
        target,
        seq,
        FitConfig {
            dim_batch,
            skip_first,
            device,
        },
    )?;
    let batched: Vec<f32> = one.repeat(dim_batch);
    let js = lens.jacobians(&batched)?;
    eprintln!(
        "fitted in {:.1}s | {}",
        start.elapsed().as_secs_f64(),
        lens.timing().summary()
    );

    // Residuals at each tapped layer, and the final representation, at batch 1.
    let stack = model.stack(&layers, target, 1, seq)?;
    let mut fwd = Session::new(device).compile(stack.tapped.graph().clone());
    for (name, data) in &stack.params {
        fwd.set_param(name, data);
    }
    let outs = fwd.run(&[(stack.token_input.as_str(), &one[..])]);
    let tok = |o: &[f32], t: usize| -> Vec<f32> { o[t * d..(t + 1) * d].to_vec() };

    println!("\ncosine to each token's own final representation");
    println!("`lens` transports through J_l first; `raw` is the same read with no transport");
    println!("(the logit-lens analogue). CLS is token 0; patches start at {first_patch}.\n");
    println!(
        "{:<7} {:<18} {:<18} {:<18} gain",
        "layer", "CLS lens / raw", "patch mean lens", "patch mean raw"
    );
    println!("{}", "─".repeat(84));

    // Saved so the fitted J can be dissected off-line, and compared against a
    // language model's J on identical statistics.
    if let Some(p) = &out_path {
        let map: std::collections::BTreeMap<usize, _> =
            layers.iter().copied().zip(js.iter().cloned()).collect();
        rlx_jlens::JacobianLens::new(map, 1, target)?.save(p)?;
        eprintln!("wrote {p}");
    }

    // Sanity first: a residual block's Jacobian should be `I` plus what the
    // block did, so the mean diagonal sits near 1 in the text case. If it does
    // not here, `J` is not "this token's own transport" and the cosine columns
    // below are measuring the wrong thing.
    eprintln!("\nJ sanity: mean diagonal and ||J||/sqrt(d) per layer");
    for (slot, &layer) in layers.iter().enumerate() {
        let j = &js[slot];
        let diag: f32 = (0..d).map(|i| j.values[i * d + i]).sum::<f32>() / d as f32;
        eprintln!(
            "  layer {layer:<3} mean diag {diag:+.4}   ||J||/sqrt(d) {:.4}",
            j.scaled_norm()
        );
    }

    let n_patch = seq - first_patch;
    for (slot, &layer) in layers.iter().enumerate() {
        let cls_l = cosine(
            &js[slot].transport(&tok(&outs[slot + 1], 0)),
            &tok(&outs[0], 0),
        );
        let cls_r = cosine(&tok(&outs[slot + 1], 0), &tok(&outs[0], 0));
        let (mut pl, mut pr) = (0.0f64, 0.0f64);
        for t in first_patch..seq {
            let h = tok(&outs[slot + 1], t);
            let f = tok(&outs[0], t);
            pl += cosine(&js[slot].transport(&h), &f) as f64;
            pr += cosine(&h, &f) as f64;
        }
        let (pl, pr) = (pl / n_patch as f64, pr / n_patch as f64);
        println!(
            "{:<7} {:<18} {:<18.4} {:<18.4} {:+.4}",
            layer,
            format!("{cls_l:.4} / {cls_r:.4}"),
            pl,
            pr,
            pl - pr
        );
    }
    println!("{}", "─".repeat(84));
    Ok(())
}
