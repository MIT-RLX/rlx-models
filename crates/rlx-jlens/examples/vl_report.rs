//! Everything the lens and the attention probe can say about one image, in one run.
//!
//! Produces, from a single model load and a single fit:
//!
//! 1. **Attention by layer** — where the answer position looks, all 36 layers.
//! 2. **Word masks by layer** — where each probe word is supported, per fitted layer.
//! 3. **Segmentation** — which probe word each patch supports most.
//! 4. **Word readout by layer** — the transported top-1 token at every *text*
//!    position, which is the Jacobian lens proper.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen25-vl,metal --release --example vl_report -- \
//!     --device metal --out /tmp/report \
//!     --mmproj .../mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
//!     --image crates/rlx-locateanything/fixtures/sample.jpg
//! ```
//!
//! Runtime is dominated by the fit (`d_model / dim_batch` backward passes);
//! attention is one forward and effectively free, so it covers every layer while
//! the lens covers every `--every`-th.

#![cfg(feature = "qwen25-vl")]

use anyhow::{Context, Result, bail};
use rlx_jlens::heatmap::{Upsample, contact_sheet, overlay_with};
use rlx_jlens::models::qwen25_vl::Qwen25VlLensModel;
use rlx_jlens::{FitConfig, LensModel, Readout, StackLens};
use rlx_runtime::{Device, Session};

const RAMP: [char; 8] = [' ', '·', ':', '-', '=', '+', '*', '#'];

fn softmax(v: &mut [f32]) {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    let inv = 1.0 / s.max(1e-30);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn zscore(v: &[f32]) -> Vec<f32> {
    let n = v.len().max(1) as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let sd = (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-9);
    v.iter().map(|x| ((*x as f64 - mean) / sd) as f32).collect()
}

fn hi_of(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn ascii(mask: &[f32], gx: usize, gy: usize, lo: f32, hi: f32) -> Vec<String> {
    let span = (hi - lo).max(1e-30);
    (0..gy)
        .map(|y| {
            (0..gx)
                .map(|x| {
                    let t = ((mask[y * gx + x] - lo) / span).clamp(0.0, 1.0);
                    RAMP[(t * (RAMP.len() - 1) as f32).round() as usize]
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<()> {
    let mut dir = "/Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct".to_string();
    let mut device = Device::Cpu;
    let mut image = "crates/rlx-locateanything/fixtures/sample.jpg".to_string();
    let mut prompt = "Describe the image.".to_string();
    let mut mmproj: Option<String> = None;
    let mut out = "jlens-report".to_string();
    let mut every = 6usize;
    let mut dim_batch = 4usize;
    let mut max_side = 448usize;
    let mut words = " crowd, man, hand, suit".to_string();

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
            "--image" => {
                image = need(i)?;
                i += 2;
            }
            "--prompt" => {
                prompt = need(i)?;
                i += 2;
            }
            "--mmproj" => {
                mmproj = Some(need(i)?);
                i += 2;
            }
            "--out" => {
                out = need(i)?;
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
            "--max-side" => {
                max_side = need(i)?.parse()?;
                i += 2;
            }
            "--words" => {
                words = need(i)?;
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }
    std::fs::create_dir_all(&out)?;

    // ── 1. vision + prompt assembly ──
    let mmproj_path = match mmproj {
        Some(p) => std::path::PathBuf::from(p),
        None => std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf"))
            })
            .context("no mmproj-*.gguf beside the weights; pass --mmproj")?,
    };
    eprintln!("[1/4] loading {dir}");
    let mut runner = rlx_qwen25_vl::runner::Qwen25VlRunner::builder()
        .weights(&dir)
        .hf_config(std::path::Path::new(&dir).join("config.json"))
        .mmproj(&mmproj_path)
        .device(device)
        .build()?;

    let img = image::open(&image)
        .with_context(|| format!("opening {image}"))?
        .to_rgb8();
    let (w0, h0) = img.dimensions();
    let scale = (max_side as f32 / w0.max(h0) as f32).min(1.0);
    let (w, h) = (
        ((w0 as f32 * scale) as u32).max(28),
        ((h0 as f32 * scale) as u32).max(28),
    );
    let small = image::imageops::resize(&img, w, h, image::imageops::FilterType::CatmullRom);
    let vision = runner.encode_image(small.as_raw(), w as usize, h as usize)?;
    let (gx, gy) = (vision.grid_x, vision.grid_y);

    let tok_path =
        rlx_qwen25_vl::resolve_tokenizer_path(std::path::Path::new(&dir)).context("tokenizer")?;
    let tokenizer = rlx_qwen25_vl::load_tokenizer(&tok_path)?;
    let templated = rlx_qwen25_vl::chat_template::qwen25_vl_chatml(
        &rlx_qwen25_vl::chat_template::user_turn_with_media(&prompt),
        rlx_qwen25_vl::chat_template::DEFAULT_SYSTEM,
    );
    let embed = runner.embed_table()?;
    let n_embd = runner.lm_config().lm.hidden_size;
    let mm = rlx_qwen25_vl::multimodal::MultimodalPrompt {
        prompt: &templated,
        vision: &vision,
    };
    let prefill = mm.assemble(
        |s: &str| rlx_qwen25_vl::encode_prompt(&tokenizer, s),
        &embed,
        n_embd,
        0,
    )?;
    let (seq, v0, nv) = (
        prefill.seq.len(),
        prefill.vision_start_idx,
        prefill.n_vision_tokens,
    );
    let ids = prefill.seq.clone();
    let sections = prefill.mrope_sections.clone();
    eprintln!("      {seq} positions, image {gx}x{gy} = {nv} tokens at {v0}");
    if nv != gx * gy {
        bail!("{nv} vision tokens but a {gx}x{gy} grid");
    }

    // ── 2. attention, every layer (one forward) ──
    eprintln!("[2/4] attention probe");
    let cfg = runner.lm_config().lm.clone();
    runner.prefill_from_assembled_probe(prefill.clone())?;
    let (q_layers, k_layers) = runner
        .last_prefill_qk()
        .context("prefill exported no Q/K")?;
    let (nh, dh) = (cfg.num_attention_heads, cfg.head_dim);
    let stride = q_layers[0].len() / seq;
    let qk_scale = (dh as f32).sqrt().recip();
    let qi = seq - 1;

    let mut attn: Vec<(Vec<f32>, f32)> = Vec::with_capacity(q_layers.len());
    for (q, k) in q_layers.iter().zip(k_layers.iter()) {
        let mut acc = vec![0f32; qi + 1];
        for head in 0..nh {
            let qh = &q[qi * stride + head * dh..qi * stride + head * dh + dh];
            let mut sc: Vec<f32> = (0..=qi)
                .map(|j| {
                    let ko = j * stride + head * dh;
                    qh.iter()
                        .zip(&k[ko..ko + dh])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * qk_scale
                })
                .collect();
            softmax(&mut sc);
            for (a, s) in acc.iter_mut().zip(&sc) {
                *a += s / nh as f32;
            }
        }
        let vis: Vec<f32> = (0..nv).map(|p| acc[v0 + p]).collect();
        let mass: f32 = vis.iter().sum();
        attn.push((vis, mass));
    }

    // The runner holds a full f32 copy of the LM. Drop it before the fit builds
    // its own two arenas, or three copies of a 3B model are live at once and the
    // OS kills the process with no diagnostic.
    drop(runner);

    let sheet_hi = {
        let mut v: Vec<f32> = attn.iter().flat_map(|(m, _)| m.iter().copied()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(v.len() as f32 * 0.99) as usize % v.len()]
    };
    let tiles: Vec<_> = attn
        .iter()
        .map(|(m, _)| overlay_with(&small, m, gx, gy, 0.0, sheet_hi, Upsample::Bilinear))
        .collect();
    if let Some(s) = contact_sheet(&tiles, 6) {
        s.save(format!("{out}/1_attention_by_layer.png"))?;
    }
    for (l, tile) in tiles.iter().enumerate() {
        tile.save(format!("{out}/attention_L{l:02}.png"))?;
    }

    // ── 3. fit the lens ──
    let model = Qwen25VlLensModel::open(&dir)?.with_sections(sections);
    let target = model.n_layers() - 1;
    let layers: Vec<usize> = (0..target).step_by(every).collect();
    let d = model.d_model();
    eprintln!(
        "[3/4] fitting J for {layers:?} ({} passes each)",
        d.div_ceil(dim_batch)
    );
    let t0 = std::time::Instant::now();
    let js = {
        let mut lens = StackLens::new(
            &model,
            &layers,
            target,
            seq,
            FitConfig {
                dim_batch,
                skip_first: 0,
                device,
            },
        )?;
        let batched: Vec<f32> = prefill.hidden.repeat(dim_batch);
        let js = lens.jacobians(&batched)?;
        eprintln!("      fitted in {:.0}s", t0.elapsed().as_secs_f64());
        js
    };

    // ── 4. read out ──
    eprintln!("[4/4] readout");
    let stack = model.stack(&layers, target, 1, seq)?;
    let mut fwd = Session::new(device).compile(stack.tapped.graph().clone());
    for (name, data) in &stack.params {
        fwd.set_param(name, data);
    }
    let mut feed: Vec<(&str, &[f32])> = vec![(stack.token_input.as_str(), &prefill.hidden[..])];
    for (name, data) in &stack.extra_feeds {
        feed.push((name.as_str(), data.as_slice()));
    }
    let outs = fwd.run(&feed);

    let mut readout = Readout::new(&model, seq, device)?;
    let vocab = readout.vocab();
    let dec = |t: u32| {
        tokenizer
            .decode(&[t], false)
            .unwrap_or_else(|_| format!("<{t}>"))
    };
    let logits: Vec<Vec<f32>> = (0..layers.len())
        .map(|s| readout.logits(&js[s].transport(&outs[s + 1])))
        .collect::<Result<_, _>>()?;
    let final_logits = readout.logits(&outs[0])?;
    let answer = (0..vocab)
        .max_by(|&a, &b| {
            final_logits[qi * vocab + a]
                .partial_cmp(&final_logits[qi * vocab + b])
                .unwrap()
        })
        .context("argmax")? as u32;

    let probes: Vec<(String, u32)> = words
        .split(',')
        .map(|wd| {
            let e = rlx_qwen25_vl::encode_prompt(&tokenizer, wd)?;
            Ok((wd.trim().to_string(), *e.first().context("empty token")?))
        })
        .collect::<Result<_>>()?;

    let mut report = String::new();
    report.push_str(&format!(
        "prompt {prompt:?}\nimage {image} -> {w}x{h}, patch grid {gx}x{gy}\n\
         model's own next token: {:?}\n\n",
        dec(answer)
    ));

    // 4a. attention summary, all layers
    report.push_str("== attention onto the image, by layer ==\n");
    report.push_str(&format!(
        "{:<7}{:>8}{:>12}{:>8}\n",
        "layer", "img%", "peak(x,y)", "top1/3"
    ));
    for (l, (m, mass)) in attn.iter().enumerate() {
        let tot: f32 = m.iter().sum::<f32>().max(1e-30);
        let peak = (0..nv)
            .max_by(|&a, &b| m[a].partial_cmp(&m[b]).unwrap())
            .unwrap_or(0);
        let top: f32 = (0..nv)
            .filter(|p| (p / gx) * 3 < gy)
            .map(|p| m[p])
            .sum::<f32>()
            / tot;
        report.push_str(&format!(
            "{l:<7}{:>7.1}%{:>12}{:>7.0}%\n",
            mass * 100.0,
            format!("({},{})", peak % gx, peak / gx),
            top * 100.0
        ));
    }

    // 4b. the Jacobian lens proper: transported top-1 at every text position
    report.push_str("\n== transported top-1 by layer, text positions ==\n");
    report.push_str(&format!("{:<5}{:<18}", "pos", "prompt token"));
    for l in &layers {
        report.push_str(&format!("{:<15}", format!("L{l}")));
    }
    report.push('\n');
    let trunc = |s: String| {
        let q = format!("{s:?}");
        q.chars().take(14).collect::<String>()
    };
    for pos in 0..seq {
        if pos >= v0 && pos < v0 + nv {
            if pos == v0 {
                report.push_str(&format!(
                    "{:<5}{:<18}(see masks)\n",
                    "…",
                    format!("<{nv} image>")
                ));
            }
            continue;
        }
        report.push_str(&format!("{pos:<5}{:<18}", trunc(dec(ids[pos]))));
        for s in 0..layers.len() {
            let row = &logits[s][pos * vocab..(pos + 1) * vocab];
            let top = (0..vocab)
                .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap())
                .unwrap();
            report.push_str(&format!("{:<15}", trunc(dec(top as u32))));
        }
        report.push('\n');
    }

    // 4c. per-word masks over the image
    report.push_str("\n== word masks over the image, by layer ==\n");
    let mut seg: Vec<Vec<Vec<f32>>> = Vec::with_capacity(layers.len());
    for (wd, id) in &probes {
        let masks: Vec<Vec<f32>> = (0..layers.len())
            .map(|s| {
                zscore(
                    &(0..nv)
                        .map(|p| logits[s][(v0 + p) * vocab + *id as usize])
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let name = wd.replace(|c: char| !c.is_alphanumeric(), "_");
        let tiles: Vec<_> = masks
            .iter()
            .map(|m| overlay_with(&small, m, gx, gy, 0.0, hi_of(m), Upsample::Bilinear))
            .collect();
        if let Some(s) = contact_sheet(&tiles, layers.len()) {
            s.save(format!("{out}/2_mask_{name}_by_layer.png"))?;
        }
        report.push_str(&format!("\n-- {wd:?} (token {id})\n"));
        let rows: Vec<Vec<String>> = masks
            .iter()
            .map(|m| ascii(m, gx, gy, 0.0, hi_of(m)))
            .collect();
        for l in &layers {
            report.push_str(&format!(
                "L{:<3}{}",
                l,
                " ".repeat(gx.saturating_sub(4) + 2)
            ));
        }
        report.push('\n');
        for y in 0..gy {
            for r in &rows {
                report.push_str(&r[y]);
                report.push_str("  ");
            }
            report.push('\n');
        }
        seg.push(masks);
    }

    // 4d. which word wins each patch
    report.push_str("\n== which probe word each patch supports most ==\n   ");
    for (k, (wd, _)) in probes.iter().enumerate() {
        report.push_str(&format!("{}={wd} ", (b'a' + k as u8) as char));
    }
    report.push_str("\n\n");
    for l in &layers {
        report.push_str(&format!(
            "L{:<3}{}",
            l,
            " ".repeat(gx.saturating_sub(4) + 2)
        ));
    }
    report.push('\n');
    for y in 0..gy {
        for s in 0..layers.len() {
            for x in 0..gx {
                let p = y * gx + x;
                let best = (0..probes.len())
                    .max_by(|&a, &b| seg[a][s][p].partial_cmp(&seg[b][s][p]).unwrap())
                    .unwrap_or(0);
                report.push((b'a' + best as u8) as char);
            }
            report.push_str("  ");
        }
        report.push('\n');
    }

    std::fs::write(format!("{out}/report.txt"), &report)?;
    print!("{report}");
    println!("\nwritten to {out}/");
    println!(
        "  1_attention_by_layer.png     all {} layers, shared scale",
        attn.len()
    );
    println!("  2_mask_<word>_by_layer.png   per probe word, fitted layers");
    println!("  report.txt                   this text");
    Ok(())
}
