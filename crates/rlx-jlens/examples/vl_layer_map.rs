//! Every position of a multimodal prompt, read out at every fitted layer.
//!
//! Two panels, from one fit:
//!
//! * **Text positions** — the transported top-1 token per prompt word per
//!   layer. This is the layer-by-layer picture of what each word is disposed to
//!   make the model say.
//! * **Image positions** — a spatial mask over the patch grid. Reading a patch
//!   by `argmax` over the vocabulary is a dead end (see the README): in a causal
//!   LM a position is only trained to predict the *next* position, and the next
//!   position at an image site is another `<|image_pad|>`, so the argmax is
//!   punctuation. What *is* meaningful is the score of a **named** token at each
//!   patch — "how much does this patch support the word `crowd`" — which is a
//!   mask over the image, one per word per layer.
//!
//! Each mask is reported with numbers, not just ASCII, so the claim is
//! checkable: the centroid of the positive mass and the share of that mass in
//! the top third of the frame. `--baseline` prints the same masks without the
//! Jacobian transport, which is the comparison that says whether transport buys
//! anything.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen25-vl,metal --release --example vl_layer_map -- \
//!     --device metal --max-side 448 --every 6 --dim-batch 4 \
//!     --words " crowd, man, hand, suit" \
//!     --mmproj .../mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
//!     --image crates/rlx-locateanything/fixtures/sample.jpg --out /tmp/masks
//! ```

#![cfg(feature = "qwen25-vl")]

use anyhow::{Context, Result, bail};
use rlx_jlens::models::qwen25_vl::Qwen25VlLensModel;
use rlx_jlens::{FitConfig, LensModel, Readout, StackLens};
use rlx_runtime::{Device, Session};

/// Darkest-to-lightest ramp for the ASCII masks.
const RAMP: [char; 8] = [' ', '·', ':', '-', '=', '+', '*', '#'];

/// One patch's standing in its own layer's mask, in units of sigma.
fn zscore(v: &[f32]) -> Vec<f32> {
    let n = v.len().max(1) as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt().max(1e-9);
    v.iter().map(|x| ((*x as f64 - mean) / sd) as f32).collect()
}

/// Where a mask's positive mass sits: `(x̄, ȳ)` in `[0,1]`, plus the share of
/// that mass in the top third of the frame. Positive-only because a z-scored
/// mask is half negative by construction and the negative half is "everywhere
/// this word is not", which has no location.
fn centroid(z: &[f32], gx: usize, gy: usize) -> (f32, f32, f32) {
    let (mut sx, mut sy, mut tot, mut top) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..gy {
        for x in 0..gx {
            let w = z[y * gx + x].max(0.0) as f64;
            sx += w * x as f64;
            sy += w * y as f64;
            tot += w;
            if y * 3 < gy {
                top += w;
            }
        }
    }
    if tot <= 0.0 {
        return (0.5, 0.5, 0.0);
    }
    (
        (sx / tot / (gx - 1).max(1) as f64) as f32,
        (sy / tot / (gy - 1).max(1) as f64) as f32,
        (top / tot) as f32,
    )
}

/// Word masks are z-scores: symmetric about zero and half negative by
/// construction. Rendering `min..max` on a *sequential* ramp puts "average for
/// this word" in the middle of the scale, which floods the frame — a signed
/// quantity on a one-hue ramp is the wrong encoding. These are drawn from `0.0`
/// to `hi_of`, the positive arm only, which is the same choice `centroid`
/// already makes and for the same reason: the negative arm is "everywhere this
/// word is not", and that has no location.
fn hi_of(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn main() -> Result<()> {
    let mut dir = "/Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct".to_string();
    let mut device = Device::Cpu;
    let mut every = 6usize;
    let mut dim_batch = 4usize;
    let mut image = "crates/rlx-locateanything/fixtures/sample.jpg".to_string();
    let mut prompt = "Describe the image.".to_string();
    let mut max_side = 448usize;
    let mut mmproj: Option<String> = None;
    let mut words = " crowd, man, hand, suit".to_string();
    let mut out: Option<String> = None;
    // Bilinear between patch centres by default; --patches keeps the lattice.
    let mut upsample = rlx_jlens::heatmap::Upsample::Bilinear;
    let mut baseline = false;

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
            "--image" => {
                image = need(i)?;
                i += 2;
            }
            "--prompt" => {
                prompt = need(i)?;
                i += 2;
            }
            "--max-side" => {
                max_side = need(i)?.parse()?;
                i += 2;
            }
            "--mmproj" => {
                mmproj = Some(need(i)?);
                i += 2;
            }
            // Comma-separated. Leading spaces matter: " man" and "man" are
            // different tokens, and the one a continuation would use is " man".
            "--words" => {
                words = need(i)?;
                i += 2;
            }
            "--out" => {
                out = Some(need(i)?);
                i += 2;
            }
            "--patches" => {
                upsample = rlx_jlens::heatmap::Upsample::Nearest;
                i += 1;
            }
            // Also print the untransported masks, as the comparison.
            "--baseline" => {
                baseline = true;
                i += 1;
            }
            other => bail!("unknown argument {other}"),
        }
    }

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
    eprintln!("loading {dir}");
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
    eprintln!(
        "image {w0}x{h0} -> {w}x{h}, patch grid {gx}x{gy} = {} tokens",
        vision.n_tokens
    );

    let embed = runner.embed_table()?;
    let n_embd = runner.lm_config().lm.hidden_size;
    let weights_path = std::path::PathBuf::from(&dir);
    let tok_path = rlx_qwen25_vl::resolve_tokenizer_path(&weights_path).context("tokenizer")?;
    let tokenizer = rlx_qwen25_vl::load_tokenizer(&tok_path)?;
    let templated = rlx_qwen25_vl::chat_template::qwen25_vl_chatml(
        &rlx_qwen25_vl::chat_template::user_turn_with_media(&prompt),
        rlx_qwen25_vl::chat_template::DEFAULT_SYSTEM,
    );
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
    let seq = prefill.seq.len();
    let v0 = prefill.vision_start_idx;
    let nv = prefill.n_vision_tokens;
    eprintln!(
        "assembled: {seq} positions, {nv} vision tokens at {v0}..{}",
        v0 + nv
    );
    if nv != gx * gy {
        bail!("{nv} vision tokens but a {gx}x{gy} grid; the mask would be misaligned");
    }

    let model = Qwen25VlLensModel::open(&dir)?.with_sections(prefill.mrope_sections.clone());
    let target = model.n_layers() - 1;
    let layers: Vec<usize> = (0..target).step_by(every).collect();
    let d = model.d_model();
    eprintln!(
        "{} layers, d_model {d}, fitting J for {:?}",
        model.n_layers(),
        layers
    );

    // Fit, then drop: the readout below builds its own copy of the trunk, and
    // two live copies of a 3B parameter set is what the OS kills.
    let start = std::time::Instant::now();
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
        eprintln!(
            "fitted in {:.1}s | {}",
            start.elapsed().as_secs_f64(),
            lens.timing().summary()
        );
        js
    };

    // One forward at batch 1 gives every layer's residual for every position.
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

    // `rows = seq` decodes the whole prompt in one unembed call per layer.
    let mut readout = Readout::new(&model, seq, device)?;
    let vocab = readout.vocab();
    let decode = |t: u32| -> String {
        tokenizer
            .decode(&[t], false)
            .unwrap_or_else(|_| format!("<{t}>"))
    };

    // Probe words, as their first token. Printed so the substitution is visible
    // when a word does not tokenize to one piece.
    let probes: Vec<(String, u32)> = words
        .split(',')
        .map(|w| {
            let ids = rlx_qwen25_vl::encode_prompt(&tokenizer, w)?;
            let id = *ids
                .first()
                .with_context(|| format!("{w:?} tokenizes to nothing"))?;
            Ok((w.to_string(), id))
        })
        .collect::<Result<_>>()?;

    // Logits for every position at every layer, transported and not.
    let mut logits_t: Vec<Vec<f32>> = Vec::with_capacity(layers.len());
    let mut logits_u: Vec<Vec<f32>> = Vec::with_capacity(layers.len());
    for slot in 0..layers.len() {
        let h = &outs[slot + 1];
        logits_t.push(readout.logits(&js[slot].transport(h))?);
        if baseline {
            logits_u.push(readout.logits(h)?);
        }
    }
    // `outs[0]` is already the residual at the target layer, so it is read out
    // directly — transporting it would apply `J` to a vector that is already
    // where `J` transports *to*.
    let final_logits = readout.logits(&outs[0])?;
    let last = seq - 1;
    let answer = (0..vocab)
        .max_by(|&a, &b| {
            final_logits[last * vocab + a]
                .partial_cmp(&final_logits[last * vocab + b])
                .unwrap()
        })
        .context("argmax")? as u32;

    // ── panel A: every text position, every layer ──
    println!("\n══ text positions: transported top-1 by layer");
    print!("{:<5} {:<16}", "pos", "prompt token");
    for l in &layers {
        print!(" {:<13}", format!("L{l}"));
    }
    println!();
    println!("{}", "─".repeat(21 + 14 * layers.len()));
    let trunc = |s: String| -> String {
        let s = format!("{s:?}");
        if s.chars().count() > 12 {
            s.chars().take(12).collect()
        } else {
            s
        }
    };
    for pos in 0..seq {
        if pos >= v0 && pos < v0 + nv {
            if pos == v0 {
                println!("{:<5} {:<16} {}", "…", format!("<{nv} image>"), "(panel B)");
            }
            continue;
        }
        // Positions after the image are the ones that can carry image content:
        // they are the only ones the model is ever asked to speak from.
        let after = if pos > v0 { ">" } else { " " };
        print!("{pos:<4}{after}{:<16}", trunc(decode(prefill.seq[pos])));
        for slot in 0..layers.len() {
            let row = &logits_t[slot][pos * vocab..(pos + 1) * vocab];
            let top = (0..vocab)
                .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap())
                .unwrap();
            print!(" {:<13}", trunc(decode(top as u32)));
        }
        println!();
    }
    println!("\nthe model's own next token is {:?}", decode(answer));

    // ── panel B: the image, as a mask per word per layer ──
    println!("\n══ image patches: per-word mask over the {gx}x{gy} grid, by layer");
    println!(
        "   z-scored across patches within each layer; ramp {:?} = low..high",
        RAMP
    );
    println!("   `top` is the share of positive mass in the top third — chance is 33%");
    for (word, id) in &probes {
        println!("\n── {:?} (token {} = {:?})", word, id, decode(*id));
        let masks: Vec<Vec<f32>> = (0..layers.len())
            .map(|slot| {
                let raw: Vec<f32> = (0..nv)
                    .map(|p| logits_t[slot][(v0 + p) * vocab + *id as usize])
                    .collect();
                zscore(&raw)
            })
            .collect();

        for l in &layers {
            print!("L{:<3}{}", l, " ".repeat(gx.saturating_sub(4) + 2));
        }
        println!();
        for y in 0..gy {
            for m in &masks {
                for x in 0..gx {
                    let z = m[y * gx + x];
                    let idx = (((z + 2.0) / 4.0 * (RAMP.len() - 1) as f32).round() as i32)
                        .clamp(0, RAMP.len() as i32 - 1) as usize;
                    print!("{}", RAMP[idx]);
                }
                print!("  ");
            }
            println!();
        }
        print!("{:<10}", "centroid");
        for m in &masks {
            let (cx, cy, top) = centroid(m, gx, gy);
            print!("x{cx:.2} y{cy:.2} top{top:.0}%  ", top = top * 100.0);
        }
        println!();

        if baseline {
            print!("{:<10}", "no J");
            for slot in 0..layers.len() {
                let raw: Vec<f32> = (0..nv)
                    .map(|p| logits_u[slot][(v0 + p) * vocab + *id as usize])
                    .collect();
                let (cx, cy, top) = centroid(&zscore(&raw), gx, gy);
                print!("x{cx:.2} y{cy:.2} top{top:.0}%  ", top = top * 100.0);
            }
            println!();
        }

        if let Some(dir_out) = &out {
            std::fs::create_dir_all(dir_out)?;
            let name = word.trim().replace(|c: char| !c.is_alphanumeric(), "_");
            let mut tiles = Vec::with_capacity(masks.len());
            for (slot, m) in masks.iter().enumerate() {
                let tile =
                    rlx_jlens::heatmap::overlay_with(&small, m, gx, gy, 0.0, hi_of(m), upsample);
                tile.save(format!("{dir_out}/{name}_L{}.png", layers[slot]))?;
                tiles.push(tile);
            }
            // One strip per word, layers left to right, so depth reads as a
            // sequence instead of a folder of files.
            if let Some(sheet) = rlx_jlens::heatmap::contact_sheet(&tiles, layers.len()) {
                sheet.save(format!("{dir_out}/{name}_by_layer.png"))?;
            }
        }
    }
    // ── panel C: which probe word each patch supports most ──
    //
    // The per-word masks above are one word at a time and hard to read against
    // each other. This is the segmentation view: z-score each word's scores
    // across patches — so a word with globally larger logits cannot win on
    // offset alone — then label every patch with whichever word stands highest
    // there. Compare it against the photo: the answer is legible or it is not.
    println!("\n══ image patches: which probe word each patch supports most");
    print!("   ");
    for (k, (word, _)) in probes.iter().enumerate() {
        print!("{}={} ", (b'a' + k as u8) as char, word.trim());
    }
    println!("\n");
    let zs_per_layer: Vec<Vec<Vec<f32>>> = (0..layers.len())
        .map(|slot| {
            probes
                .iter()
                .map(|(_, id)| {
                    let raw: Vec<f32> = (0..nv)
                        .map(|p| logits_t[slot][(v0 + p) * vocab + *id as usize])
                        .collect();
                    zscore(&raw)
                })
                .collect()
        })
        .collect();
    for l in &layers {
        print!("L{:<3}{}", l, " ".repeat(gx.saturating_sub(4) + 2));
    }
    println!();
    for y in 0..gy {
        for zs in &zs_per_layer {
            for x in 0..gx {
                let p = y * gx + x;
                let best = (0..probes.len())
                    .max_by(|&a, &b| zs[a][p].partial_cmp(&zs[b][p]).unwrap())
                    .unwrap_or(0);
                print!("{}", (b'a' + best as u8) as char);
            }
            print!("  ");
        }
        println!();
    }
    // Share of patches each word claims, per layer. A word that claims nearly
    // everything is not segmenting, it is just the highest-variance direction.
    for (k, (word, _)) in probes.iter().enumerate() {
        print!("{:<10}", word.trim());
        for zs in &zs_per_layer {
            let n = (0..nv)
                .filter(|&p| {
                    (0..probes.len())
                        .max_by(|&a, &b| zs[a][p].partial_cmp(&zs[b][p]).unwrap())
                        .unwrap_or(0)
                        == k
                })
                .count();
            print!("{:<18}", format!("{:.0}%", 100.0 * n as f32 / nv as f32));
        }
        println!();
    }

    if let Some(d) = &out {
        println!("\nmask overlays written to {d}/");
    }
    Ok(())
}
