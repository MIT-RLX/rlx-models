//! Where a Qwen2.5-VL looks at the image, layer by layer.
//!
//! This is the *attention* question, and it is separate from what the lens
//! answers. The lens says what a position is disposed to make the model **say**;
//! attention says which positions the model **reads** when it says it. Both
//! change with depth, and they do not have to change together.
//!
//! The measurement: take one query position — by default the last, the one the
//! answer is generated from — and score it against every key with the model's
//! own post-RoPE Q and GQA-expanded K, per head, softmax over the full causal
//! row. The probability mass landing on image keys is the visual attention, and
//! its shape over the patch grid is where in the picture it landed.
//!
//! Reported per layer:
//!
//! * `img%`   — share of the query's attention on image keys, mean over heads.
//!              This is "how much is it looking at the picture at all".
//! * `spread` — entropy of the image-restricted distribution as a share of its
//!              maximum. 100% is uniform over every patch, low is a spotlight.
//! * `peak`   — the single patch with the most attention, in grid coordinates.
//! * `top⅓`   — share of the visual attention in the top third of the frame.
//!
//! One forward, no fit, so every layer is affordable rather than a sampled few.
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen25-vl,metal --release --example vl_focus -- \
//!     --device metal --max-side 448 --show-every 6 \
//!     --mmproj .../mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
//!     --image crates/rlx-locateanything/fixtures/sample.jpg
//! ```

#![cfg(feature = "qwen25-vl")]

use anyhow::{Context, Result, bail};

const RAMP: [char; 8] = [' ', '·', ':', '-', '=', '+', '*', '#'];

fn softmax(v: &mut [f32]) {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    let inv = 1.0 / s.max(1e-30);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn lo_of(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::INFINITY, f32::min)
}
fn hi_of(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn main() -> Result<()> {
    let mut dir = "/Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct".to_string();
    let mut device = rlx_runtime::Device::Cpu;
    let mut image = "crates/rlx-locateanything/fixtures/sample.jpg".to_string();
    let mut prompt = "Describe the image.".to_string();
    let mut max_side = 448usize;
    let mut mmproj: Option<String> = None;
    let mut show_every = 6usize;
    // Which position does the looking. Default = the last, i.e. the query the
    // answer is actually generated from.
    let mut query: Option<usize> = None;
    let mut out: Option<String> = None;
    // Bilinear between patch centres by default; --patches keeps the lattice.
    let mut upsample = rlx_jlens::heatmap::Upsample::Bilinear;

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
            "--max-side" => {
                max_side = need(i)?.parse()?;
                i += 2;
            }
            "--mmproj" => {
                mmproj = Some(need(i)?);
                i += 2;
            }
            "--show-every" => {
                show_every = need(i)?.parse::<usize>()?.max(1);
                i += 2;
            }
            "--query" => {
                query = Some(need(i)?.parse()?);
                i += 2;
            }
            // Heatmap overlays: one PNG per layer plus a contact sheet.
            "--out" => {
                out = Some(need(i)?);
                i += 2;
            }
            "--patches" => {
                upsample = rlx_jlens::heatmap::Upsample::Nearest;
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
    let ids = prefill.seq.clone();
    if nv != gx * gy {
        bail!("{nv} vision tokens but a {gx}x{gy} grid");
    }
    let qi = query.unwrap_or(seq - 1);
    if qi >= seq {
        bail!("query position {qi} is past the end of a {seq}-position prompt");
    }
    eprintln!(
        "{seq} positions, image {gx}x{gy} at {v0}..{}, query {qi} = {:?}",
        v0 + nv,
        tokenizer.decode(&[ids[qi]], false).unwrap_or_default()
    );

    // The probe prefill is the one that exports Q/K.
    runner.prefill_from_assembled_probe(prefill)?;
    let (q_layers, k_layers) = runner
        .last_prefill_qk()
        .context("prefill exported no Q/K — is `export_aif_qk` set?")?;
    let cfg = runner.lm_config().lm.clone();
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim;
    let stride = q_layers[0].len() / seq;
    let scale = (dh as f32).sqrt().recip();
    eprintln!("{} layers of Q/K, {nh} heads x {dh}", q_layers.len());

    // Attention from `qi` to every key, per layer, averaged over heads.
    let mut per_layer: Vec<Vec<f32>> = Vec::with_capacity(q_layers.len());
    for (q, k) in q_layers.iter().zip(k_layers.iter()) {
        let mut acc = vec![0f32; qi + 1];
        for head in 0..nh {
            let qoff = qi * stride + head * dh;
            let qh = &q[qoff..qoff + dh];
            let mut scores = vec![0f32; qi + 1];
            for (j, s) in scores.iter_mut().enumerate() {
                let koff = j * stride + head * dh;
                *s = qh
                    .iter()
                    .zip(&k[koff..koff + dh])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            softmax(&mut scores);
            for (a, s) in acc.iter_mut().zip(&scores) {
                *a += s / nh as f32;
            }
        }
        per_layer.push(acc);
    }

    println!("\n══ attention from position {qi} onto the image, by layer");
    println!("   img%   = share of this query's attention landing on image keys");
    println!("   spread = entropy of the image-restricted map / its maximum (100% = uniform)");
    println!("   top⅓   = share of the visual attention in the top third of the frame\n");
    println!(
        "{:<7}{:>8}{:>9}{:>12}{:>8}",
        "layer", "img%", "spread", "peak(x,y)", "top⅓"
    );
    println!("{}", "─".repeat(44));

    let mut grids: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut all: Vec<(usize, Vec<f32>, f32)> = Vec::new();
    for (layer, att) in per_layer.iter().enumerate() {
        let vis: Vec<f32> = (0..nv)
            .map(|p| att.get(v0 + p).copied().unwrap_or(0.0))
            .collect();
        let mass: f32 = vis.iter().sum();
        // Normalized within the image, so `spread` and `peak` describe where it
        // looks *given* that it is looking at the picture, independent of how
        // much of its budget went there at all.
        let inv = 1.0 / mass.max(1e-30);
        let norm: Vec<f32> = vis.iter().map(|v| v * inv).collect();
        let ent: f32 = -norm
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f32>();
        let spread = ent / (nv as f32).log2();
        let peak = (0..nv)
            .max_by(|&a, &b| norm[a].partial_cmp(&norm[b]).unwrap())
            .unwrap_or(0);
        let top: f32 = (0..nv).filter(|p| (p / gx) * 3 < gy).map(|p| norm[p]).sum();
        println!(
            "{:<7}{:>7.1}%{:>8.0}%{:>12}{:>7.0}%",
            layer,
            mass * 100.0,
            spread * 100.0,
            format!("({},{})", peak % gx, peak / gx),
            top * 100.0
        );
        if layer % show_every == 0 || layer + 1 == per_layer.len() {
            grids.push((layer, norm.clone()));
        }
        all.push((layer, norm, mass));
    }

    println!("\n══ where on the image, at every {show_every}th layer");
    println!(
        "   each map is scaled to its own peak; ramp {:?} = low..high\n",
        RAMP
    );
    for chunk in grids.chunks(6) {
        for (layer, _) in chunk {
            print!("L{:<3}{}", layer, " ".repeat(gx.saturating_sub(4) + 2));
        }
        println!();
        for y in 0..gy {
            for (_, g) in chunk {
                let hi = g.iter().copied().fold(0f32, f32::max).max(1e-30);
                for x in 0..gx {
                    let t = (g[y * gx + x] / hi).clamp(0.0, 1.0);
                    let idx = (t * (RAMP.len() - 1) as f32).round() as usize;
                    print!("{}", RAMP[idx.min(RAMP.len() - 1)]);
                }
                print!("  ");
            }
            println!();
        }
        println!();
    }

    if let Some(dir_out) = &out {
        // The photo the model actually saw, at the resolution it saw it, so a
        // patch block lines up with the pixels that produced it.
        std::fs::create_dir_all(dir_out)?;
        // Per-tile scaling is right for one figure and wrong for a set: a 2%
        // layer and a 22% layer would both saturate, so the sheet would show
        // sharp early-layer focus that does not exist. Scale the sheet by the
        // *unnormalised* attention across every layer so darkness is comparable.
        // A percentile, not the maximum. The single hottest sink patch is many
        // times anything else, so scaling to it renders every other tile blank —
        // trading one misleading figure for another.
        let sheet_hi = {
            let mut v: Vec<f32> = all
                .iter()
                .flat_map(|(_, norm, mass)| norm.iter().map(move |x| x * mass))
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[(v.len() as f32 * 0.99) as usize % v.len()]
        };
        let mut tiles = Vec::with_capacity(all.len());
        let mut shared = Vec::with_capacity(all.len());
        for (layer, norm, mass) in &all {
            let absolute: Vec<f32> = norm.iter().map(|v| v * mass).collect();
            shared.push(rlx_jlens::heatmap::overlay_scaled(
                &small, &absolute, gx, gy, 0.0, sheet_hi,
            ));
            let tile = rlx_jlens::heatmap::overlay_with(
                &small,
                norm,
                gx,
                gy,
                lo_of(norm),
                hi_of(norm),
                upsample,
            );
            // The absolute scale is in the name: each overlay is normalised to
            // its own peak, so without it a light layer and a heavy one look
            // identical.
            tile.save(format!(
                "{dir_out}/attn_L{layer:02}_img{:.0}pct.png",
                mass * 100.0
            ))?;
            tiles.push(tile);
        }
        // Two sheets on purpose: one comparable across layers, one where each
        // layer is stretched to its own range so a faint layer's *shape* is
        // still legible. Neither alone tells the truth.
        if let Some(sheet) = rlx_jlens::heatmap::contact_sheet(&shared, 6) {
            sheet.save(format!("{dir_out}/attn_contact_sheet.png"))?;
        }
        if let Some(sheet) = rlx_jlens::heatmap::contact_sheet(&tiles, 6) {
            sheet.save(format!("{dir_out}/attn_contact_sheet_per_layer_scale.png"))?;
        }
        println!(
            "\n{} per-layer heatmaps + attn_contact_sheet.png written to {dir_out}/",
            tiles.len()
        );
        println!("each overlay is scaled to its own peak; the filename carries the absolute img%");
    }
    Ok(())
}
