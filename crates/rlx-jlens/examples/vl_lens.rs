//! The Jacobian lens on a vision-language model: image patches → words.
//!
//! A Qwen-VL projects image patches to the LM's hidden width and splices them
//! into the token sequence, so image content travels the **LM's** residual
//! stream as ordinary positions. That means the lens applies unchanged, and the
//! readout is the LM's own vocabulary — not a caption list anyone picked.
//!
//! So the text question becomes the image question verbatim:
//!
//! > at which layer does *this image patch* become disposed to make the model
//! > say "dog"?
//!
//! ```bash
//! cargo run -p rlx-jlens --features qwen25-vl,metal --release --example vl_lens -- \
//!     --device metal --weights /Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct \
//!     --image crates/rlx-locateanything/fixtures/sample.jpg --every 6
//! ```

#![cfg(feature = "qwen25-vl")]

use anyhow::{Context, Result, bail};
use rlx_jlens::models::qwen25_vl::Qwen25VlLensModel;
use rlx_jlens::{FitConfig, LensModel, Readout, StackLens, rank_of};
use rlx_runtime::{Device, Session};

fn main() -> Result<()> {
    let mut dir = "/Volumes/FOUR/weights/vision/Qwen2.5-VL-3B-Instruct".to_string();
    let mut device = Device::Cpu;
    let mut every = 6usize;
    let mut dim_batch = 8usize;
    let mut image = "crates/rlx-locateanything/fixtures/sample.jpg".to_string();
    let mut prompt = "Describe the image.".to_string();
    let mut max_side = 224usize;
    let mut mmproj: Option<String> = None;
    let mut no_fit = false;
    let mut text_only = false;
    let mut reference: Option<String> = None;

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
            // Forward + untransported readout only: verifies vision -> LM ->
            // vocabulary without the `d_model / dim_batch` backward passes.
            "--no-fit" => {
                no_fit = true;
                i += 1;
            }
            // Skip the image entirely: if the LM path is sound, a text-only
            // prompt must read out sensibly. That separates "the LM/mRoPE/
            // tokenizer plumbing is wrong" from "the vision embeddings are".
            "--text-only" => {
                text_only = true;
                no_fit = true;
                i += 1;
            }
            // Reference dump from `scripts/qwen25vl_reference_mm.py`.
            "--ref" => {
                reference = Some(need(i)?);
                i += 2;
            }
            other => bail!("unknown argument {other}"),
        }
    }

    // ── vision tower + multimodal assembly, via the model's own runner ──
    eprintln!("loading {dir}");
    // The vision tower loads from a GGUF mmproj; the LM stays safetensors, which
    // is what the lens taps. Same model either way.
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
    let mut runner = rlx_qwen25_vl::runner::Qwen25VlRunner::builder()
        .weights(&dir)
        // Safetensors carry no embedded config, unlike GGUF.
        .hf_config(std::path::Path::new(&dir).join("config.json"))
        .mmproj(&mmproj_path)
        .device(device)
        .build()?;

    let img = image::open(&image)
        .with_context(|| format!("opening {image}"))?
        .to_rgb8();
    let (w0, h0) = img.dimensions();
    // Keep the patch grid small: every image patch is a sequence position, and
    // the fit costs `d_model` backward passes over the whole sequence.
    let scale = (max_side as f32 / w0.max(h0) as f32).min(1.0);
    let (w, h) = (
        ((w0 as f32 * scale) as u32).max(28),
        ((h0 as f32 * scale) as u32).max(28),
    );
    let small = image::imageops::resize(&img, w, h, image::imageops::FilterType::CatmullRom);
    eprintln!("image {image} {w0}x{h0} -> {w}x{h}");

    // No `--gen` control here on purpose. `Qwen25VlRunner::build` hardcodes
    // `let lm = None`, so `generate_text`/`predict_logits` fail with "requires
    // .weights(...) LM GGUF" even when handed one — the crate's own text path
    // is unreachable, and the ModelFlow path the CLI uses is what this example
    // already exercises. Verify against `--ref` instead.

    let vision = runner.encode_image(small.as_raw(), w as usize, h as usize)?;
    eprintln!(
        "vision tower: grid {}x{} = {} tokens, {} floats ({} per token)",
        vision.grid_x,
        vision.grid_y,
        vision.n_tokens,
        vision.embeddings.len(),
        vision.embeddings.len() / vision.n_tokens.max(1),
    );
    if text_only {
        // Embed the prompt directly from the token table — no image, no
        // `<|image_pad|>` span, plain 1-D positions.
        let embed_tbl = runner.embed_table()?;
        let n_embd_t = runner.lm_config().lm.hidden_size;
        let wp = std::path::PathBuf::from(&dir);
        let tp = rlx_qwen25_vl::resolve_tokenizer_path(&wp).context("tokenizer")?;
        let tk = rlx_qwen25_vl::load_tokenizer(&tp)?;
        let templated =
            rlx_qwen25_vl::chat_template::qwen25_vl_chatml(&prompt, "You are a helpful assistant.");
        let ids = rlx_qwen25_vl::encode_prompt(&tk, &templated)?;
        let seq_t = ids.len();
        let mut hidden = vec![0f32; seq_t * n_embd_t];
        for (t, &id) in ids.iter().enumerate() {
            let src = id as usize * n_embd_t;
            hidden[t * n_embd_t..(t + 1) * n_embd_t]
                .copy_from_slice(&embed_tbl[src..src + n_embd_t]);
        }
        let sections: Vec<[usize; 4]> = (0..seq_t).map(|t| [t, t, t, 0]).collect();
        let model = Qwen25VlLensModel::open(&dir)?.with_sections(sections);
        let target = model.n_layers() - 1;
        let layers: Vec<usize> = (0..target).step_by(every).collect();
        let stack = model.stack(&layers, target, 1, seq_t)?;
        let mut fwd = Session::new(device).compile(stack.tapped.graph().clone());
        for (name, data) in &stack.params {
            fwd.set_param(name, data);
        }
        let mut feed: Vec<(&str, &[f32])> = vec![(stack.token_input.as_str(), &hidden[..])];
        for (name, data) in &stack.extra_feeds {
            feed.push((name.as_str(), data.as_slice()));
        }
        let outs = fwd.run(&feed);
        let d = model.d_model();
        let mut readout = Readout::new(&model, 1, device)?;
        let dec = |t: u32| tk.decode(&[t], false).unwrap_or_else(|_| format!("<{t}>"));
        let last = seq_t - 1;
        let fin = readout.logits(&outs[0][last * d..(last + 1) * d])?;
        let top = (0..fin.len())
            .max_by(|&a, &b| fin[a].partial_cmp(&fin[b]).unwrap())
            .unwrap() as u32;
        println!("\ntext-only prompt: {prompt:?} ({seq_t} tokens)");
        println!("model's own next token: {:?}", dec(top));
        let mut idx: Vec<usize> = (0..fin.len()).collect();
        idx.sort_by(|&a, &b| fin[b].partial_cmp(&fin[a]).unwrap());
        let top5: Vec<String> = idx
            .iter()
            .take(5)
            .map(|&t| format!("{:?}", dec(t as u32)))
            .collect();
        println!("top-5: {}", top5.join(" "));
        return Ok(());
    }
    let embed = runner.embed_table()?;
    let n_embd = runner.lm_config().lm.hidden_size;
    let weights_path = std::path::PathBuf::from(&dir);
    let tok_path = rlx_qwen25_vl::resolve_tokenizer_path(&weights_path)
        .context("no tokenizer.json beside the weights")?;
    let tokenizer = rlx_qwen25_vl::load_tokenizer(&tok_path)?;
    // The chat template is what puts `<|image_pad|>` where the vision tokens go.
    // `user_turn_with_media` is only the *inside* of the user turn — without the
    // ChatML frame around it the trunk never sees `<|im_start|>assistant`, and
    // answers every image with `<|im_end|>`.
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
    eprintln!(
        "assembled: {seq} positions, {} vision tokens starting at {}",
        prefill.n_vision_tokens, prefill.vision_start_idx
    );

    // ── the lens over the LM trunk ──
    let model = Qwen25VlLensModel::open(&dir)?
        .with_sections(prefill.mrope_sections.clone())
        .with_name("qwen25-vl");
    let target = model.n_layers() - 1;
    let layers: Vec<usize> = (0..target).step_by(every).collect();
    eprintln!(
        "{} LM layers, d_model {}, fitting J for {} of them …",
        model.n_layers(),
        model.d_model(),
        layers.len()
    );

    if no_fit {
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
        let d = model.d_model();
        let mut readout = Readout::new(&model, 1, device)?;
        let tok_path = rlx_qwen25_vl::resolve_tokenizer_path(&weights_path).context("tokenizer")?;
        let tk = rlx_qwen25_vl::load_tokenizer(&tok_path)?;
        let dec = |t: u32| tk.decode(&[t], false).unwrap_or_else(|_| format!("<{t}>"));
        let row = |o: &[f32], t: usize| -> Vec<f32> { o[t * d..(t + 1) * d].to_vec() };
        let top = |l: &[f32]| -> u32 {
            (0..l.len())
                .max_by(|&a, &b| l[a].partial_cmp(&l[b]).unwrap())
                .unwrap_or(0) as u32
        };
        let read_pos = seq - 1;
        let fin = readout.logits(&row(&outs[0], read_pos))?;
        println!("\nmodel's own next token: {:?}", dec(top(&fin)));
        // Against `scripts/qwen25vl_reference_mm.py`, when it has been run. The
        // whole multimodal path is in scope here: preprocessing, the vision
        // tower, the splice, the mRoPE sections and the trunk.
        if let Some(path) = &reference {
            let refs = rlx_core::load_weight_map(std::path::Path::new(path), &[])?;
            let want = refs
                .get("last_logits")
                .map(|t| t.0.to_vec())
                .context("last_logits missing from the reference dump")?;
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for (a, b) in fin.iter().zip(want.iter()) {
                dot += *a as f64 * *b as f64;
                na += (*a as f64).powi(2);
                nb += (*b as f64).powi(2);
            }
            println!(
                "reference logits cosine {:.6}",
                dot / (na.sqrt() * nb.sqrt() + 1e-30)
            );
            println!("reference next token   {:?}", dec(top(&want)));
        }
        let v0 = prefill.vision_start_idx;
        let vmid = v0 + prefill.n_vision_tokens / 2;
        println!("\nuntransported readout (logit-lens analogue), by layer:");
        println!(
            "{:<7} {:<20} {:<20} {:<20}",
            "layer", "last text", "image patch 0", "image patch mid"
        );
        for (slot, &layer) in layers.iter().enumerate() {
            let a = readout.logits(&row(&outs[slot + 1], read_pos))?;
            let b = readout.logits(&row(&outs[slot + 1], v0))?;
            let c = readout.logits(&row(&outs[slot + 1], vmid))?;
            println!(
                "{:<7} {:<20} {:<20} {:<20}",
                layer,
                format!("{:?}", dec(top(&a))),
                format!("{:?}", dec(top(&b))),
                format!("{:?}", dec(top(&c)))
            );
        }
        return Ok(());
    }

    let start = std::time::Instant::now();
    // Scoped so the fitted `StackLens` — which holds the trunk's parameters and
    // both halves of the split backward — is dropped before the forward-only
    // graph below builds a *second* copy of the same 3B parameter set. Keeping
    // both alive is what the OS kills, and it kills the process outright rather
    // than raising, so it reads as "the fit silently stopped after printing its
    // timings". Only `js` needs to survive, and that is 6 x [d, d].
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

    // Residuals at batch 1, plus the model's own final distribution.
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

    let d = model.d_model();
    let mut readout = Readout::new(&model, 1, device)?;
    let row = |o: &[f32], t: usize| -> Vec<f32> { o[t * d..(t + 1) * d].to_vec() };
    let decode = |t: u32| -> String {
        tokenizer
            .decode(&[t], false)
            .unwrap_or_else(|_| format!("<{t}>"))
    };

    let read_pos = seq - 1;
    let final_logits = readout.logits(&row(&outs[0], read_pos))?;
    let answer = (0..final_logits.len())
        .max_by(|&a, &b| final_logits[a].partial_cmp(&final_logits[b]).unwrap())
        .context("argmax")? as u32;
    println!("\nmodel's own next token: {:?}\n", decode(answer));

    // Text position: the standard readout. Vision positions: the same readout
    // applied where an image patch sits, which is the whole point.
    let v0 = prefill.vision_start_idx;
    let vmid = v0 + prefill.n_vision_tokens / 2;
    println!(
        "{:<7} {:<22} {:<10} {:<22} {:<22}",
        "layer", "last text pos", "rank", "first image patch", "middle image patch"
    );
    println!("{}", "─".repeat(88));
    let top_of = |l: &[f32]| -> u32 {
        (0..l.len())
            .max_by(|&a, &b| l[a].partial_cmp(&l[b]).unwrap())
            .unwrap_or(0) as u32
    };
    for (slot, &layer) in layers.iter().enumerate() {
        let ltext = readout.logits(&js[slot].transport(&row(&outs[slot + 1], read_pos)))?;
        let text_top = format!("{:?}", decode(top_of(&ltext)));
        let rank = rank_of(&ltext, answer);
        let lv0 = readout.logits(&js[slot].transport(&row(&outs[slot + 1], v0)))?;
        let v0_top = format!("{:?}", decode(top_of(&lv0)));
        let lvm = readout.logits(&js[slot].transport(&row(&outs[slot + 1], vmid)))?;
        let vm_top = format!("{:?}", decode(top_of(&lvm)));
        println!("{layer:<7} {text_top:<22} {rank:<10} {v0_top:<22} {vm_top:<22}");
    }
    println!("{}", "─".repeat(88));
    Ok(())
}
