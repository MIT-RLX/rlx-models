// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// RLX CLI for Qwen3.5 / Qwen3.6
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{WeightsResolveCli, parse_qwen35_device, resolve_weights_cli};
use rlx_qwen3::SampleOpts;
use std::path::PathBuf;

fn parse_prompt_id_rows(raw: &str) -> Result<Vec<Vec<u32>>> {
    raw.split(';')
        .map(|row| {
            row.split(',')
                .map(|s| s.trim().parse::<u32>())
                .collect::<std::result::Result<_, _>>()
                .context("prompt row")
        })
        .collect()
}

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt_ids: Vec<u32> = vec![1, 2, 3];
    let mut prompt_rows: Vec<Vec<u32>> = vec![vec![1, 2, 3]];
    let mut prompt_text: Option<String> = None;
    let mut system_text: Option<String> = None;
    let mut messages_json: Option<String> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut max_seq = 0usize;
    let mut max_tokens = 0usize;
    let mut enable_mtp = false;
    let mut spec_decode = false;
    let mut spec_n = 4usize;
    let mut packed_weights = false;
    let mut batch = 1usize;
    let mut temperature = 0f32;
    let mut top_p = 1f32;
    let mut fast_mtp = false;
    let mut seed = 0u64;
    let mut aot_cache: Option<PathBuf> = None;
    let mut dynamic_prefill = false;
    let mut dynamic_decode = false;
    let mut mmproj: Option<PathBuf> = None;
    let mut image: Option<PathBuf> = None;
    let mut resolve_cli = WeightsResolveCli::default();

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => weights = Some(PathBuf::from(it.next().context("--weights")?)),
            "--prefer-quant" | "--prefer" | "-p" => {
                resolve_cli.prefer_gguf = Some(it.next().context("--prefer-quant")?.clone());
            }
            "--gguf-index" => {
                resolve_cli.gguf_index = Some(
                    it.next()
                        .context("--gguf-index")?
                        .parse()
                        .context("usize")?,
                );
            }
            "--device" => device = it.next().context("--device")?.clone(),
            "--max-seq" => max_seq = it.next().context("--max-seq")?.parse()?,
            "--max-tokens" => max_tokens = it.next().context("--max-tokens")?.parse()?,
            "--batch" => batch = it.next().context("--batch")?.parse()?,
            "--mtp" => enable_mtp = true,
            "--fast-mtp" => fast_mtp = true,
            "--spec-decode" => spec_decode = true,
            "--spec-n" => spec_n = it.next().context("--spec-n")?.parse()?,
            "--packed" => packed_weights = true,
            "--dynamic-prefill" => dynamic_prefill = true,
            "--dynamic-decode" => dynamic_decode = true,
            "--temperature" => {
                temperature = it.next().context("--temperature")?.parse()?;
            }
            "--top-p" => top_p = it.next().context("--top-p")?.parse()?,
            "--seed" => seed = it.next().context("--seed")?.parse()?,
            "--aot-cache" => aot_cache = Some(PathBuf::from(it.next().context("--aot-cache")?)),
            "--mmproj" => mmproj = Some(PathBuf::from(it.next().context("--mmproj")?)),
            "--image" => image = Some(PathBuf::from(it.next().context("--image")?)),
            "--prompt" => prompt_text = Some(it.next().context("--prompt")?.clone()),
            "--system" => system_text = Some(it.next().context("--system")?.clone()),
            "--messages-json" => {
                messages_json = Some(it.next().context("--messages-json")?.clone());
            }
            "--tokenizer" => tokenizer = Some(PathBuf::from(it.next().context("--tokenizer")?)),
            "--prompt-ids" => {
                let raw = it.next().context("--prompt-ids")?;
                prompt_rows = parse_prompt_id_rows(raw)?;
                prompt_ids = prompt_rows.first().cloned().unwrap_or_default();
            }
            other => bail!("rlx-qwen35: unknown flag: {other}"),
        }
    }

    let weights = resolve_weights_cli(
        &weights
            .ok_or_else(|| anyhow!("rlx-qwen35: --weights <path or dir with .gguf> required"))?,
        &resolve_cli,
    )?;

    if let Some(raw) = messages_json {
        let msgs = crate::parse_messages_json(&raw)?;
        prompt_ids = crate::encode_chat_auto(&weights, tokenizer.as_deref(), &msgs)?;
        prompt_rows = vec![prompt_ids.clone()];
        println!(
            "[rlx-qwen35] qwen35: chat ({} turns) → {} ids",
            msgs.len(),
            prompt_ids.len()
        );
    } else if let Some(ref text) = prompt_text {
        if system_text.is_some() {
            let msgs = crate::messages_from_prompt(system_text.as_deref(), text);
            prompt_ids = crate::encode_chat_auto(&weights, tokenizer.as_deref(), &msgs)?;
        } else {
            prompt_ids = crate::encode_prompt_auto(&weights, tokenizer.as_deref(), text)?;
        }
        prompt_rows = vec![prompt_ids.clone()];
        println!(
            "[rlx-qwen35] qwen35: tokenized prompt → {} ids",
            prompt_ids.len()
        );
    }
    if batch > 1 && prompt_rows.len() == 1 {
        prompt_rows = vec![prompt_ids.clone(); batch];
    }
    if batch > 1 && prompt_rows.len() != batch {
        bail!(
            "rlx-qwen35: --batch {batch} requires {batch} prompt rows \
             (use ';' between rows in --prompt-ids, e.g. 1,2,3;4,5,6)"
        );
    }
    let dev = parse_qwen35_device(&device)?;
    if max_seq == 0 {
        max_seq = (prompt_ids.len() + max_tokens).max(8);
    }

    if (dynamic_prefill || dynamic_decode) && batch != 1 {
        bail!("rlx-qwen35: --dynamic-prefill/decode require --batch 1");
    }

    if spec_decode && !enable_mtp {
        bail!("rlx-qwen35: --spec-decode requires --mtp");
    }
    if spec_n == 0 {
        bail!("rlx-qwen35: --spec-n must be >= 1");
    }
    if image.is_some() && mmproj.is_none() {
        bail!("rlx-qwen35: --image requires --mmproj");
    }
    if image.is_some() && batch != 1 {
        bail!("rlx-qwen35: --image requires --batch 1");
    }

    println!(
        "[rlx-qwen35] qwen35: weights={:?} device={device} batch={batch} max_seq={max_seq} \
         mtp={enable_mtp} spec_decode={spec_decode} packed={packed_weights} \
         dynamic_prefill={dynamic_prefill} dynamic_decode={dynamic_decode} \
         mmproj={}",
        weights,
        mmproj
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".into()),
    );

    if fast_mtp && !enable_mtp && !spec_decode {
        bail!("rlx-qwen35: --fast-mtp requires --mtp or --spec-decode");
    }

    let build_runner = |mtp_logits_path: bool| {
        let mut b = crate::Qwen35RunnerBuilder::default()
            .weights(&weights)
            .device(dev)
            .batch(batch)
            .max_seq(max_seq)
            .enable_mtp(enable_mtp || mtp_logits_path)
            .mtp_logits_path(mtp_logits_path)
            .packed_weights(packed_weights)
            .last_logits_only(true);
        if fast_mtp && mtp_logits_path {
            b = b.fast_mtp(true);
        }
        if let Some(ref dir) = aot_cache {
            b = b.aot_cache_dir(dir);
        }
        if dynamic_prefill {
            b = b.dynamic_prefill(true);
        }
        if dynamic_decode {
            b = b.dynamic_decode(true);
        }
        if let Some(ref path) = mmproj {
            b = b.mmproj(path);
        }
        b.build()
    };

    if spec_decode && max_tokens > 0 {
        let draft_runner = build_runner(true)?;
        let target_runner = build_runner(false)?;
        let draft = crate::Qwen35MtpDraft::new(draft_runner);
        let target = crate::Qwen35TrunkTarget::new(target_runner);
        let mut dec = rlx_runtime::spec_decode::SpecDecoder::new(draft, target, spec_n, seed);

        println!(
            "[rlx-qwen35] qwen35: speculative decode {max_tokens} tokens (spec_n={spec_n}, seed={seed})…"
        );
        let mut context = prompt_ids.clone();
        let mut generated = Vec::new();
        while generated.len() < max_tokens {
            let batch = dec.step(&context);
            if batch.is_empty() {
                break;
            }
            for tok in batch {
                if generated.len() >= max_tokens {
                    break;
                }
                generated.push(tok);
                context.push(tok);
                print!("{tok} ");
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
        println!("\n[rlx-qwen35] qwen35: generated: {generated:?}");
        return Ok(());
    }

    let mut runner = build_runner(false)?;

    if let Some(image_path) = image {
        let prompt = prompt_text.as_deref().ok_or_else(|| {
            anyhow!(
                "rlx-qwen35: --image requires --prompt containing `{}`",
                crate::MEDIA_MARKER
            )
        })?;
        if !prompt.contains(crate::MEDIA_MARKER) {
            bail!(
                "rlx-qwen35: multimodal --prompt must contain `{}`",
                crate::MEDIA_MARKER
            );
        }
        #[cfg(feature = "qwen35-vlm")]
        {
            use crate::load_rgb_image;
            let (rgb, w, h) = load_rgb_image(
                image_path
                    .to_str()
                    .ok_or_else(|| anyhow!("non-utf8 --image path"))?,
            )?;
            if max_tokens == 0 {
                let out = runner.prefill_multimodal(prompt, &rgb, w, h, tokenizer.as_deref())?;
                println!(
                    "[rlx-qwen35] qwen35: multimodal prefill logits={} vocab≈{}",
                    out.trunk_logits.len(),
                    runner.lm_vocab_size(),
                );
            } else {
                let opts = if temperature <= 0.0 {
                    SampleOpts::greedy()
                } else {
                    SampleOpts::temperature(temperature, seed).with_top_p(top_p)
                };
                println!(
                    "[rlx-qwen35] qwen35: multimodal generate {max_tokens} tokens \
                     (image={image_path:?})…"
                );
                let new_ids = runner.generate_multimodal_with_opts(
                    prompt,
                    &rgb,
                    w,
                    h,
                    tokenizer.as_deref(),
                    max_tokens,
                    opts,
                    |t| {
                        print!("{t} ");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                        true
                    },
                )?;
                println!("\n[rlx-qwen35] qwen35: generated: {new_ids:?}");
            }
            return Ok(());
        }
        #[cfg(not(feature = "qwen35-vlm"))]
        {
            let _ = (prompt, image_path);
            bail!("rlx-qwen35: --image requires rebuilding with feature `qwen35-vlm`");
        }
    }

    println!(
        "[rlx-qwen35] qwen35: compiled (hidden={}, layers={}, ssm_state={}, dt_rank={})",
        runner.cfg().hidden_size,
        runner.cfg().num_hidden_layers,
        runner.cfg().ssm_state_size,
        runner.cfg().ssm_time_step_rank,
    );

    if max_tokens == 0 {
        let out = runner.predict_logits(&prompt_ids)?;
        println!(
            "[rlx-qwen35] qwen35: logits={} vocab≈{}",
            out.logits.len(),
            out.vocab_size
        );

        let mut idx: Vec<usize> = (0..out.logits.len()).collect();
        idx.sort_by(|&a, &b| {
            out.logits[b]
                .partial_cmp(&out.logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!("[rlx-qwen35] qwen35: top-5 trunk logits:");
        for &i in idx.iter().take(5) {
            println!("    token {i:6}  logit {:>12.5}", out.logits[i]);
        }
        if let Some(mtp) = &out.mtp_logits {
            let mut midx: Vec<usize> = (0..mtp.len()).collect();
            midx.sort_by(|&a, &b| {
                mtp[b]
                    .partial_cmp(&mtp[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            println!("[rlx-qwen35] qwen35: top-5 MTP logits:");
            for &i in midx.iter().take(5) {
                println!("    token {i:6}  logit {:>12.5}", mtp[i]);
            }
        }
    } else {
        let opts = if temperature <= 0.0 {
            SampleOpts::greedy()
        } else {
            SampleOpts::temperature(temperature, seed).with_top_p(top_p)
        };
        println!(
            "[rlx-qwen35] qwen35: generating {max_tokens} tokens (temp={temperature}, top_p={top_p})…"
        );
        if batch > 1 {
            let generated = runner.generate_batch_with_opts(
                &prompt_rows,
                max_tokens,
                None,
                opts,
                |_, tok| {
                    print!("{tok} ");
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                    true
                },
            )?;
            println!("\n[rlx-qwen35] qwen35: generated: {generated:?}");
        } else {
            let new_ids = runner.generate_with_opts(&prompt_ids, max_tokens, opts, |t| {
                print!("{t} ");
                std::io::Write::flush(&mut std::io::stdout()).ok();
                true
            })?;
            println!("\n[rlx-qwen35] qwen35: generated: {new_ids:?}");
        }
    }
    Ok(())
}
