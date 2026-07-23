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

//! `rlx-qwen35` CLI: GGUF generate, ChatML, `--fast`, MTP / VLM.
//!
//! See the crate README for flags and env (`RLX_QWEN35_BENCH`, warm/keep-prefill).

use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{WeightsResolveCli, parse_qwen35_device, resolve_weights_cli};
use rlx_qwen3::SampleOpts;
use std::path::PathBuf;

fn print_usage() {
    eprintln!(
        "\
Usage: rlx-qwen35 --weights PATH [options]

  --weights PATH           GGUF (or directory) path
  --device NAME            cpu|metal|mlx|cuda|rocm|gpu|vulkan|auto (default cpu)
  --packed                 Keep GGUF quants packed in the arena
  --prompt TEXT            ChatML user turn (needs tokenizer feature)
  --prompt-ids 1,2;3,4     Raw token ids (; = batch rows)
  --system TEXT            System message (with --prompt / --chat)
  --messages-json JSON     Multi-turn ChatML messages
  --chat                   Format --prompt as ChatML (thinking on by default)
  --no-think / --think     Disable / enable assistant <think> block
  --thinking-budget N      Force-close think after N tokens, then answer
  --show-thinking          Print think block separately from the answer
  --fast                   --no-think + tight max_seq + prefill_seq=prompt_len
  --max-seq N / --max-tokens N
  --mtp / --spec-decode / --spec-n N / --fast-mtp
  --dynamic-prefill / --dynamic-decode
  --mmproj PATH / --image PATH
  --temperature F / --top-p F / --seed N / --batch N
  --aot-cache DIR
  --help                   This message

Env: RLX_QWEN35_BENCH, RLX_QWEN35_DECODE_TRACE, RLX_QWEN35_WARM_DECODE,
     RLX_QWEN35_KEEP_PREFILL, RLX_LOW_MEM_COMPILE (see crate README)."
    );
}

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
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        if args.is_empty() {
            bail!("rlx-qwen35: missing --weights (pass --help for usage)");
        }
        return Ok(());
    }

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
    let mut use_chat = false;
    let mut enable_thinking = true;
    let mut thinking_budget: Option<usize> = None;
    let mut show_thinking = false;
    let mut fast = false;
    let mut max_seq_set = false;

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
            "--max-seq" => {
                max_seq = it.next().context("--max-seq")?.parse()?;
                max_seq_set = true;
            }
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
            "--chat" => use_chat = true,
            "--fast" => {
                // Low-latency QA: no chain-of-thought, tight decode max_seq,
                // and prefill_seq = prompt length (see builder below).
                fast = true;
                enable_thinking = false;
                use_chat = true;
            }
            "--no-think" => {
                enable_thinking = false;
                use_chat = true;
            }
            "--think" => {
                enable_thinking = true;
                use_chat = true;
            }
            "--thinking-budget" => {
                thinking_budget = Some(it.next().context("--thinking-budget")?.parse()?);
                enable_thinking = true;
                use_chat = true;
            }
            "--show-thinking" => show_thinking = true,
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

    let chat_opts = crate::ChatFormatOpts { enable_thinking };
    if let Some(raw) = messages_json {
        let msgs = crate::parse_messages_json(&raw)?;
        prompt_ids =
            crate::encode_chat_auto_with(&weights, tokenizer.as_deref(), &msgs, chat_opts)?;
        prompt_rows = vec![prompt_ids.clone()];
        println!(
            "[rlx-qwen35] qwen35: chat ({} turns, thinking={}) → {} ids",
            msgs.len(),
            enable_thinking,
            prompt_ids.len()
        );
    } else if let Some(ref text) = prompt_text {
        if use_chat || system_text.is_some() {
            let msgs = crate::messages_from_prompt(system_text.as_deref(), text);
            prompt_ids =
                crate::encode_chat_auto_with(&weights, tokenizer.as_deref(), &msgs, chat_opts)?;
            println!(
                "[rlx-qwen35] qwen35: chat prompt (thinking={}) → {} ids",
                enable_thinking,
                prompt_ids.len()
            );
        } else {
            prompt_ids = crate::encode_prompt_auto(&weights, tokenizer.as_deref(), text)?;
            println!(
                "[rlx-qwen35] qwen35: tokenized prompt → {} ids",
                prompt_ids.len()
            );
        }
        prompt_rows = vec![prompt_ids.clone()];
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
    if max_seq == 0 || (fast && !max_seq_set) {
        // Tight compile shape: padded decode cost scales with max_seq.
        let need = prompt_ids.len() + max_tokens.max(8) + if fast { 4 } else { 0 };
        max_seq = need.max(8);
    }
    if fast && max_seq_set && max_seq > prompt_ids.len() + max_tokens + 32 {
        eprintln!(
            "[rlx-qwen35] qwen35: --fast note: --max-seq {max_seq} is larger than needed \
             (prompt+tokens≈{}); decode pads to max_seq",
            prompt_ids.len() + max_tokens
        );
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
    if thinking_budget.is_some() && batch != 1 {
        bail!("rlx-qwen35: --thinking-budget requires --batch 1");
    }

    println!(
        "[rlx-qwen35] qwen35: weights={:?} device={device} batch={batch} max_seq={max_seq} \
         mtp={enable_mtp} spec_decode={spec_decode} packed={packed_weights} \
         thinking={enable_thinking} thinking_budget={} \
         dynamic_prefill={dynamic_prefill} dynamic_decode={dynamic_decode} \
         mmproj={}",
        weights,
        thinking_budget
            .map(|b| b.to_string())
            .unwrap_or_else(|| "none".into()),
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
        if fast {
            // Prefill GEMMs at prompt length; decode still uses max_seq.
            b = b.prefill_seq(prompt_ids.len().max(1));
        }
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
        let specials = crate::SpecialTokenIds::resolve(&weights, tokenizer.as_deref());
        if batch > 1 {
            let generated = runner.generate_batch_with_opts(
                &prompt_rows,
                max_tokens,
                None,
                opts,
                |_, tok| {
                    if specials.is_stop(tok) {
                        return false;
                    }
                    print!("{tok} ");
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                    true
                },
            )?;
            println!("\n[rlx-qwen35] qwen35: generated: {generated:?}");
        } else {
            let mut think_watch = thinking_budget.map(|b| {
                if enable_thinking {
                    crate::ThinkingBudgetWatch::new_already_thinking(specials.clone(), b)
                } else {
                    crate::ThinkingBudgetWatch::new(specials.clone(), b)
                }
            });
            let mut new_ids = runner.generate_with_opts(&prompt_ids, max_tokens, opts, |t| {
                if specials.is_stop(t) {
                    return false;
                }
                if let Some(w) = think_watch.as_mut() {
                    let cont = w.observe(t);
                    print!("{t} ");
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                    return cont;
                }
                print!("{t} ");
                std::io::Write::flush(&mut std::io::stdout()).ok();
                true
            })?;

            if think_watch
                .as_ref()
                .is_some_and(|w| w.budget_hit && !w.closed)
            {
                let remain = max_tokens.saturating_sub(new_ids.len());
                if remain > 0 {
                    println!(
                        "\n[rlx-qwen35] qwen35: thinking budget hit ({} toks) — closing think + finishing answer…",
                        thinking_budget.unwrap_or(0)
                    );
                    let mut cont_prompt = prompt_ids.clone();
                    cont_prompt.extend_from_slice(&new_ids);
                    let close_ids = crate::encode_prompt_auto(
                        &weights,
                        tokenizer.as_deref(),
                        crate::THINK_BUDGET_CLOSE,
                    )?;
                    cont_prompt.extend_from_slice(&close_ids);
                    if cont_prompt.len() >= max_seq {
                        bail!(
                            "rlx-qwen35: thinking-budget continuation exceeds --max-seq {max_seq} \
                             (prompt+think+close = {})",
                            cont_prompt.len()
                        );
                    }
                    let more = runner.generate_with_opts(&cont_prompt, remain, opts, |t| {
                        if specials.is_stop(t) {
                            return false;
                        }
                        print!("{t} ");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                        true
                    })?;
                    new_ids.extend(close_ids);
                    new_ids.extend(more);
                }
            }

            println!("\n[rlx-qwen35] qwen35: generated: {new_ids:?}");
            match crate::decode_ids_auto(&weights, tokenizer.as_deref(), &new_ids, true) {
                Ok(text) => {
                    let cleaned = text.trim();
                    let (think, answer) = crate::split_thinking(cleaned);
                    if show_thinking {
                        if let Some(t) = think.as_ref() {
                            if !t.is_empty() {
                                println!("[rlx-qwen35] qwen35: thinking>\n{t}");
                            }
                        }
                    }
                    let display = if answer.is_empty() {
                        cleaned.to_string()
                    } else {
                        answer
                    };
                    println!("[rlx-qwen35] qwen35: text: {display:?}");
                    println!("[rlx-qwen35] qwen35: text>\n{display}");
                }
                Err(e) => eprintln!("[rlx-qwen35] qwen35: detokenize skipped: {e}"),
            }
        }
    }
    Ok(())
}
