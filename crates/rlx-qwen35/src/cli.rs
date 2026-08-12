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
  --packed                 Keep GGUF quants packed (DEFAULT; F16-resident decode on Metal)
  --dense / --no-packed    Dequantize weights to full F32 (slower, most accurate)
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
    // Default ON: keep GGUF K-quants packed and dequant-on-the-fly, which on
    // Metal runs the fast F16-resident decode path (~6× the F32 dequant path,
    // token-identical). `--dense`/`--no-packed` opts back into full F32 weights.
    let mut packed_weights = true;
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
    // StreamingLLM sink+window KV eviction (0 window = off).
    let mut kv_window: usize = 0;
    let mut kv_sinks: usize = 4;
    let mut verify_selftest = false;
    let mut chunk_prefill: usize = 0;
    let mut prefix_cache_bench = false;
    let mut self_spec = false;
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
            "--dense" | "--no-packed" => packed_weights = false,
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
            "--kv-window" => kv_window = it.next().context("--kv-window")?.parse()?,
            "--kv-sinks" => kv_sinks = it.next().context("--kv-sinks")?.parse()?,
            "--verify-selftest" => verify_selftest = true,
            "--chunk-prefill" => chunk_prefill = it.next().context("--chunk-prefill")?.parse()?,
            "--prefix-cache-bench" => prefix_cache_bench = true,
            "--self-spec" => self_spec = true,
            "--swa-window" => {
                // Sliding-window prefill attention → linear TTFT (lossy on middle
                // context). Read by the builder via RLX_QWEN35_SWA_WINDOW.
                let w: usize = it.next().context("--swa-window")?.parse()?;
                // SAFETY: single-threaded CLI setup before runner construction.
                unsafe { std::env::set_var("RLX_QWEN35_SWA_WINDOW", w.to_string()) };
            }
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
        // Batched verify reads the host KV mirror; force non-resident.
        // SAFETY: single-threaded CLI setup before runner construction.
        unsafe { std::env::set_var("RLX_QWEN35_GPU_KV", "0") };
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

    if kv_window > 0 {
        // The decode graph reads RLX_QWEN35_GPU_KV at compile time; the resident
        // graph shape is incompatible with the host-fed eviction path, so force
        // non-resident BEFORE the runner (and its decode graph) is built.
        // SAFETY: single-threaded CLI setup, before any runner construction.
        unsafe { std::env::set_var("RLX_QWEN35_GPU_KV", "0") };
    }
    if verify_selftest {
        // verify_forward reads the host KV mirror, which the resident path leaves
        // stale — force non-resident so both paths see the same cache.
        // SAFETY: single-threaded CLI setup before runner construction.
        unsafe { std::env::set_var("RLX_QWEN35_GPU_KV", "0") };
        let mut runner = build_runner(false)?;
        let seed = runner.prefill_seed_for_decode(&prompt_ids)?;
        let t0 = seed
            .trunk_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let t1 = t0; // any 2nd token exercises the position-1 logits
        let cache = runner
            .decode_cache_checkpoint()
            .context("verify selftest: no cache")?;
        // Batched m=2 verify (does not mutate the cache).
        let vlog = runner.verify_forward(&cache, &[t0, t1])?;
        // Two sequential decode steps (the reference).
        runner.restore_decode_cache(Some(cache));
        let s0 = runner.decode_get_logits(t0)?;
        let s1 = runner.decode_get_logits(t1)?;
        let amax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let maxdiff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        println!(
            "[rope-dims] key_length={} rope_dim_count={} head_half={} rot_half={} runtime_mrope={}",
            runner.cfg_key_length(),
            runner.cfg_rope_dim_count(),
            runner.cfg_key_length() / 2,
            runner.cfg_rope_dim_count() / 2,
            runner.is_runtime_mrope(),
        );
        let (a0, a1) = (amax(&vlog[0]) == amax(&s0), amax(&vlog[1]) == amax(&s1));
        println!(
            "[verify-selftest] pos0: argmax_match={a0} maxdiff={:.5} | pos1: argmax_match={a1} maxdiff={:.5} | verdict={}",
            maxdiff(&vlog[0], &s0),
            maxdiff(&vlog[1], &s1),
            if a0 && a1 { "PASS ✓" } else { "FAIL ✗" }
        );
        // Component isolation: commit [t0,t1] both ways (batched verify_and_commit
        // vs two sequential decodes) and diff the committed cache per layer type,
        // so we can see whether GDN recurrent state or full-attn KV is the source.
        {
            use crate::cache::Qwen35LayerState;
            let base = runner
                .decode_cache_checkpoint()
                .context("gdn-diff: no base cache")?;
            let mut ca = base.clone();
            let _ = runner.verify_and_commit(&mut ca, &[t0, t1])?;
            runner.restore_decode_cache(Some(base.clone()));
            runner.decode_get_logits(t0)?;
            runner.decode_get_logits(t1)?;
            let cb = runner
                .decode_cache_checkpoint()
                .context("gdn-diff: no seq cache")?;
            // PER-LAYER localization: the first layer whose committed state diverges
            // pins the source. GDN ssm_state at layer L takes layer L-1's output as
            // input, so a clean ssm at L but dirty at L+k localizes to the layers
            // between. For FullAttn, diff the pos0 vs pos1 committed KV rows.
            let rows = ca.past_seq.max(1);
            for (il, (la, lb)) in ca.layers.iter().zip(cb.layers.iter()).enumerate() {
                match (la, lb) {
                    (
                        Qwen35LayerState::Linear {
                            conv_state: ca_c,
                            ssm_state: ca_s,
                        },
                        Qwen35LayerState::Linear {
                            conv_state: cb_c,
                            ssm_state: cb_s,
                        },
                    ) => {
                        println!(
                            "[gdn-diff] L{il:02} Linear   ssm={:.6} conv={:.6}",
                            maxdiff(ca_s, cb_s),
                            maxdiff(ca_c, cb_c)
                        );
                    }
                    (
                        Qwen35LayerState::FullAttn {
                            past_k: ak,
                            past_v: av,
                            ..
                        },
                        Qwen35LayerState::FullAttn {
                            past_k: bk,
                            past_v: bv,
                            ..
                        },
                    ) => {
                        // Committed KV is [past_seq, kv_cols] flattened; last two rows
                        // are pos0 (second-to-last) and pos1 (last).
                        let kv_cols = (ak.len() / rows).max(1);
                        let last = |v: &[f32], back: usize| -> Vec<f32> {
                            let end = v.len().saturating_sub((back - 1) * kv_cols);
                            let start = v.len().saturating_sub(back * kv_cols);
                            v[start..end].to_vec()
                        };
                        let k0 = maxdiff(&last(ak, 2), &last(bk, 2));
                        let k1 = maxdiff(&last(ak, 1), &last(bk, 1));
                        let v1 = maxdiff(&last(av, 1), &last(bv, 1));
                        println!(
                            "[gdn-diff] L{il:02} FullAttn k_pos0={k0:.6} k_pos1={k1:.6} v_pos1={v1:.6}"
                        );
                    }
                    _ => {}
                }
            }
        }
        // PREFILL pos>=1 rope check: does the multi-token PREFILL flow (separate
        // from decode/verify) also mis-stride partial rope? Compare prefill of
        // [prompt,d0,d1] last-token logits vs decode(d0);decode(d1) last logits.
        {
            use crate::cache::Qwen35LayerState;
            runner.reset_decode_cache();
            let seed = runner.prefill_seed_for_decode(&prompt_ids)?;
            let d0 = amax(&seed.trunk_logits) as u32;
            let l0 = runner.decode_get_logits(d0)?;
            let d1 = amax(&l0) as u32;
            let l1_decode = runner.decode_get_logits(d1)?;
            let dec_cache = runner.decode_cache_checkpoint().unwrap();
            let mut ext = prompt_ids.clone();
            ext.push(d0);
            ext.push(d1);
            runner.reset_decode_cache();
            let seed2 = runner.prefill_seed_for_decode(&ext)?;
            let l1_prefill = seed2.trunk_logits.clone();
            let pre_cache = runner.decode_cache_checkpoint().unwrap();
            println!(
                "[prefill-rope] pos>=1 argmax_match={} maxdiff={:.5}",
                amax(&l1_prefill) == amax(&l1_decode),
                maxdiff(&l1_prefill, &l1_decode),
            );
            // Per-layer: where does prefill's committed KV first diverge from decode?
            // If born at a FullAttn layer's k_pos (rope'd K) while v_pos is exact →
            // prefill has the same partial-rope stride bug. Otherwise it's attention.
            let rows = pre_cache.past_seq.max(1);
            for (il, (pa, db)) in pre_cache
                .layers
                .iter()
                .zip(dec_cache.layers.iter())
                .enumerate()
            {
                match (pa, db) {
                    (
                        Qwen35LayerState::Linear {
                            conv_state: pc,
                            ssm_state: ps,
                        },
                        Qwen35LayerState::Linear {
                            conv_state: dc,
                            ssm_state: ds,
                        },
                    ) => {
                        println!(
                            "[prefill-diff] L{il:02} Linear   ssm={:.5} conv={:.5}",
                            maxdiff(ps, ds),
                            maxdiff(pc, dc),
                        );
                    }
                    (
                        Qwen35LayerState::FullAttn {
                            past_k: ak,
                            past_v: av,
                            ..
                        },
                        Qwen35LayerState::FullAttn {
                            past_k: bk,
                            past_v: bv,
                            ..
                        },
                    ) => {
                        let kv_cols = (ak.len() / rows).max(1);
                        let last = |v: &[f32]| v[v.len().saturating_sub(kv_cols)..].to_vec();
                        println!(
                            "[prefill-diff] L{il:02} FullAttn k_pos1={:.5} v_pos1={:.5}",
                            maxdiff(&last(ak), &last(bk)),
                            maxdiff(&last(av), &last(bv)),
                        );
                    }
                    _ => {}
                }
            }
        }
        return Ok(());
    }

    if self_spec {
        // Greedy self-speculative decode built from the validated primitives:
        // verify_forward (decision, no commit) + prefill_chunk (commit). Output
        // MUST equal greedy; this validates the speculative accept/reject logic
        // before merging verify+commit into one forward for the speedup.
        // SAFETY: single-threaded CLI setup before runner construction.
        unsafe { std::env::set_var("RLX_QWEN35_GPU_KV", "0") };
        let mut runner = build_runner(false)?;
        let amax = |v: &[f32]| -> u32 {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        // Argmax + confidence margin (top1 − top2). The multi-token bias-mask
        // verify kernel is not bit-identical to single-token decode, so its argmax
        // can flip vs greedy on a near-tie. We only trust a verify-committed token
        // when its margin clears `spec_margin` (well above kernel FP noise);
        // anything closer falls back to the EXACT `prefill_chunk` decode.
        let amax_margin = |v: &[f32]| -> (u32, f32) {
            let mut best = (0usize, f32::NEG_INFINITY);
            let mut second = f32::NEG_INFINITY;
            for (i, &x) in v.iter().enumerate() {
                if x > best.1 {
                    second = best.1;
                    best = (i, x);
                } else if x > second {
                    second = x;
                }
            }
            (best.0 as u32, best.1 - second)
        };
        // Default 0.0 = accept whenever the draft matches verify's argmax. The
        // batched verify is now bit-identical to sequential decode (partial-RoPE
        // feed-stride fix), so accepts no longer drift and no margin is needed for
        // exactness. The knob stays for A/B testing.
        let spec_margin: f32 = rlx_ir::env::var("RLX_QWEN35_SPEC_MARGIN")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        // window=0 → correctness mode (no eviction; MUST equal full greedy, but
        // slow: the growing past_seq recompiles the verify graph each step).
        // window>0 → speed mode: sink+window eviction bounds past_seq so the
        // verify/chunk graphs compile ONCE (no recompile). Output is windowed,
        // so we compare against a WINDOWED greedy (same bound).
        let window = kv_window;
        if window > 0 {
            runner.set_kv_sink_window(kv_sinks, window, 0);
        }
        // Greedy reference (timed).
        let tg = std::time::Instant::now();
        let greedy = runner.generate_with_opts(
            &prompt_ids,
            max_tokens,
            rlx_qwen3::SampleOpts::greedy(),
            |_| true,
        )?;
        let greedy_s = tg.elapsed().as_secs_f64();
        // Self-spec generation. Pass 0 warms the verify/chunk graphs (one-time
        // compile, amortized across requests in a real deployment); pass 1 is the
        // timed STEADY-STATE run that reuses the cached graphs.
        let mut out: Vec<u32> = Vec::new();
        let (mut n_fwd, mut n_accept2, mut spec_s) = (0usize, 0usize, 0f64);
        for pass in 0..2 {
            runner.reset_decode_cache();
            let seed = runner.prefill_seed_for_decode(&prompt_ids)?;
            let mut cache = runner
                .decode_cache_checkpoint()
                .context("self-spec: no cache")?;
            let mut t = amax(&seed.trunk_logits);
            out = vec![t];
            n_fwd = 0;
            n_accept2 = 0;
            if window > 0 {
                runner.evict_kv_window(&mut cache, kv_sinks, window);
            }
            // 2-gram prompt-lookup draft (fallback / initial): the token that
            // followed the last prior occurrence of [prev, t] in the context.
            let lookup = |out: &[u32], t: u32| -> u32 {
                let mut full = prompt_ids.clone();
                full.extend_from_slice(out);
                if full.len() >= 2 {
                    let (a, b) = (full[full.len() - 2], full[full.len() - 1]);
                    (0..full.len().saturating_sub(2))
                        .rev()
                        .find(|&i| full[i] == a && full[i + 1] == b)
                        .and_then(|i| full.get(i + 2).copied())
                        .unwrap_or(t)
                } else {
                    t
                }
            };
            let mut d = lookup(&out, t);
            let ts = std::time::Instant::now();
            while out.len() < max_tokens {
                // ONE forward verifies [t,d] AND commits both, and its MTP head
                // emits the NEXT draft for free.
                let checkpoint = cache.clone();
                let (vlog, mtp_draft) = runner.verify_and_commit(&mut cache, &[t, d])?;
                n_fwd += 1;
                let (a0, m0) = amax_margin(&vlog[0]);
                let (a1, m1) = amax_margin(&vlog[1]);
                // Accept the 2-token commit ONLY when the draft matches AND BOTH
                // committed positions are confident (margin clears the kernel-noise
                // floor) — so a bias-verify argmax that could flip vs exact decode is
                // never committed. Everything else rolls back to an EXACT decode.
                if a0 == d && m0 >= spec_margin && m1 >= spec_margin && out.len() + 1 < max_tokens {
                    out.push(a0);
                    t = a1;
                    out.push(t);
                    n_accept2 += 1;
                    // MTP draft is for the token after `d` — exactly next-after-`t`.
                    d = mtp_draft.unwrap_or_else(|| lookup(&out, t));
                } else {
                    // Reject (draft wrong OR near-tie): roll back both committed
                    // tokens and re-decode `t` through the EXACT decode kernel — its
                    // logits (not the bias verify's) give the committed next token,
                    // so this position is bit-identical to greedy.
                    cache = checkpoint;
                    let exact = runner.prefill_chunk(&mut cache, &[t])?;
                    n_fwd += 1;
                    t = amax(&exact);
                    out.push(t);
                    d = lookup(&out, t);
                }
                if window > 0 {
                    runner.evict_kv_window(&mut cache, kv_sinks, window);
                }
            }
            out.truncate(max_tokens);
            if pass == 1 {
                spec_s = ts.elapsed().as_secs_f64();
            }
        }
        let matches = out == greedy;
        println!(
            "[self-spec] match_greedy={matches} | fwd/tok={:.2} ({n_fwd} fwd, accept2={n_accept2}) | \
             decode: greedy={greedy_s:.2}s self-spec={spec_s:.2}s ({:.2}x)",
            n_fwd as f32 / out.len().max(1) as f32,
            greedy_s / spec_s.max(1e-9),
        );
        if !matches {
            let first = out.iter().zip(&greedy).position(|(a, b)| a != b);
            println!(
                "  first divergence at {first:?}\n  greedy={:?}\n  spec  ={out:?}",
                greedy
            );
        }
        return Ok(());
    }

    if prefix_cache_bench {
        // prefill_chunk reuses the host KV mirror → non-resident.
        // SAFETY: single-threaded CLI setup before runner construction.
        unsafe { std::env::set_var("RLX_QWEN35_GPU_KV", "0") };
        let mut runner = build_runner(false)?;
        let prompt = prompt_ids.clone();
        // Split the prompt into a shared prefix + a 16-token "new query" suffix.
        let qn = 16.min(prompt.len().saturating_sub(1)).max(1);
        let split = prompt.len() - qn;
        let prefix = &prompt[..split];
        let query = &prompt[split..];
        // COLD: fresh prefill of the whole prompt.
        runner.reset_decode_cache();
        let t = std::time::Instant::now();
        let _ = runner.prefill_get_last_logits(&prompt)?;
        let cold = t.elapsed().as_secs_f64();
        // WARM: prefill the prefix ONCE (amortized), then feed only the query
        // suffix into the cached prefix via one continued forward.
        runner.reset_decode_cache();
        let _ = runner.prefill_get_last_logits(prefix)?;
        let ck = runner
            .decode_cache_checkpoint()
            .context("prefix-cache: no cache")?;
        let t = std::time::Instant::now();
        let mut cache = ck;
        let _ = runner.prefill_chunk(&mut cache, query)?;
        let warm = t.elapsed().as_secs_f64();
        println!(
            "[prefix-cache] prefix={} query={} | COLD (full prefill) TTFT={cold:.3}s | \
             WARM (reuse prefix) TTFT={warm:.3}s | speedup={:.2}x",
            prefix.len(),
            query.len(),
            cold / warm.max(1e-9)
        );
        return Ok(());
    }

    if chunk_prefill > 0 {
        // Chunked prefill needs the host KV mirror (non-resident).
        // SAFETY: single-threaded CLI setup before runner construction.
        unsafe { std::env::set_var("RLX_QWEN35_GPU_KV", "0") };
        let mut runner = build_runner(false)?;
        let window = if kv_window > 0 { kv_window } else { 512 };
        println!(
            "[chunk-prefill] prompt={} tok, chunk={chunk_prefill}, sinks={kv_sinks}, window={window}",
            prompt_ids.len()
        );
        // Chunked (linear) TTFT.
        let t0 = std::time::Instant::now();
        let (_logits, _cache) =
            runner.prefill_chunked(&prompt_ids, chunk_prefill, kv_sinks, window)?;
        let chunked = t0.elapsed().as_secs_f64();
        // Normal (quadratic) TTFT for comparison.
        runner.reset_decode_cache();
        let t1 = std::time::Instant::now();
        let _ = runner.prefill_get_last_logits(&prompt_ids)?;
        let normal = t1.elapsed().as_secs_f64();
        println!(
            "[chunk-prefill] TTFT chunked={chunked:.2}s | normal={normal:.2}s | speedup={:.2}x",
            normal / chunked.max(1e-9)
        );
        return Ok(());
    }

    let mut runner = build_runner(false)?;
    if kv_window > 0 {
        runner.set_kv_sink_window(kv_sinks, kv_window, 0);
        println!(
            "[rlx-qwen35] qwen35: sink+window KV eviction on (sinks={kv_sinks}, window={kv_window}) \
             — resident KV disabled"
        );
    }

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
            // Decode-only timing: first token = prefill/TTFT done; time the rest.
            // RLX_BENCH_NOSTOP forces the full max_tokens (ignore EOS) for clean tps.
            let bench_nostop = std::env::var_os("RLX_BENCH_NOSTOP").is_some();
            let ttft = std::cell::Cell::new(None::<std::time::Instant>);
            let ndec = std::cell::Cell::new(0usize);
            let gen_t0 = std::time::Instant::now();
            // Live text streaming: detokenize the running id list each step and
            // print only the newly-decoded suffix (BPE-safe), so the user reads the
            // answer as it generates instead of waiting for the whole thing. Falls
            // back to printing raw ids when no tokenizer is available.
            let stream_ids = std::cell::RefCell::new(Vec::<u32>::new());
            let printed = std::cell::Cell::new(0usize);
            let emit = |t: u32| {
                stream_ids.borrow_mut().push(t);
                match crate::decode_ids_auto(
                    &weights,
                    tokenizer.as_deref(),
                    &stream_ids.borrow(),
                    true,
                ) {
                    Ok(text) => {
                        // Avoid emitting an incomplete trailing UTF-8/BPE piece.
                        let stable = text
                            .char_indices()
                            .rev()
                            .find(|(_, c)| c.is_whitespace() || c.is_ascii_punctuation())
                            .map(|(i, c)| i + c.len_utf8())
                            .unwrap_or(0)
                            .max(printed.get());
                        if stable > printed.get() && text.is_char_boundary(stable) {
                            print!("{}", &text[printed.get()..stable]);
                            std::io::Write::flush(&mut std::io::stdout()).ok();
                            printed.set(stable);
                        }
                    }
                    Err(_) => {
                        print!("{t} ");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                }
            };
            let mut new_ids = runner.generate_with_opts(&prompt_ids, max_tokens, opts, |t| {
                if ttft.get().is_none() {
                    ttft.set(Some(std::time::Instant::now()));
                } else {
                    ndec.set(ndec.get() + 1);
                }
                if !bench_nostop && specials.is_stop(t) {
                    return false;
                }
                if let Some(w) = think_watch.as_mut() {
                    let cont = w.observe(t);
                    emit(t);
                    return cont;
                }
                emit(t);
                true
            })?;
            // Flush any remaining tail text after the last stable boundary.
            if let Ok(text) =
                crate::decode_ids_auto(&weights, tokenizer.as_deref(), &stream_ids.borrow(), true)
                && text.len() > printed.get()
                && text.is_char_boundary(printed.get())
            {
                print!("{}", &text[printed.get()..]);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            if let Some(ft) = ttft.get() {
                let secs = ft.elapsed().as_secs_f64();
                let n = ndec.get();
                eprintln!(
                    "\n[rlx-qwen35] BENCH: prefill/TTFT={:.3}s | decode {} tok in {:.3}s = {:.2} tok/s",
                    (ft - gen_t0).as_secs_f64(),
                    n,
                    secs,
                    n as f64 / secs.max(1e-9),
                );
            }

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
                    let close_ids = crate::encode_prompt_auto(
                        &weights,
                        tokenizer.as_deref(),
                        crate::THINK_BUDGET_CLOSE,
                    )?;
                    let total_len = prompt_ids.len() + new_ids.len() + close_ids.len();
                    if total_len >= max_seq {
                        bail!(
                            "rlx-qwen35: thinking-budget continuation exceeds --max-seq {max_seq} \
                             (prompt+think+close = {total_len})"
                        );
                    }
                    // Show the forced think-close text streaming (it's fed into
                    // the cache, not generated, so `generate_continue` won't emit
                    // it through the token callback).
                    for &t in &close_ids {
                        emit(t);
                    }
                    let more = if std::env::var_os("RLX_QWEN35_REPREFILL").is_some() {
                        // Legacy / parity path: re-prefill the whole
                        // prompt+think+close before answering.
                        let mut cont_prompt = prompt_ids.clone();
                        cont_prompt.extend_from_slice(&new_ids);
                        cont_prompt.extend_from_slice(&close_ids);
                        runner.generate_with_opts(&cont_prompt, remain, opts, |t| {
                            if specials.is_stop(t) {
                                return false;
                            }
                            emit(t);
                            true
                        })?
                    } else {
                        // Incremental path: feed `[pending, close…]` into the
                        // live decode cache (O(close), no re-prefill), then
                        // stream the answer. `new_ids.last()` is the pending
                        // budget-hit token (sampled, not yet folded into KV).
                        let mut feed = Vec::with_capacity(1 + close_ids.len());
                        feed.push(*new_ids.last().expect("budget hit implies ≥1 token"));
                        feed.extend_from_slice(&close_ids);
                        runner.generate_continue(&feed, remain, opts, |t| {
                            if specials.is_stop(t) {
                                return false;
                            }
                            emit(t);
                            true
                        })?
                    };
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
