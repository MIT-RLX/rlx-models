// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Interactive terminal chat REPL for qwen3.
//!
//! Uses the fast KV-cache decode path (`Qwen3Runner::generate_stoppable`),
//! streams tokens as they're produced, and keeps multi-turn history. Picks up
//! the decode optimizations from env — e.g.:
//!
//! ```bash
//! cargo run --release -p rlx-qwen3 --example qwen_chat --features metal -- --device metal
//! # with the fast decode stack:
//! RLX_QWEN3_F16_WEIGHTS=1 RLX_QWEN3_BAKE_WEIGHTS=1 RLX_QWEN3_GQA_NATIVE=1 \
//!   cargo run --release -p rlx-qwen3 --example qwen_chat --features metal -- --device metal
//! ```
//!
//! Flags: `--device <metal|cpu|mlx|…>` `--weights <dir>` `--max-tokens <n>`
//! `--temp <f>` `--system <text>`. Type `/reset` to clear history, `/exit` to quit.

use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;

use rlx_cli::WeightFormat;
use rlx_qwen3::{Qwen3Runner, SampleOpts};
use rlx_runtime::Device;
use tokenizers::Tokenizer;

const DEFAULT_WEIGHTS: &str = "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b";

/// Full Qwen chat-template prompt: system + completed (user, assistant) turns +
/// the pending user turn (open, ready for the assistant to continue). Used for a
/// fresh prefill on turn 1 and after eviction; normal turns feed only the delta.
fn render_prompt(system: &str, turns: &[(String, String)], pending_user: &str) -> String {
    let mut s = format!("<|im_start|>system\n{system}<|im_end|>\n");
    for (u, a) in turns {
        s.push_str(&format!(
            "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        ));
    }
    s.push_str(&format!(
        "<|im_start|>user\n{pending_user}<|im_end|>\n<|im_start|>assistant\n"
    ));
    s
}

/// Print the per-token decode-latency distribution + a sparkline, and append the
/// full time series to `csv_path`. `tok_times[i]` is the elapsed-since-t0 the
/// i-th visible token arrived; inter-token deltas are the instantaneous decode
/// latency. `ttft` is the time to the first token (feed + any compile). We report
/// the distribution (p50/p90/p99/max), NOT just an average — stutter is a tail
/// event that a mean smears away.
fn report_timings(
    tok_times: &[std::time::Duration],
    ttft: std::time::Duration,
    csv_path: &str,
    turn: usize,
) {
    if tok_times.len() < 2 {
        eprintln!("\x1b[2m  [timings: too few tokens to chart]\x1b[0m");
        return;
    }
    // Inter-token latencies (ms): skip index 0 (that gap is TTFT, not a decode step).
    let inter: Vec<f64> = tok_times
        .windows(2)
        .map(|w| (w[1].as_secs_f64() - w[0].as_secs_f64()) * 1e3)
        .collect();
    let mut sorted = inter.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let tps = |ms: f64| if ms > 0.0 { 1000.0 / ms } else { 0.0 };
    let maxv = *sorted.last().unwrap();
    eprintln!(
        "\x1b[2m  decode inter-token ms: p50={:.1} p90={:.1} p99={:.1} max={:.1}  |  tps p50={:.0} p90={:.0} min={:.0}\x1b[0m",
        pct(50.0),
        pct(90.0),
        pct(99.0),
        maxv,
        tps(pct(50.0)),
        tps(pct(90.0)),
        tps(maxv),
    );
    // Sparkline of inter-token ms over token index. Downsample to `width` columns
    // taking the MAX per column so stutter spikes survive; log scale so a 2000ms
    // stall and 15ms steps both register. A flat ▁ line = smooth; a ▇/█ = a stall.
    let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let width = 60usize.min(inter.len());
    let n = inter.len();
    let lo_ln = 1.0f64.ln();
    let hi_ln = maxv.max(2.0).ln();
    let mut line = String::with_capacity(width * 3);
    for c in 0..width {
        let a = c * n / width;
        let b = ((c + 1) * n / width).max(a + 1);
        let colmax = inter[a..b].iter().cloned().fold(0.0f64, f64::max).max(1.0);
        let frac = ((colmax.ln() - lo_ln) / (hi_ln - lo_ln).max(1e-9)).clamp(0.0, 1.0);
        let bi = ((frac * (blocks.len() - 1) as f64).round() as usize).min(blocks.len() - 1);
        line.push(blocks[bi]);
    }
    eprintln!(
        "\x1b[2m  decode latency over {} tokens (log ms, ▁={:.0} █={:.0}): \x1b[0m{}",
        n, sorted[0], maxv, line
    );
    // Append the full series to CSV for external charting.
    use std::io::Write as _;
    let mut buf = String::new();
    if !std::path::Path::new(csv_path).exists() {
        buf.push_str("turn,token_idx,elapsed_ms,inter_ms,inst_tps\n");
    }
    for (i, t) in tok_times.iter().enumerate() {
        let elapsed_ms = t.as_secs_f64() * 1e3;
        let inter_ms = if i == 0 {
            ttft.as_secs_f64() * 1e3
        } else {
            (t.as_secs_f64() - tok_times[i - 1].as_secs_f64()) * 1e3
        };
        buf.push_str(&format!(
            "{turn},{i},{elapsed_ms:.3},{inter_ms:.3},{:.2}\n",
            if inter_ms > 0.0 {
                1000.0 / inter_ms
            } else {
                0.0
            }
        ));
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)
    {
        let _ = f.write_all(buf.as_bytes());
        eprintln!("\x1b[2m  [timings appended to {csv_path}]\x1b[0m");
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // Default the token-identical **Metal-only** decode optimizations ON for this
    // chat example (f16-resident weights ≈ lossless bf16, bake-weight-concat is
    // exact, GQA-native is exact). Read at graph-build time; an explicit `=0`
    // still wins. F16-resident weights need the backend to convert f32→f16 param
    // bytes at bind (Metal does; CPU/other backends read the F16-declared param
    // as raw f32 → garbage weights → gibberish), so only enable on Metal.
    let sel_device = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|p| args.get(p + 1))
        .cloned()
        .unwrap_or_else(|| "metal".to_string());
    if sel_device.eq_ignore_ascii_case("metal") {
        for k in [
            "RLX_QWEN3_F16_WEIGHTS",
            "RLX_QWEN3_BAKE_WEIGHTS",
            "RLX_QWEN3_GQA_NATIVE",
        ] {
            if std::env::var_os(k).is_none() {
                unsafe { std::env::set_var(k, "1") };
            }
        }
    }

    let mut device = "metal".to_string();
    let mut weights = PathBuf::from(DEFAULT_WEIGHTS);
    let mut max_tokens = 256usize;
    let mut temperature = 0.0f32; // greedy by default (deterministic)
    let mut system = "You are a helpful assistant.".to_string();
    // Context window. The model supports up to 40_960; 8192 fits long chats at
    // modest KV memory. When the conversation would overflow this, the oldest
    // turns are evicted (system prompt is always kept).
    let mut max_seq = 8192usize;
    // Pre-compile decode buckets up to this context length at startup so a
    // growing conversation doesn't stall mid-reply on a first-use bucket compile.
    // Each bucket costs ~2s to compile and the ladder is logarithmic, so this is
    // ~15-20s one-time. `--warm-ctx 0` skips it (fast startup, lazy spikes);
    // `--warm-ctx 8192` pre-warms the whole window.
    let mut warm_ctx = 256usize;
    // Run each reply until the model emits EOS instead of cutting off at
    // --max-tokens. Still bounded by the window budget so one reply can't
    // overflow the KV cache. --max-tokens then acts as the room reserved for a
    // reply when deciding eviction (a soft floor), not a hard truncation.
    let mut until_eos = false;
    // Qwen3 thinking soft-switch: append `/no_think` to each user turn so the
    // model answers directly instead of emitting a long <think> block that a
    // small (0.6B) model often abandons mid-reasoning by emitting EOS early.
    let mut no_think = false;
    // Record per-token timestamps and, after each reply, print the inter-token
    // latency distribution (p50/p90/p99/max — where stutter hides) + a sparkline,
    // and append a CSV row-per-token to `timings_path` for charting.
    let mut timings = false;
    let mut timings_path = "/tmp/qwen_chat_timings.csv".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                i += 1;
                device = args[i].clone();
            }
            "--weights" => {
                i += 1;
                weights = PathBuf::from(&args[i]);
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse()?;
            }
            "--max-seq" => {
                i += 1;
                max_seq = args[i].parse()?;
            }
            "--warm-ctx" => {
                i += 1;
                warm_ctx = args[i].parse()?;
            }
            "--until-eos" => {
                until_eos = true;
            }
            "--no-think" => {
                no_think = true;
            }
            "--timings" => {
                timings = true;
            }
            "--timings-csv" => {
                i += 1;
                timings_path = args[i].clone();
                timings = true;
            }
            "--temp" => {
                i += 1;
                temperature = args[i].parse()?;
            }
            "--system" => {
                i += 1;
                system = args[i].clone();
            }
            "-h" | "--help" => {
                eprintln!(
                    "qwen_chat [--device metal] [--weights DIR] [--max-tokens 256] \
                           [--max-seq 8192] [--warm-ctx 256] [--until-eos] [--no-think] \
                           [--timings] [--timings-csv PATH] [--temp 0.0] [--system TEXT]"
                );
                return Ok(());
            }
            other => eprintln!("[qwen_chat] ignoring unknown arg: {other}"),
        }
        i += 1;
    }
    let warm_ctx = warm_ctx.min(max_seq);

    let dev = Device::from_str(&device).map_err(|e| anyhow::anyhow!("--device {device}: {e}"))?;
    let tok = Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tok.token_to_id(t))
        .collect();

    eprintln!("[qwen_chat] loading qwen3 on {dev:?} from {weights:?} …");
    let sample = SampleOpts {
        temperature,
        ..SampleOpts::greedy()
    };
    // Force the fast F32/f16 KV-cache decode path on the bf16 `model.safetensors`
    // (matches the bench). The weights dir also ships a Q4_K_M GGUF; without this
    // the runner auto-picks the GGUF + `packed_weights` DequantMatMul path, which
    // costs ~one full prefill *per token* (appears to hang) and can't take the
    // RLX_QWEN3_F16_WEIGHTS / BAKE / GQA_NATIVE decode optimizations.
    let mut runner = Qwen3Runner::builder()
        .weights(weights.clone())
        .device(dev)
        .format(WeightFormat::Safetensors)
        .packed_weights(false)
        .max_seq(max_seq)
        .sample(sample)
        .build()?;
    // Warm up: compile the decode graph now (one-time) so the first real reply
    // streams without a mid-generation bucket-compile stall, then drop the cache
    // so turn 1 starts a clean conversation rather than a continuation of the
    // warmup prompt.
    if warm_ctx > 0 {
        eprint!(
            "[qwen_chat] warming up (compiling graph + decode buckets ≤{warm_ctx} ctx; one-time, ~15-20s — use --warm-ctx 0 to skip) … "
        );
    } else {
        eprint!("[qwen_chat] warming up (compiling graph) … ");
    }
    std::io::stderr().flush().ok();
    let warm_prompt: Vec<u32> = tok
        .encode(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
            false,
        )
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    let w0 = std::time::Instant::now();
    runner.generate_stoppable(&warm_prompt, 1, |_| false)?;
    runner.reset_cache();
    // Pre-compile the decode-bucket ladder up to `warm_ctx` so a growing
    // conversation doesn't stall ~seconds mid-reply the first time it crosses a
    // new length bucket. Bounded by the cache's resident-bytes LRU cap.
    let nb = if warm_ctx > 0 {
        runner.warm_buckets(warm_ctx)
    } else {
        0
    };
    eprintln!(
        "done ({:.1}s; pre-warmed {nb} decode buckets).",
        w0.elapsed().as_secs_f64()
    );
    let reply_mode = if until_eos {
        "replies run to EOS (window-bounded)".to_string()
    } else {
        format!("replies capped at {max_tokens} tokens")
    };
    eprintln!(
        "[qwen_chat] ready — vocab={} layers={}, window={max_seq}, {reply_mode}. Multi-turn KV cache \
         is reused across turns (feeds only new tokens); oldest turns evict when full. /reset clears, /exit quits.",
        runner.config().vocab_size,
        runner.config().num_hidden_layers
    );

    // Persistent KV cache across turns: we feed only the NEW text each turn (the
    // delta), never re-encoding the prior reply — the generator keeps the exact
    // generated ids, and BPE round-tripping the reply text could retokenize it
    // differently. `first_turn` includes the system prompt; later turns just
    // close the previous assistant turn and open the new user turn.
    let mut turns: Vec<(String, String)> = Vec::new();
    let mut first_turn = true;
    let mut last_ended_on_eos = false;
    let stdin = std::io::stdin();
    loop {
        print!("\n\x1b[1myou>\x1b[0m ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break; // EOF (Ctrl-D)
        }
        let user = line.trim().to_string();
        if user.is_empty() {
            continue;
        }
        if user == "/exit" || user == "/quit" {
            break;
        }
        if user == "/reset" {
            runner.reset_cache();
            turns.clear();
            first_turn = true;
            last_ended_on_eos = false;
            eprintln!("[qwen_chat] history cleared.");
            continue;
        }
        // Qwen3 soft-switch: appended after command handling so the switched text
        // flows uniformly into the fed delta, the eviction reconstruction, and the
        // stored turn (kept consistent with what the KV cache actually saw).
        let user = if no_think {
            format!("{user} /no_think")
        } else {
            user
        };

        let enc_len =
            |s: &str| -> usize { tok.encode(s, false).map(|e| e.get_ids().len()).unwrap_or(0) };
        // Decide fresh prefill vs incremental feed, and evict oldest turns if this
        // turn would overflow the window. Reserve room for the reply (max_tokens)
        // plus the new user tokens so the growing cache never exceeds max_seq.
        let user_est = enc_len(&user) + 8;
        let mut need_fresh = first_turn;
        if !first_turn && runner.context_len() + user_est + max_tokens >= max_seq {
            need_fresh = true;
            let mut evicted = 0usize;
            while !turns.is_empty()
                && enc_len(&render_prompt(&system, &turns, &user)) + max_tokens + 8 >= max_seq
            {
                turns.remove(0); // drop the oldest turn; system prompt is always kept
                evicted += 1;
            }
            eprintln!(
                "[qwen_chat] window full — evicted {evicted} oldest turn(s); kept system + {} turn(s), re-prefilling.",
                turns.len()
            );
            runner.reset_cache();
        }

        // Fresh (turn 1 / post-eviction): re-prefill system + kept turns + new user.
        // Otherwise feed only the delta (close previous assistant turn, open this
        // user turn) into the live KV cache — no re-prefill.
        let delta_text = if need_fresh {
            render_prompt(&system, &turns, &user)
        } else {
            let close = if last_ended_on_eos {
                "\n"
            } else {
                "<|im_end|>\n"
            };
            format!("{close}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        };
        let delta_ids: Vec<u32> = tok
            .encode(delta_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();

        // Effective generation cap. Always bounded by the remaining window budget
        // so a single reply can't push the cache past max_seq (which would exhaust
        // the decode buckets). With --until-eos we let the reply run to that budget
        // and rely on the EOS callback to stop; otherwise we cap at --max-tokens.
        let budget = max_seq.saturating_sub(runner.context_len() + delta_ids.len() + 4);
        let gen_cap = if until_eos {
            budget
        } else {
            max_tokens.min(budget)
        };

        print!("\x1b[1;36mqwen>\x1b[0m ");
        std::io::stdout().flush().ok();

        // Stream: decode incrementally and print only the newly-completed text.
        let mut out_ids: Vec<u32> = Vec::new();
        let mut decode_start = 0usize; // first token not yet printed
        let mut hit_eos = false;
        // Diagnostic: RLX_QWEN_CHAT_NOSTREAM=1 skips per-token tokenizer decode +
        // print so pure generator throughput can be isolated from streaming cost.
        let nostream = std::env::var_os("RLX_QWEN_CHAT_NOSTREAM").is_some();
        let t0 = std::time::Instant::now();
        // Pre-compile the decode buckets this reply is likely to cross so the
        // ~seconds-per-bucket compile lands here (counted in TTFT) instead of
        // stalling the token stream mid-reply (the stutter). Already-compiled
        // buckets are skipped, so steady turns pay nothing. --until-eos replies
        // (esp. with thinking) run long, so reserve a generous lookahead; the
        // compiles are one-time and shared across turns.
        let lookahead = if until_eos {
            max_tokens.max(1024)
        } else {
            max_tokens
        };
        let warm_up_to = (runner.context_len() + delta_ids.len() + lookahead).min(max_seq);
        runner.warm_buckets(warm_up_to);
        let mut ttft: Option<std::time::Duration> = None;
        // Per-token elapsed-since-t0 (only when --timings). Entry i is the wall
        // time the i-th visible token arrived; inter-token deltas give the
        // instantaneous decode latency over the reply — the stutter shows as
        // spikes, which an average would smear away.
        let mut tok_times: Vec<std::time::Duration> = Vec::new();
        runner.generate_continuation_stoppable(&delta_ids, gen_cap, |t| {
            if ttft.is_none() {
                ttft = Some(t0.elapsed()); // feed new tokens + first-token latency
            }
            if eos.contains(&t) {
                hit_eos = true;
                return false; // stop before printing the end marker
            }
            out_ids.push(t);
            if timings {
                tok_times.push(t0.elapsed());
            }
            if nostream {
                return true;
            }
            // Decode ONLY the not-yet-printed tail (usually 1-3 tokens), NOT the
            // whole growing reply — re-decoding all of out_ids every token is O(N)
            // per step = O(N²) total, which made long replies crawl and starved the
            // GPU between steps. A byte-level BPE char that spans tokens leaves a
            // trailing U+FFFD (�); hold the window (don't flush / advance) until it
            // completes, so the window stays tiny and boundaries stay clean.
            if let Ok(s) = tok.decode(&out_ids[decode_start..], true) {
                if !s.is_empty() && !s.ends_with('\u{FFFD}') {
                    print!("{s}");
                    std::io::stdout().flush().ok();
                    decode_start = out_ids.len();
                }
            }
            true
        })?;
        let dt = t0.elapsed();
        // Flush any remaining tail (reply ended mid-grapheme, or --nostream held
        // everything back — decode_start is still 0 there, so this prints it all).
        if decode_start < out_ids.len() {
            if let Ok(s) = tok.decode(&out_ids[decode_start..], true) {
                print!("{s}");
                std::io::stdout().flush().ok();
            }
        }
        let reply = tok.decode(&out_ids, true).unwrap_or_default();
        // Split the report: time-to-first-token (prefill + any graph compile) vs
        // the steady-state decode rate over the *remaining* tokens. The aggregate
        // rate is dominated by TTFT on short replies, which hides real decode tps.
        let ttft = ttft.unwrap_or(dt);
        let decode_dt = dt.saturating_sub(ttft);
        let decode_toks = out_ids.len().saturating_sub(1);
        let decode_tps = if decode_toks > 0 && decode_dt.as_secs_f64() > 0.0 {
            decode_toks as f64 / decode_dt.as_secs_f64()
        } else {
            0.0
        };
        let stop = if hit_eos {
            "eos".to_string()
        } else {
            format!("cap {gen_cap}")
        };
        println!(
            "\x1b[2m  [{} tok, {:.1} tok/s overall | TTFT {:.0}ms | decode {:.1} tok/s | stop={stop}]\x1b[0m",
            out_ids.len(),
            out_ids.len() as f64 / dt.as_secs_f64().max(1e-9),
            ttft.as_secs_f64() * 1e3,
            decode_tps,
        );
        if timings {
            report_timings(&tok_times, ttft, &timings_path, turns.len() + 1);
        }
        // The reply's exact token ids already live in the generator's KV cache;
        // next turn we feed only the new user text as a delta. Keep the text too,
        // so we can rebuild a trimmed prompt if the window later fills.
        turns.push((user, reply));
        first_turn = false;
        last_ended_on_eos = hit_eos;
    }
    eprintln!("\n[qwen_chat] bye.");
    Ok(())
}
