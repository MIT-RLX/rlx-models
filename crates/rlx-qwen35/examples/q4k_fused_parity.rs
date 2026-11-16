//! CPU decode A/B: the default Q4_K path (dequant → f32 GEMV) against the
//! fused int8 kernel (`RLX_Q4K_FUSED_MIN_N`, Q8_K activations, `sdot` where the
//! CPU has `dotprod`). Same runner, same tokens: each prompt decodes greedily on
//! the default path, then replays those tokens on the fused path (teacher
//! forcing), so every step compares logits for an identical context. A free
//! greedy run on the fused path then reports where its text first diverges.
//!
//! ```text
//! cargo run -p rlx-qwen35 --example q4k_fused_parity --release -- \
//!   weights/Qwen3.5-0.8B-gguf/Qwen3.5-0.8B-Q4_K_M.gguf [tokens=48] [fused_min_n=1]
//! ```

use std::time::Instant;

use anyhow::Result;
use rlx_runtime::Device;

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "Write a haiku about the ocean.",
    "Explain what an EEG measures in one paragraph.",
    "List the first ten prime numbers.",
    "def fibonacci(n):",
    "Translate to German: The weather is nice today.",
];

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
    v.iter().map(|&x| x as f64 - lse).collect()
}

/// KL(p ‖ q) for logits `a` (reference) and `b`.
fn kl(a: &[f32], b: &[f32]) -> f64 {
    let (la, lb) = (log_softmax(a), log_softmax(b));
    la.iter().zip(&lb).map(|(&x, &y)| x.exp() * (x - y)).sum()
}

fn set_fused(on: bool, min_n: &str) {
    rlx_ir::env::set("RLX_Q4K_FUSED_MIN_N", if on { min_n } else { "18446744073709551615" });
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let weights = std::path::PathBuf::from(args.next().expect("usage: q4k_fused_parity <gguf> [tokens] [fused_min_n]"));
    let steps: usize = args.next().map(|s| s.parse().expect("tokens")).unwrap_or(48);
    let min_n = args.next().unwrap_or_else(|| "1".into());

    let mut runner = rlx_qwen35::Qwen35RunnerBuilder::default()
        .weights(&weights)
        .device(Device::Cpu)
        .max_seq(512)
        .packed_weights(true)
        .last_logits_only(true)
        .build()?;

    let (mut agree, mut total) = (0usize, 0usize);
    let (mut kl_sum, mut kl_max, mut dmax) = (0f64, 0f64, 0f32);
    let (mut t_ref, mut t_fused) = (0f64, 0f64);
    let mut free_identical = 0usize;

    for prompt in PROMPTS {
        let ids = rlx_qwen35::encode_prompt_auto(&weights, None, prompt)?;

        // Reference: default path, greedy.
        set_fused(false, &min_n);
        let mut logits = runner.prefill_seed_for_decode(&ids)?.trunk_logits;
        let (mut toks, mut ref_logits) = (Vec::new(), Vec::new());
        let t0 = Instant::now();
        for _ in 0..steps {
            let t = argmax(&logits);
            toks.push(t);
            logits = runner.decode_get_logits(t)?;
            ref_logits.push(logits.clone());
        }
        t_ref += t0.elapsed().as_secs_f64();

        // Fused path, teacher-forced on the reference tokens.
        set_fused(true, &min_n);
        runner.prefill_seed_for_decode(&ids)?;
        let t0 = Instant::now();
        for (i, &t) in toks.iter().enumerate() {
            let l = runner.decode_get_logits(t)?;
            let r = &ref_logits[i];
            total += 1;
            agree += usize::from(argmax(&l) == argmax(r));
            let k = kl(r, &l);
            kl_sum += k;
            kl_max = kl_max.max(k);
            dmax = r.iter().zip(&l).map(|(a, b)| (a - b).abs()).fold(dmax, f32::max);
        }
        t_fused += t0.elapsed().as_secs_f64();

        // Fused path, free greedy run: where does the text first differ?
        let mut logits = runner.prefill_seed_for_decode(&ids)?.trunk_logits;
        let mut diverge = None;
        for (i, &want) in toks.iter().enumerate() {
            let t = argmax(&logits);
            if t != want {
                diverge = Some(i);
                break;
            }
            logits = runner.decode_get_logits(t)?;
        }
        free_identical += usize::from(diverge.is_none());
        let text = rlx_qwen35::decode_ids_from_gguf(&weights, &toks, true).unwrap_or_default();
        println!(
            "{prompt:?}: free-run {} | {:?}",
            diverge.map_or("identical".into(), |i| format!("diverges at token {i}")),
            text.chars().take(80).collect::<String>()
        );
    }

    let n = (PROMPTS.len() * steps) as f64;
    println!("---");
    println!("steps compared        : {total}");
    println!("top-1 agreement       : {agree}/{total} ({:.2}%)", 100.0 * agree as f64 / total as f64);
    println!("KL(ref‖fused) mean/max: {:.2e} / {:.2e}", kl_sum / total as f64, kl_max);
    println!("max |Δlogit|          : {dmax:.4}");
    println!("free runs identical   : {free_identical}/{}", PROMPTS.len());
    println!("decode tok/s ref/fused: {:.2} / {:.2}", n / t_ref, n / t_fused);
    Ok(())
}
