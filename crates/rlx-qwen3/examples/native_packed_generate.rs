//! native_packed_generate — end-to-end runner check.
//!
//! In packed mode the runner now builds a **native packed generator**: no F32
//! weights, and the prefill-seed + decode graphs lower projections to
//! `Op::DequantMatMul` straight from the GGUF (a real m=1 decode loop, not the
//! old O(N²) repeated-prefill). Greedy generation must therefore produce the
//! exact same token stream as the F32 runner.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3 --example native_packed_generate -- \
//!       /Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf

use rlx_qwen3::Qwen3Runner;
use rlx_runtime::Device;

fn parse_device(s: &str) -> Device {
    match s.to_ascii_lowercase().as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cpu" => Device::Cpu,
        other => panic!("unknown device '{other}' (use cpu|metal|mlx)"),
    }
}

fn run_gen(
    gguf: &str,
    device: Device,
    packed: bool,
    prompt: &[u32],
    n: usize,
) -> anyhow::Result<Vec<u32>> {
    let max_seq = (prompt.len() + n + 16).max(256);
    let mut runner = Qwen3Runner::builder()
        .weights(gguf)
        .device(device)
        .packed_weights(packed)
        .max_seq(max_seq)
        .build()?;
    let mut out = Vec::with_capacity(n);
    // Warm: first token pays prefill + first bucket compile. Time the rest.
    let mut first = true;
    let prefill_start = std::time::Instant::now();
    let mut prefill_secs = 0f64;
    let mut t0 = std::time::Instant::now();
    let mut timed = 0usize;
    runner.generate(prompt, n, |t| {
        if first {
            first = false;
            prefill_secs = prefill_start.elapsed().as_secs_f64(); // time-to-first-token = prefill
            t0 = std::time::Instant::now(); // reset clock after the seed token
        } else {
            timed += 1;
        }
        out.push(t);
    })?;
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[e2e]   {} generate: prefill(TTFT {} tok)={prefill_secs:.3}s | {timed} steady tokens in {secs:.2}s = {:.1} tok/s (device={device:?})",
        if packed { "packed" } else { "f32   " },
        prompt.len(),
        timed as f64 / secs.max(1e-9),
    );
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let gguf = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf".to_string()
    });
    let device = std::env::args()
        .nth(2)
        .map(|s| parse_device(&s))
        .unwrap_or(Device::Metal);
    // Long prompt so decode starts in the TOP compile bucket → no mid-stream
    // bucket-boundary recompiles → the timed window is pure steady-state decode.
    // E2E_PROMPT_LEN overrides the prompt length to probe long-context decode.
    let prompt_len: u32 = std::env::var("E2E_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(140);
    // E2E_PROMPT_TEXT: tokenize a real coherent prompt (via the gguf-dir's
    // tokenizer.json) → a confident greedy stream, the fair parity test (the
    // synthetic prompts below are knife-edge/degenerate worst cases).
    let prompt: Vec<u32> = if let Ok(text) = std::env::var("E2E_PROMPT_TEXT") {
        let tokpath = std::path::Path::new(&gguf)
            .parent()
            .unwrap()
            .join("tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&tokpath).expect("load tokenizer.json");
        let ids = tok.encode(text, false).expect("encode").get_ids().to_vec();
        eprintln!("[e2e] tokenized prompt: {} tokens", ids.len());
        ids
    } else if std::env::var("E2E_PROMPT_VARIED").is_ok() {
        (1u32..=prompt_len)
            .map(|i| ((i.wrapping_mul(2_654_435_761)) % 151_000).max(1))
            .collect()
    } else {
        (1u32..=prompt_len).collect()
    };
    let n = 96;
    eprintln!(
        "[e2e] GGUF: {gguf} | device={device:?} | prompt={} tokens",
        prompt.len()
    );

    // E2E_PACKED_ONLY: profile/measure the packed runner alone (skip the F32
    // reference + comparison).
    if std::env::var("E2E_PACKED_ONLY").is_ok() {
        eprintln!("[e2e] packed runner ONLY greedy generate…");
        let toks = run_gen(&gguf, device, true, &prompt, n)?;
        eprintln!("[e2e] packed toks: {toks:?}");
        return Ok(());
    }
    eprintln!("[e2e] F32 runner (dequant-at-load) greedy generate…");
    let f32_toks = run_gen(&gguf, device, false, &prompt, n)?;
    let packed = std::env::var("E2E_PACKED").is_ok();
    let packed_toks = if packed {
        eprintln!("[e2e] packed runner (native m=1 decode) greedy generate…");
        run_gen(&gguf, device, true, &prompt, n)?
    } else {
        eprintln!("[e2e] (packed run skipped; set E2E_PACKED=1 to include)");
        f32_toks.clone()
    };

    eprintln!("[e2e] f32   : {f32_toks:?}");
    eprintln!("[e2e] packed: {packed_toks:?}");

    if f32_toks == packed_toks {
        eprintln!("[e2e] PASS ✓ (native packed generate == F32 greedy stream, {n} tokens)");
        Ok(())
    } else {
        let diverge = f32_toks.iter().zip(&packed_toks).position(|(a, b)| a != b);
        anyhow::bail!("E2E FAIL: streams diverge at index {diverge:?}");
    }
}
