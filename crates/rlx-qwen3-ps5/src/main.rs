// RLX models — distributed inference.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Qwen3 GGUF decode on the PS5's CPU, timed: load, lock the weights, prefill
//! a prompt, stream the answer, report prefill and decode tokens/s.
//!
//! A payload starts with no arguments, so settings come from
//! `/data/rlx-ps5/qwen3.conf` (`KEY=VALUE` lines; an environment variable of
//! the same name wins, so the binary also runs on a host):
//!
//! ```text
//! MODEL=/data/rlx-ps5/models/Qwen3-0.6B-Q4_K_M.gguf
//! N=64            # tokens to generate (stops early at <|im_end|>)
//! MAX_SEQ=512
//! PACKED=1        # keep K-quant weights packed (0 = dequant to F32: ~4x memory)
//! PROMPT_IDS=…    # comma-separated ids; default: a short chat question
//! RLX_…=…         # any RLX option, e.g. RLX_PACKED_LM_HEAD=0
//! ```
//!
//! Output text is decoded with the GGUF's own vocabulary, so no tokenizer
//! library is needed. Results also go to `/data/rlx-ps5/qwen3.txt`.

use anyhow::{Context, Result, anyhow};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_qwen3::Qwen3Runner;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::time::Instant;

const CONF: &str = "/data/rlx-ps5/qwen3.conf";
const OUT: &str = "/data/rlx-ps5/qwen3.txt";
const DEFAULT_MODEL: &str = "/data/rlx-ps5/models/Qwen3-0.6B-Q4_K_M.gguf";

/// `<|im_start|>user\nWhat is the capital of France? Answer in one
/// sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` in the
/// Qwen3 vocabulary (thinking off).
const DEFAULT_PROMPT: &[u32] = &[
    151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 21806, 304, 825, 11652, 13, 151645, 198,
    151644, 77091, 198, 151667, 271, 151668, 271,
];

fn main() {
    let mut log = String::new();
    let code = match run(&mut log) {
        Ok(()) => 0,
        Err(e) => {
            let _ = writeln!(log, "error: {e:#}");
            println!("error: {e:#}");
            1
        }
    };
    let _ = std::fs::create_dir_all("/data/rlx-ps5");
    let _ = std::fs::write(OUT, &log);
    rlx_ps5::exit(code);
}

fn setting(conf: &HashMap<String, String>, key: &str) -> Option<String> {
    rlx_ir::env::read(key).ok().or_else(|| conf.get(key).cloned())
}

fn run(log: &mut String) -> Result<()> {
    let conf: HashMap<String, String> = std::fs::read_to_string(CONF)
        .unwrap_or_default()
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let model = setting(&conf, "MODEL").unwrap_or_else(|| DEFAULT_MODEL.into());
    let n: usize = setting(&conf, "N")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let max_seq: usize = setting(&conf, "MAX_SEQ")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let packed = setting(&conf, "PACKED").is_none_or(|v| v != "0");
    let prompt: Vec<u32> = match setting(&conf, "PROMPT_IDS") {
        Some(s) => s
            .split(',')
            .map(|t| t.trim().parse().context("PROMPT_IDS"))
            .collect::<Result<_>>()?,
        None => DEFAULT_PROMPT.to_vec(),
    };

    let mut say = |s: String| {
        println!("{s}");
        let _ = writeln!(log, "{s}");
    };
    for (k, v) in conf.iter().filter(|(k, _)| k.starts_with("RLX_")) {
        rlx_ps5::set_default(k, v);
    }
    // Memory first: without these, every compiled graph (prefill + each decode
    // bucket) holds an F32 copy of the 151936-row embedding table, and
    // superseded buckets stay resident — ~4 GB at 96 tokens.
    // - the head reads the packed `output.weight` (or the packed embedding
    //   table when the file has none) instead of an F32 copy;
    rlx_ps5::set_default("RLX_PACKED_LM_HEAD", "1");
    // - token rows are dequantized from the mmapped GGUF and fed in;
    rlx_ps5::set_default("RLX_QWEN3_HOST_EMBED", "1");
    // - only the current decode bucket stays compiled (decode never revisits
    //   a smaller one within a generation).
    rlx_ps5::set_default("RLX_KV_CACHE_MAX_RESIDENT", "1");
    // Attention reads the 8 shared KV heads in place instead of copying the
    // whole cache out to 16 heads every token (`repeat_kv`): O(context) work
    // gone, same tokens.
    rlx_ps5::set_default("RLX_QWEN3_GQA_NATIVE", "1");
    // One decode bucket for the whole context: a rung costs a compile stall
    // and its own weights, and padded keys are masked and skipped, so a wide
    // bucket costs ~1 ms/token over a tight one.
    rlx_ps5::set_default("RLX_QWEN3_DECODE_BUCKET_MIN", &max_seq.to_string());
    // The prompt goes through that decode graph too, one token per step: no
    // prefill graph to compile and hold a second copy of the weights. A
    // short prompt is also faster this way; a long one is slower.
    rlx_ps5::set_default("RLX_QWEN3_PREFILL_DECODE", "1");
    // A Q4_0 file stores a few `ffn_down` tensors as Q4_1; keep them packed
    // too (the int8 GEMV covers it) rather than as F32 matrices.
    rlx_ps5::set_default("RLX_GGUF_PACKED_Q4_1", "1");
    // Run each step on a pool worker: the ~225 parallel kernels of a decoded
    // token then join by spinning instead of parking and waking this thread.
    rlx_ps5::set_default("RLX_CPU_EXEC_IN_POOL", "1");
    let workers = rlx_ps5::init();
    say(format!(
        "qwen3-ps5: {workers} CPU worker(s), {} MB available, model {model}",
        rlx_ps5::available_memory().map_or(-1, |b| (b >> 20) as i64)
    ));

    let vocab = Vocab::from_gguf(&model)?;
    let t = Instant::now();
    let mut runner = Qwen3Runner::builder()
        .weights(&model)
        .max_seq(max_seq)
        .packed_weights(packed)
        .build()
        .with_context(|| format!("load {model}"))?;
    say(format!(
        "load: {:.2} s ({}), peak resident {} MB",
        t.elapsed().as_secs_f64(),
        if packed { "packed K-quant" } else { "F32" },
        rlx_ps5::peak_resident_bytes() >> 20
    ));
    // Compile the decode graph now rather than on the first decoded token,
    // where it stalls the answer for most of a second. Its bucket covers the
    // whole context (`RLX_QWEN3_DECODE_BUCKET_MIN` above), so this is the
    // only compile of the run.
    let t = Instant::now();
    let warmed = runner.warm_buckets(prompt.len());
    say(format!(
        "decode graph: {warmed} compiled in {:.2} s",
        t.elapsed().as_secs_f64()
    ));
    // Weights must stay resident: the system pages a payload out while the
    // browser is in front, and decode then runs at disk speed.
    match rlx_ps5::lock_resident() {
        Ok(b) => say(format!("locked: {} MB resident", b >> 20)),
        Err(e) => say(format!("not locked: {e}")),
    }

    let eos = vocab.eos;
    let mut text = String::new();
    let mut first: Option<Instant> = None;
    let mut stamps: Vec<Instant> = Vec::with_capacity(n);
    let mut count = 0usize;
    let t0 = Instant::now();
    runner.generate_stoppable(&prompt, n, |tok| {
        let now = Instant::now();
        first.get_or_insert(now);
        stamps.push(now);
        count += 1;
        if Some(tok) == eos {
            return false;
        }
        let piece = vocab.decode(tok);
        print!("{piece}");
        let _ = std::io::stdout().flush();
        text.push_str(&piece);
        true
    })?;
    let end = Instant::now();
    println!();
    let first = first.ok_or_else(|| anyhow!("no tokens generated"))?;
    let prefill = (first - t0).as_secs_f64();
    let decode = (end - first).as_secs_f64();
    say(format!("answer: {}", text.trim()));
    say(format!(
        "prefill: {} tokens in {:.3} s ({:.1} tok/s, incl. the first decode step)",
        prompt.len(),
        prefill,
        prompt.len() as f64 / prefill
    ));
    if count > 1 {
        // A step that crosses into a bigger KV bucket compiles that bucket's
        // decode graph first (seconds), so report the steady rate — the median
        // step — next to the average, and count those stalls.
        let mut steps: Vec<f64> = stamps
            .windows(2)
            .map(|w| (w[1] - w[0]).as_secs_f64())
            .collect();
        steps.sort_by(f64::total_cmp);
        let median = steps[steps.len() / 2];
        let stalls = steps.iter().filter(|&&d| d > 10.0 * median).count();
        say(format!(
            "decode: {} tokens in {:.3} s = {:.2} tok/s average, {:.2} tok/s steady \
             (median step {:.1} ms; {stalls} bucket-compile stall(s))",
            count - 1,
            decode,
            (count - 1) as f64 / decode,
            1.0 / median,
            median * 1e3
        ));
    }
    // The load figure above is taken before any graph compiles; each compiled
    // decode bucket and the prefill graph add their own weight arenas.
    say(format!(
        "peak resident over the run: {} MB",
        rlx_ps5::peak_resident_bytes() >> 20
    ));
    Ok(())
}

/// Token-id → text from the GGUF's own `tokenizer.ggml.tokens`, undoing
/// GPT-2 byte-level BPE (each byte stored as a printable code point).
struct Vocab {
    tokens: Vec<String>,
    unbyte: HashMap<char, u8>,
    eos: Option<u32>,
}

impl Vocab {
    fn from_gguf(path: &str) -> Result<Self> {
        let g = GgufFile::from_path_mmap(path).with_context(|| format!("open {path}"))?;
        let tokens = match g.metadata.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(v)) => v
                .iter()
                .map(|m| match m {
                    MetaValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => return Err(anyhow!("{path}: no tokenizer.ggml.tokens")),
        };
        let eos = match g.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(MetaValue::U32(v)) => Some(*v),
            Some(MetaValue::I32(v)) => u32::try_from(*v).ok(),
            Some(MetaValue::U64(v)) => u32::try_from(*v).ok(),
            _ => None,
        };
        Ok(Self {
            tokens,
            unbyte: gpt2_unbyte(),
            eos,
        })
    }

    fn decode(&self, id: u32) -> String {
        let Some(tok) = self.tokens.get(id as usize) else {
            return String::new();
        };
        // Special tokens (`<|im_end|>`, `<think>`) are stored verbatim.
        if tok.starts_with("<|") || tok.starts_with("<think") || tok.starts_with("</think") {
            return String::new();
        }
        let bytes: Vec<u8> = tok
            .chars()
            .filter_map(|c| self.unbyte.get(&c).copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Inverse of GPT-2's `bytes_to_unicode`: printable Latin-1 bytes map to
/// themselves, the other 68 bytes to U+0100 onward, in order.
fn gpt2_unbyte() -> HashMap<char, u8> {
    let mut map = HashMap::new();
    let mut next = 256u32;
    for b in 0u32..256 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            next += 1;
            next - 1
        };
        map.insert(char::from_u32(c).unwrap(), b as u8);
    }
    map
}
