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

//! Speed + precision sweep across Gemma 4 12B GGUF quantizations.
//!
//! Discovers `gemma-4-12b-it-*.gguf` under `RLX_GEMMA4_FIXTURE` (or downloads
//! via `RLX_GEMMA4_QUANTS` when set). Compares each quant against the best
//! available reference (Q8_0 > UD-Q8_K_XL > Q6_K > Q4_K_M).
//!
//! ```bash
//! RLX_GEMMA4_FIXTURE=/path/to/gemma4-12B-it \
//!   RLX_GEMMA4_QUANTS=all \
//!   RLX_GEMMA4_MAX_SEQ=128 RLX_GEMMA4_REAL_DECODE_STEPS=8 \
//!   cargo test -p rlx-gemma --release --features apple-silicon \
//!   --test gemma4_gguf_quant_sweep bench_all_quants -- --nocapture --test-threads=1
//!
//! # Optional: move Metal prefill JIT to session build (RLX_GEMMA_PACKED_WARM_PREFILL=1)
//! ```

mod gemma4_bench_common;

use anyhow::{Context, Result, bail};
use gemma4_bench_common::bench_device_from_env;
use rlx_gemma::{GemmaConfig, GemmaConfigSource, GemmaRunner, encode_prompt_auto};
use rlx_qwen3::SampleOpts;
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const PROMPT: &str = "User: What is quantum entanglement? Answer in two sentences.\nAssistant:";

const REF_PRIORITY: &[&str] = &[
    "Q8_0",
    "UD-Q8_K_XL",
    "Q6_K",
    "UD-Q6_K_XL",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_0",
    "Q5_1",
    "UD-Q5_K_XL",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_0",
    "Q4_1",
    "UD-Q4_K_XL",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
];

/// Q* filenames on `unsloth/gemma-4-12B-it-GGUF` (verified against HF).
const ALL_Q_QUANTS: &[&str] = &[
    "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
];

fn quant_list_from_env() -> Option<Vec<String>> {
    let raw = std::env::var("RLX_GEMMA4_QUANTS").ok()?;
    if raw.eq_ignore_ascii_case("all") || raw == "*" {
        return Some(ALL_Q_QUANTS.iter().map(|s| (*s).to_string()).collect());
    }
    Some(
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os("RLX_GEMMA4_FIXTURE").map(PathBuf::from)
}

fn bench_max_seq() -> usize {
    std::env::var("RLX_GEMMA4_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128)
}

fn decode_steps() -> usize {
    std::env::var("RLX_GEMMA4_REAL_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
}

fn quant_label(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("gemma-4-12b-it-"))
        .and_then(|s| s.strip_suffix(".gguf"))
        .unwrap_or("unknown")
        .to_string()
}

fn discover_quants(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("gguf")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("gemma-4-12b-it-"))
        })
        .collect();
    out.sort_by_cached_key(|p| quant_label(p));
    if let Some(want) = quant_list_from_env() {
        out.retain(|p| want.iter().any(|w| quant_label(p) == *w));
    }
    out
}

fn maybe_download_quants(dir: &Path) -> Result<()> {
    let Some(want) = quant_list_from_env() else {
        return Ok(());
    };
    std::fs::create_dir_all(dir)?;
    for q in &want {
        let fname = format!("gemma-4-12b-it-{q}.gguf");
        let dest = dir.join(&fname);
        if dest.is_file() {
            continue;
        }
        eprintln!("[gemma4 quant sweep] downloading {fname} …");
        let status = Command::new("hf")
            .args([
                "download",
                "unsloth/gemma-4-12B-it-GGUF",
                &fname,
                "--local-dir",
                dir.to_str().unwrap_or("."),
            ])
            .status()
            .context("hf download")?;
        if !status.success() {
            eprintln!("[gemma4 quant sweep] warning: failed to download {fname}");
        }
    }
    Ok(())
}

fn top_k_argmax(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().take(k).map(|i| (i, logits[i])).collect()
}

fn logits_metrics(ref_logits: &[f32], logits: &[f32]) -> (f32, f32, bool) {
    let n = ref_logits.len().min(logits.len());
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let d = (logits[i] - ref_logits[i]).abs();
        max_abs = max_abs.max(d);
        sum_sq += d * d;
    }
    let rmse = (sum_sq / n.max(1) as f32).sqrt();
    let top1_match = top_k_argmax(ref_logits, 1)[0].0 == top_k_argmax(logits, 1)[0].0;
    (max_abs, rmse, top1_match)
}

#[derive(Clone)]
struct QuantRow {
    label: String,
    disk_gb: f64,
    session_ms: f64,
    prefill_ms: f64,
    steady_ms_per_tok: f64,
    e2e_tok_s: f64,
    max_logit_diff: f32,
    logit_rmse: f32,
    top1_match: bool,
    greedy_match: bool,
}

fn ref_artifact_dir(dir: &Path) -> PathBuf {
    dir.join(".gemma4_quant_ref")
}

fn save_ref_artifacts(dir: &Path, logits: &[f32], tokens: &[u32]) -> Result<()> {
    let d = ref_artifact_dir(dir);
    std::fs::create_dir_all(&d)?;
    let mut logits_bytes = Vec::with_capacity(logits.len() * 4);
    for &v in logits {
        logits_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(d.join("ref_logits.f32"), logits_bytes)?;
    let mut tokens_bytes = Vec::with_capacity(tokens.len() * 4);
    for &v in tokens {
        tokens_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(d.join("ref_tokens.u32"), tokens_bytes)?;
    Ok(())
}

fn load_ref_artifacts(dir: &Path) -> Result<(Vec<f32>, Vec<u32>)> {
    let d = ref_artifact_dir(dir);
    let logits_bytes = std::fs::read(d.join("ref_logits.f32"))?;
    if !logits_bytes.len().is_multiple_of(4) {
        bail!("ref_logits.f32 length mismatch");
    }
    let logits: Vec<f32> = logits_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let tokens_bytes = std::fs::read(d.join("ref_tokens.u32"))?;
    if !tokens_bytes.len().is_multiple_of(4) {
        bail!("ref_tokens.u32 length mismatch");
    }
    let tokens: Vec<u32> = tokens_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((logits, tokens))
}

fn row_to_line(r: &QuantRow) -> String {
    format!(
        "{}\t{:.4}\t{:.2}\t{:.2}\t{:.2}\t{:.4}\t{:.6}\t{:.6}\t{}\t{}",
        r.label,
        r.disk_gb,
        r.session_ms,
        r.prefill_ms,
        r.steady_ms_per_tok,
        r.e2e_tok_s,
        r.max_logit_diff,
        r.logit_rmse,
        u8::from(r.top1_match),
        u8::from(r.greedy_match),
    )
}

fn row_from_line(line: &str) -> Result<QuantRow> {
    let p: Vec<&str> = line.split('\t').collect();
    if p.len() != 10 {
        bail!("bad QUANT_ROW ({})", p.len());
    }
    Ok(QuantRow {
        label: p[0].into(),
        disk_gb: p[1].parse()?,
        session_ms: p[2].parse()?,
        prefill_ms: p[3].parse()?,
        steady_ms_per_tok: p[4].parse()?,
        e2e_tok_s: p[5].parse()?,
        max_logit_diff: p[6].parse()?,
        logit_rmse: p[7].parse()?,
        top1_match: p[8] != "0",
        greedy_match: p[9] != "0",
    })
}

fn spawn_bench_quant(gguf: &Path, fixture: &Path) -> Result<QuantRow> {
    let exe = std::env::current_exe().context("current_exe")?;
    let output = Command::new(exe)
        .args(["bench_quant_isolated", "--exact", "--nocapture"])
        .env("RLX_GEMMA4_FIXTURE", fixture)
        .env("RLX_GEMMA4_QUANT_PATH", gguf)
        .env("RLX_GEMMA4_MAX_SEQ", bench_max_seq().to_string())
        .env("RLX_GEMMA4_REAL_DECODE_STEPS", decode_steps().to_string())
        .output()
        .context("spawn bench_quant_isolated")?;
    if !output.status.success() {
        bail!(
            "bench_quant_isolated failed for {}: {}",
            quant_label(gguf),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for line in combined.lines() {
        if let Some(rest) = line.strip_prefix("QUANT_ROW\t") {
            return row_from_line(rest);
        }
    }
    bail!("QUANT_ROW missing for {}", quant_label(gguf));
}

fn bench_one_quant(
    device: Device,
    gguf: &Path,
    config_path: &Path,
    tokenizer: &Path,
    ref_logits: Option<&[f32]>,
    ref_tokens: Option<&[u32]>,
) -> Result<QuantRow> {
    let max_seq = bench_max_seq();
    let steps = decode_steps();
    let label = quant_label(gguf);
    let disk_gb = gguf
        .metadata()
        .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0);

    let t_session = Instant::now();
    let mut runner = GemmaRunner::builder()
        .weights(gguf)
        .device(device)
        .max_seq(max_seq)
        .stream(false)
        .sample(SampleOpts::greedy())
        .packed_weights(true)
        .config(GemmaConfigSource::JsonFile(config_path.to_path_buf()))
        .build()?;
    let session_ms = t_session.elapsed().as_secs_f64() * 1000.0;

    let ids = encode_prompt_auto(gguf, Some(tokenizer), PROMPT)?;

    let t_prefill = Instant::now();
    let logits = runner.predict_logits(&ids)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let t_decode = Instant::now();
    let mut greedy_out = Vec::new();
    runner.generate(&ids, steps, |t| greedy_out.push(t))?;
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let steady_ms_per_tok = if steps > 1 {
        decode_ms / steps as f64
    } else {
        decode_ms
    };
    let e2e_tok_s = greedy_out.len() as f64 / (decode_ms / 1000.0);

    let (max_logit_diff, logit_rmse, top1_match) = if let Some(r) = ref_logits {
        logits_metrics(r, &logits)
    } else {
        (0.0, 0.0, true)
    };
    let greedy_match = ref_tokens.is_none_or(|r| r == greedy_out.as_slice());

    Ok(QuantRow {
        label,
        disk_gb,
        session_ms,
        prefill_ms,
        steady_ms_per_tok,
        e2e_tok_s,
        max_logit_diff,
        logit_rmse,
        top1_match,
        greedy_match,
    })
}

fn pick_reference(quants: &[PathBuf]) -> Option<&Path> {
    for pref in REF_PRIORITY {
        if let Some(p) = quants.iter().find(|q| quant_label(q) == *pref) {
            return Some(p.as_path());
        }
    }
    quants.first().map(|p| p.as_path())
}

fn print_table(rows: &[QuantRow], ref_label: &str, prompt_tokens: usize) {
    eprintln!(
        "\n[gemma4 quant sweep] reference={ref_label} prompt_tokens={prompt_tokens} decode_steps={}",
        decode_steps()
    );
    eprintln!(
        "{:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>5} {:>5}",
        "quant", "GB", "session", "prefill", "ms/tok", "e2e", "max|Δ|", "rmse", "top1", "greedy"
    );
    for r in rows {
        eprintln!(
            "{:<18} {:>6.1} {:>7.0}ms {:>7.0}ms {:>7.1}ms {:>7.2} {:>8.2} {:>8.4} {:>5} {:>5}",
            r.label,
            r.disk_gb,
            r.session_ms,
            r.prefill_ms,
            r.steady_ms_per_tok,
            r.e2e_tok_s,
            r.max_logit_diff,
            r.logit_rmse,
            if r.top1_match { "yes" } else { "no" },
            if r.greedy_match { "yes" } else { "no" },
        );
    }
}

#[test]
fn bench_all_quants() -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!("[gemma4 quant sweep] RLX_GEMMA4_FIXTURE unset — skip");
        return Ok(());
    };
    maybe_download_quants(&dir)?;

    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        bail!("missing config.json in fixture dir");
    }
    let tokenizer = dir.join("tokenizer.json");
    let cfg = GemmaConfig::from_file(&config_path)?;

    let quants = discover_quants(&dir);
    if quants.is_empty() {
        eprintln!("[gemma4 quant sweep] no gemma-4-12b-it-*.gguf in {dir:?} — skip");
        return Ok(());
    }

    let device = bench_device_from_env();

    eprintln!(
        "\n[gemma4 quant sweep] {} quants on {device:?} layers={} hidden={} max_seq={}",
        quants.len(),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        bench_max_seq()
    );

    let ref_path = pick_reference(&quants).context("no reference quant")?;
    let ref_label = quant_label(ref_path);
    eprintln!("[gemma4 quant sweep] building reference from {ref_label} …");

    let ref_ids = encode_prompt_auto(ref_path, Some(tokenizer.as_path()), PROMPT)?;
    let t_session = Instant::now();
    let mut ref_runner = GemmaRunner::builder()
        .weights(ref_path)
        .device(device)
        .max_seq(bench_max_seq())
        .stream(false)
        .sample(SampleOpts::greedy())
        .packed_weights(true)
        .config(GemmaConfigSource::JsonFile(config_path.clone()))
        .build()?;
    let ref_session_ms = t_session.elapsed().as_secs_f64() * 1000.0;

    let t_prefill = Instant::now();
    let ref_logits = ref_runner.predict_logits(&ref_ids)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let t_decode = Instant::now();
    let ref_tokens = ref_runner.generate(&ref_ids, decode_steps(), |_| {})?;
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let ref_row = QuantRow {
        label: ref_label.clone(),
        disk_gb: ref_path
            .metadata()
            .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0),
        session_ms: ref_session_ms,
        prefill_ms,
        steady_ms_per_tok: decode_ms / decode_steps().max(1) as f64,
        e2e_tok_s: ref_tokens.len() as f64 / (decode_ms / 1000.0),
        max_logit_diff: 0.0,
        logit_rmse: 0.0,
        top1_match: true,
        greedy_match: true,
    };
    drop(ref_runner);
    save_ref_artifacts(&dir, &ref_logits, &ref_tokens)?;

    let mut rows = vec![ref_row];
    for gguf in &quants {
        if gguf.as_path() == ref_path {
            continue;
        }
        eprintln!(
            "\n[gemma4 quant sweep] --- {} (isolated) ---",
            quant_label(gguf)
        );
        rows.push(spawn_bench_quant(gguf, &dir)?);
    }

    rows.sort_by_cached_key(|r| r.label.clone());
    print_table(&rows, &ref_label, ref_ids.len());
    Ok(())
}

#[test]
fn bench_quant_isolated() -> Result<()> {
    let Some(fixture) = fixture_dir() else {
        eprintln!("[gemma4 quant sweep] bench_quant_isolated: RLX_GEMMA4_FIXTURE unset — skip");
        return Ok(());
    };
    let Some(quant_path) = std::env::var_os("RLX_GEMMA4_QUANT_PATH").map(PathBuf::from) else {
        eprintln!("[gemma4 quant sweep] bench_quant_isolated: RLX_GEMMA4_QUANT_PATH unset — skip");
        return Ok(());
    };
    let config_path = fixture.join("config.json");
    let tokenizer = fixture.join("tokenizer.json");
    let (ref_logits, ref_tokens) = load_ref_artifacts(&fixture)?;
    let row = bench_one_quant(
        bench_device_from_env(),
        &quant_path,
        &config_path,
        &tokenizer,
        Some(&ref_logits),
        Some(&ref_tokens),
    )?;
    println!("QUANT_ROW\t{}", row_to_line(&row));
    Ok(())
}
