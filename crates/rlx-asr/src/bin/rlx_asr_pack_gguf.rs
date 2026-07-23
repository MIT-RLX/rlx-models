// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Pack `weights/asr` into a single GGUF.
//!
//! ```text
//! rlx-asr-pack-gguf [--dir weights/asr] [--out weights/asr/model.gguf]
//! ```
//!
//! Pack sources: published tree, `RLX_ASR_PACK_SRC`, or `.cache/asr`.

use anyhow::Result;
use rlx_asr::gguf_io::{pack_asr_gguf, DEFAULT_GGUF_NAME};
use rlx_asr::AsrPaths;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("Usage: rlx-asr-pack-gguf [--dir DIR] [--out FILE]");
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    let root = dir.unwrap_or_else(|| AsrPaths::resolve().root);
    let out = out.unwrap_or_else(|| root.join(DEFAULT_GGUF_NAME));
    let report = pack_asr_gguf(&root, &out)?;
    println!(
        "{}",
        serde_json::json!({
            "out": report.path,
            "n_tensors": report.n_tensors,
            "bytes": report.bytes,
            "mb": (report.bytes as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0,
            "skipped": report.skipped,
        })
    );
    Ok(())
}
