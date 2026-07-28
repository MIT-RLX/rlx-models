// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPLv3.

//! Pack `weights/asr` into a single `.rlxp` or legacy GGUF.
//!
//! ```text
//! rlx-asr-pack-gguf [--dir weights/asr] [--out weights/asr/model.rlxp] [--rlxp]
//! ```
//!
//! Pack sources: published tree, `RLX_ASR_PACK_SRC`, or `.cache/asr`.

use anyhow::Result;
use rlx_asr::AsrPaths;
use rlx_asr::gguf_io::{DEFAULT_GGUF_NAME, DEFAULT_RLXP_NAME, pack_asr_gguf, pack_asr_rlxp};
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
    let mut rlxp = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--rlxp" => rlxp = true,
            "-h" | "--help" => {
                eprintln!(
                    "Usage: rlx-asr-pack-gguf [--dir DIR] [--out FILE] [--rlxp]\n\
                     Default out: model.rlxp with --rlxp, else model.gguf"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    let root = dir.unwrap_or_else(|| AsrPaths::resolve().root);
    let out = out.unwrap_or_else(|| {
        root.join(if rlxp {
            DEFAULT_RLXP_NAME
        } else {
            DEFAULT_GGUF_NAME
        })
    });
    let report = if rlxp
        || out
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("rlxp"))
    {
        pack_asr_rlxp(&root, &out)?
    } else {
        pack_asr_gguf(&root, &out)?
    };
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
