// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Native-import feasibility probe for Parler-TTS: can `rlx-onnx-import` import
//! (and compile) the T5 `text_encoder` + `decoder` ONNX graphs? Reports op gaps
//! so the native (ort-free) port path can be scoped. No ort at runtime.
//!
//! ```text
//! cargo run -p rlx-parlertts --example import_probe --features native
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_runtime::{AotCache, CompileOptions, Device};

fn probe(dir: &Path, component: &str, named: &[(&str, usize)], seq: usize) {
    let path = dir.join("onnx").join(format!("{component}.onnx"));
    let mut named_lengths: HashMap<String, usize> =
        named.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    named_lengths.entry("batch_size".into()).or_insert(1);
    let opts = ImportOptions {
        sequence_length: seq,
        named_lengths,
        strict: false,
        ..Default::default()
    };
    print!("[{component}] import… ");
    let t0 = std::time::Instant::now();
    match build_hir_from_onnx_file(&path, opts) {
        Ok((hir, params, report, manifest)) => {
            let nodes = hir.len();
            println!(
                "OK  {nodes} nodes, {} params, {} outputs  ({:?})  [lowered={} stubbed={} skipped={}]",
                params.len(),
                manifest.outputs.len(),
                t0.elapsed(),
                report.lowered,
                report.stubbed,
                report.skipped,
            );
            if report.stubbed > 0 {
                println!("   STUBBED nodes: {:?}", report.stubbed_nodes);
            }
            if !report.unsupported.is_empty() {
                println!("   UNSUPPORTED ops: {:?}", report.unsupported);
            }
            if std::env::var_os("RLX_DUMP_NARROW").is_some() {
                use std::collections::BTreeMap;
                let mut hist: BTreeMap<String, usize> = BTreeMap::new();
                let mut narrows = 0;
                for n in hir.nodes() {
                    let dbg = format!("{:?}", n.op);
                    let key = dbg.split([' ', '{', '(']).next().unwrap_or("?").to_string();
                    *hist.entry(key).or_default() += 1;
                    if dbg.contains("Narrow") {
                        narrows += 1;
                        if narrows <= 12 {
                            println!("   NARROW {:?}  shape={:?}", n.op, n.shape.dims());
                        }
                    }
                }
                println!(
                    "   op histogram (top): {:?}",
                    hist.iter().rev().take(14).collect::<Vec<_>>()
                );
                println!("   total Narrow ops: {narrows}");
            }
            print!("[{component}] compile(Cpu)… ");
            let cache = AotCache::new(std::env::temp_dir().join("rlx_parler_probe"));
            let key = format!("parler_{component}_probe");
            let t1 = std::time::Instant::now();
            match cache.compile_hir_cached(&key, Device::Cpu, hir, &CompileOptions::default()) {
                Ok(_) => println!("OK  ({:?})", t1.elapsed()),
                Err(e) => println!("FAIL: {e}"),
            }
        }
        Err(e) => println!("FAIL: {e}"),
    }
}

fn main() -> Result<()> {
    let dir =
        std::env::var("RLX_PARLER_DIR").unwrap_or_else(|_| "weights/tts/parlertts".to_string());
    let dir = Path::new(&dir);
    if !dir.join("onnx/decoder.onnx").exists() {
        eprintln!(
            "skip: no Parler ONNX at {} (set RLX_PARLER_DIR)",
            dir.display()
        );
        return Ok(());
    }
    // text_encoder: T5 relative-position-bias is baked at length 128 → bind seq=128.
    probe(
        dir,
        "text_encoder",
        &[("sequence_length", 128), ("t", 128)],
        128,
    );
    // decoder: decoder_input_ids [b,9,t], encoder_hidden_states [b,et=128,1024]
    probe(
        dir,
        "decoder",
        &[
            ("t", 8),
            ("et", 128),
            ("sequence_length", 8),
            ("encoder_sequence_length", 128),
        ],
        8,
    );
    Ok(())
}
