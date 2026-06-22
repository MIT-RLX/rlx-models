// Diagnostic: print per-layer attention tensor shapes + the
// per-layer metadata array. Confirms whether attention dims are
// uniform or if some layers have different head counts.

use anyhow::{Context, Result};
use rlx_gguf::{GgufFile, MetaValue};
use std::path::PathBuf;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .context("usage: diag_per_layer <gguf>")?
        .into();
    let f = GgufFile::from_path(&path).context("load gguf")?;

    if let Some(MetaValue::Array(arr)) = f.metadata.get("gemma4.attention.head_count_kv") {
        let vals: Vec<i64> = arr
            .iter()
            .map(|v| match v {
                MetaValue::I32(n) => *n as i64,
                MetaValue::U32(n) => *n as i64,
                _ => -1,
            })
            .collect();
        println!("head_count_kv per layer: {:?}", vals);
    }
    if let Some(MetaValue::Array(arr)) = f.metadata.get("gemma4.attention.sliding_window_pattern") {
        let vals: Vec<i64> = arr
            .iter()
            .map(|v| match v {
                MetaValue::I32(n) => *n as i64,
                MetaValue::U32(n) => *n as i64,
                MetaValue::Bool(b) => *b as i64,
                _ => -1,
            })
            .collect();
        println!("sliding_window_pattern: {:?}", vals);
    }

    println!("\nper-layer attention tensor shapes:");
    println!("layer | attn_q          | attn_k          | attn_v          | attn_output");
    for n in 0..48 {
        let q = f
            .tensors
            .get(&format!("blk.{n}.attn_q.weight"))
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        let k = f
            .tensors
            .get(&format!("blk.{n}.attn_k.weight"))
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        let v = f
            .tensors
            .get(&format!("blk.{n}.attn_v.weight"))
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        let o = f
            .tensors
            .get(&format!("blk.{n}.attn_output.weight"))
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        println!("  {n:2}  | {q:>16?} | {k:>16?} | {v:>16?} | {o:>16?}");
    }
    Ok(())
}
