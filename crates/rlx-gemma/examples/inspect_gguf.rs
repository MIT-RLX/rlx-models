use anyhow::Result;
use rlx_gguf::GgufFile;
use std::path::PathBuf;

fn main() -> Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: inspect_gguf <path>")
        .into();
    let f = GgufFile::from_path(&path)?;
    println!("metadata keys parsed by rlx-gguf: {}", f.metadata.len());
    let mut keys: Vec<&str> = f.metadata.keys().map(String::as_str).collect();
    keys.sort();
    for k in keys {
        let v = match f.metadata.get(k) {
            Some(rlx_gguf::MetaValue::U32(n)) => format!("{n}"),
            Some(rlx_gguf::MetaValue::I32(n)) => format!("{n}"),
            Some(rlx_gguf::MetaValue::U64(n)) => format!("{n}"),
            Some(rlx_gguf::MetaValue::I64(n)) => format!("{n}"),
            Some(rlx_gguf::MetaValue::F32(x)) => format!("{x}"),
            Some(rlx_gguf::MetaValue::F64(x)) => format!("{x}"),
            Some(rlx_gguf::MetaValue::Bool(b)) => format!("{b}"),
            Some(rlx_gguf::MetaValue::String(s)) => {
                if s.len() > 80 {
                    format!("\"{}…\" ({} chars)", &s[..60], s.len())
                } else {
                    format!("\"{s}\"")
                }
            }
            Some(rlx_gguf::MetaValue::Array(a)) => format!("array({})", a.len()),
            _ => "?".to_string(),
        };
        println!("  {k} = {v}");
    }

    println!("\ntensors: {}", f.tensors.len());
    let mut by_dtype: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    let mut rows: Vec<(String, String, Vec<usize>, u64)> = Vec::with_capacity(f.tensors.len());
    for (name, t) in &f.tensors {
        let nelem: u64 = t.shape.iter().map(|d| *d as u64).product();
        let bytes_per_elem: u64 = match t.dtype {
            rlx_gguf::GgmlType::F32 => 4,
            rlx_gguf::GgmlType::F16 => 2,
            rlx_gguf::GgmlType::BF16 => 2,
            _ => 1,
        };
        let approx_bytes = nelem * bytes_per_elem;
        rows.push((
            name.clone(),
            format!("{:?}", t.dtype),
            t.shape.clone(),
            approx_bytes,
        ));
        let e = by_dtype.entry(format!("{:?}", t.dtype)).or_default();
        e.0 += 1;
        e.1 += approx_bytes;
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.3));
    println!("\ntop 25 tensors by approx-size:");
    for (n, dt, sh, b) in rows.iter().take(25) {
        println!("  {:>8} MB  {:>8}  {:?}  {}", b / (1024 * 1024), dt, sh, n);
    }
    // Also: every blk.0.* tensor, sorted by name. Reveals if there are
    // arch-specific tensors like altup_proj or per_layer_input that
    // rlx-gemma's loader is silently ignoring.
    // Print attn_k/attn_q/attn_v/layer_output_scale shapes for every layer
    // to spot per-layer-type differences (SWA vs FULL in Gemma 4).
    println!("\nattn weights per layer:");
    let mut attn_rows: Vec<(usize, String, Vec<usize>, String)> = f
        .tensors
        .iter()
        .filter_map(|(n, t)| {
            let lid_pos = n.strip_prefix("blk.")?;
            let dot = lid_pos.find('.')?;
            let lid: usize = lid_pos[..dot].parse().ok()?;
            let kind = &lid_pos[dot + 1..];
            if matches!(
                kind,
                "attn_q.weight"
                    | "attn_k.weight"
                    | "attn_v.weight"
                    | "attn_output.weight"
                    | "layer_output_scale.weight"
                    | "attn_q_norm.weight"
                    | "attn_k_norm.weight"
                    | "attn_norm.weight"
                    | "ffn_norm.weight"
                    | "post_attention_norm.weight"
                    | "post_ffw_norm.weight"
            ) {
                Some((
                    lid,
                    kind.to_string(),
                    t.shape.clone(),
                    format!("{:?}", t.dtype),
                ))
            } else {
                None
            }
        })
        .collect();
    attn_rows.sort();
    for (lid, kind, shape, dt) in &attn_rows {
        println!("  blk.{lid:>2}.{kind:<28} {dt:>6} {shape:?}");
    }

    println!("\nblk.0.* tensors (full):");
    let mut blk0: Vec<(String, String, Vec<usize>)> = f
        .tensors
        .iter()
        .filter(|(n, _)| {
            n.starts_with("blk.0.")
                || n.starts_with("model.layers.0.")
                || n == &&String::from("token_embd.weight")
        })
        .map(|(n, t)| (n.clone(), format!("{:?}", t.dtype), t.shape.clone()))
        .collect();
    blk0.sort();
    for (n, dt, sh) in &blk0 {
        println!("  {:>8}  {:?}  {}", dt, sh, n);
    }
    println!("\nby dtype:");
    for (dt, (n, b)) in by_dtype {
        println!("  {dt:>8}  {n:>4} tensors  ~{} MB", b / (1024 * 1024));
    }
    Ok(())
}
