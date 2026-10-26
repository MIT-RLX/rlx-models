//! Human-readable and JSON summaries of a TFLite model.

use std::collections::BTreeMap;
use std::fmt::Write;

use anyhow::Result;
use serde_json::{Value, json};

use crate::proto;
use crate::tflite::{BufferLoc, TfliteModel};

/// Op-name → count across all subgraphs, sorted by name.
pub fn op_histogram(model: &TfliteModel) -> BTreeMap<String, usize> {
    let mut hist = BTreeMap::new();
    for sg in &model.subgraphs {
        for op in &sg.operators {
            let name = model
                .operator_codes
                .get(op.opcode_index as usize)
                .map_or_else(|| format!("opcode#{}", op.opcode_index), |c| c.name());
            *hist.entry(name).or_insert(0) += 1;
        }
    }
    hist
}

fn metadata_text(model: &TfliteModel, buffer: u32) -> String {
    let bytes = match model.buffer_bytes(buffer) {
        Ok(b) => b,
        Err(e) => return format!("<unreadable: {e}>"),
    };
    if bytes.is_empty() {
        return "<empty>".into();
    }
    if proto::is_printable(bytes) {
        return format!("{:?}", String::from_utf8_lossy(bytes));
    }
    if proto::looks_like_message(bytes)
        && let Ok(s) = proto::render(bytes)
    {
        return format!("protobuf ({} B)\n{s}", bytes.len());
    }
    format!("<{} bytes>", bytes.len())
}

/// Multi-line text report. `tensor_limit` caps the per-tensor listing.
pub fn report(model: &TfliteModel, tensor_limit: usize) -> Result<String> {
    let mut s = String::new();
    writeln!(
        s,
        "TFLite v{}  {} B  description={:?}",
        model.version,
        model.file_len(),
        model.description
    )?;
    let external = model
        .buffers
        .iter()
        .filter(|b| matches!(b, BufferLoc::External { .. }))
        .count();
    writeln!(
        s,
        "buffers: {} ({} external)  subgraphs: {}  opcodes: {}",
        model.buffers.len(),
        external,
        model.subgraphs.len(),
        model.operator_codes.len()
    )?;

    writeln!(s, "\nmetadata:")?;
    for m in &model.metadata {
        let text = metadata_text(model, m.buffer);
        writeln!(s, "  {} -> {}", m.name, text.replace('\n', "\n    "))?;
    }

    let hist = op_histogram(model);
    if !hist.is_empty() {
        writeln!(s, "\nops:")?;
        for (name, n) in &hist {
            writeln!(s, "  {n:6}  {name}")?;
        }
    }

    for (i, sg) in model.subgraphs.iter().enumerate() {
        writeln!(
            s,
            "\nsubgraph {i} {:?}: {} tensors, {} ops, inputs {:?}, outputs {:?}",
            sg.name,
            sg.tensors.len(),
            sg.operators.len(),
            sg.inputs,
            sg.outputs
        )?;
        for t in sg.tensors.iter().take(tensor_limit) {
            let stored = model.tensor_bytes(t).map(<[u8]>::len).unwrap_or(0);
            let quant = t.quantization.as_ref().map_or(String::new(), |q| {
                format!(
                    "  q[dim={} scales={} zp={}]",
                    q.quantized_dimension,
                    q.scale.len(),
                    q.zero_point.len()
                )
            });
            writeln!(
                s,
                "  {:<72} {:<6} {:?} {} B{}",
                t.name, t.dtype, t.shape, stored, quant
            )?;
        }
        if sg.tensors.len() > tensor_limit {
            writeln!(s, "  … {} more", sg.tensors.len() - tensor_limit)?;
        }
    }
    Ok(s)
}

/// Machine-readable manifest: every tensor's name, shape, dtype and size.
pub fn manifest_json(model: &TfliteModel) -> Value {
    let subgraphs: Vec<Value> = model
        .subgraphs
        .iter()
        .map(|sg| {
            let tensors: Vec<Value> = sg
                .tensors
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "shape": t.shape,
                        "dtype": t.dtype.to_string(),
                        "bytes": model.tensor_bytes(t).map(<[u8]>::len).unwrap_or(0),
                        "quant_scales": t.quantization.as_ref().map_or(0, |q| q.scale.len()),
                        "quant_dim": t.quantization.as_ref().map(|q| q.quantized_dimension),
                    })
                })
                .collect();
            json!({ "name": sg.name, "tensors": tensors, "ops": sg.operators.len() })
        })
        .collect();
    let metadata: Vec<Value> = model
        .metadata
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "bytes": model.buffer_bytes(m.buffer).map(<[u8]>::len).unwrap_or(0),
            })
        })
        .collect();
    json!({
        "version": model.version,
        "description": model.description,
        "file_bytes": model.file_len(),
        "metadata": metadata,
        "ops": op_histogram(model),
        "subgraphs": subgraphs,
    })
}
