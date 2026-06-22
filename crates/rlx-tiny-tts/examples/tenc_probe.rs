//! Text-encoder tap probe: import text_encoder.onnx, optionally expose an
//! intermediate node (RLX_TAP=onnx_node_name) with no shape perturbation, run on
//! the chosen backend with the dumped phone/tone/lang/sid inputs, dump outputs.

use std::path::PathBuf;

use rlx_runtime::{AotCache, CompileOptions, DType, Device};
use rlx_tiny_tts::model::import_graph;

fn rd_i64(name: &str) -> Vec<i64> {
    let b = std::fs::read(format!("/tmp/ttsdump/{name}.f32")).expect("read dump");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
        .collect()
}
fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let device = match std::env::args().nth(1).as_deref() {
        Some(d) => rlx_runtime::parse_device(d).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => Device::Cpu,
    };
    let phone = rd_i64("phone");
    let tone = rd_i64("tone");
    let lang = rd_i64("lang");
    let sid = rd_i64("sid");
    let t = phone.len();

    let path = PathBuf::from("/tmp/tenc_probe.onnx");
    let (mut hir, params, report) = import_graph(&path, "text_encoder", t, true)?;
    eprintln!(
        "[report] lowered={} stubbed={} unsupported={:?}",
        report.lowered, report.stubbed, report.unsupported
    );
    if std::env::var("RLX_REACH").is_ok() {
        // Reachability from m_p (output index 1) back to inputs.
        let mp = hir.outputs[1];
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![mp];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            for &inp in &hir.node(id).inputs {
                stack.push(inp);
            }
        }
        let mut reaches_convq = false;
        let mut reaches_phone = false;
        for id in &seen {
            if let Some(nm) = hir.node(*id).name.as_deref() {
                if nm.contains("attn_layers.0/conv_q") {
                    reaches_convq = true;
                }
            }
            if let rlx_ir::hir::HirOp::Input { name } = &hir.node(*id).op {
                if name.contains("phone") {
                    reaches_phone = true;
                }
                eprintln!("[reach-input] {name}");
            }
        }
        eprintln!(
            "[reach] m_p reaches conv_q={reaches_convq} phone_input={reaches_phone} ({} nodes)",
            seen.len()
        );
    }
    if let Ok(tap) = std::env::var("RLX_TAP") {
        let mut found = None;
        for (i, n) in hir.nodes().iter().enumerate() {
            if n.name.as_deref() == Some(tap.as_str()) {
                found = Some(rlx_ir::hir::HirNodeId(i as u32));
            }
        }
        match found {
            Some(id) => {
                eprintln!(
                    "[tap] {tap} -> {id:?} shape {:?}",
                    hir.node(id).shape.dims()
                );
                hir.outputs.push(id);
            }
            None => eprintln!("[tap] NOT FOUND: {tap}"),
        }
    }

    if std::env::var("RLX_LISTPARAMS").is_ok() {
        for (k, v) in &params {
            if k.contains("conv_q.weight") {
                let am = v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32;
                eprintln!("[param] {k} len={} absmean={am:.4}", v.len());
            }
        }
        // Is there an HIR Param node with this exact name?
        for n in hir.nodes() {
            if n.name
                .as_deref()
                .map(|s| s.contains("conv_q"))
                .unwrap_or(false)
            {
                eprintln!(
                    "[hir-named] {:?} op={:?}",
                    n.name,
                    std::mem::discriminant(&n.op)
                );
            }
        }
    }
    let cache = AotCache::new(PathBuf::from("/tmp/tenc_probe_aot"));
    let _ = std::fs::remove_dir_all("/tmp/tenc_probe_aot");
    let key = format!("tenc_probe_{device:?}_s{t}");
    let mut compiled = cache
        .compile_hir_cached(&key, device, hir, &CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("compile: {e}"))?;
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    compiled.finalize_params();

    let bert = vec![0.0f32; 1024 * t];
    let ja_bert = vec![0.0f32; 768 * t];
    let out = compiled.run_typed(&[
        ("phone_ids", &i64_bytes(&phone), DType::I64),
        ("phone_lengths", &i64_bytes(&[t as i64]), DType::I64),
        ("tone_ids", &i64_bytes(&tone), DType::I64),
        ("language_ids", &i64_bytes(&lang), DType::I64),
        ("bert", &f32_bytes(&bert), DType::F32),
        ("ja_bert", &f32_bytes(&ja_bert), DType::F32),
        ("speaker_id", &i64_bytes(&sid), DType::I64),
    ]);
    std::fs::create_dir_all("/tmp/rlx_tenc")?;
    eprintln!("[probe] {} outputs", out.len());
    for (i, (bytes, dt)) in out.iter().enumerate() {
        // Convert every output to f32 for dumping so i64/bool taps (mask chain)
        // are comparable, not skipped.
        let v: Vec<f32> = match dt {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::I64 => bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            DType::I32 => bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            DType::Bool | DType::U8 => bytes.iter().map(|&b| b as f32).collect(),
            _ => {
                eprintln!("  out{i} dtype {dt:?} (skip)");
                continue;
            }
        };
        let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32;
        std::fs::write(format!("/tmp/rlx_tenc/out{i}.f32"), f32_bytes(&v))?;
        eprintln!("  out{i}: dt={dt:?} len={} mean|x|={mean:.4}", v.len());
    }
    Ok(())
}
