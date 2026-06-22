//! Flow-graph bisection probe: import /tmp/flow_probe.onnx (flow.onnx + boundary
//! tensors appended as extra graph outputs), run it on the chosen backend with the
//! dumped z_p/y_mask/g inputs, and write every output to /tmp/rlx_flow/<name>.f32
//! for correlation against ORT.

use std::path::{Path, PathBuf};

use rlx_runtime::{AotCache, CompileOptions, DType, Device};
use rlx_tiny_tts::model::import_graph;

fn rd(name: &str) -> Vec<f32> {
    let b = std::fs::read(format!("/tmp/ttsdump/{name}.f32")).expect("read dump");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let device = match std::env::args().nth(1).as_deref() {
        Some(d) => rlx_runtime::parse_device(d).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => Device::Mlx,
    };
    // Length = z_p frames (z_p is [1, 80, L]); falls back to 85.
    let length = std::fs::read("/tmp/ttsdump/z_p.f32")
        .map(|b| (b.len() / 4) / 80)
        .unwrap_or(85);
    eprintln!("[probe] flow length={length}");
    let path = PathBuf::from("/tmp/flow_probe.onnx");

    let (mut hir, params, report) = import_graph(&path, "flow", length, true)?;
    // Perturbation-free tap: import_graph already ran propagate_shapes with only the
    // real graph output, so appending an HIR output here cannot change any resolved
    // shape. Find the node by its ONNX name and expose it as the last output.
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
                    "[tap] {tap} -> node {id:?} shape {:?}",
                    hir.node(id).shape.dims()
                );
                hir.outputs.push(id);
            }
            None => eprintln!("[tap] NODE NOT FOUND: {tap}"),
        }
    }
    eprintln!(
        "[probe] stubbed={} unsupported={:?}",
        report.stubbed, report.unsupported
    );
    eprintln!("[probe] graph outputs (HIR order):");
    for (i, o) in hir.outputs.iter().enumerate() {
        eprintln!("    [{i}] {o:?}");
    }

    let cache = AotCache::new(PathBuf::from("/tmp/flow_probe_aot"));
    let _ = std::fs::remove_dir_all("/tmp/flow_probe_aot");
    let cache_key = format!("flow_probe_{device:?}_s{length}");
    eprintln!("[probe] compiling...");
    let t0 = std::time::Instant::now();
    let mut compiled = cache
        .compile_hir_cached(&cache_key, device, hir, &CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("compile: {e}"))?;
    eprintln!("[probe] compiled in {:?}", t0.elapsed());
    for (k, v) in &params {
        if k.contains("rev_perm") {
            eprintln!("[probe] {k} -> {} elems", v.len());
        }
    }
    eprintln!("[probe] setting {} params...", params.len());
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    eprintln!("[probe] finalizing...");
    compiled.finalize_params();
    eprintln!("[probe] finalized");

    eprintln!(
        "[probe] params: {} (rev_perm: {})",
        params.len(),
        params.keys().filter(|k| k.contains("rev_perm")).count()
    );
    let z_p = rd("z_p");
    let y_mask = rd("y_mask");
    let g = rd("g");
    eprintln!("[probe] running...");
    let out = compiled.run_typed(&[
        ("z_p", &f32_bytes(&z_p), DType::F32),
        ("y_mask", &f32_bytes(&y_mask), DType::F32),
        ("g", &f32_bytes(&g), DType::F32),
    ]);

    let dir = Path::new("/tmp/rlx_flow");
    std::fs::create_dir_all(dir)?;
    eprintln!("[probe] {} outputs returned", out.len());
    for (i, (bytes, dt)) in out.iter().enumerate() {
        anyhow::ensure!(*dt == DType::F32, "output {i} dtype {dt:?}");
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32;
        std::fs::write(dir.join(format!("out{i}.f32")), f32_bytes(&v))?;
        eprintln!("    out{i}: len={} mean|x|={mean:.4}", v.len());
    }
    eprintln!("[probe] wrote outputs to {}", dir.display());
    Ok(())
}
