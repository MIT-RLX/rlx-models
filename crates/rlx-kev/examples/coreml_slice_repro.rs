//! Minimal repro for the CoreML `narrow`-on-axis-1 defect hit by the
//! Qwen3.5 Gated DeltaNet prologue.
//!
//! The GDN block projects a fused `[tokens, 8208]` tensor and narrows out
//! three slices (`[.., 0..6144]`, `[.., 6144..8192]`, `[.., 8192..8208]`).
//! CoreML emits `slice_by_size` with the right MIL type annotation, but
//! MPSGraph materializes the wrong shape and the model fails verification.
//!
//! ```text
//! cargo run --release -p rlx-kev --features coreml --example coreml_slice_repro
//! ```
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn try_case(device: Device, tokens: usize, wide: usize, start: usize, len: usize) -> String {
    let mut g = Graph::new("coreml_slice_repro");
    let x = g.input("x", Shape::new(&[tokens, wide], DType::F32));
    let s = g.narrow_(x, 1, start, len);
    let a = g.sigmoid(s);
    let out = g.reshape_(a, vec![1, tokens as i64, len as i64]);
    g.set_outputs(vec![out]);
    run_graph(device, g, tokens, wide)
}

/// The shape the GDN prologue actually builds: one matmul against a weight
/// that is itself a `concat` of three consts, then three narrows off the
/// result. This is what the builder emits when it fuses qkv / z / beta into
/// a single projection.
fn try_fused(device: Device, tokens: usize, hidden: usize, parts: [usize; 3]) -> String {
    let mut g = Graph::new("coreml_fused_repro");
    let wide: usize = parts.iter().sum();
    let x = g.input("x", Shape::new(&[tokens, hidden], DType::F32));
    let mut ws = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        let w = g.param(format!("w{i}"), Shape::new(&[hidden, *p], DType::F32));
        ws.push(w);
    }
    let wcat = g.concat_(ws, 1);
    let y = g.matmul(x, wcat, Shape::new(&[tokens, wide], DType::F32));
    let mut outs = Vec::new();
    let mut off = 0usize;
    for p in parts {
        let sl = g.narrow_(y, 1, off, p);
        let act = g.sigmoid(sl);
        outs.push(g.reshape_(act, vec![1, tokens as i64, p as i64]));
        off += p;
    }
    g.set_outputs(outs);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut c = Session::new(device).compile(g);
        for (i, p) in parts.iter().enumerate() {
            let w = vec![0.01f32; hidden * p];
            c.set_param(&format!("w{i}"), &w);
        }
        let xs = vec![0.1f32; tokens * hidden];
        c.run(&[("x", &xs[..])]).len()
    }));
    match r {
        Ok(n) => format!("ok ({n} outputs)"),
        Err(_) => "FAILED".to_string(),
    }
}

fn run_graph(device: Device, g: Graph, tokens: usize, wide: usize) -> String {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut c = Session::new(device).compile(g);
        let xs = vec![0.1f32; tokens * wide];
        let out = c.run(&[("x", &xs[..])]);
        out[0].len()
    }));
    match r {
        Ok(n) => format!("ok (out {n})"),
        Err(_) => "FAILED".to_string(),
    }
}

fn main() {
    let cases = [
        ("narrow start=0   len=6144", 0usize, 6144usize),
        ("narrow start=6144 len=2048", 6144, 2048),
        ("narrow start=8192 len=16", 8192, 16),
        ("narrow start=16   len=16", 16, 16),
    ];
    println!("--- narrow only");
    for (label, start, len) in cases {
        let cpu = try_case(Device::Cpu, 128, 8208, start, len);
        #[cfg(feature = "coreml")]
        let ane = try_case(Device::Ane, 128, 8208, start, len);
        #[cfg(not(feature = "coreml"))]
        let ane = "skipped (no coreml feature)".to_string();
        println!("{label:28}  cpu: {cpu:14}  ane: {ane}");
    }

    println!("--- concat(weights) -> matmul -> narrow  (the GDN prologue shape)");
    for parts in [[6144usize, 2048, 16], [64, 32, 16]] {
        let cpu = try_fused(Device::Cpu, 128, 1024, parts);
        #[cfg(feature = "coreml")]
        let ane = try_fused(Device::Ane, 128, 1024, parts);
        #[cfg(not(feature = "coreml"))]
        let ane = "skipped".to_string();
        println!("parts {parts:?}          cpu: {cpu:14}  ane: {ane}");
    }
}
