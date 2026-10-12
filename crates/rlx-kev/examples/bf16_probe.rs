//! Locate where a bf16 autocast policy goes wrong on a real Qwen3.5 trunk.
//!
//! `bf16_sweep` says *which op kind* breaks; this says *which node*. It
//! re-roots the graph at one node at a time, so the answer is the first op that
//! actually produces the bad value rather than the downstream one that happens
//! to be observed. That distinction is the whole point: the
//! `ArenaWidthPolicy`/kernel-layout mismatch it was written to chase surfaced as
//! NaN at an innocent `Softplus` whose own input was intact.
//!
//! ```text
//! bf16_probe <kind> [device]        first non-finite node (linear scan)
//! RLX_KEV_DIFF=1 bf16_probe ...     first node where <device> disagrees with CPU
//! RLX_KEV_PAIR=a,b,c bf16_probe ... read those nodes in ONE run (values, not
//!                                   finiteness — re-rooting can hide an
//!                                   in-place or aliasing bug)
//! RLX_KEV_RANGE=lo-hi bf16_probe    list ops lo..=hi with dtype and finiteness
//! RLX_KEV_ROOT=i bf16_probe ...     evaluate just node i
//! ```
//!
//! `<kind>` is `compute` | `datamovement` | `elementwise` | `reduction`.
//!
use rlx_compile::precision::{AutoMixedPrecision, OpKind, Precision, PrecisionPolicy};
use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, NodeId};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "compute".into());
    let dev = std::env::args().nth(2).map_or(Device::Cpu, |d| {
        rlx_runtime::parse_device(&d).expect("device")
    });
    let kind = match which.as_str() {
        "compute" => OpKind::Compute,
        "datamovement" => OpKind::DataMovement,
        "elementwise" => OpKind::Elementwise,
        "reduction" => OpKind::Reduction,
        other => anyhow::bail!("unknown op kind {other}"),
    };

    let cfg = rlx_qwen35::synth::tiny_cfg();
    let w = rlx_qwen35::synth::synth_weights(&cfg);
    let (g, params, _) = rlx_qwen35::build_qwen35_graph_sized_ext(
        &cfg, w, 1, 64, false, false, false, false, None, true, false, true, false,
    )?;
    let mut m: HashMap<OpKind, Precision> = HashMap::new();
    m.insert(kind, Precision::BF16);
    let amp = AutoMixedPrecision::new(PrecisionPolicy::Custom(m)).run(g);

    let ids: Vec<f32> = (0..64).map(|i| (i % 20 + 1) as f32).collect();
    let (cos, sin) = rlx_qwen35::mrope_prefill_feeds(&cfg, 64, None, cfg.rope_dim_count / 2);

    let values_at = |target: NodeId, device: Device| -> Option<Vec<f32>> {
        let mut gg: Graph = amp.clone();
        gg.set_outputs(vec![target]);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut c = Session::new(device).compile(gg);
            for (n, v) in &params {
                c.set_param(n, v);
            }
            c.run(&[
                ("input_ids", &ids[..]),
                ("rope_cos", &cos[..]),
                ("rope_sin", &sin[..]),
            ])[0]
                .clone()
        }))
        .ok()
    };
    let finite_at = |target: NodeId| -> bool {
        let mut gg: Graph = amp.clone();
        gg.set_outputs(vec![target]);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut c = Session::new(dev).compile(gg);
            for (n, v) in &params {
                c.set_param(n, v);
            }
            let out = c.run(&[
                ("input_ids", &ids[..]),
                ("rope_cos", &cos[..]),
                ("rope_sin", &sin[..]),
            ]);
            let mx = out[0].iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
            eprintln!("    [node max|v| = {mx}]");
            out[0].iter().all(|x| x.is_finite())
        }));
        r.unwrap_or(false)
    };

    if std::env::var_os("RLX_KEV_DIFF").is_some() {
        // First node where `dev` disagrees with the CPU under the same policy.
        for (i, nd) in amp.nodes().iter().enumerate() {
            if nd
                .shape
                .dims()
                .iter()
                .any(|d| !matches!(d, rlx_ir::Dim::Static(_)))
            {
                continue;
            }
            let (Some(a), Some(b)) = (values_at(nd.id, Device::Cpu), values_at(nd.id, dev)) else {
                continue;
            };
            let d = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            if d > 1e-2 {
                println!("first divergence at index {i}: max|d| = {d}");
                println!("  op    : {:?}", nd.op);
                println!(
                    "  dtype : {:?} shape {:?}",
                    nd.shape.dtype(),
                    nd.shape.dims()
                );
                for (j, inp) in nd.inputs.iter().enumerate() {
                    let m = amp.node(*inp);
                    println!(
                        "  in[{j}] : {:?} {:?} {:?}",
                        m.op,
                        m.shape.dtype(),
                        m.shape.dims()
                    );
                }
                return Ok(());
            }
        }
        println!("no divergence");
        return Ok(());
    }
    if let Ok(pair) = std::env::var("RLX_KEV_PAIR") {
        let idx: Vec<usize> = pair.split(',').map(|t| t.parse().unwrap()).collect();
        let mut gg: Graph = amp.clone();
        gg.set_outputs(idx.iter().map(|&i| amp.nodes()[i].id).collect());
        let mut c = Session::new(dev).compile(gg);
        for (nm, v) in &params {
            c.set_param(nm, v);
        }
        let out = c.run(&[
            ("input_ids", &ids[..]),
            ("rope_cos", &cos[..]),
            ("rope_sin", &sin[..]),
        ]);
        for (o, i) in out.iter().zip(&idx) {
            println!(
                "  node {i} {:?}: finite={} first={:?}",
                amp.nodes()[*i].op,
                o.iter().all(|v| v.is_finite()),
                &o[..o.len().min(4)]
            );
        }
        return Ok(());
    }
    if let Ok(r) = std::env::var("RLX_KEV_RANGE") {
        let (a, b) = r.split_once('-').expect("lo-hi");
        for i in a.parse::<usize>()?..=b.parse::<usize>()? {
            let nd = &amp.nodes()[i];
            let ins: Vec<String> = nd
                .inputs
                .iter()
                .map(|x| {
                    format!(
                        "{}",
                        amp.nodes().iter().position(|q| q.id == *x).unwrap_or(9999)
                    )
                })
                .collect();
            println!(
                "{i:>4} {:<28} {:?}{:?} <- [{}]  finite={}",
                format!("{:?}", nd.op).chars().take(28).collect::<String>(),
                nd.shape.dtype(),
                nd.shape.dims().len(),
                ins.join(","),
                finite_at(nd.id)
            );
        }
        return Ok(());
    }
    if let Ok(idx) = std::env::var("RLX_KEV_ROOT") {
        let i: usize = idx.parse()?;
        let id = amp.nodes()[i].id;
        println!(
            "rooted at {i}: {:?} -> finite={}",
            amp.nodes()[i].op,
            finite_at(id)
        );
        return Ok(());
    }
    // Linear scan: the finite/non-finite predicate is NOT monotone in
    // topological order (a later op can rescue an intermediate), so a bisect
    // lands on an arbitrary bad node rather than the first one.
    let n = amp.nodes().len();
    eprintln!("{which} -> BF16 on {dev:?}: {n} nodes, scanning…");
    for (i, nd) in amp.nodes().iter().enumerate() {
        if nd
            .shape
            .dims()
            .iter()
            .any(|d| !matches!(d, rlx_ir::Dim::Static(_)))
        {
            continue;
        }
        if finite_at(nd.id) {
            continue;
        }
        println!("first non-finite node: index {i}");
        println!("  op    : {:?}", nd.op);
        println!(
            "  dtype : {:?}  shape {:?}",
            nd.shape.dtype(),
            nd.shape.dims()
        );
        for (j, inp) in nd.inputs.iter().enumerate() {
            let m = amp.node(*inp);
            println!(
                "  in[{j}] : {:?} dtype {:?} shape {:?} (finite: {})",
                m.op,
                m.shape.dtype(),
                m.shape.dims(),
                finite_at(*inp)
            );
        }
        return Ok(());
    }
    println!("all nodes finite");
    Ok(())
}
