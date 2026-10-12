//! What gradient checkpointing buys on the real training graph.
//!
//! Reports the planned arena for the kev training backward at several
//! segment counts, on the shapes `decision-v7` actually produces.
use anyhow::Result;
use rlx_autodiff::checkpoint::{checkpoint_backward, suggest_segments};
use rlx_autodiff::grad_with_loss;
use rlx_compile::memory::{ArenaWidthPolicy, plan_memory_with_policy};
use rlx_ir::NodeId;
use rlx_kev::train::{TrainConfig, TrainShape, build_train_graph};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let base = &a[0];
    let rows: usize = a.get(1).map_or(8, |v| v.parse().unwrap());
    let seq: usize = a.get(2).map_or(448, |v| v.parse().unwrap());
    let opts: usize = a.get(3).map_or(8, |v| v.parse().unwrap());

    let kw = rlx_kev::weights::load_merged_base_only(std::path::Path::new(base))?;
    let cfg = kw.cfg.clone();
    let tcfg = TrainConfig::default();
    let shape = TrainShape {
        rows,
        seq,
        max_options: opts,
    };
    let g = build_train_graph(&cfg, kw.weights, shape, &tcfg)?;
    let wrt: Vec<NodeId> = g.wrt.iter().map(|s| s.node).collect();
    let bwd = grad_with_loss(&g.forward, &wrt);
    let plan =
        |x: &rlx_ir::Graph| plan_memory_with_policy(x, 256, ArenaWidthPolicy::Native).arena_size;

    let base_bytes = plan(&bwd);
    let gb = |b: usize| b as f64 / 1_073_741_824.0;
    println!(
        "shape {rows}x{seq} x{opts} options | forward {} nodes | backward {} nodes",
        g.forward.nodes().len(),
        bwd.nodes().len()
    );
    println!("  no checkpointing : {:.2} GB", gb(base_bytes));
    for segs in [4usize, 8, 16, 32, 64] {
        let ck = checkpoint_backward(&g.forward, &bwd, segs);
        println!(
            "  {segs:3} segments    : {:.2} GB  ({:.0}% of baseline, {} nodes)",
            gb(plan(&ck)),
            100.0 * plan(&ck) as f64 / base_bytes as f64,
            ck.nodes().len()
        );
    }
    // bf16 autocast, same shape.
    let kw2 = rlx_kev::weights::load_merged_base_only(std::path::Path::new(base))?;
    let bf = TrainConfig {
        bf16: true,
        ..TrainConfig::default()
    };
    let g2 = build_train_graph(&kw2.cfg, kw2.weights, shape, &bf)?;
    let wrt2: Vec<NodeId> = g2.wrt.iter().map(|s| s.node).collect();
    let bwd2 = grad_with_loss(&g2.forward, &wrt2);
    println!(
        "  bf16 autocast    : {:.2} GB  ({:.0}% of baseline)",
        gb(plan(&bwd2)),
        100.0 * plan(&bwd2) as f64 / base_bytes as f64
    );

    let s = suggest_segments(g.forward.nodes().len());
    let ck = checkpoint_backward(&g.forward, &bwd, s);
    println!(
        "  suggest_segments({s}) : {:.2} GB  ({:.0}% of baseline)",
        gb(plan(&ck)),
        100.0 * plan(&ck) as f64 / base_bytes as f64
    );
    Ok(())
}
