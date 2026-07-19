// RLX models — distributed MNIST training CLI.
// SPDX-License-Identifier: GPL-3.0-only

//! `rlx-vision-bench [--world N] [--epochs E] [--batch B] [--lr L] [--hidden H]`
//!                 `[--model mlp|cnn] [--async] [--augment] [--seed S]`
//!                 `[--no-deterministic]`  (deterministic gradient reduce is on
//!                 by default: reproducible across runs and node counts)
//!
//! Two run modes:
//!
//! * **single machine** (default): spawns `--world N` data-parallel ranks as
//!   threads over a loopback transport.
//! * **multi-node**: if `RANK`/`WORLD` are set in the environment, this process
//!   is exactly one rank and joins the real cross-machine group via
//!   `Node::from_env()` (`PEERS=host:port,…` or `DISCOVER=1`). Launch one
//!   process per rank on each machine.
//!
//! Every rank prints a banner showing which host/pid owns it and its data
//! shard, then per-rank throughput + comm/compute metrics.

use rlx_vision_bench::{
    Config, DatasetKind, ModelKind, Report, datasets, harness, run_alldev, run_coordinate,
    run_distributed, run_node_from_env, training_device,
};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let mut world = 2u32;
    let mut coordinate = false;
    let mut alldev = false;
    let mut cfg = Config::default();
    let mut dataset = DatasetKind::Mnist;
    let mut suite: Option<Vec<DatasetKind>> = None;
    let mut model_set = false;
    let mut max_train: Option<usize> = None;
    let mut max_test: Option<usize> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let val = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_default()
    };
    while i < args.len() {
        match args[i].as_str() {
            "--world" => world = val(&mut i).parse().unwrap_or(world),
            "--epochs" => cfg.epochs = val(&mut i).parse().unwrap_or(cfg.epochs),
            "--batch" => cfg.batch = val(&mut i).parse().unwrap_or(cfg.batch),
            "--lr" => cfg.lr = val(&mut i).parse().unwrap_or(cfg.lr),
            "--hidden" => cfg.hidden = val(&mut i).parse().unwrap_or(cfg.hidden),
            "--seed" => cfg.seed = val(&mut i).parse().unwrap_or(cfg.seed),
            "--async" => cfg.async_overlap = true,
            "--augment" => cfg.augment = true,
            "--deterministic" => cfg.deterministic = true,
            "--no-deterministic" => cfg.deterministic = false,
            "--coordinate" => coordinate = true,
            "--alldev" => alldev = true,
            "--max-train" => max_train = val(&mut i).parse().ok(),
            "--max-test" => max_test = val(&mut i).parse().ok(),
            "--dataset" => {
                dataset = DatasetKind::parse(&val(&mut i)).unwrap_or_else(|| {
                    eprintln!("unknown --dataset; using mnist");
                    DatasetKind::Mnist
                })
            }
            // `--suite [ds1,ds2,…]`: sweep the (dataset × model) matrix. With no
            // list, all datasets. Restrict models with `--model`.
            "--suite" => {
                let next = args.get(i + 1).cloned().unwrap_or_default();
                let list = if !next.is_empty() && !next.starts_with("--") {
                    i += 1;
                    next
                } else {
                    String::new()
                };
                suite = Some(if list.is_empty() {
                    DatasetKind::all().to_vec()
                } else {
                    list.split(',').filter_map(DatasetKind::parse).collect()
                });
            }
            "--model" => {
                cfg.model = ModelKind::parse(&val(&mut i)).unwrap_or(ModelKind::Mlp);
                model_set = true;
            }
            other => eprintln!("ignoring unknown arg: {other}"),
        }
        i += 1;
    }

    // Configuration-sweep harness: (dataset × model) on one device.
    if let Some(ds_list) = suite {
        let opts = harness::SuiteOpts {
            datasets: ds_list,
            models: if model_set {
                vec![cfg.model]
            } else {
                ModelKind::all().to_vec()
            },
            device: training_device(),
            epochs: cfg.epochs,
            batch: cfg.batch,
            hidden: cfg.hidden,
            augment: cfg.augment,
            max_train: max_train.or(Some(20_000)),
            max_test: max_test.or(Some(4_000)),
        };
        let rows = harness::run_suite(&opts);
        harness::print_table(&rows);
        return;
    }

    eprintln!("loading {} …", dataset.name());
    let data = Arc::new(datasets::load(dataset, None, None).unwrap_or_else(|e| {
        eprintln!("{} load failed: {e}", dataset.name());
        std::process::exit(1);
    }));
    cfg.spec = data.spec;

    let multinode = std::env::var("RANK").is_ok();
    eprintln!(
        "dataset={} · train {} / test {} · input {}×{}×{} · classes {} · model={} · batch={} · lr={} · hidden={} · epochs={} · async={} · augment={} · deterministic={} · mode={}",
        dataset.name(),
        data.train.len(),
        data.test.len(),
        cfg.spec.c,
        cfg.spec.h,
        cfg.spec.w,
        cfg.spec.classes,
        cfg.model.name(),
        cfg.batch,
        cfg.lr,
        cfg.hidden,
        cfg.epochs,
        cfg.async_overlap,
        cfg.augment,
        cfg.deterministic,
        if multinode {
            if matches!(std::env::var("TOPOLOGY").as_deref(), Ok("iroh"))
                || matches!(std::env::var("RLX_TRANSPORT").as_deref(), Ok("iroh"))
            {
                "multi-node over iroh (relays + discovery)"
            } else {
                "multi-node over tcp/thunderbolt"
            }
        } else {
            "single-machine threads"
        },
    );

    let t0 = Instant::now();

    // Intra-node all-backends: one node, every backend a lane, no networking.
    if alldev {
        let rep = run_alldev(&cfg, &data.train, &data.test);
        println!(
            "intra-node all-backends MNIST · test accuracy = {:.2}% · {:.1}s",
            rep.accuracy * 100.0,
            t0.elapsed().as_secs_f64(),
        );
        return;
    }

    // Master/coordinator role: this process owns the MNIST code and drives a
    // fleet of GENERIC workers (rlx-collectives `dist_node --example`,
    // MODE=trainserve) that have no model code. Only rank 0 runs this.
    if coordinate {
        use rlx_vision_bench::hostname;
        let node = rlx_driver::node::Node::from_env().unwrap_or_else(|e| {
            eprintln!("coordinator needs RANK/WORLD/PEERS: {e}");
            std::process::exit(2);
        });
        let (rank, w) = (node.rank(), node.world());
        let group = node.connect().unwrap_or_else(|e| {
            eprintln!("coordinator connect failed: {e}");
            std::process::exit(1);
        });
        eprintln!(
            "  [master rank {rank}/{w}] host={} — driving {} generic worker(s) via ship-graph training",
            hostname(),
            w.saturating_sub(1)
        );
        let rep = run_coordinate(&cfg, group, &data.train, &data.test);
        print_metrics(&[(rank, rep)], w);
        println!(
            "distributed MNIST (ship-graph, generic workers) · {w} ranks · master accuracy = {:.2}% · {:.1}s",
            rep.accuracy * 100.0,
            t0.elapsed().as_secs_f64(),
        );
        return;
    }

    if multinode {
        // One process = one rank. Join the real cross-machine group.
        match run_node_from_env(&cfg, &data.train, &data.test) {
            Ok((rank, w, rep)) => {
                print_metrics(&[(rank, rep)], w);
                if rank == 0 {
                    println!(
                        "distributed MNIST (multi-node) · {w} ranks · rank0 accuracy = {:.2}% · {:.1}s",
                        rep.accuracy * 100.0,
                        t0.elapsed().as_secs_f64(),
                    );
                }
            }
            Err(e) => {
                eprintln!("multi-node join failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Single-machine: spawn `world` ranks as threads.
    let reports = run_distributed(&cfg, world, &data.train, &data.test);
    let indexed: Vec<(u32, Report)> = reports
        .iter()
        .enumerate()
        .map(|(r, rep)| (r as u32, *rep))
        .collect();
    print_metrics(&indexed, world);
    println!(
        "distributed MNIST · {world} ranks · test accuracy = {:.2}% · {:.1}s",
        reports[0].accuracy * 100.0,
        t0.elapsed().as_secs_f64(),
    );
}

/// Print the per-rank metrics table + an aggregate throughput line.
fn print_metrics(reports: &[(u32, Report)], world: u32) {
    eprintln!("─── per-rank metrics ──────────────────────────────────────────");
    eprintln!("  rank │  samples │  wall  │ compute │  comm   │  throughput");
    let mut total_thru = 0.0f64;
    for (rank, r) in reports {
        let thru = r.samples as f64 / r.wall_s.max(1e-9);
        total_thru += thru;
        let comm_pct = if r.compute_s + r.comm_s > 0.0 {
            100.0 * r.comm_s / (r.compute_s + r.comm_s)
        } else {
            0.0
        };
        eprintln!(
            "  {:>4} │ {:>8} │ {:>5.1}s │ {:>6.2}s │ {:>5.2}s │ {:>7.0} samp/s  (comm {:.0}%)",
            rank, r.samples, r.wall_s, r.compute_s, r.comm_s, thru, comm_pct,
        );
    }
    if reports.len() == world as usize && world > 1 {
        eprintln!(
            "  aggregate throughput across {world} ranks: {:.0} samp/s",
            total_thru
        );
    }
    eprintln!("───────────────────────────────────────────────────────────────");
}
