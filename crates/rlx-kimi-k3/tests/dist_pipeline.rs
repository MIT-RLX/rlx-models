// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! The distributed layer pipeline over loopback TCP, against single-node
//! execution, on a synthetic checkpoint: inference (prefix + greedy
//! generation), the agreement checks, worker-side errors, the unresponsive-peer
//! timeout, and pipeline-parallel training.

mod common;
use common::TempDir;
use rlx_kimi_k3::dist::{
    Cluster, DistOptions, PipelineOptions, PipelineTrainer, ServeOptions, StageSpec, serve,
};
use rlx_kimi_k3::loader::CheckpointLoader;
use rlx_kimi_k3::runner::{run_generate, run_layer_range_streaming};
use rlx_kimi_k3::source::CheckpointSource;
use rlx_kimi_k3::synth::{SynthSpec, write_synthetic_checkpoint};
use rlx_kimi_k3::train::{Batch, TrainSpec};
use rlx_kimi_k3::train_stream::{StreamOptions, StreamTrainer};
use rlx_runtime::Device;
use std::path::Path;
use std::time::Duration;

fn synth(tag: &str) -> TempDir {
    let d = TempDir::new(&format!("rlx-kimi-dist-{tag}"));
    write_synthetic_checkpoint(&d, &SynthSpec::default()).unwrap();
    d
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Start one worker per range on loopback threads; returns the stage specs.
fn spawn_workers(dir: &Path, ranges: &[std::ops::Range<usize>]) -> Vec<StageSpec> {
    ranges
        .iter()
        .map(|r| {
            let addr = format!("127.0.0.1:{}", free_port());
            let (a, d) = (addr.clone(), dir.to_string_lossy().to_string());
            std::thread::spawn(move || serve(&a, &d, ServeOptions::default()).unwrap());
            StageSpec {
                addr,
                layers: r.clone(),
            }
        })
        .collect()
}

fn opts() -> DistOptions {
    DistOptions {
        timeout: Duration::from_secs(120),
        connect_attempts: 50,
        connect_backoff: Duration::from_millis(100),
    }
}

#[test]
fn distributed_inference_matches_single_node() {
    let dir = synth("infer");
    let src = CheckpointSource::open(&dir).unwrap();
    let tc = src.text_config().clone();
    let stages = spawn_workers(&dir, &[0..2, 2..4]);
    let mut cluster = Cluster::connect(&stages, Some(&tc), &opts()).unwrap();

    let tokens = [3u32, 14, 15, 9, 2];
    let mut ck = CheckpointLoader::open(&dir).unwrap();
    let cfg = tc.flow_config(1, tokens.len());
    let h0 = ck
        .gather_embed(
            "language_model.model.embed_tokens.weight",
            &tokens,
            cfg.hidden,
        )
        .unwrap();
    let (h_d, s_d) = cluster.prefix(h0.clone(), Vec::new()).unwrap();
    let (h_l, s_l) =
        run_layer_range_streaming(&mut ck, &tc, &cfg, h0, Vec::new(), 0, 4, Device::Cpu).unwrap();
    assert_eq!(h_d, h_l, "distributed prefix hidden");
    assert_eq!(s_d, s_l, "distributed prefix snapshots");

    let make = |seq: usize| tc.flow_config(1, seq);
    let want = run_generate(&mut ck, &tc, make, &tokens, 4, 4, Device::Cpu).unwrap();
    let got =
        rlx_kimi_k3::dist::generate(&mut cluster, &mut ck, make, &tokens, 4, Device::Cpu).unwrap();
    assert_eq!(got, want, "distributed greedy tokens");

    // Several sequences pipelined through the stages at once: each gets exactly
    // the tokens it gets alone.
    let prompts = vec![tokens.to_vec(), vec![7, 1, 30], vec![12, 12, 4, 60, 2, 8]];
    let batch =
        rlx_kimi_k3::dist::generate_batch(&mut cluster, &mut ck, make, &prompts, 4, Device::Cpu)
            .unwrap();
    for (p, got) in prompts.iter().zip(&batch) {
        let want = run_generate(&mut ck, &tc, make, p, 4, 4, Device::Cpu).unwrap();
        assert_eq!(got, &want, "batched sequence {p:?}");
    }
    let info = cluster.conns[0].ping().unwrap();
    assert_eq!(
        info["sequences"].as_u64(),
        Some(1),
        "batch states freed (the default sequence remains)"
    );

    // A gap in the stage ranges is refused before any work.
    let gap = vec![
        stages[0].clone(),
        StageSpec {
            addr: stages[1].addr.clone(),
            layers: 3..4,
        },
    ];
    drop(cluster);
    let err = Cluster::connect(&gap, Some(&tc), &opts())
        .err()
        .expect("gap must fail");
    assert!(
        err.to_string().contains("expected a non-empty range"),
        "{err}"
    );

    // A worker-side error comes back as a message; the worker keeps serving.
    let mut c = rlx_kimi_k3::dist::Conn::connect(&stages[0].addr, &opts()).unwrap();
    let mut cl = Cluster {
        stages: vec![StageSpec {
            addr: stages[0].addr.clone(),
            layers: 0..9,
        }],
        conns: vec![rlx_kimi_k3::dist::Conn::connect(&stages[0].addr, &opts()).unwrap()],
        next_seq: 1,
    };
    let Err(e) = cl.prefix(vec![0.0; cfg.hidden], Vec::new()) else {
        panic!("a range past the checkpoint must fail");
    };
    assert!(e.to_string().contains("not inside this checkpoint"), "{e}");
    assert!(c.ping().is_ok(), "worker still serves after an error");
    drop(cl);
    for s in &stages {
        rlx_kimi_k3::dist::Conn::connect(&s.addr, &opts())
            .unwrap()
            .shutdown()
            .unwrap();
    }
}

#[test]
fn silent_peer_times_out() {
    // A "worker" that accepts and then never answers.
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let (_s, _) = l.accept().unwrap();
        std::thread::sleep(Duration::from_secs(30));
    });
    let o = DistOptions {
        timeout: Duration::from_millis(500),
        ..opts()
    };
    let mut c = rlx_kimi_k3::dist::Conn::connect(&addr, &o).unwrap();
    let t = std::time::Instant::now();
    let Err(e) = c.ping() else {
        panic!("a silent peer must time out");
    };
    assert!(t.elapsed() < Duration::from_secs(5));
    assert!(e.to_string().contains("stopped responding"), "{e}");
}

#[test]
fn pipeline_training_matches_single_node_streamed() {
    let dir = synth("train");
    let src = CheckpointSource::open(&dir).unwrap();
    let tc = src.text_config().clone();
    let (batch, seq) = (1usize, 4usize);
    let micro: Vec<Batch> = (0..2)
        .map(|m| Batch {
            tokens: (0..seq)
                .map(|i| ((i * 5 + m * 11 + 1) % tc.vocab_size) as u32)
                .collect(),
            labels: (0..seq)
                .map(|i| ((i * 3 + m * 7 + 2) % tc.vocab_size) as f32)
                .collect(),
            mask: vec![1.0; seq],
        })
        .collect();
    let spec = TrainSpec::lora(4, 8.0, &[]);
    let (lr, clip, seed) = (5e-3f32, Some(0.5f32), 3u64);

    // Single node.
    let cfg = tc.flow_config(batch, seq);
    let mut st = StreamTrainer::new(
        Box::new(CheckpointSource::open(&dir).unwrap()),
        cfg,
        spec.clone(),
        StreamOptions {
            lr,
            weight_decay: 0.0,
            clip,
            device: Device::Cpu,
            seed,
            cache_layers: false,
        },
        None,
    );
    let local = st.train_step(&micro).unwrap();

    // Two-stage pipeline.
    let stages = spawn_workers(&dir, &[0..1, 1..4]);
    let mut pt = PipelineTrainer::new(
        &dir,
        &stages,
        &PipelineOptions {
            spec,
            batch,
            seq,
            lr,
            weight_decay: 0.0,
            clip,
            seed,
            cache_layers: true,
            resume: None,
            device: Device::Cpu,
            dist: opts(),
        },
    )
    .unwrap();
    let dist = pt.train_step(&micro).unwrap();
    assert!(
        (dist.loss - local.loss).abs() < 1e-5
            && (dist.grad_norm - local.grad_norm).abs() < 1e-4 * local.grad_norm,
        "pipeline step {dist} vs single node {local}"
    );
    let (trained, shapes) = pt.fetch_trained().unwrap();
    let want = st.trained();
    assert_eq!(trained.len(), want.len(), "same trained params");
    for (k, v) in want {
        let g = &trained[k];
        let err = g.iter().zip(v).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-5, "{k}: pipeline vs single node differ by {err}");
        assert!(shapes.contains_key(k), "{k} has a shape for export");
    }
    // A second step keeps working (optimizer state lives on the workers).
    let again = pt.train_step(&micro).unwrap();
    assert!(again.loss.is_finite());
    pt.cluster.shutdown().unwrap();
}
