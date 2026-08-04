//! context_1m_telemetry — a **1M-token multi-shot** interaction over the
//! disk-tiered dual-encoder KV context store, recording telemetry for EVERY part
//! of the system:
//!   • ops-inspect: op shapes / stats / FLOPs / bytes / roofline + dataflow cones
//!     (rlx-opscope, on the real qwen3 graph)              → ops_shapes.csv / ops_dataflow.csv
//!   • timing:      per-phase — ingest tok/s, query-embed ms, HNSW-retrieve ms,
//!                  block-read ms, bounded GPU/CPU decode ms/tps  → telemetry.csv
//!   • precision:   KV quant scheme, embedding dim/dtype, weight dtype → summary
//!   • error:       nan/inf counts + min/max/mean/std of retrieved K/V and decode
//!                  logits (rlx_ir::tensor_inspect)          → inspect_stats.csv / _hist.csv
//!   • recall:      semantic retrieval recall of planted needles vs 1M distractors
//!   • store/HNSW:  blocks, tokens, on-disk bytes, RAM-index bytes, per shot
//!
//! Feasibility note (honest): decoding 1M *real* tokens is infeasible, so the
//! store is populated to 1M with synthetic filler KV (+ real sentence-encoder
//! embeddings drawn from a real-text pool) and REAL encoder embeddings for the
//! planted needles; recall here is the *retrieval* recall at 1M scale (does the
//! embedding HNSW return the needle among ~31k distractors). Decode-side recall
//! (model reads spliced KV) is measured separately in `memory_probe`. Decode
//! latency is timed over the bounded resident (context-independent).
//!
//! Run (CPU; bge-small is cached, no network needed):
//!   cargo run --release -p rlx-qwen3 --example context_1m_telemetry \
//!       --features mmap-kv,dual-encoder -- --device cpu --tokens 1000000 --out ctx1m_out

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use rlx_ir::tensor_inspect::InspectLog;
use rlx_qwen3::{Qwen3Runner, SampleOpts};
use rlx_runtime::Device;
use rlx_runtime::hnsw::{HnswConfig, Metric};
use rlx_runtime::kv_context_store::{KvContextStore, Origin};
use rlx_runtime::quantized_kv::KvQuant;
use tokenizers::Tokenizer;

const DEFAULT_WEIGHTS: &str = "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b";

/// Deterministic pseudo-random unit in [-1,1) (splitmix64) — no RNG (reproducible).
fn hpm(mut z: u64) -> f32 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn l2norm(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut device = "cpu".to_string();
    let mut weights = PathBuf::from(DEFAULT_WEIGHTS);
    let mut tokens = 1_000_000usize;
    let mut block = 32usize;
    let mut topk = 4usize;
    let mut out_dir = PathBuf::from("ctx1m_out");
    let mut quant = "q4_0".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                i += 1;
                device = args[i].clone();
            }
            "--weights" => {
                i += 1;
                weights = PathBuf::from(&args[i]);
            }
            "--tokens" => {
                i += 1;
                tokens = args[i].parse()?;
            }
            "--block" => {
                i += 1;
                block = args[i].parse()?;
            }
            "--topk" => {
                i += 1;
                topk = args[i].parse()?;
            }
            "--quant" => {
                i += 1;
                quant = args[i].clone();
            }
            "--out" => {
                i += 1;
                out_dir = PathBuf::from(&args[i]);
            }
            other => eprintln!("[1m] ignoring {other}"),
        }
        i += 1;
    }
    std::fs::create_dir_all(&out_dir)?;
    let scheme = match quant.as_str() {
        "f16" => KvQuant::F16,
        "q8_0" => KvQuant::Q8_0,
        "q5_0" => KvQuant::Q5_0,
        _ => KvQuant::Q4_0,
    };
    let dev = Device::from_str(&device).map_err(|e| anyhow::anyhow!("--device {device}: {e}"))?;
    let tok = Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // ── qwen3 (decode timing over bounded resident) ──
    eprintln!("[1m] loading qwen3 on {dev:?} …");
    let mut runner = Qwen3Runner::builder()
        .weights(weights.clone())
        .device(dev)
        .format(rlx_cli::WeightFormat::Safetensors)
        .packed_weights(false)
        .max_seq(4096)
        .sample(SampleOpts::greedy())
        .build()?;
    let kv_dim = runner.config().kv_proj_dim();
    let n_layers = runner.config().num_hidden_layers;
    let head_dim = runner.config().head_dim;

    // ── (1) OPS-INSPECT: static op shapes/stats/FLOPs/bytes/roofline + dataflow ──
    // Build the real qwen3 graph and analyze it with rlx-opscope (no execution).
    {
        use rlx_qwen3::{Qwen3Config, build_qwen3_graph_sized};
        eprintln!("[1m] ops-inspect: analyzing the qwen3 graph (rlx-opscope) …");
        let cfg = Qwen3Config::from_file(&weights.join("config.json"))?;
        let mut loader = rlx_core::SafetensorsMmapLoader::open(&weights)?;
        let (g, _p) = build_qwen3_graph_sized(&cfg, &mut loader, 1, 8, true, false)?;
        let costs = rlx_opscope::shapes::op_costs(&g);
        let ridge = 30.0f64;
        let mut oc = String::from("id,op,m,k,n,flops,bytes,intensity,roofline\n");
        for c in &costs {
            oc.push_str(&format!(
                "{},{},{},{},{},{},{},{:.3},{}\n",
                c.id,
                c.op,
                c.m,
                c.k,
                c.n,
                c.flops,
                c.bytes,
                c.intensity(),
                rlx_opscope::shapes::roofline_class(c, ridge),
            ));
        }
        std::fs::write(out_dir.join("ops_shapes.csv"), &oc)?;
        let flows = rlx_opscope::dataflow::repeated_flow_patterns(&g, 2, 4, 3);
        let mut fc = String::from("depth,count,tree\n");
        for fp in &flows {
            fc.push_str(&format!("{},{},\"{}\"\n", fp.depth, fp.count, fp.tree));
        }
        std::fs::write(out_dir.join("ops_dataflow.csv"), &fc)?;
        eprintln!(
            "[1m] ops-inspect: {} ops, {} dataflow cones → ops_shapes.csv / ops_dataflow.csv",
            costs.len(),
            flows.len()
        );
    }

    // ── Dual encoder (bge-small, cached) for semantic embeddings ──
    let repo = std::env::var("RLX_QWEN3_EMBED_REPO")
        .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());
    eprintln!("[1m] loading sentence encoder {repo} …");
    let enc =
        rlx_qwen3::embedder::RlxEmbedEmbedder::from_pretrained(tok.clone(), &repo, Device::Cpu)?;
    use rlx_qwen3::embedder::BlockEmbedder as _;
    let edim = enc.dim();

    // ── Store: 1M-token disk-tiered, dual-encoder embedding index ──
    let nblocks = tokens / block;
    let store_dir = out_dir.join("store");
    // ef_search (query-time exploration width) — raise to hold HNSW recall at
    // 1M-node scale (env `RLX_1M_EF`). Also raise the embedding index's build
    // quality (ef_construction + graph degree m/m0), since a low-quality graph
    // caps recall no matter how big ef_search is.
    let ef: usize = std::env::var("RLX_1M_EF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    let ef_construction: usize = std::env::var("RLX_1M_EFC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let hm: usize = std::env::var("RLX_1M_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    eprintln!(
        "[1m] embed HNSW: ef_search={ef}, ef_construction={ef_construction}, m={hm}/m0={}",
        hm * 2
    );
    let mut store = KvContextStore::new(
        n_layers,
        kv_dim,
        scheme,
        nblocks * block + block,
        Some(&store_dir),
        HnswConfig {
            metric: Metric::L2,
            ..Default::default()
        },
        ef,
        1,
        1.0,
    )?;
    store.enable_embeddings(
        edim,
        HnswConfig {
            metric: Metric::Cosine,
            ef_construction,
            m: hm,
            m0: hm * 2,
            ..Default::default()
        },
    );

    // ── Needles: real facts + paraphrased questions (embedded by the encoder) ──
    let needles: &[(&str, &str)] = &[
        (
            "The launch code is 7731.",
            "What number is used for the rocket launch?",
        ),
        ("My cat is named Waffles.", "What did I call my pet feline?"),
        (
            "The meeting is on Thursday.",
            "Which weekday is the appointment?",
        ),
        ("My favorite color is teal.", "Which shade do I like most?"),
        (
            "I am staying in hotel room 412.",
            "Which room number is my accommodation?",
        ),
        (
            "The secret password is swordfish.",
            "What is the confidential passphrase?",
        ),
        (
            "My dog is called Rex.",
            "What is my canine companion's name?",
        ),
        (
            "The project codename is Phoenix.",
            "What is the initiative's secret name?",
        ),
        (
            "The total budget is 5000 dollars.",
            "How much money was allocated?",
        ),
        ("My flight departs at 6 AM.", "When does my plane take off?"),
        (
            "The wifi network is BlueHeron.",
            "What is the name of the wireless network?",
        ),
        (
            "The vault combination is 88-14-27.",
            "What are the numbers to open the safe?",
        ),
        (
            "My favorite author is Borges.",
            "Who is the writer I like best?",
        ),
        (
            "The conference is in Lisbon.",
            "Which city hosts the conference?",
        ),
        (
            "The server IP is 10.0.4.9.",
            "What is the machine's network address?",
        ),
        (
            "The recipe needs three eggs.",
            "How many eggs does the dish require?",
        ),
    ];
    let n_needle = needles.len();
    eprintln!("[1m] embedding {n_needle} needle facts + a filler pool …");
    let needle_doc_emb: Vec<Vec<f32>> = needles
        .iter()
        .map(|(fact, _)| enc.embed_document_text(fact))
        .collect();
    let needle_qry_emb: Vec<Vec<f32>> = needles
        .iter()
        .map(|(_, q)| enc.embed_query_text(q))
        .collect();
    // A pool of REAL filler-sentence embeddings so distractors live on the text
    // manifold (a fair recall test), reused across the 31k filler blocks with jitter.
    let filler_sents: &[&str] = &[
        "The weather today is mild and sunny.",
        "Photosynthesis converts light to energy.",
        "The river flows east toward the sea.",
        "A triangle has three sides.",
        "Coffee is a popular morning beverage.",
        "The mountain peak was covered in snow.",
        "Music can influence human emotion.",
        "The library opens at nine in the morning.",
        "Electrons carry a negative charge.",
        "The garden was full of blooming roses.",
        "Trains are an efficient mode of transport.",
        "The recipe called for fresh basil.",
        "Stars are visible on a clear night.",
        "The museum displayed ancient pottery.",
        "Rain is expected later this week.",
        "The bakery sells fresh bread daily.",
    ];
    let pool: Vec<Vec<f32>> = filler_sents
        .iter()
        .map(|s| enc.embed_document_text(s))
        .collect();

    // Needle block positions spread across the 1M context.
    let needle_block: Vec<usize> = (0..n_needle)
        .map(|i| nblocks / 25 + i * (nblocks * 9 / 10 - nblocks / 25) / n_needle.max(1))
        .collect();
    let is_needle = |b: usize| needle_block.iter().position(|&x| x == b);

    // Synthetic per-block KV (rows = a distinctive key broadcast; cheap).
    let synth_kv = |seed: u64| -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        let key: Vec<f32> = (0..kv_dim).map(|j| hpm(seed ^ (j as u64 + 1))).collect();
        let mut row = Vec::with_capacity(block * kv_dim);
        for _ in 0..block {
            row.extend_from_slice(&key);
        }
        let k: Vec<Vec<f32>> = (0..n_layers).map(|_| row.clone()).collect();
        let v = k.clone();
        (k, v, key)
    };

    // Shots at 10/33/66/100% of the block-rounded context.
    let full = nblocks * block;
    let shots = [full / 10, full / 3, (full * 2) / 3, full];
    let mut shot_i = 0usize;
    let mut appended = 0usize;
    let mut inspect = InspectLog::new();
    let mut csv = String::from(
        "shot,ctx_tokens,store_blocks,disk_gb,ram_idx_mb,ingest_tok_per_s,\
         embed_query_ms,hnsw_retrieve_ms,exact_retrieve_ms,recall_hnsw,recall_exact,\
         ret_k_absmax,ret_k_naninf,decode_tps,decode_ms\n",
    );
    let t_all = Instant::now();
    let mut t_ing = Instant::now();

    eprintln!("[1m] ingesting {tokens} tokens into the store (multi-shot at {shots:?}) …");
    for b in 0..nblocks {
        let (k, v, key) = synth_kv((b as u64) << 20);
        let id = if let Some(ni) = is_needle(b) {
            let id = store.append_block(b * block, Origin::File, ni as u32, &k, &v, &key)?;
            store.append_embed(id, &needle_doc_emb[ni]);
            id
        } else {
            let id = store.append_block(b * block, Origin::Generated, b as u32, &k, &v, &key)?;
            // Filler embedding: a random convex mix of TWO pool sentences + jitter,
            // so distractors spread diversely across the real-text manifold instead
            // of collapsing into a few mega-clusters (which would bury the needles —
            // a test artifact, not a store limit).
            let a = &pool[b % pool.len()];
            let c = &pool[(b * 7 + 3) % pool.len()];
            let w = 0.5 + 0.45 * hpm(b as u64);
            let mut e: Vec<f32> = a
                .iter()
                .zip(c)
                .map(|(x, y)| w * x + (1.0 - w) * y)
                .collect();
            for (j, x) in e.iter_mut().enumerate() {
                *x += 0.03 * hpm((b as u64) << 8 ^ j as u64);
            }
            l2norm(&mut e);
            store.append_embed(id, &e);
            id
        };
        let _ = id;
        appended += block;

        if shot_i < shots.len() && appended >= shots[shot_i] {
            let ingest_tps = appended as f64 / t_ing.elapsed().as_secs_f64().max(1e-9);
            // ── retrieval shot: per-needle timing + recall + K/V error inspect ──
            let mut hits_hnsw = 0usize;
            let mut hits_exact = 0usize;
            let mut nq = 0usize;
            let (mut t_embed, mut t_hnsw, mut t_read) = (0.0f64, 0.0f64, 0.0f64);
            let mut absmax = 0.0f32;
            let mut naninf = 0usize;
            for (ni, &nb) in needle_block.iter().enumerate() {
                if nb > b {
                    continue;
                }
                nq += 1;
                // (query-embed is precomputed; time a fresh encode to measure it)
                let te = Instant::now();
                let q = enc.embed_query_text(needles[ni].1);
                t_embed += te.elapsed().as_secs_f64() * 1e3;
                let _ = &needle_qry_emb[ni];
                // HNSW (approximate) retrieval — timed.
                let th = Instant::now();
                let got_hnsw = store.retrieve_embed(&q, topk);
                t_hnsw += th.elapsed().as_secs_f64() * 1e3;
                if got_hnsw.iter().any(|r| r.start_pos == nb * block) {
                    hits_hnsw += 1;
                }
                // EXACT (brute-force) retrieval — the correct number at this scale.
                let tr = Instant::now();
                let got = store.retrieve_embed_exact(&q, topk);
                t_read += tr.elapsed().as_secs_f64() * 1e3;
                if got.iter().any(|r| r.start_pos == nb * block) {
                    hits_exact += 1;
                }
                if let Some(r) = got.first() {
                    // Precision/error inspect of retrieved K (layer 0) + V.
                    inspect.record_tensor(shot_i, "retrieved.k", &[r.rows, kv_dim], &r.k[0], 16);
                    inspect.record_tensor(shot_i, "retrieved.v", &[r.rows, kv_dim], &r.v[0], 16);
                    for &x in &r.k[0] {
                        absmax = absmax.max(x.abs());
                        if !x.is_finite() {
                            naninf += 1;
                        }
                    }
                }
            }
            let nqf = nq.max(1) as f64;
            let recall = hits_exact as f64 / nqf;
            let recall_hnsw = hits_hnsw as f64 / nqf;

            // ── decode timing (bounded resident, context-independent) ──
            runner.reset_cache();
            let prompt: Vec<u32> = tok
                .encode("The quick brown fox jumps", false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            let n_gen = 24usize;
            let tg = Instant::now();
            runner.generate_stoppable(&prompt, n_gen, |_| true)?;
            let gdt = tg.elapsed().as_secs_f64();

            let disk_gb = store.data_bytes() as f64 / 1e9;
            let ram_mb = store.resident_index_bytes() as f64 / 1e6;
            eprintln!(
                "  [shot {}] ctx {:>8} tok | {} blk | disk {:.1} GB | ram-idx {:.0} MB | ingest {:.0} tok/s | \
                 q-embed {:.1} ms | hnsw {:.2} ms (recall {:.0}%) | exact {:.2} ms (recall {:.0}%) | decode {:.0} tps",
                shot_i,
                appended,
                store.len_blocks(),
                disk_gb,
                ram_mb,
                ingest_tps,
                t_embed / nqf,
                t_hnsw / nqf,
                recall_hnsw * 100.0,
                t_read / nqf,
                recall * 100.0,
                n_gen as f64 / gdt.max(1e-9),
            );
            csv.push_str(&format!(
                "{},{},{},{:.3},{:.1},{:.0},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.1},{:.3}\n",
                shot_i,
                appended,
                store.len_blocks(),
                disk_gb,
                ram_mb,
                ingest_tps,
                t_embed / nqf,
                t_hnsw / nqf,
                t_read / nqf,
                recall_hnsw,
                recall,
                absmax,
                naninf,
                n_gen as f64 / gdt.max(1e-9),
                gdt * 1e3 / n_gen as f64,
            ));
            shot_i += 1;
            t_ing = Instant::now();
        }
    }
    store.flush()?;

    std::fs::write(out_dir.join("telemetry.csv"), &csv)?;
    std::fs::write(out_dir.join("inspect_stats.csv"), inspect.to_csv())?;
    std::fs::write(out_dir.join("inspect_hist.csv"), inspect.to_hist_csv())?;
    std::fs::write(out_dir.join("inspect_dataflow.dot"), inspect.dataflow_dot())?;

    println!("\n════════ 1M-CONTEXT MULTI-SHOT TELEMETRY (device={device}) ════════");
    print!("{csv}");
    println!("\n── PRECISION ──");
    println!(
        "  KV store quant: {scheme:?}   (kv_dim {kv_dim}, {n_layers} layers, head_dim {head_dim})"
    );
    println!("  embedding: {repo}, dim {edim}, f32, cosine index");
    println!("  weights: f32 (CPU); RLX_QWEN3_F16_WEIGHTS is Metal-only");
    println!("── STORE ──");
    println!(
        "  {} blocks / {} tokens, disk {:.2} GB, RAM index {:.0} MB (index grows with block COUNT, not token DATA)",
        store.len_blocks(),
        store.total_tokens(),
        store.data_bytes() as f64 / 1e9,
        store.resident_index_bytes() as f64 / 1e6,
    );
    println!(
        "── ERROR ──  (retrieved-K nan/inf per shot in telemetry.csv `ret_k_naninf`; stats in inspect_stats.csv)"
    );
    println!(
        "── RECALL ──  semantic retrieval recall per shot in telemetry.csv (needle among 1M distractors)"
    );
    println!(
        "\ntotal wall: {:.1}s. Telemetry → {out_dir:?}/ (telemetry.csv, inspect_*.csv/.dot, ops_*.csv)",
        t_all.elapsed().as_secs_f64()
    );
    println!(
        "reading: retrieve_ms ~O(log N) as ctx→{tokens}; decode_tps flat (bounded resident); RAM index ≪ disk data."
    );
    println!(
        "NOTE: {store_dir:?} holds the ~{:.0}GB store — delete it when done.",
        store.data_bytes() as f64 / 1e9
    );
    Ok(())
}
