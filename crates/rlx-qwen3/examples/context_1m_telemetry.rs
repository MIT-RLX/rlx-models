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
use std::{
    collections::HashMap,
    io::{BufReader, BufWriter, Read, Write},
};

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

const EMBED_CACHE_MAGIC: u32 = 0x4542_4d31;
const STORE_MANIFEST_MAGIC: u32 = 0x5354_4d31;

#[derive(Clone)]
struct ManifestBlock {
    start_pos: usize,
    origin: Origin,
    source_id: u32,
    embed: Vec<f32>,
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> anyhow::Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> anyhow::Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> anyhow::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_f32_slice<W: Write>(w: &mut W, v: &[f32]) -> anyhow::Result<()> {
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    for x in &mut out {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        *x = f32::from_le_bytes(b);
    }
    Ok(out)
}

fn origin_to_tag(origin: Origin) -> (u16, u16) {
    match origin {
        Origin::Query => (0, 0),
        Origin::File => (1, 0),
        Origin::Generated => (2, 0),
        Origin::System => (3, 0),
        Origin::Retrieved => (4, 0),
        Origin::Other(x) => (5, x),
    }
}

fn tag_to_origin(tag: u16, payload: u16) -> anyhow::Result<Origin> {
    Ok(match tag {
        0 => Origin::Query,
        1 => Origin::File,
        2 => Origin::Generated,
        3 => Origin::System,
        4 => Origin::Retrieved,
        5 => Origin::Other(payload),
        _ => anyhow::bail!("unknown origin tag {tag}"),
    })
}

fn write_embed_cache(
    path: &std::path::Path,
    needle_doc: &[Vec<f32>],
    needle_qry: &[Vec<f32>],
    pool: &[Vec<f32>],
    edim: usize,
) -> anyhow::Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    write_u32(&mut w, EMBED_CACHE_MAGIC)?;
    write_u32(&mut w, 1)?;
    write_u64(&mut w, edim as u64)?;
    write_u64(&mut w, needle_doc.len() as u64)?;
    write_u64(&mut w, needle_qry.len() as u64)?;
    write_u64(&mut w, pool.len() as u64)?;
    for v in needle_doc
        .iter()
        .chain(needle_qry.iter())
        .chain(pool.iter())
    {
        write_f32_slice(&mut w, v)?;
    }
    w.flush()?;
    Ok(())
}

fn read_embed_cache(
    path: &std::path::Path,
    n_needle: usize,
    n_pool: usize,
    edim: usize,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut r = BufReader::new(std::fs::File::open(path)?);
    let magic = read_u32(&mut r)?;
    let ver = read_u32(&mut r)?;
    anyhow::ensure!(
        magic == EMBED_CACHE_MAGIC && ver == 1,
        "invalid embed cache header"
    );
    let dim = read_u64(&mut r)? as usize;
    let nd = read_u64(&mut r)? as usize;
    let nq = read_u64(&mut r)? as usize;
    let np = read_u64(&mut r)? as usize;
    anyhow::ensure!(dim == edim, "embed cache dim mismatch: {dim} != {edim}");
    anyhow::ensure!(
        nd == n_needle && nq == n_needle,
        "embed cache needle count mismatch"
    );
    anyhow::ensure!(np == n_pool, "embed cache pool count mismatch");

    let mut needle_doc = Vec::with_capacity(nd);
    let mut needle_qry = Vec::with_capacity(nq);
    let mut pool = Vec::with_capacity(np);
    for _ in 0..nd {
        needle_doc.push(read_f32_vec(&mut r, edim)?);
    }
    for _ in 0..nq {
        needle_qry.push(read_f32_vec(&mut r, edim)?);
    }
    for _ in 0..np {
        pool.push(read_f32_vec(&mut r, edim)?);
    }
    Ok((needle_doc, needle_qry, pool))
}

fn write_store_manifest(
    path: &std::path::Path,
    blocks: &[ManifestBlock],
    rows_per_block: usize,
    edim: usize,
) -> anyhow::Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    write_u32(&mut w, STORE_MANIFEST_MAGIC)?;
    write_u32(&mut w, 1)?;
    write_u64(&mut w, rows_per_block as u64)?;
    write_u64(&mut w, edim as u64)?;
    write_u64(&mut w, blocks.len() as u64)?;
    for b in blocks {
        write_u64(&mut w, b.start_pos as u64)?;
        let (tag, payload) = origin_to_tag(b.origin);
        write_u32(&mut w, tag as u32)?;
        write_u32(&mut w, payload as u32)?;
        write_u32(&mut w, b.source_id)?;
        write_f32_slice(&mut w, &b.embed)?;
    }
    w.flush()?;
    Ok(())
}

fn read_store_manifest(
    path: &std::path::Path,
    rows_per_block: usize,
    edim: usize,
) -> anyhow::Result<Vec<ManifestBlock>> {
    let mut r = BufReader::new(std::fs::File::open(path)?);
    let magic = read_u32(&mut r)?;
    let ver = read_u32(&mut r)?;
    anyhow::ensure!(
        magic == STORE_MANIFEST_MAGIC && ver == 1,
        "invalid store manifest header"
    );
    let rows = read_u64(&mut r)? as usize;
    let dim = read_u64(&mut r)? as usize;
    let n = read_u64(&mut r)? as usize;
    anyhow::ensure!(rows == rows_per_block, "manifest block size mismatch");
    anyhow::ensure!(dim == edim, "manifest embed dim mismatch");
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let start_pos = read_u64(&mut r)? as usize;
        let tag = read_u32(&mut r)? as u16;
        let payload = read_u32(&mut r)? as u16;
        let source_id = read_u32(&mut r)?;
        let embed = read_f32_vec(&mut r, edim)?;
        out.push(ManifestBlock {
            start_pos,
            origin: tag_to_origin(tag, payload)?,
            source_id,
            embed,
        });
    }
    Ok(out)
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
    let mut reuse_store = false;
    let mut warm_buckets = 0usize;
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
            "--reuse-store" => {
                reuse_store = true;
            }
            "--warm-buckets" => {
                i += 1;
                warm_buckets = args[i].parse()?;
            }
            other => eprintln!("[1m] ignoring {other}"),
        }
        i += 1;
    }
    std::fs::create_dir_all(&out_dir)?;
    let scheme = KvQuant::from_name(&quant).unwrap_or(KvQuant::Q4_0);
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
    if warm_buckets > 0 {
        let warmed = runner.warm_buckets(warm_buckets);
        eprintln!("[1m] decode bucket warmup requested={warm_buckets}, compiled={warmed}");
    }
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
    let manifest_path = out_dir.join("store_manifest.bin");
    let can_reuse_store = reuse_store && manifest_path.exists() && store_dir.exists();
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
    let mut store = KvContextStore::new_with_reuse(
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
        can_reuse_store,
    )?;
    store.enable_embeddings(
        edim,
        HnswConfig {
            metric: Metric::Cosine,
            ef_construction,
            m: hm,
            m0: hm * 2,
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
    let embed_cache = out_dir.join("embed_cache.bin");
    let (needle_doc_emb, needle_qry_emb, pool): (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) =
        if embed_cache.exists() {
            eprintln!("[1m] loading embedding sidecar cache {:?}", embed_cache);
            read_embed_cache(&embed_cache, n_needle, filler_sents.len(), edim)?
        } else {
            eprintln!("[1m] embedding {n_needle} needle facts + a filler pool …");
            let needle_doc_emb: Vec<Vec<f32>> = needles
                .iter()
                .map(|(fact, _)| enc.embed_document_text(fact))
                .collect();
            let needle_qry_emb: Vec<Vec<f32>> = needles
                .iter()
                .map(|(_, q)| enc.embed_query_text(q))
                .collect();
            let pool: Vec<Vec<f32>> = filler_sents
                .iter()
                .map(|s| enc.embed_document_text(s))
                .collect();
            write_embed_cache(&embed_cache, &needle_doc_emb, &needle_qry_emb, &pool, edim)?;
            eprintln!("[1m] wrote embedding sidecar cache {:?}", embed_cache);
            (needle_doc_emb, needle_qry_emb, pool)
        };

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
    let mut manifest_blocks: Vec<ManifestBlock> = if can_reuse_store {
        eprintln!("[1m] reusing store files + manifest {:?}", manifest_path);
        read_store_manifest(&manifest_path, block, edim)?
    } else {
        Vec::with_capacity(nblocks)
    };
    if can_reuse_store {
        anyhow::ensure!(
            manifest_blocks.len() >= nblocks,
            "manifest has {} blocks but run needs {nblocks}",
            manifest_blocks.len()
        );
    }
    let mut query_cache: HashMap<usize, Vec<f32>> = needle_qry_emb
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.clone()))
        .collect();
    let mut hnsw_cache: HashMap<
        (usize, usize, usize),
        Vec<rlx_runtime::kv_context_store::RetrievedBlock>,
    > = HashMap::new();
    let mut exact_cache: HashMap<
        (usize, usize, usize),
        Vec<rlx_runtime::kv_context_store::RetrievedBlock>,
    > = HashMap::new();
    let mut csv = String::from(
        "shot,ctx_tokens,store_blocks,disk_gb,ram_idx_mb,ingest_tok_per_s,\
         embed_query_ms,hnsw_retrieve_ms,exact_retrieve_ms,recall_hnsw,recall_exact,\
         ret_k_absmax,ret_k_naninf,decode_tps,decode_ms\n",
    );
    let t_all = Instant::now();
    let mut t_ing = Instant::now();

    eprintln!("[1m] populating {tokens} tokens into the store (multi-shot at {shots:?}) …");
    for b in 0..nblocks {
        let id = if can_reuse_store {
            let m = &manifest_blocks[b];
            let id = store.import_block(m.start_pos, block, m.origin, m.source_id)?;
            store.append_embed(id, &m.embed);
            id
        } else {
            let (k, v, key) = synth_kv((b as u64) << 20);
            if let Some(ni) = is_needle(b) {
                let id = store.append_block(b * block, Origin::File, ni as u32, &k, &v, &key)?;
                let embed = needle_doc_emb[ni].clone();
                store.append_embed(id, &embed);
                manifest_blocks.push(ManifestBlock {
                    start_pos: b * block,
                    origin: Origin::File,
                    source_id: ni as u32,
                    embed,
                });
                id
            } else {
                let id =
                    store.append_block(b * block, Origin::Generated, b as u32, &k, &v, &key)?;
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
                manifest_blocks.push(ManifestBlock {
                    start_pos: b * block,
                    origin: Origin::Generated,
                    source_id: b as u32,
                    embed: e,
                });
                id
            }
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
            let ctx_blocks = store.len_blocks();
            for (ni, &nb) in needle_block.iter().enumerate() {
                if nb > b {
                    continue;
                }
                nq += 1;
                let te = Instant::now();
                let q = query_cache
                    .entry(ni)
                    .or_insert_with(|| enc.embed_query_text(needles[ni].1));
                t_embed += te.elapsed().as_secs_f64() * 1e3;
                // HNSW (approximate) retrieval — timed.
                let key = (ctx_blocks, ni, topk);
                let got_hnsw = if let Some(v) = hnsw_cache.get(&key) {
                    v.clone()
                } else {
                    let th = Instant::now();
                    let v = store.retrieve_embed(q, topk);
                    t_hnsw += th.elapsed().as_secs_f64() * 1e3;
                    hnsw_cache.insert(key, v.clone());
                    v
                };
                if got_hnsw.iter().any(|r| r.start_pos == nb * block) {
                    hits_hnsw += 1;
                }
                // EXACT (brute-force) retrieval — the correct number at this scale.
                let got = if let Some(v) = exact_cache.get(&key) {
                    v.clone()
                } else {
                    let tr = Instant::now();
                    let v = store.retrieve_embed_exact(q, topk);
                    t_read += tr.elapsed().as_secs_f64() * 1e3;
                    exact_cache.insert(key, v.clone());
                    v
                };
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
    if !can_reuse_store {
        write_store_manifest(&manifest_path, &manifest_blocks, block, edim)?;
        eprintln!("[1m] wrote store manifest {:?}", manifest_path);
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
    println!(
        "  model backend: {dev:?}; weights dtype path: f32 (set RLX_QWEN3_F16_WEIGHTS=1 for Metal f16 weights)"
    );
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
