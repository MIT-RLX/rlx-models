//! memory_probe — a multi-shot **memory / context-retrieval** test that records
//! and analyzes the whole picture of an rlx model's KV cache during a live
//! session.
//!
//! It plants facts, buries them under unrelated "filler" turns until they fall
//! out of the bounded resident KV budget, then asks about them. A naive
//! recency-only policy (`sinks`) forgets; `retrieval` / `auto` pull the evicted
//! facts back from the store. The *recall accuracy* across policies is the
//! functional signal that selective retention beats sliding-window amnesia.
//!
//! While it runs it records — and then analyzes together —:
//!   • **cache/context** telemetry (resident / evict / retrieve / store,
//!     effective context, extension factor) via the retention recorder
//!     (`rlx_runtime::kv_metrics`),
//!   • **KV cache + selection-preference DATA** (shape / stats / histograms /
//!     dataflow) via the inspect log (`rlx_ir::tensor_inspect`),
//!   • **per-turn throughput** (TTFT, decode tok/s) against the growing context,
//!   • and (optional) **op-level** shape/stats/histogram/dataflow captured on a
//!     CPU forward pass through the `RLX_INSPECT_OPS` op tap.
//!
//! Everything is dumped to CSV/DOT under `--out` for plotting, and summarized on
//! screen. The recorders live in rlx core, so the same harness inspects any rlx
//! model that decodes through a KV cache.
//!
//! Run:
//!   cargo run --release -p rlx-qwen3 --example memory_probe --features metal -- \
//!       --device metal
//! Flags: --device <d> --weights <dir> --policies <a,b,c> --max-tokens <n>
//!        --out <dir> --op-inspect / --no-op-inspect

use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use rlx_cli::WeightFormat;
use rlx_qwen3::{Qwen3Runner, SampleOpts};
use rlx_runtime::Device;
use tokenizers::Tokenizer;

const DEFAULT_WEIGHTS: &str = "/Users/Shared/rlx-models/weights/lm/qwen3-0.6b";

/// A fact to plant and later recall.
struct Needle {
    plant: &'static str,
    ask: &'static str,
    /// Any of these substrings (case-insensitive) in the reply counts as recall.
    expect: &'static [&'static str],
}

/// One conversation turn.
enum Turn {
    Plant(usize),
    Filler(&'static str),
    Recall(usize),
}

impl Turn {
    fn label(&self) -> &'static str {
        match self {
            Turn::Plant(_) => "plant",
            Turn::Filler(_) => "filler",
            Turn::Recall(_) => "recall",
        }
    }
}

/// Per-turn throughput vs. context growth.
struct TurnPerf {
    turn: usize,
    kind: &'static str,
    ctx_before: usize,
    gen_toks: usize,
    ttft_ms: f64,
    decode_tps: f64,
}

/// Everything recorded for one policy run.
struct PolicyResult {
    policy: String,
    recall: Vec<(String, bool, String)>,
    retention_report: String,
    resident_spark: String,
    ctx_spark: String,
    extension: f32,
    ctx_peak: usize,
    retention_csv: String,
    inspect_report: String,
    inspect_csv: String,
    inspect_hist_csv: String,
    dataflow_dot: String,
    perf: Vec<TurnPerf>,
}

/// Build a **greedy top-k** sampler. `temp <= 0` ⇒ deterministic argmax over the
/// (top-k restricted) distribution — the right default for a reproducible
/// pass/fail recall test, so policies are compared on identical decoding. A
/// positive `temp` switches to seeded top-k / top-p sampling (still
/// deterministic for a fixed `seed`).
fn make_sampler(temp: f32, top_k: usize, top_p: f32, seed: u64) -> SampleOpts {
    let mut s = SampleOpts::greedy();
    s.top_k = top_k;
    s.top_p = top_p;
    s.seed = seed;
    if temp > 0.0 {
        s.temperature = temp;
        s.greedy = false;
    }
    s
}

fn render_prompt(system: &str, pending_user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{pending_user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Format retrieved spans as labeled notes (`[1] …\n[2] …`), one per line.
#[cfg(feature = "mmap-kv")]
fn notes_from_spans(tok: &Tokenizer, spans: &[(Vec<u32>, f32)]) -> String {
    let mut s = String::new();
    for (j, (ids, _)) in spans.iter().enumerate() {
        let t = tok.decode(ids, true).unwrap_or_default();
        let t = t.replace('\n', " ");
        let t = t.trim();
        if !t.is_empty() {
            s.push_str(&format!("[{}] {t}\n", j + 1));
        }
    }
    s
}

/// **Interleaved retrieve-in-the-loop** recall (the dynamic mid-reasoning path).
///
/// Seeds with a retrieval on the question, then repeatedly: generate a short
/// reasoning chunk; if the model emits `SEARCH: <q>` fetch `<q>` and feed the
/// result back; if it emits `ANSWER: <a>` stop; otherwise (no marker) fall back
/// to IRCoT — re-query the store with `question + reasoning-so-far` and feed the
/// result — so a fact the first pass missed can still be pulled in. Bounded by
/// `max_hops`. Returns `(answer, full_transcript, notes_seen_lower, hops)`.
///
/// The store stays SUSPENDED throughout (no auto-splice); retrieval is driven
/// here via `retrieve_context_spans`, which reads the frozen retrieval stream so
/// it survives this loop's per-hop re-prefills.
#[cfg(feature = "mmap-kv")]
#[allow(clippy::too_many_arguments)]
fn interleave_recall(
    runner: &mut Qwen3Runner,
    tok: &Tokenizer,
    system: &str,
    question: &str,
    eos: &[u32],
    tm_topk: usize,
    tm_margin: usize,
    max_hops: usize,
    hop_tokens: usize,
) -> anyhow::Result<(String, String, String, usize)> {
    runner.set_kv_store_suspended(true);
    let enc = |s: &str| -> anyhow::Result<Vec<u32>> {
        Ok(tok
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec())
    };
    // Hop 0 seed: retrieve on the question itself.
    runner.set_retrieval_query(Some(question.to_string()));
    let seed = runner.retrieve_context_spans(Some(tm_topk), tm_margin);
    let mut notes_seen = notes_from_spans(tok, &seed).to_lowercase();
    let instr = format!(
        "Answer the question using the memory notes. If the notes do NOT contain the fact, \
         write on a new line exactly `SEARCH: <keywords>` and I will look it up and add the \
         result; then continue. When you know the answer, write `ANSWER: <answer>`.\n\n\
         NOTES:\n{}\nQUESTION: {question}\n",
        notes_from_spans(tok, &seed)
    );
    let mut transcript = render_prompt(system, &instr);
    let mut answer = String::new();
    let mut hops = 0usize;
    for _ in 0..=max_hops {
        hops += 1;
        let ids = enc(&transcript)?;
        let mut chunk_toks: Vec<u32> = Vec::new();
        runner.generate_stoppable(&ids, hop_tokens, |t| {
            if eos.contains(&t) {
                return false;
            }
            chunk_toks.push(t);
            let s = tok.decode(&chunk_toks, true).unwrap_or_default();
            // Stop the moment a SEARCH: or ANSWER: line is complete (has a newline).
            if let Some(p) = s.rfind("SEARCH:") {
                if s[p + 7..].contains('\n') {
                    return false;
                }
            }
            if let Some(p) = s.rfind("ANSWER:") {
                if s[p + 7..].contains('\n') {
                    return false;
                }
            }
            true
        })?;
        let chunk = tok.decode(&chunk_toks, true).unwrap_or_default();
        transcript.push_str(&chunk);
        // ANSWER wins.
        if let Some(p) = chunk.find("ANSWER:") {
            answer = chunk[p + 7..]
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !answer.is_empty() {
                break;
            }
        }
        // Explicit SEARCH → fetch and feed the result.
        let (query, tag) = if let Some(p) = chunk.rfind("SEARCH:") {
            (
                chunk[p + 7..]
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                "SEARCH",
            )
        } else if !chunk.trim().is_empty() {
            // IRCoT fallback: re-query with the question + reasoning so far.
            (format!("{question} {}", chunk.replace('\n', " ")), "IRCOT")
        } else {
            break; // nothing generated
        };
        if query.is_empty() {
            break;
        }
        runner.set_retrieval_query(Some(query.clone()));
        let hits = runner.retrieve_context_spans(Some(tm_topk), tm_margin);
        let notes = notes_from_spans(tok, &hits);
        notes_seen.push_str(&notes.to_lowercase());
        eprintln!(
            "      [hop {hops} {tag}] q={:?}",
            query.chars().take(60).collect::<String>()
        );
        transcript.push_str(&format!("\nRESULT:\n{notes}\n"));
    }
    // Force a final answer if none was emitted.
    if answer.is_empty() {
        transcript.push_str("\nANSWER:");
        let ids = enc(&transcript)?;
        let mut a: Vec<u32> = Vec::new();
        runner.generate_stoppable(&ids, 24, |t| {
            if eos.contains(&t) {
                return false;
            }
            a.push(t);
            true
        })?;
        answer = tok.decode(&a, true).unwrap_or_default().trim().to_string();
    }
    runner.set_kv_store_suspended(false);
    Ok((answer, transcript, notes_seen, hops))
}

/// Log-scaled unicode sparkline of a `usize` series.
fn sparkline_usize(series: &[usize]) -> String {
    let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if series.is_empty() {
        return String::new();
    }
    let maxln = series
        .iter()
        .map(|&c| ((c as f64) + 1.0).ln())
        .fold(0.0f64, f64::max)
        .max(1e-9);
    // Downsample to <=60 columns, taking the max per column so peaks survive.
    let width = 60usize.min(series.len().max(1));
    let n = series.len();
    (0..width)
        .map(|c| {
            let a = c * n / width;
            let b = ((c + 1) * n / width).max(a + 1);
            let colmax = series[a..b.min(n)].iter().copied().max().unwrap_or(0);
            let frac = ((colmax as f64) + 1.0).ln() / maxln;
            let bi = ((frac * (blocks.len() - 1) as f64).round() as usize).min(blocks.len() - 1);
            blocks[bi]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_policy(
    spec: &str,
    weights: &Path,
    dev: Device,
    enc_dev: Device,
    rr_dev: Device,
    tok: &Tokenizer,
    eos: &[u32],
    max_seq: usize,
    max_tokens: usize,
    sample: SampleOpts,
    text_memory: bool,
    tm_topk: usize,
    tm_margin: usize,
    rerank_repo: &str,
    rerank_overfetch: usize,
    tm_think: bool,
    tm_think_tokens: usize,
    interleave: bool,
    max_hops: usize,
    hop_tokens: usize,
    script: &[Turn],
    needles: &[Needle],
) -> anyhow::Result<PolicyResult> {
    // The generator reads its retention policy from the env at construction.
    // `full`/`none` = the O(context) upper-bound baseline (retention off).
    // `kvstore[:BLOCK:SINKS:RECENT:TOPK:NEIGH]` = disk-tiered context store.
    let is_kvstore = spec.eq_ignore_ascii_case("kvstore") || spec.starts_with("kvstore:");
    if spec.eq_ignore_ascii_case("full") || spec.eq_ignore_ascii_case("none") || is_kvstore {
        unsafe { std::env::remove_var("RLX_QWEN3_RETENTION") };
    } else {
        unsafe { std::env::set_var("RLX_QWEN3_RETENTION", spec) };
    }

    let mut runner = Qwen3Runner::builder()
        .weights(weights.to_path_buf())
        .device(dev)
        .format(WeightFormat::Safetensors)
        .packed_weights(false)
        .max_seq(max_seq)
        .sample(sample)
        .build()?;

    if is_kvstore {
        #[cfg(feature = "mmap-kv")]
        {
            // kvstore:BLOCK:SINKS:RECENT:TOPK:NEIGH[:METRIC[:CENTROIDS[:DECAY[:LEXICAL[:QSCORE]]]]]
            // (defaults 16:4:32:12:2:l2:4:1.0:0.0:0). METRIC ∈ l2|dot|cos.
            // QSCORE ∈ 0|1|q — 1/q scores blocks by the model's query (Q·K).
            let fields: Vec<&str> = spec
                .strip_prefix("kvstore:")
                .unwrap_or("")
                .split(':')
                .collect();
            let g = |i: usize, d: usize| fields.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
            let metric = match fields.get(5).copied().unwrap_or("l2") {
                "dot" => rlx_runtime::hnsw::Metric::Dot,
                "cos" | "cosine" => rlx_runtime::hnsw::Metric::Cosine,
                _ => rlx_runtime::hnsw::Metric::L2,
            };
            let centroids = g(6, 4);
            let decay: f32 = fields.get(7).and_then(|x| x.parse().ok()).unwrap_or(1.0);
            let lexical: f32 = fields.get(8).and_then(|x| x.parse().ok()).unwrap_or(0.0);
            let qscore = matches!(fields.get(9).copied().unwrap_or("0"), "1" | "q" | "Q");
            let scheme = match fields.get(10).copied().unwrap_or("q8") {
                "f16" | "F16" => rlx_runtime::quantized_kv::KvQuant::F16,
                "q4" | "Q4" | "q4_0" => rlx_runtime::quantized_kv::KvQuant::Q4_0,
                _ => rlx_runtime::quantized_kv::KvQuant::Q8_0,
            };
            // Field 11 = MAXSIM (1/m = late-interaction re-rank), 12 = ROWKEYS
            // (1/r = salient-row HNSW index). Both attack mean-pool dilution.
            let maxsim = matches!(fields.get(11).copied().unwrap_or("0"), "1" | "m" | "M");
            let row_keys = matches!(fields.get(12).copied().unwrap_or("0"), "1" | "r" | "R");
            // Field 13 = EXACT (1/e = brute-force retrieval, bypass HNSW).
            let exact = matches!(fields.get(13).copied().unwrap_or("0"), "1" | "e" | "E");
            // Field 14 = QUERY_POOL width (mean of last-N K rows as the query).
            let query_pool: usize = fields.get(14).and_then(|x| x.parse().ok()).unwrap_or(1);
            // Field 15 = MULTILAYER (1/L = all-layer exact MaxSim scoring).
            let multilayer = matches!(fields.get(15).copied().unwrap_or("0"), "1" | "l" | "L");
            // Field 16 = EMBEDDER (tm = TokenMean self-contained, enc = dedicated
            // encoder, 0/none = off). Semantic dual-encoder retrieval; blends with
            // the LEXICAL weight (field 8). This is the selective, 1M-scale signal.
            let embed_kind = match fields.get(16).copied().unwrap_or("0") {
                "tm" | "token" | "1" => rlx_qwen3::embedder::EmbedderKind::TokenMean,
                "enc" | "encoder" | "2" => rlx_qwen3::embedder::EmbedderKind::Encoder,
                _ => rlx_qwen3::embedder::EmbedderKind::None,
            };
            // Field 17 = embed_weight, 18 = dense(K·K)_weight for hybrid3 blend
            // (lexical uses field 8). Defaults 1.0 / 0.0.
            let embed_w: f32 = fields.get(17).and_then(|x| x.parse().ok()).unwrap_or(1.0);
            let dense_w: f32 = fields.get(18).and_then(|x| x.parse().ok()).unwrap_or(0.0);
            // Field 19 = relevance_gate ∈ [0,1): drop retrieved blocks below
            // gate×top_score (noise minimizer / adaptive-k). 0 = keep all topk.
            let gate: f32 = fields.get(19).and_then(|x| x.parse().ok()).unwrap_or(0.0);
            // Field 20 = RRF (1 = reciprocal-rank-fuse embed+lexical(+dense)).
            let rrf = matches!(fields.get(20).copied().unwrap_or("0"), "1" | "r" | "R");
            // Field 21 = query_window (recent tokens forming the retrieval query;
            // 0 = auto). Smaller = cleaner query (just the question tail).
            let qwin: usize = fields.get(21).and_then(|x| x.parse().ok()).unwrap_or(0);
            let cfg = rlx_qwen3::KvStoreConfig::new()
                .capacity_tokens(max_seq * 4)
                .block(g(0, 16))
                .sinks(g(1, 4))
                .recent(g(2, 32))
                .topk(g(3, 12))
                .neighbors(g(4, 2))
                .metric(metric)
                .centroids_per_block(centroids)
                .decay(decay)
                .lexical_weight(lexical)
                .query_scoring(qscore)
                .scheme(scheme)
                .maxsim(maxsim)
                .row_keys(row_keys)
                .exact(exact)
                .query_pool(query_pool)
                .multilayer(multilayer)
                .embedder(embed_kind)
                .embed_weight(embed_w)
                .dense_weight(dense_w)
                .relevance_gate(gate)
                .rrf(rrf)
                .query_window(qwin);
            runner.enable_kv_store(cfg)?;
            // Dual-encoder: build the dedicated MiniLM sentence encoder (downloads
            // weights) and inject it, overriding the self-contained fallback. Needs
            // the `dual-encoder` feature; without it, `enc` falls back to TokenMean.
            #[cfg(feature = "dual-encoder")]
            if embed_kind == rlx_qwen3::embedder::EmbedderKind::Encoder {
                use rlx_qwen3::embedder::BlockEmbedder as _;
                // Encoder repo (env-configurable). Default is bge-small-en-v1.5
                // (384-d retrieval encoder); any HF sentence encoder works.
                let repo = std::env::var("RLX_QWEN3_EMBED_REPO")
                    .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());
                // Run the retrieval encoder on the SAME device as the LM (so
                // `--device metal` keeps everything on the GPU — no pure-CPU leg).
                let enc = rlx_qwen3::embedder::RlxEmbedEmbedder::from_pretrained(
                    tok.clone(),
                    &repo,
                    enc_dev,
                )?;
                eprintln!(
                    "[kvstore] dual-encoder: {repo} loaded on {enc_dev:?} (dim {})",
                    enc.dim()
                );
                runner.set_kv_store_embedder(Box::new(enc));
            }
            eprintln!(
                "[kvstore] enabled: block {} sinks {} recent {} topk {} neigh {} metric {metric:?} centroids {centroids} decay {decay} lexical {lexical} qscore {qscore} scheme {scheme:?} maxsim {maxsim} rowkeys {row_keys} exact {exact} qpool {query_pool} multilayer {multilayer} embed {embed_kind:?}",
                g(0, 16),
                g(1, 4),
                g(2, 32),
                g(3, 12),
                g(4, 2)
            );
        }
        #[cfg(not(feature = "mmap-kv"))]
        eprintln!("[kvstore] requires --features mmap-kv; running without retention");
    }

    // Cross-encoder reranker (C): built once, reused across recall turns. Only
    // used with text-memory (it re-orders D's over-fetched candidate note list;
    // the winning notes then become the labeled prompt).
    #[cfg(feature = "dual-encoder")]
    let mut reranker: Option<rlx_embed::RlxReranker> = None;
    #[cfg(feature = "dual-encoder")]
    if text_memory && !rerank_repo.is_empty() {
        // Cross-encoder on its own device (default = LM device).
        let rr = rlx_embed::RlxReranker::from_pretrained(rerank_repo, rr_dev, 192)?;
        eprintln!(
            "[rerank] cross-encoder {rerank_repo} loaded on {rr_dev:?} (hidden {})",
            rr.hidden_size()
        );
        reranker = Some(rr);
    }
    #[cfg(not(feature = "dual-encoder"))]
    let _ = (rerank_repo, rerank_overfetch);

    // Warm the decode graph once, then start a clean conversation.
    let warm: Vec<u32> = tok
        .encode(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
            false,
        )
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    runner.generate_stoppable(&warm, 1, |_| false)?;
    runner.reset_cache();
    runner.enable_retention_recording();
    runner.enable_inspect();

    let system = "You are a helpful assistant with perfect memory. Answer concisely.";
    let mut first = true;
    let mut last_eos = false;
    let mut recall = Vec::new();
    let mut perf = Vec::new();
    // Text-reinjection (D): retrieved TEXT spans per needle, captured ONCE at the
    // first recall (while the token history the spans are recovered from is still
    // intact — a fresh D generation clears it). Keyed by needle index.
    let d_spans: std::collections::HashMap<usize, Vec<(Vec<u32>, f32)>> =
        std::collections::HashMap::new();
    #[allow(unused_mut)] // mutated only under the mmap-kv / dual-encoder feature
    let mut d_captured = false;
    #[allow(unused_mut)] // mutated only under the mmap-kv / dual-encoder feature
    let mut il_snap = false;
    // Used only under mmap-kv / dual-encoder; keep the base build warning-clean.
    let _ = (
        &d_spans,
        d_captured,
        text_memory,
        tm_topk,
        tm_margin,
        tm_think,
        tm_think_tokens,
    );
    let _ = (il_snap, interleave, max_hops, hop_tokens, enc_dev, rr_dev);

    for (ti, turn) in script.iter().enumerate() {
        let user_raw = match turn {
            Turn::Plant(i) => needles[*i].plant,
            Turn::Filler(f) => f,
            Turn::Recall(i) => needles[*i].ask,
        };
        // /no_think keeps the 0.6B model from abandoning a <think> block mid-reasoning.
        let user = format!("{user_raw} /no_think");
        let delta_text = if first {
            render_prompt(system, &user)
        } else {
            let close = if last_eos { "\n" } else { "<|im_end|>\n" };
            format!("{close}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        };
        first = false;
        let delta_ids: Vec<u32> = tok
            .encode(delta_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();

        let ctx_before = runner.context_len();
        runner.warm_buckets((ctx_before + delta_ids.len() + max_tokens).min(max_seq));

        // Clean retrieval query: on recall turns, target the ACTUAL question text
        // (not the noisy decode-position token window). Cleared otherwise.
        match turn {
            Turn::Recall(i) => runner.set_retrieval_query(Some(needles[*i].ask.to_string())),
            _ => runner.set_retrieval_query(None),
        }

        let is_interleave_recall = interleave && matches!(turn, Turn::Recall(_));
        let is_d_recall = text_memory && !interleave && matches!(turn, Turn::Recall(_));
        let mut out: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        // For D recalls: the retrieved notes' text (for retrieval-vs-gen attribution).
        #[allow(unused_mut)] // assigned only under the interleave / dual-encoder feature
        let mut d_ret_text: Option<String> = None;
        let t0 = Instant::now();
        let mut ttft: Option<std::time::Duration> = None;
        if is_interleave_recall {
            // Dynamic retrieve-in-the-loop (interleave): reason, pull facts mid-think.
            #[cfg(feature = "mmap-kv")]
            {
                let i = match turn {
                    Turn::Recall(i) => *i,
                    _ => unreachable!(),
                };
                // Freeze the original stream ONCE (retrieval recovers span text from
                // it; the loop's per-hop re-prefills clobber the live tokens).
                if !il_snap {
                    il_snap = true;
                    runner.snapshot_retrieval_stream();
                }
                let (answer, transcript, notes_seen, hops) = interleave_recall(
                    &mut runner,
                    &tok,
                    system,
                    needles[i].ask,
                    &eos,
                    tm_topk,
                    tm_margin,
                    max_hops,
                    hop_tokens,
                )?;
                ttft = Some(t0.elapsed());
                d_ret_text = Some(notes_seen);
                eprintln!("      [INTERLEAVE needle[{i}] hops={hops}] answer={answer:?}");
                if std::env::var_os("RLX_PROBE_DEBUG_NOTES").is_some() {
                    eprintln!("{transcript}");
                }
                out = tok
                    .encode(answer.as_str(), false)
                    .map(|e| e.get_ids().to_vec())
                    .unwrap_or_default();
            }
        } else if is_d_recall {
            // Text-reinjection path (D). The store was populated by the write phase;
            // now answer over retrieved TEXT instead of spliced raw KV.
            #[cfg(feature = "mmap-kv")]
            {
                let i = match turn {
                    Turn::Recall(i) => *i,
                    _ => unreachable!(),
                };
                // Capture every recall needle's top-k spans ONCE, while the full
                // token history is still resident (read-only; the first D generation
                // below clears it).
                if !d_captured {
                    d_captured = true;
                    for t in script {
                        if let Turn::Recall(k) = t {
                            runner.set_retrieval_query(Some(needles[*k].ask.to_string()));
                            // With a reranker: over-fetch by bi-encoder, jointly
                            // score (question, note), keep the top tm_topk. Without:
                            // straight bi-encoder top-k.
                            #[cfg(feature = "dual-encoder")]
                            let picked = if let Some(rr) = reranker.as_mut() {
                                let cands = runner
                                    .retrieve_context_spans(Some(rerank_overfetch), tm_margin);
                                let texts: Vec<String> = cands
                                    .iter()
                                    .map(|(ids, _)| tok.decode(ids, true).unwrap_or_default())
                                    .collect();
                                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                                match rr.rerank(needles[*k].ask, &refs) {
                                    Ok(order) => order
                                        .into_iter()
                                        .take(tm_topk)
                                        .filter_map(|(idx, sc)| {
                                            cands.get(idx).map(|(ids, _)| (ids.clone(), sc))
                                        })
                                        .collect::<Vec<_>>(),
                                    Err(e) => {
                                        eprintln!("[rerank] failed: {e}; bi-encoder top-k");
                                        runner.retrieve_context_spans(Some(tm_topk), tm_margin)
                                    }
                                }
                            } else {
                                runner.retrieve_context_spans(Some(tm_topk), tm_margin)
                            };
                            #[cfg(not(feature = "dual-encoder"))]
                            let picked = runner.retrieve_context_spans(Some(tm_topk), tm_margin);
                            d_spans.insert(*k, picked);
                        }
                    }
                }
                // Detokenize the retrieved blocks into labeled notes (best first).
                let spans = d_spans.get(&i).cloned().unwrap_or_default();
                let mut notes = String::new();
                for (j, (ids, _score)) in spans.iter().enumerate() {
                    let t = tok.decode(ids, true).unwrap_or_default();
                    let t = t.replace('\n', " ");
                    let t = t.trim();
                    if !t.is_empty() {
                        notes.push_str(&format!("[{}] {t}\n", j + 1));
                    }
                }
                d_ret_text = Some(notes.to_lowercase());
                if std::env::var_os("RLX_PROBE_DEBUG_NOTES").is_some() {
                    eprintln!(
                        "      [D notes for needle[{i}] ask={:?}]\n{}",
                        needles[i].ask, notes
                    );
                }
                // Thinking budget: with `--tm-think` we DROP `/no_think` so the model
                // reasons in a <think> block before answering — the test of whether
                // reasoning disambiguates a competing fact. Otherwise reflexive.
                let think_suffix = if tm_think { "" } else { " /no_think" };
                let user = format!(
                    "Notes you saved earlier:\n{notes}\nUsing only the notes above, answer this: \
                     {}{think_suffix}",
                    needles[i].ask
                );
                let prompt = render_prompt(system, &user);
                let prompt_ids: Vec<u32> = tok
                    .encode(prompt.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                    .get_ids()
                    .to_vec();
                let budget = if tm_think {
                    tm_think_tokens
                } else {
                    max_tokens
                };
                // Isolated clean generation: suspend the store's splice/eviction so
                // the model attends to the labeled TEXT only (full attention over
                // the short prompt), then re-enable it.
                runner.set_kv_store_suspended(true);
                runner.generate_stoppable(&prompt_ids, budget, |t| {
                    if ttft.is_none() {
                        ttft = Some(t0.elapsed());
                    }
                    if eos.contains(&t) {
                        hit_eos = true;
                        return false;
                    }
                    out.push(t);
                    true
                })?;
                runner.set_kv_store_suspended(false);
                if tm_think {
                    // Show the reasoning trace + the number of think steps (tokens
                    // inside the <think> block) so we can see HOW it (mis)binds.
                    let full = tok.decode(&out, true).unwrap_or_default();
                    let think_toks = full
                        .split_once("</think>")
                        .map(|(t, _)| t.replace("<think>", "").split_whitespace().count())
                        .unwrap_or(out.len());
                    eprintln!(
                        "      [D THINK needle[{i}] budget={budget} used={} think_steps≈{think_toks}]\n{}",
                        out.len(),
                        full.trim()
                    );
                }
            }
        } else {
            runner.generate_continuation_stoppable(&delta_ids, max_tokens, |t| {
                if ttft.is_none() {
                    ttft = Some(t0.elapsed());
                }
                if eos.contains(&t) {
                    hit_eos = true;
                    return false;
                }
                out.push(t);
                true
            })?;
        }
        let dt = t0.elapsed();
        last_eos = hit_eos;

        let reply = tok.decode(&out, true).unwrap_or_default();
        let ttft_d = ttft.unwrap_or(dt);
        let decode_toks = out.len().saturating_sub(1);
        let decode_dt = dt.saturating_sub(ttft_d);
        let decode_tps = if decode_toks > 0 && decode_dt.as_secs_f64() > 0.0 {
            decode_toks as f64 / decode_dt.as_secs_f64()
        } else {
            0.0
        };
        perf.push(TurnPerf {
            turn: ti,
            kind: turn.label(),
            ctx_before,
            gen_toks: out.len(),
            ttft_ms: ttft_d.as_secs_f64() * 1e3,
            decode_tps,
        });

        if let Turn::Recall(i) = turn {
            // When thinking is on, judge the FINAL answer (after the last
            // </think>), not the reasoning trace — the trace may explore the wrong
            // fact before settling. Falls back to the whole reply if no marker.
            let answer_text = if tm_think {
                reply
                    .rsplit_once("</think>")
                    .map(|(_, a)| a)
                    .unwrap_or(&reply)
            } else {
                reply.as_str()
            };
            let lower = answer_text.to_lowercase();
            let hit = needles[*i]
                .expect
                .iter()
                .any(|e| lower.contains(&e.to_lowercase()));
            // Retrieval-vs-generation attribution: did the RETRIEVED blocks contain
            // the needle's answer text? (retrieved & !answered = GENERATION miss;
            // !retrieved = RETRIEVAL miss.) For D, use the labeled notes' text; for
            // the splice path, the last decode step's retrieved tokens.
            let ret_text = match &d_ret_text {
                Some(t) => t.clone(),
                None => {
                    let ret_toks = runner.last_retrieved_tokens();
                    tok.decode(&ret_toks, true)
                        .unwrap_or_default()
                        .to_lowercase()
                }
            };
            let retrieved_fact = needles[*i]
                .expect
                .iter()
                .any(|e| ret_text.contains(&e.to_lowercase()));
            let attrib = match (retrieved_fact, hit) {
                (true, true) => "OK",
                (true, false) => "GEN-miss (fact retrieved, model failed)",
                (false, _) => "RETRIEVAL-miss (fact not in retrieved blocks)",
            };
            eprintln!(
                "      needle[{i}] expect={:?} retrieved_fact={retrieved_fact} answered={hit} → {attrib}",
                needles[*i].expect
            );
            recall.push((needles[*i].ask.to_string(), hit, reply.clone()));
        }

        let snippet: String = reply.replace('\n', " ").chars().take(48).collect();
        eprintln!(
            "  [{ti:2}] {:7} ctx {ctx_before:5} -> gen {:3} tok @ {decode_tps:5.0} tps  {snippet}",
            turn.label(),
            out.len(),
        );
    }

    // Pull the recorded picture back out.
    let rec = runner.take_retention_recorder();
    let insp = runner.take_inspect_log();

    let (retention_report, resident_spark, ctx_spark, extension, ctx_peak, retention_csv) =
        match &rec {
            Some(r) => {
                let s = r.summary();
                let resident: Vec<usize> = r.records().iter().map(|x| x.resident).collect();
                let ctx: Vec<usize> = r.records().iter().map(|x| x.effective_context()).collect();
                (
                    s.report(),
                    sparkline_usize(&resident),
                    sparkline_usize(&ctx),
                    s.context_extension,
                    s.effective_context_max,
                    r.to_csv(),
                )
            }
            None => (
                "(no retention recorder — policy kept everything)\n".to_string(),
                String::new(),
                String::new(),
                1.0,
                0,
                String::new(),
            ),
        };

    let (inspect_report, inspect_csv, inspect_hist_csv, dataflow_dot) = match &insp {
        Some(l) => (l.report(), l.to_csv(), l.to_hist_csv(), l.dataflow_dot()),
        None => (String::new(), String::new(), String::new(), String::new()),
    };

    Ok(PolicyResult {
        policy: spec.to_string(),
        recall,
        retention_report,
        resident_spark,
        ctx_spark,
        extension,
        ctx_peak,
        retention_csv,
        inspect_report,
        inspect_csv,
        inspect_hist_csv,
        dataflow_dot,
        perf,
    })
}

/// Op-level inspection of the **real qwen3 graph** via rlx-opscope static
/// analysis: per-op shape (m·k·n) / FLOPs / bytes / arithmetic intensity /
/// roofline class, plus recurring dataflow cones. Writes CSV for plotting.
///
/// (For op *value* distributions there is also `rlx_ir::tensor_inspect`'s op
/// tap — `RLX_INSPECT_OPS=1` — which records shape/stats/histograms from the
/// CPU reference executor; opscope's static costs cover shape+dataflow on the
/// compiled path without executing.)
/// Report how well a model's forward FUSES on the target backend — the count of
/// fused ops vs missed fusions from rlx's fusion-coverage analysis. This is the
/// "fwd for each model fused" view: intra-model op fusion (linear+bias+act,
/// layernorm, attention SDPA) is compiler-driven, and on Metal MPSGraph fuses
/// further; this shows the coverage per model so regressions surface.
fn fusion_report(g: &rlx_ir::Graph, tag: &str, backend: &str) {
    use rlx_runtime::check::{CheckOptions, check_graph, parse_backend};
    let Some(target) = parse_backend(backend) else {
        return;
    };
    let opts = CheckOptions {
        backends: vec![target],
        dispatch: false,
        fusion: true,
        numeric: false,
    };
    let report = check_graph(g, &opts);
    for b in &report.backends {
        eprintln!(
            "[fusion:{tag}] {}: {} fused ops, {} missed (of {} nodes)",
            b.backend, b.fused_ops, b.missed_fusions, report.nodes,
        );
    }
    // Which structural fusions actually FIRED (the breakdown behind the total).
    let (_g2, rep) =
        rlx_opt::rlx_compile::fusion_pipeline::Fuse::new(target).run_with_report(g.clone());
    eprintln!(
        "[fusion:{tag}] fired: matmul_bias_act={} swiglu={} residual_ln={} residual_rms_norm={} attention_block={} transformer_layer={}",
        rep.fused_matmul_bias_act,
        rep.fused_swiglu,
        rep.fused_residual_ln,
        rep.fused_residual_rms_norm,
        rep.fused_attention_block,
        rep.fused_transformer_layer,
    );
    // Breakdown of the missed fusions (pattern + reason) — the actual headroom.
    let mut miss: std::collections::BTreeMap<String, (usize, Option<String>)> =
        std::collections::BTreeMap::new();
    for d in report
        .diagnostics
        .iter()
        .filter(|d| d.code == "missed-fusion")
    {
        let e = miss.entry(d.message.clone()).or_insert((0, d.hint.clone()));
        e.0 += 1;
    }
    if !miss.is_empty() {
        let mut v: Vec<_> = miss.into_iter().collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1.0));
        eprintln!("[fusion:{tag}] missed-fusion patterns (top):");
        for (msg, (count, hint)) in v.into_iter().take(8) {
            let h = hint.map(|h| format!("  — {h}")).unwrap_or_default();
            eprintln!("    {count:>3}× {msg}{h}");
        }
    }
}

/// Static op-cost/dataflow inspection of a retrieval **BERT** graph (the bge
/// encoder or the cross-encoder reranker) via the SAME rlx-opscope analysis used
/// for the LM — so the ops inspection spans the WHOLE pipeline, not just the LM.
/// Fetches config+weights via hf-hub (cached), rebuilds the graph, and writes
/// `ops_shapes_{tag}.csv` / `ops_dataflow_{tag}.csv`. No execution (device-free).
#[cfg(feature = "dual-encoder")]
fn bert_op_inspect(repo: &str, seq: usize, out: &std::path::Path, tag: &str) -> anyhow::Result<()> {
    use rlx_core::config::BertConfig;
    use rlx_core::weight_map::WeightMap;
    let api = hf_hub::api::sync::ApiBuilder::new().build()?;
    let r = api.model(repo.to_string());
    let config = r.get("config.json")?;
    let weights = r.get("model.safetensors")?;
    let cfg = BertConfig::from_file(&config)?;
    let mut wm = WeightMap::from_file(
        weights
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-utf8 weights path"))?,
    )?;
    let built = rlx_bert::flow::build_bert_built(&cfg, &mut wm, 1, seq)?;
    let (g, _params) = rlx_core::flow_util::graph_from_built(built)?;
    fusion_report(&g, tag, "metal");
    let costs = rlx_opscope::shapes::op_costs(&g);
    let ridge = 30.0f64;
    let mut csv = String::from("id,op,m,k,n,flops,bytes,intensity,roofline\n");
    for c in &costs {
        csv.push_str(&format!(
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
    fs::write(out.join(format!("ops_shapes_{tag}.csv")), &csv)?;
    let flows = rlx_opscope::dataflow::repeated_flow_patterns(&g, 2, 4, 3);
    let mut fcsv = String::from("depth,count,score,tree\n");
    for p in &flows {
        fcsv.push_str(&format!(
            "{},{},{},\"{}\"\n",
            p.depth,
            p.count,
            p.score(),
            p.tree
        ));
    }
    fs::write(out.join(format!("ops_dataflow_{tag}.csv")), &fcsv)?;
    let total_flops: u128 = costs.iter().map(|c| c.flops as u128).sum();
    let mut top: Vec<_> = costs.iter().collect();
    top.sort_by_key(|c| std::cmp::Reverse(c.flops));
    eprintln!(
        "[op-inspect:{tag}] {repo}: {} ops, {:.2} GFLOP @ seq {seq}; heaviest:",
        costs.len(),
        total_flops as f64 / 1e9,
    );
    for c in top.iter().take(4) {
        eprintln!(
            "    {:<16} {}x{}x{}  {:.3} GFLOP  [{}]",
            c.op,
            c.m,
            c.k,
            c.n,
            c.flops as f64 / 1e9,
            rlx_opscope::shapes::roofline_class(c, ridge),
        );
    }
    Ok(())
}

fn op_inspect_pass(weights: &Path, out: &std::path::Path) -> anyhow::Result<()> {
    use rlx_qwen3::{Qwen3Config, build_qwen3_graph_sized};

    eprintln!("\n[op-inspect] building the qwen3 graph for static op analysis (rlx-opscope) …");
    let cfg = Qwen3Config::from_file(&weights.join("config.json"))?;
    let mut loader = rlx_core::SafetensorsMmapLoader::open(weights)?;
    // batch=1, a short seq; lm_head on so the output projection op is included.
    let (g, _params) = build_qwen3_graph_sized(&cfg, &mut loader, 1, 8, true, false)?;
    fusion_report(&g, "lm", "metal");

    // Per-op shape / cost / roofline.
    let costs = rlx_opscope::shapes::op_costs(&g);
    let ridge = 30.0f64; // FLOP:byte ridge for compute- vs memory-bound classing
    let mut csv = String::from("id,op,m,k,n,flops,bytes,intensity,roofline\n");
    for c in &costs {
        csv.push_str(&format!(
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
    fs::write(out.join("ops_shapes.csv"), &csv)?;

    // Recurring dataflow cones (the repeated compute structure).
    let flows = rlx_opscope::dataflow::repeated_flow_patterns(&g, 2, 4, 3);
    let mut flowcsv = String::from("depth,count,score,tree\n");
    for p in &flows {
        flowcsv.push_str(&format!(
            "{},{},{},\"{}\"\n",
            p.depth,
            p.count,
            p.score(),
            p.tree
        ));
    }
    fs::write(out.join("ops_dataflow.csv"), &flowcsv)?;

    // Packed-decode rewrite preview: does `rewrite_matmuls_to_packed` actually
    // fire on the REAL qwen3 decode graph (the integration risk — the flow could
    // emit weights behind a transpose, so a MatMul(x, Param) match would miss)?
    {
        use rlx_qwen3::packed_decode::{PackedWeightInfo, rewrite_matmuls_to_packed};
        let mut loader2 = rlx_core::SafetensorsMmapLoader::open(weights)?;
        match rlx_qwen3::builder::build_qwen3_decode_graph_sized(&cfg, &mut loader2, 1, 32) {
            Ok((mut dg, _params)) => {
                let matmuls = dg
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.op, rlx_ir::Op::MatMul))
                    .count();
                let keys = rewrite_matmuls_to_packed(&mut dg, &|name| {
                    (name.ends_with("_proj.weight") || name.ends_with("lm_head.weight")).then_some(
                        PackedWeightInfo {
                            scheme: rlx_ir::quant::QuantScheme::GgufQ4K,
                            nbytes: 1,
                            n: 0,
                            n_groups: 0,
                        },
                    )
                });
                let dq = dg
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.op, rlx_ir::Op::DequantMatMul { .. }))
                    .count();
                eprintln!(
                    "[packed-decode] decode graph: {matmuls} MatMul → {} rewritten to \
                     DequantMatMul ({dq} total); sample keys: {:?}",
                    keys.len(),
                    keys.iter().take(3).collect::<Vec<_>>(),
                );
            }
            Err(e) => eprintln!("[packed-decode] decode graph build failed: {e}"),
        }
    }

    eprintln!(
        "[op-inspect] {} ops analyzed -> ops_shapes.csv; {} recurring dataflow cones -> ops_dataflow.csv",
        costs.len(),
        flows.len(),
    );
    // On-screen digest: the heaviest ops by FLOPs.
    let mut top: Vec<_> = costs.iter().collect();
    top.sort_by_key(|c| std::cmp::Reverse(c.flops));
    eprintln!("[op-inspect] heaviest ops by FLOPs:");
    for c in top.iter().take(8) {
        eprintln!(
            "    {:<18} {}x{}x{}  {:.1} GFLOP  intensity {:.1}  [{}]",
            c.op,
            c.m,
            c.k,
            c.n,
            c.flops as f64 / 1e9,
            c.intensity(),
            rlx_opscope::shapes::roofline_class(c, ridge),
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // #3: default the token-identical peak-decode levers ON (f16-resident weights
    // ≈ lossless bf16, bake-weight-concat + GQA-native are exact). 71% of decode
    // bytes are weights; f16 halves them. `--no-f16` or an explicit `=0` opts out.
    // #4: mixed-precision KV (K-f16 / V-int8). Forces the host KV path so the
    // quantized mirror actually feeds attention (the Metal GPU-resident path
    // keeps its own on-device KV; the native int8-V kernel is the follow-up).
    if args.iter().any(|a| a == "--kv-quant") {
        unsafe {
            std::env::set_var("RLX_QWEN3_KV_QUANT", "1");
            std::env::set_var("RLX_QWEN3_NO_GPU_KV", "1");
        }
    }
    // These are Metal-only peak-decode levers: F16-resident weights need the
    // backend to convert f32→f16 param bytes at bind (Metal does via
    // `write_weight_from_f32`; the CPU/other backends read the F16-declared
    // param as raw f32 bytes → garbage weights → gibberish). BAKE_WEIGHTS /
    // GQA_NATIVE are likewise Metal decode paths. Only auto-enable on Metal;
    // on other devices leave them off so decode is coherent by default. A user
    // can still force any of them via the env var explicitly.
    let sel_device = {
        let mut d = "metal".to_string();
        let mut j = 1;
        while j < args.len() {
            if args[j] == "--device" && j + 1 < args.len() {
                d = args[j + 1].clone();
                break;
            }
            j += 1;
        }
        d
    };
    let f16_off = args.iter().any(|a| a == "--no-f16");
    if !f16_off && sel_device.eq_ignore_ascii_case("metal") {
        for k in [
            "RLX_QWEN3_F16_WEIGHTS",
            "RLX_QWEN3_BAKE_WEIGHTS",
            "RLX_QWEN3_GQA_NATIVE",
        ] {
            if std::env::var_os(k).is_none() {
                unsafe { std::env::set_var(k, "1") };
            }
        }
    } else if !f16_off && !sel_device.eq_ignore_ascii_case("metal") {
        eprintln!(
            "[memory_probe] device={sel_device}: F16_WEIGHTS/BAKE_WEIGHTS/GQA_NATIVE \
             are Metal-only peak-decode levers — left OFF (they corrupt weights on \
             other backends). Use --device metal to enable, or set the env var to force."
        );
    }
    let mut device = "metal".to_string();
    let mut weights = PathBuf::from(DEFAULT_WEIGHTS);
    let mut policies = "sinks:4:24,retrieval:8:6:4:24,auto:64".to_string();
    let mut max_tokens = 24usize;
    let mut max_seq = 8192usize;
    let mut out_dir = PathBuf::from("memory_probe_out");
    let mut op_inspect = true;
    let mut temp = 0.0f32;
    let mut top_k = 40usize;
    let mut top_p = 1.0f32;
    let mut seed = 0u64;
    // Text-reinjection (D): on recall, retrieve the top-k blocks as TEXT and
    // generate over a clean labeled prompt (no raw-KV splice) so the small LM
    // binds the right entity among competing facts. `--tm-topk` caps how many
    // notes are shown (fewer competitors = better binding).
    let mut text_memory = false;
    let mut tm_topk = 3usize;
    // Widen each retrieved block by this many tokens on both sides so a fact that
    // straddles the fixed block boundary (e.g. a 4-digit code) isn't truncated.
    let mut tm_margin = 16usize;
    // Cross-encoder reranker (C): over-fetch candidates by bi-encoder, then jointly
    // score (question, note) with this HF reranker and keep the top `tm_topk`.
    // Empty = off. Only applies with --text-memory.
    let mut rerank_repo = String::new();
    let mut rerank_overfetch = 16usize;
    // Thinking budget (D recall only): drop `/no_think` so the model reasons in a
    // <think> block before answering, with `tm_think_tokens` of budget. Tests
    // whether reasoning lets a small LM disambiguate a competing fact that plain
    // reflexive decoding binds wrong (canine→dog→Rex vs the salient "Waffles").
    let mut tm_think = false;
    let mut tm_think_tokens = 200usize;
    // Interleaved retrieve-in-the-loop (dynamic mid-reasoning retrieval): the model
    // reasons, requests facts with `SEARCH:` (or the harness re-queries IRCoT-style
    // with the growing reasoning), notes are fed back, repeat up to --max-hops.
    let mut interleave = false;
    let mut max_hops = 3usize;
    let mut hop_tokens = 64usize;
    // Per-aux-model device placement (default = the LM's --device). Lets the
    // retrieval encoder and cross-encoder reranker run on a DIFFERENT accelerator
    // than the LM — e.g. LM on Metal + reranker on ANE (CoreML), or encoder on
    // CPU where f32 GEMM already dispatches to Accelerate/AMX. Empty = follow --device.
    let mut encoder_device = String::new();
    let mut reranker_device = String::new();
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
            "--policies" => {
                i += 1;
                policies = args[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse()?;
            }
            "--max-seq" => {
                i += 1;
                max_seq = args[i].parse()?;
            }
            "--out" => {
                i += 1;
                out_dir = PathBuf::from(&args[i]);
            }
            "--op-inspect" => op_inspect = true,
            "--no-op-inspect" => op_inspect = false,
            "--no-f16" => {}   // handled before arg parse (env defaults)
            "--kv-quant" => {} // handled before arg parse (env)
            "--temp" => {
                i += 1;
                temp = args[i].parse()?;
            }
            "--top-k" => {
                i += 1;
                top_k = args[i].parse()?;
            }
            "--top-p" => {
                i += 1;
                top_p = args[i].parse()?;
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse()?;
            }
            "--text-memory" => text_memory = true,
            "--tm-topk" => {
                i += 1;
                tm_topk = args[i].parse()?;
            }
            "--tm-margin" => {
                i += 1;
                tm_margin = args[i].parse()?;
            }
            "--rerank" => {
                i += 1;
                rerank_repo = args[i].clone();
            }
            "--rerank-overfetch" => {
                i += 1;
                rerank_overfetch = args[i].parse()?;
            }
            "--tm-think" => tm_think = true,
            "--tm-think-tokens" => {
                i += 1;
                tm_think_tokens = args[i].parse()?;
            }
            "--interleave" => interleave = true,
            "--max-hops" => {
                i += 1;
                max_hops = args[i].parse()?;
            }
            "--hop-tokens" => {
                i += 1;
                hop_tokens = args[i].parse()?;
            }
            "--encoder-device" => {
                i += 1;
                encoder_device = args[i].clone();
            }
            "--reranker-device" => {
                i += 1;
                reranker_device = args[i].clone();
            }
            "-h" | "--help" => {
                eprintln!(
                    "memory_probe [--device metal] [--weights DIR] \
                     [--policies sinks:4:24,retrieval:8:6:4:24,auto:64] \
                     [--max-tokens 24] [--max-seq 8192] [--out DIR] [--no-op-inspect] \
                     [--temp 0.0] [--top-k 40] [--top-p 1.0] [--seed 0] \
                     [--text-memory] [--tm-topk 3] [--tm-margin 16] \
                     [--rerank REPO] [--rerank-overfetch 16] \
                     [--tm-think] [--tm-think-tokens 200] \
                     [--interleave] [--max-hops 3] [--hop-tokens 64] \
                     [--encoder-device D] [--reranker-device D]"
                );
                return Ok(());
            }
            other => eprintln!("[memory_probe] ignoring unknown arg: {other}"),
        }
        i += 1;
    }
    fs::create_dir_all(&out_dir)?;

    // Text-reinjection needs the disk-tiered store + a block embedder (mmap-kv).
    #[cfg(not(feature = "mmap-kv"))]
    if text_memory {
        eprintln!(
            "[memory_probe] --text-memory needs --features mmap-kv (or dual-encoder); \
             ignoring (running the raw-KV splice path)."
        );
        text_memory = false;
    }
    if text_memory {
        eprintln!(
            "[memory_probe] text-memory (D) ON: recall retrieves top-{tm_topk} blocks as \
             TEXT and generates over a clean labeled prompt (no raw-KV splice)."
        );
    }
    if !rerank_repo.is_empty() && !text_memory {
        eprintln!(
            "[memory_probe] --rerank only applies with --text-memory (it re-orders the D \
             candidate notes); ignoring."
        );
        rerank_repo.clear();
    }
    if !rerank_repo.is_empty() {
        eprintln!(
            "[memory_probe] rerank (C) ON: over-fetch {rerank_overfetch} notes, cross-encode \
             ({rerank_repo}) vs the question, keep top-{tm_topk}."
        );
    }
    #[cfg(not(feature = "mmap-kv"))]
    if interleave {
        eprintln!(
            "[memory_probe] --interleave needs --features mmap-kv (or dual-encoder); ignoring."
        );
        interleave = false;
    }
    if interleave {
        eprintln!(
            "[memory_probe] interleave ON: dynamic retrieve-in-the-loop, up to {max_hops} hops \
             × {hop_tokens} tok/hop (SEARCH:/ANSWER: protocol + IRCoT fallback)."
        );
    }

    let dev = Device::from_str(&device).map_err(|e| anyhow::anyhow!("--device {device}: {e}"))?;
    // Aux-model devices default to the LM device; override to split across accelerators.
    let enc_dev = if encoder_device.is_empty() {
        dev
    } else {
        Device::from_str(&encoder_device).map_err(|e| anyhow::anyhow!("--encoder-device: {e}"))?
    };
    let rr_dev = if reranker_device.is_empty() {
        dev
    } else {
        Device::from_str(&reranker_device).map_err(|e| anyhow::anyhow!("--reranker-device: {e}"))?
    };
    if enc_dev != dev || rr_dev != dev {
        eprintln!("[memory_probe] aux devices: LM={dev:?} encoder={enc_dev:?} reranker={rr_dev:?}");
    }
    let tok = Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tok.token_to_id(t))
        .collect();

    // The needles and the conversation that buries them. Small budgets + a dozen
    // turns are enough to evict the early facts before they are recalled.
    // 10 needles with PARAPHRASED recall questions (the ask shares few surface
    // words with the plant) — this favors SEMANTIC retrieval (the dual encoder)
    // over lexical/K·K, and measures toward literal 10/10 at selective top-k.
    let needles = [
        Needle {
            plant: "Please remember this important detail: the launch code is 7731.",
            ask: "What number was I told to use for the rocket launch? Just the number.",
            expect: &["7731"],
        },
        Needle {
            plant: "Also remember: my cat is named Waffles.",
            ask: "What did I call my pet feline? One word.",
            expect: &["waffles"],
        },
        Needle {
            plant: "One more thing to remember: the meeting is on Thursday.",
            ask: "Which weekday is the appointment scheduled for? One word.",
            expect: &["thursday"],
        },
        Needle {
            plant: "Note this down: my favorite color is teal.",
            ask: "Which shade do I like the most? One word.",
            expect: &["teal"],
        },
        Needle {
            plant: "Keep in mind: I am staying in hotel room 412.",
            ask: "Which room number is my accommodation? Just the number.",
            expect: &["412"],
        },
        Needle {
            plant: "Remember this: the secret password is swordfish.",
            ask: "What was the confidential passphrase I set? One word.",
            expect: &["swordfish"],
        },
        Needle {
            plant: "Also note: my dog is called Rex.",
            ask: "What is the name of my canine companion? One word.",
            expect: &["rex"],
        },
        Needle {
            plant: "Important: the project codename is Phoenix.",
            ask: "What is the secret name of the initiative? One word.",
            expect: &["phoenix"],
        },
        Needle {
            plant: "For the record: the total budget is 5000 dollars.",
            ask: "How much money was allocated overall? Just the number.",
            expect: &["5000"],
        },
        Needle {
            plant: "Don't forget: my flight departs at 6 AM.",
            ask: "At what time does my plane take off?",
            expect: &["6"],
        },
    ];
    // Filler pool (varied, to add unrelated context tokens between/after plants).
    const FILLER: &[&str] = &[
        "Tell me one fun fact about the ocean.",
        "What is 15 times 12?",
        "Describe a mountain in one short sentence.",
        "What is the capital of France?",
        "Give me a synonym for happy.",
        "Name a musical instrument.",
        "Write a short sentence about the weather.",
        "What is the square root of 144?",
        "Name a type of tree.",
        "Suggest a healthy breakfast.",
        "What color is the sky at noon?",
        "Give me a word that rhymes with cat.",
        "Name a planet in the solar system.",
        "What is two plus two?",
        "Describe a river in one sentence.",
        "Name a common house pet.",
        "What language is spoken in Brazil?",
        "Give me a sentence about music.",
        "Name a fruit that is red.",
        "What is the opposite of hot?",
    ];
    // Plant each needle, each followed by 2 filler turns (bury it), then a final
    // burst of filler, then recall all 10 in order.
    let mut script: Vec<Turn> = Vec::new();
    let mut f = 0usize;
    // One filler per plant keeps the buried context long enough to evict the early
    // needles past the resident window, while keeping the total short enough that
    // qwen3-0.6B still generates coherently at recall time (it collapses to an
    // immediate EOS past ~1k tokens — a model limit, not a retrieval one).
    let fillers_per_plant: usize = std::env::var("RLX_PROBE_FILL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    for i in 0..needles.len() {
        script.push(Turn::Plant(i));
        for _ in 0..fillers_per_plant {
            script.push(Turn::Filler(FILLER[f % FILLER.len()]));
            f += 1;
        }
    }
    for i in 0..needles.len() {
        script.push(Turn::Recall(i));
    }

    let specs: Vec<String> = policies.split(',').map(|s| s.trim().to_string()).collect();
    let sampler = make_sampler(temp, top_k, top_p, seed);
    let sampler_desc = if temp <= 0.0 {
        format!("greedy(argmax), top_k={top_k} (deterministic)")
    } else {
        format!("top-k sample temp={temp} top_k={top_k} top_p={top_p} seed={seed}")
    };
    eprintln!(
        "[memory_probe] device={device} weights={weights:?} policies={:?} out={out_dir:?}\n\
         [memory_probe] sampler: {sampler_desc}",
        specs
    );

    let mut results = Vec::new();
    for spec in &specs {
        eprintln!("\n=== policy: {spec} ===");
        let r = run_policy(
            spec,
            &weights,
            dev,
            enc_dev,
            rr_dev,
            &tok,
            &eos,
            max_seq,
            max_tokens,
            sampler,
            text_memory,
            tm_topk,
            tm_margin,
            &rerank_repo,
            rerank_overfetch,
            tm_think,
            tm_think_tokens,
            interleave,
            max_hops,
            hop_tokens,
            &script,
            &needles,
        )?;
        // Write per-policy CSV/DOT.
        let tag = spec.replace([':', '/'], "_");
        if !r.retention_csv.is_empty() {
            fs::write(
                out_dir.join(format!("retention_{tag}.csv")),
                &r.retention_csv,
            )?;
        }
        if !r.inspect_csv.is_empty() {
            fs::write(
                out_dir.join(format!("inspect_stats_{tag}.csv")),
                &r.inspect_csv,
            )?;
            fs::write(
                out_dir.join(format!("inspect_hist_{tag}.csv")),
                &r.inspect_hist_csv,
            )?;
            fs::write(out_dir.join(format!("dataflow_{tag}.dot")), &r.dataflow_dot)?;
        }
        results.push(r);
    }

    if op_inspect {
        if let Err(e) = op_inspect_pass(&weights, &out_dir) {
            eprintln!("[op-inspect] skipped: {e}");
        }
        // Whole-pipeline ops: also cost the retrieval encoder + reranker BERT
        // graphs (the LM graph + context-construction telemetry are already
        // captured above / per-policy).
        #[cfg(feature = "dual-encoder")]
        {
            let enc_repo = std::env::var("RLX_QWEN3_EMBED_REPO")
                .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());
            if let Err(e) = bert_op_inspect(&enc_repo, 256, &out_dir, "encoder") {
                eprintln!("[op-inspect:encoder] skipped: {e}");
            }
            if !rerank_repo.is_empty() {
                if let Err(e) = bert_op_inspect(&rerank_repo, 192, &out_dir, "reranker") {
                    eprintln!("[op-inspect:reranker] skipped: {e}");
                }
            }
        }
    }

    // ── Unified analysis ────────────────────────────────────────────────────
    println!("\n\n════════════════════════ ANALYSIS ════════════════════════");
    for r in &results {
        let hits = r.recall.iter().filter(|(_, h, _)| *h).count();
        println!("\n── policy: {} ──", r.policy);
        println!("recall: {hits}/{} facts recalled", r.recall.len());
        for (q, hit, reply) in &r.recall {
            let mark = if *hit { "HIT " } else { "MISS" };
            let snip: String = reply.replace('\n', " ").chars().take(56).collect();
            println!("  [{mark}] {q}\n         -> {snip}");
        }
        println!("\ncache/context:");
        for line in r.retention_report.lines() {
            println!("  {line}");
        }
        if !r.resident_spark.is_empty() {
            println!("  resident │{}│", r.resident_spark);
            println!(
                "  eff.ctx  │{}│  peak {} tokens → {:.2}x extension over resident",
                r.ctx_spark, r.ctx_peak, r.extension
            );
        }
        println!("\nKV / selection data (latest snapshot per stream):");
        for line in r.inspect_report.lines() {
            println!("  {line}");
        }
        println!("\nthroughput (per turn):");
        println!("  turn kind     ctx_before  gen  ttft_ms   tps");
        for p in &r.perf {
            println!(
                "  {:>4} {:7} {:>10} {:>4} {:>8.0} {:>5.0}",
                p.turn, p.kind, p.ctx_before, p.gen_toks, p.ttft_ms, p.decode_tps
            );
        }
    }

    // Cross-policy recall comparison — the headline: does selective retention
    // beat amnesia?
    println!("\n── recall comparison ──");
    println!("  {:<28} recall   ctx-extension", "policy");
    for r in &results {
        let hits = r.recall.iter().filter(|(_, h, _)| *h).count();
        println!(
            "  {:<28} {}/{}      {:.2}x",
            r.policy,
            hits,
            r.recall.len(),
            r.extension
        );
    }
    println!(
        "\nCSV/DOT written under {out_dir:?} (retention_*, inspect_stats_*, inspect_hist_*, dataflow_*, ops_*)."
    );
    Ok(())
}
