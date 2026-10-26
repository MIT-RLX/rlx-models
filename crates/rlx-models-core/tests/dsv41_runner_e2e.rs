// RLX — versatile ML compiler + runtime. GPLv3.
//! End-to-end: a DeepSeek-V4.1 checkpoint on disk, text out.
//!
//! Everything else in this port tests a graph against a reference. This tests
//! the *runner*: that a directory containing `config.json`, `*.safetensors` and
//! `tokenizer.json` can be opened, stepped, sampled from and detokenized — the
//! whole path a user actually takes.
//!
//! The checkpoint is synthesized rather than downloaded, at toy dimensions but
//! with the **real architecture**: CSA2 sources, the Engram, the hyper-connection
//! stream, the MoE. Its tensor list comes from
//! [`DeepseekV41Spec::expected_tensors`], so the file contains exactly what the
//! builders ask for and nothing else — a name or shape the port gets wrong fails
//! here as a load error rather than silently reading zeros.
//!
//! What this does *not* cover is scale: the released model is 552 B across 48
//! shards, and nothing here says anything about running that.

use anyhow::Result;
use rlx_models_core::dsv41::DeepseekV41Spec;
use rlx_models_core::dsv41_runner::{MoeExecution, RunnerOptions, SampleOpts, V41Runner, Vocab};
use rlx_models_core::weight_loader::SyntheticLoader;
use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype as StDtype, TensorView};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A text-only V4.1 config small enough to run in a test, with every structural
/// feature the released model has: two KV sources, an index source above them,
/// an Engram layer, hyper-connections, and a routed MoE.
fn toy_config(vocab: usize) -> Value {
    json!({
        "model_type": "deepseek_v41",
        "vocab_size": vocab,
        "hidden_size": 16,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "head_dim": 16,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 8,
        "o_groups": 1,
        "o_lora_rank": 4,
        "hc_mult": 2,
        "compress_ratios": [0, 0, 2, 2, 2, 2],
        "kv_source_layer_ids": [2],
        "index_source_layer_ids": [2, 4],
        "index_head_dim": 8,
        "index_n_heads": 1,
        "index_topk": 2,
        "sliding_window": 8,
        "moe_intermediate_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "engram_layer_ids": [1],
        "engram_num_embeddings": [64],
        "engram_max_ngram_size": 2,
        "engram_vocab_size": vocab,
        "engram_compressed_vocab_size": 16,
        "engram_n_heads": 1,
        "engram_head_dim": 4,
        "engram_pad_token_id": 0,
    })
}

/// A minimal word-level tokenizer whose vocabulary is exactly `vocab` ids.
///
/// Word-level keeps the mapping legible: the prompt is whitespace-split and each
/// word is one id, so a test can reason about positions directly.
fn tokenizer_json(vocab: usize) -> String {
    let mut map = serde_json::Map::new();
    map.insert("<unk>".into(), json!(0));
    for i in 1..vocab {
        map.insert(format!("w{i}"), json!(i));
    }
    serde_json::to_string(&json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false},
        "model": {
            "type": "WordLevel",
            "vocab": Value::Object(map),
            "unk_token": "<unk>"
        }
    }))
    .unwrap()
}

/// Write a complete checkpoint directory.
///
/// Tensors are stored as plain F32 — the quantized layouts have their own
/// coverage against real checkpoint bytes, and mixing them in here would test
/// the dequantizer rather than the runner.
fn write_checkpoint(dir: &Path, cfg: &Value) -> Result<DeepseekV41Spec> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(cfg)?)?;
    let spec = DeepseekV41Spec::from_config(cfg)?;
    spec.validate()?;
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json(spec.vocab_size))?;

    let mut data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for need in spec.expected_tensors() {
        let v = SyntheticLoader::values(&need.name, &need.shape);
        data.push((
            need.name.clone(),
            need.shape.clone(),
            bytemuck::cast_slice(&v).to_vec(),
        ));
    }
    let views: HashMap<String, TensorView> = data
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                TensorView::new(StDtype::F32, s.clone(), b).expect("tensor view"),
            )
        })
        .collect();
    serialize_to_file(&views, None, &dir.join("model.safetensors"))?;
    Ok(spec)
}

/// Pins `RLX_ARENA_NO_REUSE=1` for the duration, under a process-wide lock.
///
/// The arena-reuse nondeterminism tracked by
/// `arena_reuse_is_deterministic` in the parity suite reaches this file too: the
/// paged and single-graph prefill paths agree *exactly* nine runs in ten and
/// differ by ~8% of the logit scale on the tenth. That is the known bug, not a
/// tolerance question — the deviation is bimodal, never in between — so the test
/// pins the arena rather than loosening a bound that would hide it.
///
/// The env var is process-global, hence the lock; it only ever makes a
/// concurrent compile *more* conservative.
struct ArenaPin(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl ArenaPin {
    fn new() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        rlx_ir::env::set("RLX_ARENA_NO_REUSE", "1");
        ArenaPin(g)
    }
}

impl Drop for ArenaPin {
    fn drop(&mut self) {
        rlx_ir::env::remove("RLX_ARENA_NO_REUSE");
    }
}

fn scratch(name: &str) -> PathBuf {
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let d = PathBuf::from(base).join(format!("rlx_dsv41_e2e_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// The word-level vocabulary the toy tokenizer defines, as the Engram needs it.
///
/// Supplied explicitly rather than read from `tokenizer.json` so these tests run
/// with or without the `tokenizer` feature — the id-level API is the one that
/// always exists, and it deserves the coverage.
fn word_vocab(vocab: usize) -> Vocab {
    let pieces: Vec<String> = (0..vocab)
        .map(|i| {
            if i == 0 {
                "<unk>".into()
            } else {
                format!("w{i}")
            }
        })
        .collect();
    Vocab {
        decoded: pieces.clone(),
        pieces,
    }
}

fn runner(dir: &Path, moe: MoeExecution) -> V41Runner {
    let cfg: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let vocab = cfg["vocab_size"].as_u64().unwrap() as usize;
    V41Runner::open_with_vocab(
        dir,
        RunnerOptions {
            moe,
            expert_budget_bytes: 1 << 20,
            ..Default::default()
        },
        word_vocab(vocab),
    )
    .expect("runner opens the checkpoint")
}

/// The whole path: open a checkpoint directory, generate from a text prompt,
/// get text back.
#[cfg(feature = "tokenizer")]
#[test]
fn generates_text_from_a_checkpoint_directory() {
    let dir = scratch("text");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let mut r = runner(&dir, MoeExecution::Resident);

    let out = r
        .generate("w3 w7 w11", &SampleOpts::greedy(4))
        .expect("generation succeeds");
    assert_eq!(out.prompt_tokens, 3, "three whitespace-separated words");
    assert_eq!(out.tokens.len(), 4, "asked for four new tokens");
    for &t in &out.tokens {
        assert!(
            (t as usize) < r.spec().vocab_size,
            "generated id {t} is outside the vocabulary"
        );
    }
    assert!(!out.text.is_empty(), "detokenized text should not be empty");
}

/// Greedy decoding must be reproducible: same checkpoint, same prompt, same
/// tokens. A runner whose output wanders is untestable, and the arena-reuse
/// nondeterminism this port already tracks would show up right here.
#[test]
fn greedy_generation_is_reproducible() {
    let dir = scratch("repro");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");

    let a = runner(&dir, MoeExecution::Resident)
        .generate_ids(&[3, 7, 11], &SampleOpts::greedy(6))
        .expect("first run");
    let b = runner(&dir, MoeExecution::Resident)
        .generate_ids(&[3, 7, 11], &SampleOpts::greedy(6))
        .expect("second run");
    assert_eq!(a.tokens, b.tokens, "greedy decoding is not reproducible");
}

/// Priming the cache from a batched prefill must leave it exactly where
/// stepping the same prompt one token at a time would have.
///
/// This is what makes prompt processing one graph run instead of `seq` of them,
/// and it is the kind of change that is silently *almost* right: a window off by
/// one row, or a partial compressor group dropped, still decodes fluently and
/// still produces plausible logits. So the check is not "does it run" but "does
/// the next token's logits match the slow path, element for element".
#[test]
fn a_primed_cache_decodes_identically_to_a_stepped_one() {
    use rlx_models_core::dsv41_decode::V41DecodeCache;

    let dir = scratch("prime");
    let spec = write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    // long enough to fill the 8-slot window and leave a partial compressor group
    let ids: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11, 1, 6, 13, 4];
    let vocab = spec.vocab_size;

    let stepped = {
        let mut r = runner(&dir, MoeExecution::Resident);
        let mut cache = V41DecodeCache::new(&spec);
        r.step_prompt_for_test(&ids, &mut cache)
            .expect("stepped prompt")
    };
    let primed = {
        let mut r = runner(&dir, MoeExecution::Resident);
        let mut cache = V41DecodeCache::new(&spec);
        r.prefill_into_test(&ids, &mut cache)
            .expect("primed prompt")
    };
    assert_eq!(stepped.len(), vocab);
    for (i, (a, b)) in primed.iter().zip(&stepped).enumerate() {
        assert!(
            (a - b).abs() <= 2e-4 * b.abs().max(1.0),
            "logit {i}: primed {a} vs stepped {b}"
        );
    }

    // and the caches must continue the same way, not merely agree once
    let mut r = runner(&dir, MoeExecution::Resident);
    let mut c_step = V41DecodeCache::new(&spec);
    r.step_prompt_for_test(&ids, &mut c_step).unwrap();
    let mut c_prime = V41DecodeCache::new(&spec);
    r.prefill_into_test(&ids, &mut c_prime).unwrap();
    let nxt = 8u32;
    let a = r.step_for_test(ids.len(), nxt, &ids, &mut c_step).unwrap();
    let b = r.step_for_test(ids.len(), nxt, &ids, &mut c_prime).unwrap();
    for (i, (x, y)) in b.iter().zip(&a).enumerate() {
        assert!(
            (x - y).abs() <= 2e-4 * y.abs().max(1.0),
            "one step past the prompt, logit {i}: primed {x} vs stepped {y}"
        );
    }
}

/// Sampling is seeded, so a temperature run is reproducible too — and a
/// different seed should generally take a different path.
#[test]
fn sampling_is_seeded() {
    let dir = scratch("seed");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let opts = |seed: u64| SampleOpts {
        max_new_tokens: 8,
        temperature: 1.0,
        top_k: 8,
        top_p: 0.95,
        seed,
        stop: Vec::new(),
    };
    let a = runner(&dir, MoeExecution::Resident)
        .generate_ids(&[3, 7], &opts(1))
        .unwrap();
    let b = runner(&dir, MoeExecution::Resident)
        .generate_ids(&[3, 7], &opts(1))
        .unwrap();
    assert_eq!(a.tokens, b.tokens, "same seed must replay exactly");
}

/// A stop token ends generation and is reported, rather than being emitted.
#[test]
fn a_stop_token_ends_generation() {
    let dir = scratch("stop");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let mut r = runner(&dir, MoeExecution::Resident);
    let first = r
        .generate_ids(&[3, 7, 11], &SampleOpts::greedy(1))
        .expect("one token")
        .tokens[0];

    let out = r
        .generate_ids(
            &[3, 7, 11],
            &SampleOpts {
                stop: vec![first],
                ..SampleOpts::greedy(4)
            },
        )
        .expect("stopped generation");
    assert!(out.tokens.is_empty(), "the stop token must not be emitted");
    assert_eq!(out.stopped_on, Some(first));
}

/// A prompt token outside the vocabulary is an error naming the position, not a
/// panic inside the embedding gather.
#[test]
fn an_out_of_range_prompt_token_is_rejected() {
    let dir = scratch("range");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let e = runner(&dir, MoeExecution::Resident)
        .generate_ids(&[3, 999], &SampleOpts::greedy(1))
        .unwrap_err()
        .to_string();
    assert!(e.contains("999"), "error should name the offending id: {e}");
    assert!(e.contains('1'), "error should name the position: {e}");
}

/// Paged decode must produce the same logits as the single-graph decode, token
/// for token.
///
/// This is the claim that makes paging worth anything: the runner routes on the
/// host, pages in only `num_experts_per_tok` experts per layer, stitches five
/// graphs per layer together across host-held state — and must land on exactly
/// the answer the monolithic graph gives. Anything less and paging is a
/// different model, not a cheaper one.
#[test]
fn paged_decode_matches_the_single_graph_decode() {
    use rlx_models_core::dsv41_decode::{V41DecodeCache, V41DecodePlan, build_deepseek_v41_decode};
    use rlx_models_core::dsv41_engram::EngramHashPlan;
    use rlx_models_core::dsv41_paged::PagedStepper;
    use rlx_models_core::dsv41_pager::ExpertPager;
    use rlx_models_core::dsv41_weights::{StreamingLoader, WeightIndex};
    use rlx_runtime::{Device, Session};
    use std::sync::Arc;

    let dir = scratch("paged");
    let spec = write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let ids: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11, 1];

    // the Engram row ids, exactly as the runner computes them
    let e = spec.engram.as_ref().expect("toy has an engram");
    let vocab = spec.vocab_size;
    let pieces: Vec<String> = (0..vocab).map(|i| format!("w{i}")).collect();
    let (map, _) = rlx_models_core::dsv41_engram::compress_token_map(&pieces, &pieces);
    let hash = EngramHashPlan::new(e, &map).expect("hash plan");
    let compressed: Vec<u32> = ids.iter().map(|&i| map[i as usize]).collect();

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );

    // ── paged ──
    let pager = Arc::new(ExpertPager::new(
        WeightIndex::open(&dir).unwrap(),
        32,
        // deliberately tiny: a couple of experts, so eviction actually happens
        4 * 1024,
    ));
    let mut pw = StreamingLoader::open(&dir, 32).unwrap();
    let mut stepper =
        PagedStepper::new(&spec, &mut pw, Arc::clone(&pager), Device::Cpu).expect("stepper");
    let mut paged_cache = V41DecodeCache::new(&spec);

    // ── monolithic ──
    let mut mono_cache = V41DecodeCache::new(&spec);

    for (pos, &tok) in ids.iter().enumerate() {
        let plan = V41DecodePlan::new(&spec, pos);
        let rows = hash.hash_ids(&compressed[pos..pos + 1], None, &compressed[..pos]);

        // monolithic
        let mut w = StreamingLoader::open(&dir, 32).unwrap();
        let mut packed = HashMap::new();
        let inputs = rlx_models_core::dsv41_graph::V41Inputs {
            engram_rows: rows.clone(),
            image_positions: Vec::new(),
            ..Default::default()
        };
        let (g, params, names) =
            build_deepseek_v41_decode(&spec, &mut w, pos, &inputs, &mut packed).unwrap();
        let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        let idf = [tok as f32];
        let cached = mono_cache.step_inputs(&plan);
        let cols_m = e.n_hash_cols();
        let eng_m: Vec<(String, Vec<f32>)> = e
            .layer_ids
            .iter()
            .enumerate()
            .map(|(k, &il)| {
                (
                    rlx_models_core::dsv41_decode::names::engram_rows(il),
                    rows[k * cols_m..(k + 1) * cols_m]
                        .iter()
                        .map(|&v| v as f32)
                        .collect(),
                )
            })
            .collect();
        let mut feed: Vec<(&str, &[f32])> = vec![("input_ids", idf.as_slice())];
        for (n, v) in &eng_m {
            feed.push((n.as_str(), v.as_slice()));
        }
        for (n, v) in &cached {
            feed.push((n.as_str(), *v));
        }
        let mono_out = sess.run(&feed);
        mono_cache.apply(&plan, &names, &mono_out).unwrap();

        // paged
        let pc: Vec<(String, &[f32])> = paged_cache.step_inputs(&plan).into_iter().collect();
        let key = format!("{}|{:?}", plan.cache_len, plan.sources);
        // the Engram rows are a graph input, one per Engram layer
        let cols = e.n_hash_cols();
        let eng: Vec<(String, Vec<f32>)> = e
            .layer_ids
            .iter()
            .enumerate()
            .map(|(k, &il)| {
                (
                    rlx_models_core::dsv41_decode::names::engram_rows(il),
                    rows[k * cols..(k + 1) * cols]
                        .iter()
                        .map(|&v| v as f32)
                        .collect(),
                )
            })
            .collect();
        let (logits, cnames, cvals) = stepper
            .step(&mut pw, &plan, tok, &eng, &pc, &key)
            .unwrap_or_else(|e| panic!("paged step at {pos}: {e}"));
        drop(pc);
        paged_cache.apply(&plan, &cnames, &cvals).unwrap();

        assert_eq!(logits.len(), vocab, "pos {pos}: logits width");
        let want = &mono_out[0];
        for (i, (a, b)) in logits.iter().zip(want).enumerate() {
            assert!(
                (a - b).abs() <= 2e-4 * b.abs().max(1.0),
                "pos {pos}, logit {i}: paged {a} vs single-graph {b}"
            );
        }
    }

    let s = pager.stats();
    assert!(s.misses > 0, "the pager was never exercised");
    assert!(
        s.evictions > 0,
        "a 4 KB budget should have forced evictions; stats {s:?}"
    );
    assert!(
        s.resident_bytes <= pager.budget_bytes().max(4 * 1024),
        "resident {} exceeded the budget",
        s.resident_bytes
    );
    println!(
        "paged decode matched on {} tokens; pager {} hits / {} misses, {} evictions",
        ids.len(),
        s.hits,
        s.misses,
        s.evictions
    );
}

/// The two execution modes are the same model.
///
/// `MoeExecution::Paged` exists so a checkpoint whose expert banks do not fit
/// can still be run — not so it can be run differently. Driving both through the
/// public API and requiring identical tokens is what keeps that an honest claim.
#[test]
fn paged_and_resident_generate_the_same_tokens() {
    let dir = scratch("modes");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let prompt = [5u32, 9, 2, 14];

    let resident = runner(&dir, MoeExecution::Resident)
        .generate_ids(&prompt, &SampleOpts::greedy(6))
        .expect("resident run");

    let mut paged_runner = runner(&dir, MoeExecution::Paged);
    let paged = paged_runner
        .generate_ids(&prompt, &SampleOpts::greedy(6))
        .expect("paged run");

    assert_eq!(
        paged.tokens, resident.tokens,
        "paged and resident execution disagree"
    );
    let stats = paged_runner
        .pager_stats()
        .expect("a paged runner reports pager stats");
    assert!(stats.misses > 0, "the paged run never touched the pager");
    assert!(
        stats.resident_bytes <= 1 << 20,
        "resident {} exceeded the 1 MiB budget the test asked for",
        stats.resident_bytes
    );
    println!(
        "paged == resident on {} tokens; pager hit rate {:.2}, {} bytes resident",
        paged.tokens.len(),
        stats.hit_rate(),
        stats.resident_bytes
    );
}

/// A checkpoint with an Engram cannot be driven from ids alone, and must say so
/// clearly — at the point generation is attempted, not at open, because
/// `open_with_vocab` opens first and supplies the vocabulary second.
#[test]
fn an_engram_checkpoint_without_a_vocabulary_says_so() {
    let dir = scratch("novocab");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");

    // no vocabulary, and without the feature no tokenizer.json is read either
    let mut r = V41Runner::open(&dir, RunnerOptions::default()).expect("opening still succeeds");
    #[cfg(feature = "tokenizer")]
    {
        // with the feature, tokenizer.json supplies the vocabulary and it works
        assert!(r.generate_ids(&[3, 7], &SampleOpts::greedy(1)).is_ok());
    }
    #[cfg(not(feature = "tokenizer"))]
    {
        let e = r
            .generate_ids(&[3, 7], &SampleOpts::greedy(1))
            .unwrap_err()
            .to_string();
        assert!(e.contains("Engram"), "unhelpful error: {e}");
        assert!(e.contains("open_with_vocab"), "no way forward offered: {e}");
        // and supplying one makes it work
        r.set_vocab(word_vocab(32)).expect("vocab accepted");
        assert!(r.generate_ids(&[3, 7], &SampleOpts::greedy(1)).is_ok());
    }
}
/// Decode must keep matching prefill *past* the point where compiled steps start
/// being reused.
///
/// The runner caches a compiled session per step shape, and once the sliding
/// window fills, different positions start sharing one. Two things then have to
/// be true: the shape key must capture everything that changes an input's width
/// (the compressed cache grows while `fires`/`group_filled` repeat), and nothing
/// position-dependent may be baked into the graph (the Engram's n-gram rows are
/// an input for exactly this reason).
///
/// Getting either wrong is invisible until a shape actually collides — here at
/// sequence length 10, where position 9 reuses position 7's graph. Before this
/// test, that reuse put the error at 7e-2 while every shorter prompt agreed to
/// 1e-8.
#[test]
fn decode_matches_prefill_across_compiled_step_reuse() {
    let dir = scratch("reuse");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let all: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11, 1, 6, 13, 4];

    // window_size is 8, so 7..=11 spans the first collision
    for n in [7usize, 9, 10, 11] {
        let ids = &all[..n];
        let mut r = runner(&dir, MoeExecution::Resident);
        let mut c = rlx_models_core::dsv41_decode::V41DecodeCache::new(r.spec());
        let stepped = r.step_prompt_for_test(ids, &mut c).unwrap();
        let pre = r.prefill_logits(ids).unwrap();
        let maxd = pre
            .iter()
            .zip(&stepped)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            maxd < 1e-4,
            "seq {n}: decode and prefill differ by {maxd:.3e}"
        );
    }
}

/// Paged prefill must produce the same logits *and the same cache* as the
/// single-graph prefill.
///
/// A prompt is where paging is hardest: a decode step's expert union is exactly
/// `top_k`, but `seq` rows pick a union that can reach the whole bank, so the
/// gathered bank's width varies per layer and the tail graph is compiled per
/// shape. Getting the slot remapping wrong there is invisible in the logits of a
/// short prompt and wrong everywhere else — so this checks the logits *and*
/// continues decoding from the primed cache.
#[test]
fn paged_prefill_matches_the_single_graph_prefill() {
    use rlx_models_core::dsv41_decode::V41DecodeCache;

    let _arena = ArenaPin::new();

    let dir = scratch("pagedpre");
    let spec = write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let ids: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11, 1, 6, 13];

    let mut resident = runner(&dir, MoeExecution::Resident);
    let mut c_res = V41DecodeCache::new(&spec);
    let want = resident.prefill_into_test(&ids, &mut c_res).unwrap();

    let mut paged = runner(&dir, MoeExecution::Paged);
    let mut c_pag = V41DecodeCache::new(&spec);
    let got = paged.prefill_into_test(&ids, &mut c_pag).unwrap();

    assert_eq!(got.len(), spec.vocab_size);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let maxd = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "paged vs single-graph prefill: max |Δ| {maxd:.3e}, vector scale {scale:.3e}, rel {:.3e}",
        maxd / scale.max(1e-9)
    );
    // Tight on purpose. The two paths do the same arithmetic in the same order,
    // so the honest expectation is bit-equality; the deviation when this fails is
    // ~8% of the vector scale, not the 1e-4 drift a looser bound would excuse.
    assert!(
        maxd <= 1e-4 * scale.max(1.0),
        "paged prefill differs from the single-graph path by {maxd:.3e} on a scale of {scale:.3e}"
    );

    // the caches must continue the same way too, not merely agree once
    let nxt = 8u32;
    let a = resident
        .step_for_test(ids.len(), nxt, &ids, &mut c_res)
        .unwrap();
    let b = paged
        .step_for_test(ids.len(), nxt, &ids, &mut c_pag)
        .unwrap();
    for (i, (x, y)) in b.iter().zip(&a).enumerate() {
        assert!(
            (x - y).abs() <= 2e-4 * y.abs().max(1.0),
            "one step past a paged prefill, logit {i}: {x} vs {y}"
        );
    }

    let s = paged.pager_stats().expect("paged runner reports stats");
    assert!(s.misses > 0, "the prompt pass never touched the pager");
    println!(
        "paged prefill matched on a {}-token prompt; pager {} hits / {} misses",
        ids.len(),
        s.hits,
        s.misses
    );
}

/// End to end in paged mode: the prompt goes through the paged prefill and
/// generation continues through paged decode, landing on the resident answer.
#[test]
fn fully_paged_generation_matches_resident() {
    let dir = scratch("fullpaged");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let prompt = [5u32, 9, 2, 14, 7, 3];

    let resident = runner(&dir, MoeExecution::Resident)
        .generate_ids(&prompt, &SampleOpts::greedy(5))
        .expect("resident");
    let mut pr = runner(&dir, MoeExecution::Paged);
    let paged = pr
        .generate_ids(&prompt, &SampleOpts::greedy(5))
        .expect("paged");
    assert_eq!(
        paged.tokens, resident.tokens,
        "fully paged generation disagrees with resident"
    );
}

/// The runner on a **quantized** checkpoint, which is the only kind the released
/// model ships as.
///
/// Every other test here runs on dense F32 weights, so none of them touch the
/// three scale layouts a real load goes through: FP8 tiles for the projections,
/// FP4 nibble pairs for the routed experts, FP8 row-groups for the Engram table.
/// Those are covered against real checkpoint bytes in isolation, but "the
/// dequantizer is right" and "the runner reads a quantized directory correctly"
/// are different claims.
///
/// The comparison is against a dense checkpoint holding *exactly the values the
/// quantized one decodes to*, so a mismatch means a layout was misread rather
/// than that quantization is lossy.
#[test]
fn the_runner_reads_a_quantized_checkpoint() {
    use rlx_models_core::dsv41_weights::{CheckpointFormat, write_synthetic_checkpoint_as};

    let cfg = {
        let mut c = toy_config(32);
        // widths that are whole multiples of the block, so every tensor the
        // released model quantizes is quantized here too
        c["hidden_size"] = serde_json::json!(32);
        c["head_dim"] = serde_json::json!(32);
        c["moe_intermediate_size"] = serde_json::json!(32);
        c["quantization_config"] = serde_json::json!({ "weight_block_size": [8, 8] });
        c
    };

    let qdir = scratch("quant");
    let decoded = write_synthetic_checkpoint_as(&qdir, &cfg, CheckpointFormat::Quantized)
        .expect("quantized checkpoint written");
    std::fs::write(qdir.join("tokenizer.json"), tokenizer_json(32)).unwrap();
    assert!(
        decoded.values().any(|v| !v.is_empty()),
        "nothing was written"
    );

    let mut r = V41Runner::open_with_vocab(
        &qdir,
        RunnerOptions {
            block: 8,
            ..Default::default()
        },
        word_vocab(32),
    )
    .expect("the runner opens a quantized checkpoint");
    let ids = [5u32, 9, 2, 14];
    let got = r.prefill_logits(&ids).expect("quantized prefill");

    // the same model, stored densely at the values the quantized one decodes to
    let ddir = scratch("quant_ref");
    std::fs::create_dir_all(&ddir).unwrap();
    std::fs::write(
        ddir.join("config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();
    std::fs::write(ddir.join("tokenizer.json"), tokenizer_json(32)).unwrap();
    {
        use safetensors::serialize_to_file;
        use safetensors::tensor::{Dtype as StDtype, TensorView};
        let spec = rlx_models_core::dsv41::DeepseekV41Spec::from_config(&cfg).unwrap();
        let data: Vec<(String, Vec<usize>, Vec<u8>)> = spec
            .expected_tensors()
            .into_iter()
            .map(|n| {
                let v = decoded.get(&n.name).expect("every tensor was decoded");
                (n.name, n.shape, bytemuck::cast_slice(v).to_vec())
            })
            .collect();
        let views: HashMap<String, TensorView> = data
            .iter()
            .map(|(n, s, b)| {
                (
                    n.clone(),
                    TensorView::new(StDtype::F32, s.clone(), b).unwrap(),
                )
            })
            .collect();
        serialize_to_file(&views, None, &ddir.join("model.safetensors")).unwrap();
    }
    let want = V41Runner::open_with_vocab(&ddir, RunnerOptions::default(), word_vocab(32))
        .expect("dense reference opens")
        .prefill_logits(&ids)
        .expect("dense prefill");

    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 2e-4 * b.abs().max(1.0),
            "logit {i}: quantized {a} vs its own decoded values {b}"
        );
    }
}

/// Paged execution on a quantized checkpoint — the combination the released
/// model actually needs, and the one where the pager's packed cache pays off.
#[test]
fn paged_execution_on_a_quantized_checkpoint() {
    use rlx_models_core::dsv41_weights::{CheckpointFormat, write_synthetic_checkpoint_as};

    let cfg = {
        let mut c = toy_config(32);
        c["hidden_size"] = serde_json::json!(32);
        c["head_dim"] = serde_json::json!(32);
        c["moe_intermediate_size"] = serde_json::json!(32);
        c["quantization_config"] = serde_json::json!({ "weight_block_size": [8, 8] });
        c
    };
    let dir = scratch("quantpaged");
    write_synthetic_checkpoint_as(&dir, &cfg, CheckpointFormat::Quantized).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json(32)).unwrap();

    let opts = |moe| RunnerOptions {
        moe,
        block: 8,
        expert_budget_bytes: 1 << 16,
        ..Default::default()
    };
    let resident = V41Runner::open_with_vocab(&dir, opts(MoeExecution::Resident), word_vocab(32))
        .unwrap()
        .generate_ids(&[5, 9, 2, 14], &SampleOpts::greedy(4))
        .expect("resident on quantized");
    let mut pr =
        V41Runner::open_with_vocab(&dir, opts(MoeExecution::Paged), word_vocab(32)).unwrap();
    let paged = pr
        .generate_ids(&[5, 9, 2, 14], &SampleOpts::greedy(4))
        .expect("paged on quantized");
    assert_eq!(
        paged.tokens, resident.tokens,
        "paged and resident disagree on a quantized checkpoint"
    );

    // the cache holds packed bytes, so the budget buys ~8x more experts than it
    // would if the pager kept f32
    let s = pr.pager_stats().expect("stats");
    assert!(s.misses > 0, "the pager was never used");
    assert!(
        s.resident_bytes <= 1 << 16,
        "resident {} exceeds the 64 KiB budget",
        s.resident_bytes
    );
    println!(
        "quantized + paged: {} hits / {} misses, {} B read, {} B resident",
        s.hits, s.misses, s.bytes_read, s.resident_bytes
    );
}

/// A vision-enabled config: the same text stack plus a small ViT and aligner.
fn vision_config(vocab: usize) -> Value {
    let mut c = toy_config(vocab);
    // no Engram — image positions have no token id to hash, and the runner says
    // so rather than guessing
    c["engram_layer_ids"] = json!([]);
    c["vision_n_layers"] = json!(1);
    c["vision_dim"] = json!(16);
    c["vision_n_heads"] = json!(2);
    c["vision_inter_dim"] = json!(8);
    c["vision_patch_size"] = json!(2);
    c["vision_downsample_ratio"] = json!(2);
    c["vision_rope_theta"] = json!(10000.0);
    c["image_token_id"] = json!(1);
    c
}

/// Splicing embeddings in must reproduce the id path exactly when the spliced
/// rows *are* the embedding table's rows.
///
/// This is the invariant that pins the whole `inputs_embeds` route without
/// needing a reference: same model, same values, two ways of getting them in.
/// If the reshape, the hyper-connection expansion or the position handling
/// differs between the two paths, it shows up here as a mismatch rather than as
/// a subtly worse image caption.
#[test]
fn spliced_embeddings_reproduce_the_id_path() {
    let dir = scratch("embeds");
    let spec = write_checkpoint(&dir, &vision_config(32)).expect("checkpoint written");
    let ids = [5u32, 9, 2, 14];
    let d = spec.dim;

    let mut r = runner(&dir, MoeExecution::Resident);
    let want = r.prefill_logits(&ids).expect("id-path prefill");

    // the same prompt, fed as embeddings looked up by hand
    let mut loader =
        rlx_models_core::dsv41_weights::StreamingLoader::open(&dir, 32).expect("loader");
    let (table, _) =
        rlx_models_core::weight_loader::WeightLoader::take(&mut loader, "embed.weight").unwrap();
    let mut embeds = vec![0f32; ids.len() * d];
    for (i, &t) in ids.iter().enumerate() {
        embeds[i * d..(i + 1) * d].copy_from_slice(&table[t as usize * d..(t as usize + 1) * d]);
    }
    let got = r
        .prefill_embeds_for_test(ids.len(), &embeds, &[])
        .expect("embeds-path prefill");

    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
            "logit {i}: embeds path {a} vs id path {b}"
        );
    }
}

/// The vision tower runs through the runner and its rows land in the `IMAGE`
/// slots of a correctly-shaped span.
#[test]
fn an_image_generates_and_occupies_its_prompt_slots() {
    use rlx_models_core::dsv41_runner::ImagePatches;

    let dir = scratch("image");
    let spec = write_checkpoint(&dir, &vision_config(32)).expect("checkpoint written");
    let vs = spec.vision.as_ref().expect("vision config present");
    let (n_vit_h, n_vit_w) = (4usize, 4usize);
    let patch_in = 3 * vs.patch_size * vs.patch_size;
    let patches: Vec<f32> = (0..n_vit_h * n_vit_w * patch_in)
        .map(|i| ((i as f32) * 0.07).sin())
        .collect();

    let mut r = runner(&dir, MoeExecution::Resident);
    let img = ImagePatches {
        patches: patches.clone(),
        n_vit_h,
        n_vit_w,
        at: 1,
    };
    let rows = r.image_embeddings(&img).expect("vision tower runs");
    let ratio = vs.downsample_ratio.max(1);
    let (g_h, g_w) = (n_vit_h.div_ceil(ratio), n_vit_w.div_ceil(ratio));
    assert_eq!(
        rows.len(),
        g_h * g_w * spec.dim,
        "the aligner should fold a {ratio}x{ratio} neighbourhood into one token"
    );
    assert!(
        rows.iter().all(|v| v.is_finite()),
        "vision output is not finite"
    );

    // the span is START + (IMAGE x n_w + NEWLINE) x n_h + END, so it is wider
    // than the aligner's row count
    let span = img.span_len(&spec).unwrap();
    assert_eq!(span, g_h * (g_w + 1) + 2, "span layout");
    assert!(
        span > g_h * g_w,
        "the delimiters and newlines are unaccounted for"
    );
    let prompt: Vec<u32> = std::iter::once(5u32)
        .chain(std::iter::repeat_n(1u32, span))
        .chain(std::iter::once(9u32))
        .collect();

    let out = r
        .generate_with_images(&prompt, &[img], &SampleOpts::greedy(3))
        .expect("multimodal generation");
    assert_eq!(out.tokens.len(), 3);
    assert_eq!(out.prompt_tokens, prompt.len());

    // moving the image changes the answer — proof the rows are actually used
    let moved = ImagePatches {
        patches,
        n_vit_h,
        n_vit_w,
        at: 0,
    };
    let other = r
        .generate_with_images(&prompt, &[moved], &SampleOpts::greedy(3))
        .expect("multimodal generation");
    assert_ne!(
        out.tokens, other.tokens,
        "moving the image span changed nothing, so its embeddings are being ignored"
    );
}

/// An image that does not fit the prompt is an error naming both sizes, not an
/// out-of-bounds write.
#[test]
fn an_oversized_image_is_rejected() {
    use rlx_models_core::dsv41_runner::ImagePatches;

    let dir = scratch("imgfit");
    let spec = write_checkpoint(&dir, &vision_config(32)).expect("checkpoint written");
    let vs = spec.vision.as_ref().unwrap();
    let (n_h, n_w) = (4usize, 4usize);
    let img = ImagePatches {
        patches: vec![0.0; n_h * n_w * 3 * vs.patch_size * vs.patch_size],
        n_vit_h: n_h,
        n_vit_w: n_w,
        at: 0,
    };
    let span = img.span_len(&spec).unwrap();
    let e = runner(&dir, MoeExecution::Resident)
        .generate_with_images(&[5, 9], &[img], &SampleOpts::greedy(1))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains(&format!("needs {span} tokens")),
        "unhelpful error: {e}"
    );
}

/// Chunked prompt processing must not change a single token.
///
/// The point of chunking is a graph sized for the chunk rather than the prompt,
/// which is what lets a long prompt run at all. It is only worth having if the
/// answer is the same, so this drives the public API both ways — including a
/// chunk size that does not divide the prompt, and one of 1, which is the
/// degenerate case where every chunk is a decode step.
#[test]
fn chunked_prompt_processing_changes_nothing() {
    let _arena = ArenaPin::new();
    let dir = scratch("chunked");
    write_checkpoint(&dir, &toy_config(32)).expect("checkpoint written");
    let prompt: Vec<u32> = vec![5, 9, 2, 14, 7, 3, 11];

    let whole = runner(&dir, MoeExecution::Resident)
        .generate_ids(&prompt, &SampleOpts::greedy(4))
        .expect("one pass");

    for chunk in [1usize, 2, 3, 4, 6] {
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let vocab = cfg["vocab_size"].as_u64().unwrap() as usize;
        let mut r = V41Runner::open_with_vocab(
            &dir,
            RunnerOptions {
                prefill_chunk: Some(chunk),
                ..Default::default()
            },
            word_vocab(vocab),
        )
        .expect("runner opens");
        let got = r
            .generate_ids(&prompt, &SampleOpts::greedy(4))
            .unwrap_or_else(|e| panic!("chunk size {chunk}: {e}"));
        assert_eq!(
            got.tokens, whole.tokens,
            "chunk size {chunk} changed the generated tokens"
        );
    }
}

/// A config with DSpark draft stages, for speculative decoding.
fn dspark_config(vocab: usize) -> Value {
    let mut c = toy_config(vocab);
    c["num_nextn_predict_layers"] = json!(2);
    c["dspark_block_size"] = json!(3);
    c["dspark_noise_token_id"] = json!(0);
    c["dspark_target_layer_ids"] = json!([3, 5]);
    c["dspark_markov_rank"] = json!(8);
    c["dspark_n_routed_experts"] = json!(2);
    c["dspark_num_experts_per_tok"] = json!(2);
    // the draft stages reuse the layer layout, so the ratio list covers them too
    c["compress_ratios"] = json!([0, 0, 2, 2, 2, 2, 0, 0]);
    c
}

/// Greedy speculative decoding must produce **exactly** the tokens plain greedy
/// decoding produces.
///
/// That is what makes speculation a speedup rather than a different model: a
/// drafted token is only committed when the backbone would have chosen it
/// anyway, and the first disagreement is resolved the backbone's way. So this is
/// an identity, not a tolerance — any divergence means the acceptance rule or
/// the cache rollback is wrong.
///
/// It also asserts the accounting is honest: drafts are actually being proposed,
/// and the reported token count matches what came out.
#[test]
fn speculative_decoding_matches_plain_greedy() {
    let _arena = ArenaPin::new();
    let dir = scratch("spec");
    write_checkpoint(&dir, &dspark_config(32)).expect("checkpoint written");
    let prompt: Vec<u32> = vec![5, 9, 2, 14, 7];

    let plain = runner(&dir, MoeExecution::Resident)
        .generate_ids(&prompt, &SampleOpts::greedy(8))
        .expect("plain greedy");

    let mut r = runner(&dir, MoeExecution::Resident);
    let (spec_out, stats) = r
        .generate_speculative(&prompt, 8, &[])
        .expect("speculative");

    assert_eq!(
        spec_out.tokens, plain.tokens,
        "speculative decoding produced different tokens than plain greedy"
    );
    assert!(stats.rounds > 0, "no speculative rounds ran");
    assert!(stats.proposed > 0, "the draft head proposed nothing");
    assert!(
        stats.backbone_passes > 0 && stats.backbone_passes <= stats.tokens + stats.rounds,
        "implausible pass accounting: {stats:?}"
    );
    println!(
        "speculative: {} tokens in {} backbone passes ({:.2} tok/pass), \
         acceptance {:.0}% of {} proposed",
        stats.tokens,
        stats.backbone_passes,
        stats.tokens_per_pass(),
        stats.acceptance_rate() * 100.0,
        stats.proposed
    );
}

/// A stop token must still end generation mid-block: a speculative round commits
/// several tokens at once, and the stop has to be honoured at the token that
/// carries it rather than at the round boundary.
#[test]
fn speculative_decoding_honours_a_stop_token() {
    let _arena = ArenaPin::new();
    let dir = scratch("specstop");
    write_checkpoint(&dir, &dspark_config(32)).expect("checkpoint written");
    let prompt: Vec<u32> = vec![5, 9, 2, 14, 7];

    let plain = runner(&dir, MoeExecution::Resident)
        .generate_ids(&prompt, &SampleOpts::greedy(6))
        .expect("plain greedy");
    assert!(plain.tokens.len() >= 3, "need a few tokens to cut into");
    let stop = plain.tokens[2];

    let mut r = runner(&dir, MoeExecution::Resident);
    let (out, _) = r
        .generate_speculative(&prompt, 6, &[stop])
        .expect("speculative with a stop");
    assert_eq!(out.stopped_on, Some(stop));
    assert_eq!(
        out.tokens,
        plain.tokens[..2].to_vec(),
        "the stop token must end generation exactly where plain greedy would"
    );
}

/// The acceptance path itself: full, partial, and none.
///
/// `speculative_decoding_matches_plain_greedy` proves the *output* is right, but
/// on a synthetic checkpoint the draft head agrees with the backbone essentially
/// never — so that test only ever exercises the reject-everything branch. The
/// multi-token commit, which is the entire point, needs drafts that are actually
/// correct. Here they are supplied: the backbone's own continuation for full
/// acceptance, a truncated version for partial, and a wrong one for none.
///
/// Each case is checked two ways — the tokens committed, and that decoding
/// continues identically afterwards, which is what catches a cache left in the
/// wrong state by a partial commit.
#[test]
fn speculative_acceptance_commits_the_right_prefix() {
    use rlx_models_core::dsv41_decode::V41DecodeCache;
    use rlx_models_core::dsv41_speculative::{SpeculativeDecoder, SpeculativeState};
    use rlx_models_core::dsv41_weights::StreamingLoader;

    let _arena = ArenaPin::new();
    let dir = scratch("specaccept");
    let spec = write_checkpoint(&dir, &dspark_config(32)).expect("checkpoint written");
    let prompt: Vec<u32> = vec![5, 9, 2, 14, 7];
    let block = spec.dspark_block_size;

    // ground truth: what plain greedy produces from this prompt
    let truth = runner(&dir, MoeExecution::Resident)
        .generate_ids(&prompt, &SampleOpts::greedy(block + 4))
        .expect("plain greedy")
        .tokens;
    assert!(truth.len() > block, "need more than one block of truth");

    // drafts, and how many of them should survive: the first is confirmed by the
    // backbone's own next token, the rest by the verification pass
    let wrong = |t: u32| if t == 0 { 1 } else { t - 1 };
    let cases: Vec<(&str, Vec<u32>, usize)> = vec![
        ("all accepted", truth[..block].to_vec(), block),
        (
            "partial",
            truth[..block - 1]
                .iter()
                .copied()
                .chain(std::iter::once(wrong(truth[block - 1])))
                .collect(),
            block - 1,
        ),
        ("first wrong", vec![wrong(truth[0]); block], 1),
    ];

    for (label, drafts, want_commit) in cases {
        let mut r = runner(&dir, MoeExecution::Resident);
        let mut cache = V41DecodeCache::new(&spec);
        let (logits, mh) = r
            .prefill_with_main_hidden_for_test(&prompt, &mut cache)
            .expect("prefill");
        let mut dec = SpeculativeDecoder::new(&spec, rlx_runtime::Device::Cpu).unwrap();
        let mut loader = StreamingLoader::open(&dir, 32).unwrap();
        dec.seed(&mut loader, &mh, prompt.len()).expect("seed");

        let d = spec.dim * spec.dspark_target_layer_ids.len();
        let n = prompt.len();
        let mut state = SpeculativeState {
            pos: n - 1,
            token: prompt[n - 1],
            main_hidden: mh[(n - 1) * d..n * d].to_vec(),
            next_logits: logits,
        };
        let mut history = prompt.clone();
        // the checkpoint has an Engram, so the chunks need real n-gram rows
        let e = spec.engram.as_ref().expect("engram");
        let vocab = word_vocab(spec.vocab_size);
        let (map, _) =
            rlx_models_core::dsv41_engram::compress_token_map(&vocab.decoded, &vocab.pieces);
        let hash = rlx_models_core::dsv41_engram::EngramHashPlan::new(e, &map).expect("hash plan");
        let engram = |hist: &[u32], chunk: &[u32]| -> Vec<i64> {
            let c = |ids: &[u32]| -> Vec<u32> { ids.iter().map(|&i| map[i as usize]).collect() };
            hash.hash_ids(&c(chunk), None, &c(hist))
        };

        let round = dec
            .round_with(
                &mut loader,
                &mut state,
                &mut history,
                &mut cache,
                &engram,
                Some(&drafts),
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));

        assert_eq!(
            round.tokens.len(),
            want_commit,
            "{label}: committed {:?}, expected {want_commit} tokens",
            round.tokens
        );
        assert_eq!(
            round.tokens,
            truth[..want_commit].to_vec(),
            "{label}: committed the wrong tokens"
        );

        // The cache must describe exactly the committed tokens. A rejected
        // block's updates left in place are invisible in the next few tokens —
        // the window self-trims and an over-long compressed cache is truncated
        // to the declared input width — so the lengths are checked directly.
        let committed_total = prompt.len() + want_commit;
        for il in 0..spec.n_layers {
            assert_eq!(
                cache.window_len(il),
                committed_total.min(spec.window_size - 1),
                "{label}: layer {il} window holds the wrong number of rows"
            );
        }
        for src in 0..spec.n_layers {
            if !spec.is_kv_source(src) || spec.ratio(src) == 0 {
                continue;
            }
            let ratio = spec.ratio(src);
            assert_eq!(
                cache.compressed_len(src),
                committed_total / ratio,
                "{label}: source {src} has the wrong latent count"
            );
            assert_eq!(
                cache.group_len(src),
                committed_total % ratio,
                "{label}: source {src} carries the wrong partial group"
            );
        }

        // and the cache must be left able to continue correctly
        let mut next = Vec::new();
        for _ in 0..2 {
            let rd = dec
                .round_with(
                    &mut loader,
                    &mut state,
                    &mut history,
                    &mut cache,
                    &engram,
                    Some(&vec![wrong(0); block]),
                )
                .unwrap_or_else(|e| panic!("{label} continuation: {e}"));
            next.extend(rd.tokens);
        }
        let got: Vec<u32> = round.tokens.iter().copied().chain(next).collect();
        assert_eq!(
            got,
            truth[..got.len()].to_vec(),
            "{label}: decoding diverged after the round, so the cache is wrong"
        );
    }
}
