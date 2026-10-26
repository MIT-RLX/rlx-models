// RLX — versatile ML compiler + runtime. GPLv3.
//! End-to-end parity of the **DeepSeek-V4.1** port against the released
//! reference implementation.
//!
//! `deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py` was run on CPU at toy
//! scale with its tilelang kernels replaced by numerically-identical torch
//! transliterations, and its outputs captured in
//! `tests/fixtures/dsv41_toy_ref.json`. Every parameter is drawn from a
//! name-keyed PRNG that both sides reproduce bit for bit, so the fixture only has
//! to carry the *shapes* and the *outputs* — no weights.
//!
//! The toy config is small but deliberately covers every path the GA checkpoint
//! takes: pure sliding-window layers, ratio-2 and ratio-1 compressed layers, a
//! KV source that is not an index source, an index source that owns no KV, the
//! hierarchical candidate pre-filter, an active (non-degenerate) `index_topk`,
//! Engram at two layers, and a routed MoE with a shared expert.

use rlx_models_core::device_capabilities::available_devices;
use rlx_models_core::dsv41::DeepseekV41Spec;
use rlx_models_core::dsv41_engram::EngramHashPlan;
use rlx_models_core::dsv41_graph::{V41Inputs, build_deepseek_v41_prefill};
use rlx_models_core::parity::Deviation;
use rlx_models_core::weight_loader::SyntheticLoader;
use rlx_runtime::{Device, Session};
use serde_json::Value;
use std::collections::BTreeMap;

/// The debug taps ([`RLX_DSV41_DBG`]) are process-global environment variables
/// and `cargo test` runs tests on many threads, so every test that builds a
/// V4.1 graph holds this while it does.
///
/// Without it, a test that sets a tap makes a concurrently-running test emit
/// some layer's intermediate instead of logits — which is exactly how this was
/// found, and only when `RLX_DSV41_WEIGHTS` was set, because that is the one
/// test that sets a tap.
static GRAPH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn graph_guard() -> std::sync::MutexGuard<'static, ()> {
    // a poisoned lock just means some other test panicked; the tap state is
    // reset by `TapGuard` regardless, so carry on
    GRAPH_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets an environment variable for its lifetime and always restores it,
/// including on panic. Callers must hold [`graph_guard`], since the environment
/// is process-global.
struct EnvGuard(&'static str, Option<String>);

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = rlx_ir::env::read(key).ok();
        rlx_ir::env::set(key, value);
        EnvGuard(key, prev)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.1 {
            Some(v) => rlx_ir::env::set(self.0, v),
            None => rlx_ir::env::unset(self.0),
        }
    }
}

/// Sets a debug tap for its lifetime and always clears it, including on panic.
struct TapGuard;

impl TapGuard {
    fn set(stage: &str, layer: usize) -> Self {
        rlx_ir::env::set("RLX_DSV41_DBG", stage);
        rlx_ir::env::set("RLX_DSV41_DBGLAYER", layer.to_string());
        TapGuard
    }
}

impl Drop for TapGuard {
    fn drop(&mut self) {
        rlx_ir::env::remove("RLX_DSV41_DBG");
        rlx_ir::env::remove("RLX_DSV41_DBGLAYER");
    }
}

fn fixture() -> Value {
    // An override points the test at a full (untrimmed) dump while iterating.
    let path = rlx_ir::env::var_result("RLX_DSV41_REF").unwrap_or_else(|_| {
        format!(
            "{}/tests/fixtures/dsv41_toy_ref.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).expect("fixture parses")
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("float array")
        .iter()
        .map(|x| x.as_f64().expect("float") as f32)
        .collect()
}

/// Compare against the reference, failing with the offending element.
fn compare(got: &[f32], want: &[f32], label: &str, tol: f32) {
    let d = Deviation::between(got, want);
    assert!(d.is_within(tol), "{label}: {d} (tolerance {tol:e})");
}

fn spec_and_inputs(fx: &Value) -> (DeepseekV41Spec, Vec<i32>, V41Inputs) {
    let spec = DeepseekV41Spec::from_config(&fx["config"]).expect("toy config parses");
    spec.validate().expect("toy config is structurally sound");
    let ids: Vec<i32> = fx["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let engram_rows: Vec<i64> = fx
        .get("engram_rows")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_i64().unwrap()).collect())
        .unwrap_or_default();
    (
        spec,
        ids,
        V41Inputs {
            engram_rows,
            ..Default::default()
        },
    )
}

fn run_prefill(fx: &Value) -> (Vec<f32>, Vec<String>) {
    run_prefill_on(fx, Device::Cpu)
}

fn run_prefill_on(fx: &Value, device: Device) -> (Vec<f32>, Vec<String>) {
    let (spec, ids, inputs) = spec_and_inputs(fx);
    let shapes: BTreeMap<String, Vec<usize>> = fx["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let mut loader = SyntheticLoader::new(shapes);
    let mut packed = std::collections::HashMap::new();
    let (g, params, _) =
        build_deepseek_v41_prefill(&spec, &mut loader, ids.len(), &inputs, &mut packed)
            .expect("prefill graph builds");
    let asked = loader.asked().to_vec();

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        device,
    );
    let mut compiled = Session::new(device).compile_with(g, &opts);
    for (n, d) in &params {
        compiled.set_param(n, d);
    }
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let out = compiled.run(&[("input_ids", ids_f32.as_slice())]);
    (out[0].clone(), asked)
}

/// The whole stack: `logits[seq, vocab]` against the reference's.
#[test]
fn prefill_logits_match_reference() {
    let _graph = graph_guard();
    let fx = fixture();
    let (got, _) = run_prefill(&fx);
    let want = floats(&fx["logits"]);
    compare(&got, &want, "logits", 2e-4);
}

/// The host-side n-gram hashing must land on the same table rows the reference's
/// `NgramHashState` produces — a silent mismatch here reads 384M random rows.
#[test]
fn engram_hash_ids_match_reference() {
    let fx = fixture();
    let spec = DeepseekV41Spec::from_config(&fx["config"]).unwrap();
    let e = spec.engram.as_ref().expect("toy has engram");
    // the toy's compressed token map is `id % COMPRESSED_VOCAB`
    let cv = e.compressed_vocab_size;
    let map: Vec<u32> = (0..spec.vocab_size).map(|i| (i % cv) as u32).collect();
    let plan = EngramHashPlan::new(e, &map).unwrap();
    let ids: Vec<u32> = fx["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| map[v.as_u64().unwrap() as usize])
        .collect();
    let got = plan.hash_ids(&ids, None, &[]);
    let want: Vec<i64> = fx["engram_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(got, want, "engram hash row ids");
}

/// Every tensor the builder requests must exist in the reference checkpoint, and
/// the ones it must *not* request (a `wgate` on a ratio-1 compressor, an
/// `indexer.wk` on a layer that owns no KV) must stay untouched — those are
/// exactly the tensors the released checkpoint omits.
#[test]
fn requested_tensor_names_match_the_checkpoint_layout() {
    let _graph = graph_guard();
    let fx = fixture();
    let (_, asked) = run_prefill(&fx);
    let available: Vec<String> = fx["shapes"].as_object().unwrap().keys().cloned().collect();
    for k in &asked {
        assert!(available.contains(k), "asked for unknown tensor `{k}`");
    }
    let spec = DeepseekV41Spec::from_config(&fx["config"]).unwrap();
    for il in 0..spec.n_layers {
        let has = |suffix: &str| asked.iter().any(|k| k == &format!("layers.{il}.{suffix}"));
        let is_kv = spec.is_kv_source(il);
        assert_eq!(
            has("attn.compressor.wkv.weight"),
            is_kv,
            "layer {il}: only a kv source has a compressor"
        );
        assert_eq!(
            has("attn.compressor.wgate.weight"),
            is_kv && spec.ratio(il) > 1,
            "layer {il}: ratio-1 compressors have no gate"
        );
        assert_eq!(
            has("attn.indexer.wk.weight"),
            is_kv,
            "layer {il}: only a kv source derives index keys"
        );
        assert_eq!(
            has("attn.indexer.wq_b.weight"),
            spec.is_index_source(il),
            "layer {il}: only an index source scores"
        );
        assert_eq!(
            has("engram.wkv.weight"),
            spec.engram
                .as_ref()
                .is_some_and(|e| e.layer_ids.contains(&il)),
            "layer {il}: engram placement"
        );
    }
}

/// A stage split must reproduce the single-shot prefill exactly — which it only
/// can if the boundary carries the Hyper-Connection pre-mix as well as the
/// hidden state.
#[test]
fn split_stages_reproduce_single_shot_prefill() {
    let _graph = graph_guard();
    use rlx_models_core::dsv41_graph::{StageSpan, build_deepseek_v41_stage};
    let fx = fixture();
    let (spec, ids, inputs) = spec_and_inputs(&fx);
    let shapes: BTreeMap<String, Vec<usize>> = fx["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let seq = ids.len();
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let ids_f32: Vec<f32> = ids.iter().map(|&i| i as f32).collect();

    // The CSA2 sharing only works within a stage, so split where no consumer is
    // separated from its source: layers 0..2 own nothing, 2..6 owns both sources.
    let split = 2;
    let mut l1 = SyntheticLoader::new(shapes.clone());
    let mut packed = std::collections::HashMap::new();
    let span1 = StageSpan {
        layers: 0..split,
        first: true,
        last: false,
    };
    let (g1, p1, _) = build_deepseek_v41_stage(&spec, &mut l1, seq, &span1, &inputs, &mut packed)
        .expect("stage 1 builds");
    let mut s1 = Session::new(Device::Cpu).compile_with(g1, &opts);
    for (n, d) in &p1 {
        s1.set_param(n, d);
    }
    let mid = s1.run(&[("input_ids", ids_f32.as_slice())]);
    let (hidden, pre_mix) = (mid[0].clone(), mid[1].clone());
    assert_eq!(hidden.len(), seq * spec.hc_mult * spec.dim);
    assert_eq!(pre_mix.len(), seq * spec.hc_mult);

    let mut l2 = SyntheticLoader::new(shapes);
    let mut packed2 = std::collections::HashMap::new();
    let span2 = StageSpan {
        layers: split..spec.n_layers,
        first: false,
        last: true,
    };
    let (g2, p2, _) = build_deepseek_v41_stage(&spec, &mut l2, seq, &span2, &inputs, &mut packed2)
        .expect("stage 2 builds");
    let mut s2 = Session::new(Device::Cpu).compile_with(g2, &opts);
    for (n, d) in &p2 {
        s2.set_param(n, d);
    }
    let out = s2.run(&[
        ("hidden_in", hidden.as_slice()),
        ("pre_mix_in", pre_mix.as_slice()),
    ]);
    compare(&out[0], &floats(&fx["logits"]), "split-stage logits", 2e-4);
}

/// The vision tower: ViT encoder + aligner, against the reference's
/// `vision(patches)` and `aligner(...)`. The grid is 3×4 with `downsample_ratio`
/// 2, so the aligner's zero-pad of the odd dimension is exercised.
#[test]
fn vision_tower_matches_reference() {
    let _graph = graph_guard();
    use rlx_models_core::dsv41::DeepseekV41Spec;
    use rlx_models_core::dsv41_vision::build_v41_vision;

    let path = format!(
        "{}/tests/fixtures/dsv41_vision_ref.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let cfg = &fx["vision_config"];
    // the flat `vision_*` keys are the reference's own config spelling
    let spec = DeepseekV41Spec::from_config(&serde_json::json!({
        "vocab_size": 64, "dim": cfg["dim"], "num_hidden_layers": 1, "head_dim": 8,
        "num_attention_heads": 1, "o_lora_rank": 4, "n_routed_experts": 2,
        "moe_intermediate_size": 4,
        "vision_n_layers": cfg["vision_n_layers"], "vision_dim": cfg["vision_dim"],
        "vision_n_heads": cfg["vision_n_heads"], "vision_inter_dim": cfg["vision_inter_dim"],
        "vision_patch_size": cfg["vision_patch_size"],
        "vision_downsample_ratio": cfg["vision_downsample_ratio"],
        "vision_rope_theta": cfg["vision_rope_theta"],
        "vision_max_n_token": cfg["vision_max_n_token"],
        "vision_min_pixels": cfg["vision_min_pixels"],
    }))
    .unwrap();
    assert!(spec.vision.is_some(), "vision config present");

    let (n_h, n_w) = (
        fx["n_h"].as_u64().unwrap() as usize,
        fx["n_w"].as_u64().unwrap() as usize,
    );
    let shapes: BTreeMap<String, Vec<usize>> = fx["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let mut loader = SyntheticLoader::new(shapes);
    let mut packed = std::collections::HashMap::new();
    let (g, params) =
        build_v41_vision(&spec, &mut loader, n_h, n_w, &mut packed).expect("vision graph builds");

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    for (k, d) in &params {
        compiled.set_param(k, d);
    }
    let px = floats(&fx["patches"]);
    let got = compiled.run(&[("patches", px.as_slice())]);
    compare(&got[0], &floats(&fx["embeds"]), "aligner embeds", 2e-4);
}

/// Decode with a KV cache must reproduce the prefill logits token for token.
///
/// This is the induction the cache rests on, and it is the only check that
/// covers the three asymmetric pieces of cross-step state at once: the rolling
/// window (which must evict exactly one position per step once full), the
/// compressed cache shared from a source layer, and the compressor's partial
/// group on the `ratio - 1` steps that produce no latent.
#[test]
fn decode_matches_prefill() {
    let _graph = graph_guard();
    use rlx_models_core::dsv41_decode::{V41DecodeCache, V41DecodePlan, build_deepseek_v41_decode};
    use rlx_models_core::dsv41_engram::EngramHashPlan;

    let fx = fixture();
    let (spec, ids, prefill_inputs) = spec_and_inputs(&fx);
    let shapes: BTreeMap<String, Vec<usize>> = fx["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let _ = prefill_inputs;
    let want = floats(&fx["logits"]);
    let seq = ids.len();
    let vocab = spec.vocab_size;

    // the host recomputes the engram row ids for each step from the history
    let e = spec.engram.as_ref().expect("toy has engram");
    let cv = e.compressed_vocab_size;
    let map: Vec<u32> = (0..spec.vocab_size).map(|i| (i % cv) as u32).collect();
    let hash = EngramHashPlan::new(e, &map).unwrap();
    let compressed: Vec<u32> = ids.iter().map(|&i| map[i as usize]).collect();

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut cache = V41DecodeCache::new(&spec);

    for pos in 0..seq {
        let plan = V41DecodePlan::new(&spec, pos);
        let inputs = V41Inputs {
            engram_rows: hash.hash_ids(&compressed[pos..pos + 1], None, &compressed[..pos]),
            image_positions: Vec::new(),
            ..Default::default()
        };
        let mut loader = SyntheticLoader::new(shapes.clone());
        let mut packed = std::collections::HashMap::new();
        let (g, params, names) =
            build_deepseek_v41_decode(&spec, &mut loader, pos, &inputs, &mut packed)
                .unwrap_or_else(|e| panic!("decode graph at pos {pos}: {e}"));
        let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        let id = [ids[pos] as f32];
        let cached = cache.step_inputs(&plan);
        // the Engram rows are a graph input, so a cached compiled step cannot
        // carry another position's n-grams
        let cols = e.n_hash_cols();
        let eng: Vec<(String, Vec<f32>)> = e
            .layer_ids
            .iter()
            .enumerate()
            .map(|(k, &il)| {
                (
                    rlx_models_core::dsv41_decode::names::engram_rows(il),
                    inputs.engram_rows[k * cols..(k + 1) * cols]
                        .iter()
                        .map(|&v| v as f32)
                        .collect(),
                )
            })
            .collect();
        let mut feed: Vec<(&str, &[f32])> = vec![("input_ids", id.as_slice())];
        for (n, v) in &eng {
            feed.push((n.as_str(), v.as_slice()));
        }
        for (n, v) in &cached {
            feed.push((n.as_str(), *v));
        }
        let out = sess.run(&feed);
        cache.apply(&plan, &names, &out).unwrap();

        let got = &out[0];
        assert_eq!(got.len(), vocab, "pos {pos}: logits width");
        compare(
            got,
            &want[pos * vocab..(pos + 1) * vocab],
            &format!("decode logits at pos {pos}"),
            2e-4,
        );
    }
}

/// DSpark: seeding the draft head's window caches from a prompt, one draft step,
/// and the Markov-bias / confidence heads.
///
/// The draft head is driven by a synthetic `main_hidden` so the fixture pins
/// DSpark alone rather than re-testing the backbone through it.
#[test]
fn dspark_draft_head_matches_reference() {
    let _graph = graph_guard();
    use rlx_models_core::dsv41_dspark::{
        build_v41_dspark_markov_step, build_v41_dspark_seed, build_v41_dspark_step,
        dspark_draft_ids, names,
    };

    let path = format!(
        "{}/tests/fixtures/dsv41_dspark_ref.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let spec = DeepseekV41Spec::from_config(&fx["config"]).unwrap();
    spec.validate().unwrap();
    assert_eq!(spec.n_mtp_layers, 2);
    assert_eq!(
        spec.moe_dims(spec.n_layers),
        (2, 2),
        "DSpark has its own bank"
    );

    let shapes: BTreeMap<String, Vec<usize>> = fx["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let seq = fx["seq"].as_u64().unwrap() as usize;
    let pos = fx["pos"].as_u64().unwrap() as usize;
    let hd = spec.head_dim;

    // ── seed: the prompt's main_hidden fills each stage's ring ──
    let mut loader = SyntheticLoader::new(shapes.clone());
    let mut packed = std::collections::HashMap::new();
    let (g, params, seed_names) =
        build_v41_dspark_seed(&spec, &mut loader, seq, &mut packed).expect("seed graph builds");
    let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        sess.set_param(n, d);
    }
    let mh_seed = floats(&fx["mh_seed"]);
    let rings = sess.run(&[("main_hidden", mh_seed.as_slice())]);
    assert_eq!(seed_names.len(), spec.n_mtp_layers);
    let want_rings = fx["rings"].as_array().unwrap();
    for stage in 0..spec.n_mtp_layers {
        // the reference ring is `window_size` slots; after a `seq`-token seed with
        // `seq % window == 0` those are the last `window_size` positions in order
        let want = floats(&want_rings[stage]);
        assert_eq!(want.len(), spec.window_size * hd);
        compare(&rings[stage], &want, &format!("dspark ring {stage}"), 2e-4);
    }

    // ── one draft step ──
    let cache_len = pos.min(spec.window_size.saturating_sub(1));
    let mut loader = SyntheticLoader::new(shapes.clone());
    let mut packed = std::collections::HashMap::new();
    let (g, params, step_names) =
        build_v41_dspark_step(&spec, &mut loader, pos, cache_len, &mut packed)
            .expect("step graph builds");
    let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        sess.set_param(n, d);
    }
    let mh_step = floats(&fx["mh_step"]);
    let id_step = fx["id_step"].as_i64().unwrap() as i32;
    let draft: Vec<f32> = dspark_draft_ids(&spec, id_step)
        .iter()
        .map(|&i| i as f32)
        .collect();
    // this step overwrites the oldest ring slot, so it is fed the newest
    // `window_size - 1` entries and contributes the main token itself
    let tails: Vec<Vec<f32>> = (0..spec.n_mtp_layers)
        .map(|s| rings[s][(spec.window_size - cache_len) * hd..].to_vec())
        .collect();
    let mut feed: Vec<(&str, &[f32])> = vec![
        ("main_hidden", mh_step.as_slice()),
        ("draft_ids", draft.as_slice()),
    ];
    let names_owned: Vec<String> = (0..spec.n_mtp_layers).map(names::window_kv).collect();
    for (s, t) in tails.iter().enumerate() {
        feed.push((names_owned[s].as_str(), t.as_slice()));
    }
    let out = sess.run(&feed);
    assert_eq!(step_names[0], "logits");
    assert_eq!(step_names[1], "hidden");
    compare(&out[1], &floats(&fx["hidden"]), "dspark hidden", 2e-4);
    compare(&out[0], &floats(&fx["logits"]), "dspark draft logits", 2e-4);

    // ── Markov bias + confidence, on the reference's own token sequence ──
    let out_ids: Vec<i32> = fx["output_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let block = spec.dspark_block_size;
    let mut loader = SyntheticLoader::new(shapes);
    let (g, params) =
        build_v41_dspark_markov_step(&spec, &mut loader, block).expect("markov graph builds");
    let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        sess.set_param(n, d);
    }
    // slot i is biased by the token that precedes it
    let toks: Vec<f32> = out_ids[..block].iter().map(|&i| i as f32).collect();
    let mk = sess.run(&[
        ("token_ids", toks.as_slice()),
        ("hidden", out[1].as_slice()),
    ]);
    let biased: Vec<f32> = out[0].iter().zip(&mk[0]).map(|(a, b)| a + b).collect();
    compare(
        &biased,
        &floats(&fx["biased_logits"]),
        "dspark biased logits",
        2e-4,
    );
    compare(
        &mk[1],
        &floats(&fx["confidence"]),
        "dspark confidence",
        2e-4,
    );

    // and greedy sampling of the biased logits reproduces the reference's draft
    let vocab = spec.vocab_size;
    for i in 0..block {
        let row = &biased[i * vocab..(i + 1) * vocab];
        let arg = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap();
        assert_eq!(arg, out_ids[i + 1], "draft token {i}");
    }
}

/// The port's tensor manifest against the **real** `DeepSeek-V4.1-Flash`
/// checkpoint.
///
/// `tests/fixtures/dsv41_checkpoint_inventory.json` is the complete inventory of
/// all 96,085 tensors — name, dtype and shape — recovered from the 48 shards'
/// safetensors headers by HTTP range request. It costs ~11 MB to rebuild and
/// nothing to ship (15 KB compressed into patterns), and it pins every shape
/// the port assumes without downloading 510 GB of weights.
///
/// Both directions are checked: nothing the port reads is missing or
/// mis-shaped, and nothing in the checkpoint is left unexplained.
#[test]
fn tensor_manifest_matches_the_real_checkpoint() {
    use rlx_models_core::dsv41_quant;
    use std::collections::{BTreeMap, BTreeSet};

    let path = format!(
        "{}/tests/fixtures/dsv41_checkpoint_inventory.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    // expand the pattern table back into every concrete tensor
    let mut inv: BTreeMap<String, (String, Vec<usize>)> = BTreeMap::new();
    for p in fx["patterns"].as_array().unwrap() {
        let pat = p["name"].as_str().unwrap();
        let dtype = p["dtype"].as_str().unwrap().to_string();
        let shape_of = |l: i64| -> (String, Vec<usize>) {
            let node = p
                .get("per_layer")
                .and_then(|m| m.get(l.to_string()))
                .unwrap_or(p);
            (
                node["dtype"].as_str().unwrap_or(&dtype).to_string(),
                node["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect(),
            )
        };
        let layers: Vec<i64> = p
            .get("layers")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|x| x.as_i64().unwrap()).collect())
            .unwrap_or_default();
        let experts = p.get("experts").and_then(Value::as_u64).unwrap_or(0);
        if layers.is_empty() {
            inv.insert(pat.to_string(), shape_of(0));
            continue;
        }
        for l in layers {
            let per_l = pat.replace("{L}", &l.to_string());
            if experts == 0 {
                inv.insert(per_l, shape_of(l));
            } else {
                for e in 0..experts {
                    inv.insert(per_l.replace("{E}", &e.to_string()), shape_of(l));
                }
            }
        }
    }
    assert_eq!(
        inv.len(),
        fx["n_tensors"].as_u64().unwrap() as usize,
        "pattern expansion must reproduce the checkpoint's tensor count"
    );

    // the released `config.json`, verbatim — so this also checks the parser
    // against the real file rather than a transcription of it
    let cfg_path = format!(
        "{}/tests/fixtures/dsv41_config.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    assert_eq!(cfg["model_type"].as_str(), Some("deepseek_v41"));
    let spec = DeepseekV41Spec::from_config(&cfg).unwrap();
    spec.validate().unwrap();
    assert_eq!(spec.weight_block_size, 32);
    assert!(spec.expert_fp4);

    let needs = spec.expected_tensors();
    let mut explained: BTreeSet<String> = BTreeSet::new();
    for n in &needs {
        let (dtype, stored) = inv
            .get(&n.name)
            .unwrap_or_else(|| panic!("checkpoint has no tensor `{}`", n.name));
        assert_eq!(
            *stored,
            n.stored_shape(),
            "{}: checkpoint stores {stored:?}, port expects {:?}",
            n.name,
            n.stored_shape()
        );
        explained.insert(n.name.clone());
        match n.scale_name() {
            Some(sk) => {
                assert!(
                    n.packed_fp4 && dtype == "I8" || !n.packed_fp4 && dtype == "F8_E4M3",
                    "{}: dtype {dtype} does not match packed_fp4={}",
                    n.name,
                    n.packed_fp4
                );
                let (sdt, sshape) = inv
                    .get(&sk)
                    .unwrap_or_else(|| panic!("`{}` is quantized but `{sk}` is missing", n.name));
                assert_eq!(sdt, "F8_E8M0", "{sk}: scale_fmt is ue8m0");
                // and the scale's shape must resolve to one of the three layouts
                dsv41_quant::plan(stored, n.packed_fp4, sshape, spec.weight_block_size)
                    .unwrap_or_else(|e| panic!("{}: {e}", n.name));
                explained.insert(sk);
            }
            None => assert!(
                dtype == "BF16" || dtype == "F32",
                "{}: expected an unquantized dtype, got {dtype}",
                n.name
            ),
        }
    }

    // Reverse: the only tensors the port never reads are the DSpark stages' VL
    // routing bias — drafts are text-only, so the reference passes no image mask
    // through `mtp.*`.
    let unread: Vec<&String> = inv.keys().filter(|k| !explained.contains(*k)).collect();
    let expected_unread: Vec<String> = (0..spec.n_mtp_layers)
        .map(|s| format!("mtp.{s}.ffn.gate.bias_vl"))
        .collect();
    assert_eq!(
        unread.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        expected_unread
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        "every checkpoint tensor should be read or knowingly skipped"
    );

    // sanity on the scale of what was just checked
    assert_eq!(inv.len(), 96_085);
    assert!(
        needs.len() > 46_000,
        "manifest covers {} tensors",
        needs.len()
    );
}

/// The manifest and the builder must not drift apart: at toy scale, everything
/// the builder actually asks for has to appear in
/// [`DeepseekV41Spec::expected_tensors`].
#[test]
fn builder_requests_match_the_manifest() {
    let _graph = graph_guard();
    let fx = fixture();
    let (spec, _, _) = spec_and_inputs(&fx);
    let (_, asked) = run_prefill(&fx);
    let manifest: std::collections::BTreeSet<String> = spec
        .expected_tensors()
        .into_iter()
        .map(|t| t.name)
        .collect();
    for k in &asked {
        assert!(
            manifest.contains(k),
            "builder asked for `{k}`, not in the manifest"
        );
    }
    // and the manifest's shapes agree with what the reference model allocated
    let shapes = fx["shapes"].as_object().unwrap();
    for t in spec.expected_tensors() {
        if let Some(s) = shapes.get(&t.name) {
            let want: Vec<usize> = s
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(t.shape, want, "{}", t.name);
        }
    }
}

/// Attention on **real** weights, for every layer a subset was fetched for.
///
/// `scripts/dsv41_ref/fetch_subset.py --layer N` range-fetches just that layer's
/// tensors (~130-145 MB of the 510 GB — a safetensors header gives every
/// tensor's byte range) plus a slice of `embed.weight`, and `dump_real_layer.py`
/// runs the released reference over them. This drives the port's own
/// `DsV41Loader` across the same file, so the FP8 block dequant, the real
/// geometry (64 heads x 512, RoPE 64, 8 o-LoRA groups) and the sliding window at
/// `seq > window_size` are exercised on actual trained bytes.
///
/// Fetching **layer 2** as well as layer 0 is what makes this cover the parts
/// V4.1 actually added: layer 0 is sliding-window only, while layer 2 is both a
/// KV source and an index source, so it runs the compressor's gated pooling, the
/// index keys, and the YaRN-scaled compressed RoPE table. (The Indexer's *top-k*
/// stays a no-op below 1024 tokens, where every reachable position already fits
/// the 512 budget — that path is covered at toy scale.)
///
/// Set `RLX_DSV41_WEIGHTS` to the directory holding the subsets; the committed
/// digests are checked either way, and the full element-wise comparison runs
/// when `dsv41_real_layer{N}_ref.json` sits beside the weights.
#[test]
fn real_layer_attention_matches_reference() {
    let _graph = graph_guard();
    use rlx_models_core::dsv41_graph::{StageSpan, build_deepseek_v41_stage};
    use rlx_models_core::dsv41_quant::{DEFAULT_BLOCK, DsV41Loader};
    use rlx_models_core::weight_loader::WeightLoader;

    let fixtures = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
    let mut digests: Vec<(usize, Value)> = std::fs::read_dir(&fixtures)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            let name = p.file_name()?.to_str()?.to_string();
            let n: usize = name
                .strip_prefix("dsv41_real_layer")?
                .strip_suffix("_digest.json")?
                .parse()
                .ok()?;
            Some((
                n,
                serde_json::from_str(&std::fs::read_to_string(&p).ok()?).ok()?,
            ))
        })
        .collect();
    digests.sort_by_key(|(n, _)| *n);
    assert!(!digests.is_empty(), "no real-layer digests committed");
    // a sliding-window-only layer AND a compressed source layer, or this covers
    // none of what V4.1 changed
    assert!(
        digests.iter().any(|(_, d)| d["compress_ratio"] == 0)
            && digests.iter().any(|(_, d)| d["compress_ratio"] != 0),
        "real-weight coverage must include both a plain and a compressed layer"
    );

    let cfg_path = format!("{fixtures}/dsv41_config.json");
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    let spec = DeepseekV41Spec::from_config(&cfg).unwrap();

    let Some(dir) = rlx_ir::env::var_os("RLX_DSV41_WEIGHTS") else {
        for (n, dg) in &digests {
            // the digests still have to be self-consistent without the weights
            let seq = dg["seq"].as_u64().unwrap() as usize;
            let a = &dg["attn_out"];
            assert_eq!(
                a["shape"].as_array().unwrap()[0].as_u64().unwrap() as usize,
                seq
            );
            assert_eq!(a["row_norms"].as_array().unwrap().len(), seq, "layer {n}");
            assert!(a["absmean"].as_f64().unwrap() > 0.0, "layer {n}");
        }
        eprintln!("skipping real-weight run: set RLX_DSV41_WEIGHTS (see scripts/dsv41_ref)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    // measured agreement is 7.5e-7 relative (f32 round-off over a 5120-wide,
    // 64-head matmul chain); this leaves ~7x for a different BLAS
    let tol = 5e-6f32;

    let mut ran = 0;
    for (layer, dg) in &digests {
        let layer = *layer;
        let seq = dg["seq"].as_u64().unwrap() as usize;
        assert_eq!(
            spec.ratio(layer) as u64,
            dg["compress_ratio"].as_u64().unwrap(),
            "layer {layer}: the digest and the config disagree on compress_ratio"
        );
        assert!(seq > spec.window_size, "the window must actually evict");

        let mut loader = match DsV41Loader::open(&dir, DEFAULT_BLOCK) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("skipping layer {layer}: {e}");
                continue;
            }
        };
        let have: std::collections::BTreeSet<String> =
            loader.remaining_keys().into_iter().collect();
        if !have.contains(&format!("layers.{layer}.attn.wq_a.weight")) {
            eprintln!("note: no subset for layer {layer}, skipping");
            continue;
        }
        ran += 1;

        // the same real embedding rows the reference was driven with, expanded to
        // `hc_mult` copies; the one-hot initial pre-mix makes copy 0 the input
        let (rows, rshape) = loader.take("embed.rows").expect("embed.rows in the subset");
        assert!(rshape[0] >= seq && rshape[1] == spec.dim);
        let mut hidden = vec![0f32; seq * spec.hc_mult * spec.dim];
        for t in 0..seq {
            for c in 0..spec.hc_mult {
                let dst = (t * spec.hc_mult + c) * spec.dim;
                hidden[dst..dst + spec.dim]
                    .copy_from_slice(&rows[t * spec.dim..(t + 1) * spec.dim]);
            }
        }
        let mut pre_mix = vec![0f32; seq * spec.hc_mult];
        for t in 0..seq {
            pre_mix[t * spec.hc_mult] = 1.0;
        }

        // Pin the arena: `arena_reuse_is_deterministic` below shows that slot
        // reuse miscompiles this graph on roughly one compile in ten, and this
        // test is here to measure the *port* against the reference, not to
        // re-discover that. Without the pin it fails at random.
        let _arena = EnvGuard::set("RLX_ARENA_NO_REUSE", "1");
        let _tap = TapGuard::set("attn", layer);
        let mut loader = DsV41Loader::open(&dir, DEFAULT_BLOCK).unwrap();
        let mut packed = std::collections::HashMap::new();
        let inputs = V41Inputs::default();
        let span = StageSpan::middle(layer..layer + 1);
        let built = build_deepseek_v41_stage(&spec, &mut loader, seq, &span, &inputs, &mut packed);
        drop(_tap);
        let (g, params, _) =
            built.unwrap_or_else(|e| panic!("layer {layer} stage builds from the subset: {e}"));

        assert_params_sound(&g, &params, &format!("real layer {layer}"));
        let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        let out = sess.run(&[
            ("hidden_in", hidden.as_slice()),
            ("pre_mix_in", pre_mix.as_slice()),
        ]);
        // the same compiled session, run twice, must agree with itself
        let again = sess.run(&[
            ("hidden_in", hidden.as_slice()),
            ("pre_mix_in", pre_mix.as_slice()),
        ]);
        assert_eq!(
            out[0], again[0],
            "layer {layer}: two runs of the same session disagree — execution is not deterministic"
        );
        let got = &out[0];
        assert_eq!(got.len(), seq * spec.dim, "layer {layer}");

        // digest first — it is what the repo ships, so it must be what fails
        let a = &dg["attn_out"];
        let want_mean = a["absmean"].as_f64().unwrap();
        // accumulate in f64: an f32 running sum over 819k terms loses the addends
        // once the total passes ~4e5, which biases the mean by whole percent
        let mean = got.iter().map(|v| v.abs() as f64).sum::<f64>() / got.len() as f64;
        assert!(
            (mean - want_mean).abs() / want_mean < tol as f64,
            "layer {layer}: attn_out absmean {mean} vs {want_mean}"
        );
        for (t, w) in floats(&a["row_norms"]).iter().enumerate() {
            let n = got[t * spec.dim..(t + 1) * spec.dim]
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt();
            let w = *w as f64;
            assert!(
                (n - w).abs() / w.max(1e-6) < tol as f64,
                "layer {layer}: row {t} norm {n} vs {w}"
            );
        }
        let stride = a["stride"].as_u64().unwrap() as usize;
        let sampled: Vec<f32> = got.iter().step_by(stride).copied().collect();
        compare(
            &sampled,
            &floats(&a["samples"]),
            &format!("layer {layer} real attn_out samples"),
            tol,
        );

        // and the full element-wise comparison when the dump is beside the weights
        let full_path = dir.join(format!("dsv41_real_layer{layer}_ref.json"));
        if let Ok(raw) = std::fs::read_to_string(&full_path) {
            let full: Value = serde_json::from_str(&raw).unwrap();
            compare(
                got,
                &floats(&full["attn_out"]),
                &format!("layer {layer} real attn_out"),
                tol,
            );
        }
        eprintln!(
            "layer {layer} (ratio {}): matched on real weights",
            spec.ratio(layer)
        );
    }
    assert!(ran > 0, "RLX_DSV41_WEIGHTS held no layer subset");
}

/// The Engram's compressed token map, against the **real** tokenizer.
///
/// `engram_compressed_vocab_size` is not a bound check — every hash multiplier
/// is derived from it — so if this normalization differs from training's by even
/// one merge, every n-gram lands in a different bucket of a 384-million-row
/// table and the model just degrades. The released config says **99092**, which
/// makes the real 129,280-token vocab a complete, binary check on the whole
/// pipeline: NFKC → NFD → strip accents → lowercase → collapse spaces → strip.
///
/// Needs `tokenizer.json` (6.4 MB) under `RLX_DSV41_WEIGHTS`; see
/// `scripts/dsv41_ref/fetch_subset.py --tokenizer`.
#[test]
fn engram_token_map_matches_the_real_tokenizer() {
    use rlx_models_core::dsv41_engram::compress_token_map;

    let cfg_path = format!(
        "{}/tests/fixtures/dsv41_config.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    let spec = DeepseekV41Spec::from_config(&cfg).unwrap();
    let e = spec.engram.as_ref().expect("GA config has Engram");
    assert_eq!(e.compressed_vocab_size, 99_092);

    let Some(dir) = rlx_ir::env::var_os("RLX_DSV41_WEIGHTS") else {
        eprintln!("skipping: set RLX_DSV41_WEIGHTS to a dir holding tokenizer.json");
        return;
    };
    let tk_path = std::path::PathBuf::from(&dir).join("tokenizer.json");
    if !tk_path.is_file() {
        eprintln!("skipping: {tk_path:?} absent");
        return;
    }
    let tk = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer.json");
    let n = tk.get_vocab_size(true);
    assert_eq!(
        n, spec.vocab_size,
        "tokenizer and config disagree on vocab size"
    );

    // what the reference feeds `build_compressed_token_map`: the raw decode with
    // specials kept, and the raw piece for byte-fallback tokens
    let mut decoded = Vec::with_capacity(n);
    let mut pieces = Vec::with_capacity(n);
    for i in 0..n as u32 {
        decoded.push(tk.decode(&[i], false).unwrap_or_default());
        pieces.push(tk.id_to_token(i).unwrap_or_default());
    }

    let (map, size) = compress_token_map(&decoded, &pieces);
    assert_eq!(map.len(), n);
    assert_eq!(
        size, e.compressed_vocab_size,
        "compressed vocab is {size}, the checkpoint was trained with {}",
        e.compressed_vocab_size
    );
    // The count alone is a weak check — two different normalizations can land on
    // the same number of buckets. Fingerprint every merge decision.
    let cases = tokenmap_cases();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in &map {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let want: u64 = cases["map_fnv1a"].as_str().unwrap().parse().unwrap();
    assert_eq!(
        h, want,
        "the map differs from the reference's on some token"
    );
    assert!((map[e.pad_token_id] as usize) < size);
}

fn tokenmap_cases() -> Value {
    let p = format!(
        "{}/tests/fixtures/dsv41_tokenmap_cases.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

/// Merge decisions the real tokenizer makes, checked without needing it.
///
/// These groups are lifted from the released 129,280-token vocab, one set per
/// normalization step that can go wrong — and in particular both mark
/// categories. `StripAccents` filters `General_Category=Mark`, so **Mc**
/// (spacing combining marks, all over Bengali and Thai) goes too; a predicate
/// that only strips `Mn`, or one that tests the canonical combining *class*
/// instead of the category, splits these groups and silently rehashes the whole
/// Engram table.
#[test]
fn token_map_reproduces_real_vocab_merges() {
    use rlx_models_core::dsv41_engram::compress_token_map;
    let cases = tokenmap_cases();

    for g in cases["merge_groups"].as_array().unwrap() {
        let kind = g["kind"].as_str().unwrap();
        let texts: Vec<String> = g["texts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        let (map, _) = compress_token_map(&texts, &texts);
        assert!(
            map.iter().all(|&x| x == map[0]),
            "{kind}: {texts:?} must all collapse together, got {map:?}"
        );
    }

    for g in cases["must_not_merge"].as_array().unwrap() {
        let texts: Vec<String> = g["texts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        let pieces: Vec<String> = g["pieces"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_string())
            .collect();
        let (map, _) = compress_token_map(&texts, &pieces);
        assert!(
            map.iter().collect::<std::collections::BTreeSet<_>>().len() == map.len(),
            "byte-fallback tokens {pieces:?} must stay distinct, got {map:?}"
        );
    }
}

/// `DsV41Loader` against **real** quantized bytes — one tensor per scale layout.
///
/// The three layouts hide behind one `weight_block_size: [32, 32]` and are told
/// apart only by the scale's row count, so getting one wrong is silent: FP4 read
/// with the nibbles swapped, or the Engram table read as tiled, both produce a
/// plausible-looking tensor. These are real trained bytes, decoded independently
/// by `dump_real_quant.py` using torch's own fp8 decoders and the FP4 table
/// lifted verbatim from the reference `convert.py`:
///
/// * `attn.wq_a` — FP8, one scale per 32×32 tile,
/// * `ffn.experts.0.w1` — FP4 nibble pairs, one scale per row per 32 columns,
/// * 64 rows of the 384-million-row Engram table — FP8 but *row-wise* scaled,
///   which is the layout a tiled reading would smear over 32 rows at a time.
///
/// The Engram slice costs 17 KB of a 98 GB tensor: a safetensors header gives
/// the byte range, and a row-major row range is contiguous.
#[test]
fn dequant_matches_reference_on_real_bytes() {
    use rlx_models_core::dsv41_quant::{DEFAULT_BLOCK, DsV41Loader};
    use rlx_models_core::weight_loader::WeightLoader;

    let dpath = format!(
        "{}/tests/fixtures/dsv41_real_quant_digest.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let dg: Value = serde_json::from_str(&std::fs::read_to_string(&dpath).unwrap()).unwrap();
    let entries = dg.as_object().unwrap();
    // the fixture itself must cover all three layouts, with or without weights
    let layouts: std::collections::BTreeSet<&str> = entries
        .values()
        .map(|v| v["layout"].as_str().unwrap())
        .collect();
    assert!(layouts.contains("tile") && layouts.contains("row_groups"));
    assert!(
        entries
            .values()
            .any(|v| v["stored_dtype"].as_str() == Some("int8")),
        "no FP4 tensor in the digest"
    );

    let Some(dir) = rlx_ir::env::var_os("RLX_DSV41_WEIGHTS") else {
        eprintln!("skipping: set RLX_DSV41_WEIGHTS (see scripts/dsv41_ref, --what quant)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let mut loader = match DsV41Loader::open(&dir, DEFAULT_BLOCK) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };

    // Presence is decided from the checkpoint index, not from whether `take`
    // happened to work: treating a load error as "absent" would let a broken
    // layout inference silently skip the very tensor it breaks.
    let present: std::collections::BTreeSet<String> = loader.remaining_keys().into_iter().collect();

    let mut checked = 0;
    for (name, want) in entries {
        if !present.contains(name) {
            eprintln!("note: {name} is not in this subset, skipping");
            continue;
        }
        let (got, shape) = loader
            .take(name)
            .unwrap_or_else(|e| panic!("{name} is in the subset but would not load: {e}"));
        checked += 1;
        let want_shape: Vec<usize> = want["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(shape, want_shape, "{name}: dequantized shape");

        // the layout the loader inferred has to be the one the reference saw
        let scale_rows = want["scale_shape"].as_array().unwrap()[0].as_u64().unwrap() as usize;
        let inferred = if scale_rows == want_shape[0] {
            "row_groups"
        } else {
            "tile"
        };
        assert_eq!(inferred, want["layout"].as_str().unwrap(), "{name}: layout");

        // FP4 and FP8 are exact given the right table and scale, so this is an
        // equality check, not an approximation
        let stride = want["stride"].as_u64().unwrap() as usize;
        let samples = floats(&want["samples"]);
        let mine: Vec<f32> = got.iter().step_by(stride).copied().collect();
        assert_eq!(mine.len(), samples.len(), "{name}: sample count");
        for (i, (a, b)) in mine.iter().zip(&samples).enumerate() {
            assert_eq!(
                a,
                b,
                "{name}: element {} differs ({a} vs {b}) — check the nibble order \
                 and the scale layout",
                i * stride
            );
        }
        let want_mean = want["absmean"].as_f64().unwrap();
        let mean = got.iter().map(|v| v.abs() as f64).sum::<f64>() / got.len() as f64;
        assert!(
            (mean - want_mean).abs() / want_mean < 1e-6,
            "{name}: absmean {mean} vs {want_mean}"
        );
    }
    assert!(
        checked > 0,
        "RLX_DSV41_WEIGHTS held none of the digest's tensors"
    );
    eprintln!("checked {checked} real quantized tensors");
}

/// The whole stack on every backend this build can reach.
///
/// V4.1 leans on ops that have historically gone wrong per-backend rather than
/// everywhere — `TopK` and `ScatterElements` (the Indexer's exact-`k` mask),
/// `GroupedMatMul` (the routed experts), `Activation::Sign` (the Engram gate,
/// which is deliberately unfused), a masked softmax with an appended sink
/// column, and a partial GPT-J RoPE on the tail of each head. A CPU-only port is
/// a port that has not been tested.
///
/// Backends beyond CPU need their feature enabled at build time *and* the device
/// present: `cargo test -p rlx-models-core --features metal,mlx`.
#[test]
fn prefill_matches_reference_on_all_backends() {
    let _graph = graph_guard();
    let fx = fixture();
    let want = floats(&fx["logits"]);
    let devices = available_devices();
    for d in &devices {
        let (got, _) = run_prefill_on(&fx, *d);
        // GPU backends reassociate reductions and may run parts in a narrower
        // intermediate, so this is looser than the CPU comparison
        let tol = if *d == Device::Cpu { 2e-4 } else { 2e-3 };
        compare(&got, &want, &format!("{d:?} logits"), tol);
        eprintln!("{d:?}: prefill matches the reference");
    }
    eprintln!("checked {} backend(s): {devices:?}", devices.len());
    // A feature can be compiled in without the driver being present — vulkan on
    // a Mac with no MoltenVK, say. Reporting that is the difference between
    // "checked 4 backends" and a green run that only ever touched the CPU.
    let missing = rlx_models_core::device_capabilities::unavailable_compiled_devices();
    if !missing.is_empty() {
        println!("compiled but unreachable here: {missing:?}");
    }
    for want in rlx_models_core::device_capabilities::required_devices() {
        assert!(
            devices
                .iter()
                .any(|d| format!("{d:?}").to_ascii_lowercase() == want),
            "RLX_REQUIRE_DEVICES asked for `{want}` but only {devices:?} were reachable"
        );
    }
}

/// Every `Param` node must have data bound to it, and no two may share a name
/// unless they are the same tensor.
///
/// Both failures are silent and *non-deterministic*: params are bound by name at
/// session time by walking a `HashMap`, so a collision lets one node take
/// another's data depending on iteration order, and a missing binding leaves the
/// node reading whatever the arena happened to hold. Either way the same binary
/// gives different answers on different runs.
fn assert_params_sound(
    g: &rlx_ir::graph::Graph,
    params: &std::collections::HashMap<String, Vec<f32>>,
    what: &str,
) {
    use rlx_ir::op::Op;
    use std::collections::BTreeMap;
    let mut seen: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for n in g.nodes() {
        if let Op::Param { name } = &n.op {
            seen.entry(name.as_str())
                .or_default()
                .push(n.shape.num_elements().unwrap_or(0));
        }
    }
    let clashes: Vec<String> = seen
        .iter()
        .filter(|(_, sizes)| sizes.len() > 1 && sizes.iter().any(|s| *s != sizes[0]))
        .map(|(name, sizes)| format!("`{name}` used for {} nodes, sizes {sizes:?}", sizes.len()))
        .collect();
    assert!(
        clashes.is_empty(),
        "{what}: param name reused:\n  {}",
        clashes.join("\n  ")
    );
    let missing: Vec<String> = seen
        .iter()
        .filter(|(name, sizes)| params.get(**name).map(Vec::len) != Some(sizes[0]))
        .map(|(name, sizes)| {
            format!(
                "`{name}` needs {} elements, map holds {:?}",
                sizes[0],
                params.get(*name).map(Vec::len)
            )
        })
        .collect();
    assert!(
        missing.is_empty(),
        "{what}: param not bound:\n  {}",
        missing.join("\n  ")
    );
}

/// No two `Param` nodes may share a name unless they are the same tensor.
///
/// Params are bound by name at session time, so a collision means one node
/// silently gets another's data — and since the binding walks a `HashMap`, which
/// of the two wins changes per process. That is a heisenbug: right on one run,
/// subtly wrong on the next, with no diff in the source.
#[test]
fn no_param_name_collisions() {
    let _graph = graph_guard();
    let fx = fixture();
    let (spec, ids, inputs) = spec_and_inputs(&fx);
    let shapes: BTreeMap<String, Vec<usize>> = fx["shapes"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let mut loader = SyntheticLoader::new(shapes);
    let mut packed = std::collections::HashMap::new();
    let (g, params, _) =
        build_deepseek_v41_prefill(&spec, &mut loader, ids.len(), &inputs, &mut packed).unwrap();
    assert_params_sound(&g, &params, "toy prefill");
}

/// **Known upstream defect**: CPU arena slot reuse miscompiles a real-scale V4.1
/// attention graph.
///
/// The same graph, compiled and run repeatedly with the same inputs on the same
/// device, returns a different answer on roughly one attempt in ten. The wrong
/// answer is badly wrong — a contiguous band of query rows, every column, max
/// |Δ| ≈ 3.5 against activations whose absmean is 0.5 — so this is not
/// reassociation noise.
///
/// What is known:
///
/// * `RLX_ARENA_NO_REUSE=1` fixes it completely (0 failures in hundreds of
///   runs), so it is slot reuse, not a kernel;
/// * it reproduces single-threaded, so it is not a data race;
/// * it reproduces with a fresh session per run *and* with one reused session;
/// * `RLX_MEM_VERIFY=1` reports no overlap, no read-after-death and no
///   view-past-root — so the verifier's liveness model does not describe
///   whatever is actually happening;
/// * disabling fusion (`RLX_DECOMPOSE_FUSION_REGIONS`) or shared-input matmul
///   only reduces the rate, it does not remove it;
/// * it needs the surrounding layer: `build_v4_sink_attention` on its own, at
///   the same shape, is deterministic over hundreds of runs.
///
/// It is `#[ignore]`d so the suite stays usable, not because it is unimportant:
/// every parity number on a real-scale graph is a coin flip until the underlying
/// arena bug is fixed, and the real-weight tests pin `RLX_ARENA_NO_REUSE=1` to
/// work around it. Run with `cargo test -- --ignored`.
///
/// What is known, after a round of hunting:
///
/// * it depends on the **arena assignment**, not on timing — 90+ runs under
///   `RLX_ARENA_NO_REUSE=1` are identical, while reuse fails roughly one attempt
///   in ten;
/// * it is **not a race**: `RAYON_NUM_THREADS=1` reproduces at the same rate;
/// * it is **not BLAS aliasing**: a central detector on every GEMM entry point
///   (`RLX_BLAS_ALIAS_CHECK=1` in `rlx-cpu`) reports nothing on a failing run;
/// * it is **not concat aliasing** either, though hunting it did turn up two
///   genuine latent bugs there — a batched-GEMM guard that compared pointers
///   instead of ranges and so missed the cross-item case, and a concat with no
///   guard at all, which was undefined behaviour (`copy_nonoverlapping` on
///   overlapping ranges). Both are fixed upstream with regression tests, and
///   neither cures this;
/// * the damage is always a **contiguous band of query rows across every
///   column**, which says an output buffer is being read or written through the
///   wrong base rather than an arithmetic slip.
///
/// The next step is probably a differential harness that captures the arena
/// assignment of a failing compile and replays it, rather than more probing.
#[test]
#[ignore = "known upstream: CPU arena reuse miscompiles this graph ~10% of compiles"]
fn arena_reuse_is_deterministic() {
    use rlx_models_core::dsv41_graph::{StageSpan, build_deepseek_v41_stage};
    use rlx_models_core::dsv41_quant::{DEFAULT_BLOCK, DsV41Loader};
    use rlx_models_core::weight_loader::WeightLoader;

    let _graph = graph_guard();
    let Some(dir) = rlx_ir::env::var_os("RLX_DSV41_WEIGHTS") else {
        eprintln!("skipping: set RLX_DSV41_WEIGHTS (see scripts/dsv41_ref)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg_path = format!(
        "{}/tests/fixtures/dsv41_config.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    let spec = DeepseekV41Spec::from_config(&cfg).unwrap();
    let seq = 160;

    let mut loader = DsV41Loader::open(&dir, DEFAULT_BLOCK).expect("weight subset");
    let (rows, _) = loader.take("embed.rows").expect("embed.rows");
    let mut hidden = vec![0f32; seq * spec.hc_mult * spec.dim];
    for t in 0..seq {
        for c in 0..spec.hc_mult {
            let dst = (t * spec.hc_mult + c) * spec.dim;
            hidden[dst..dst + spec.dim].copy_from_slice(&rows[t * spec.dim..(t + 1) * spec.dim]);
        }
    }
    let mut pre_mix = vec![0f32; seq * spec.hc_mult];
    for t in 0..seq {
        pre_mix[t * spec.hc_mult] = 1.0;
    }
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );

    let mut first: Option<Vec<f32>> = None;
    let mut bad = 0;
    const ATTEMPTS: usize = 20;
    for _ in 0..ATTEMPTS {
        let _tap = TapGuard::set("attn", 0);
        let mut l = DsV41Loader::open(&dir, DEFAULT_BLOCK).unwrap();
        let mut packed = std::collections::HashMap::new();
        let span = StageSpan::middle(0..1);
        let built = build_deepseek_v41_stage(
            &spec,
            &mut l,
            seq,
            &span,
            &V41Inputs::default(),
            &mut packed,
        );
        drop(_tap);
        let (g, params, _) = built.expect("layer-0 stage builds");
        let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
        for (n, d) in &params {
            sess.set_param(n, d);
        }
        let out = sess.run(&[
            ("hidden_in", hidden.as_slice()),
            ("pre_mix_in", pre_mix.as_slice()),
        ])[0]
            .clone();
        match &first {
            None => first = Some(out),
            Some(f) if *f != out => bad += 1,
            Some(_) => {}
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad} of {} compiles of the same graph returned a different answer; \
         RLX_ARENA_NO_REUSE=1 makes it 0",
        ATTEMPTS - 1
    );
}

/// The image resize plan and span layout, against the reference processor.
///
/// Both are pure functions of a pixel size and five config values, so they can
/// be pinned exactly with no image and no checkpoint — and they need to be: the
/// plan decides how many tokens an image costs and what grid the ViT sees, and
/// getting it wrong produces a working model that looks at the wrong thing.
///
/// The fixture covers the branches that are easy to miss: the aspect-ratio cap,
/// the `min_pixels` floor, and the two degenerate collapses where a very tall or
/// very wide image would otherwise round to an empty grid.
#[test]
fn image_plan_matches_the_reference_processor() {
    use rlx_models_core::dsv41::VisionSpec;
    use rlx_models_core::dsv41_vision::{
        ImageTokenType, image_token_types, num_image_tokens, plan_image_grid,
    };

    let path = format!(
        "{}/tests/fixtures/dsv41_image_plan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    let spec_of = |name: &str| -> VisionSpec {
        let a = &fx["args"][name];
        VisionSpec {
            n_layers: 1,
            dim: 16,
            n_heads: 1,
            inter_dim: 8,
            patch_size: a["patch_size"].as_u64().unwrap() as usize,
            rope_theta: 10000.0,
            downsample_ratio: a["downsample_ratio"].as_u64().unwrap() as usize,
            max_n_token: a["max_n_token"].as_u64().unwrap() as usize,
            min_pixels: a["min_pixels"].as_u64().unwrap() as usize,
            max_wh_ratio: a["max_wh_ratio"].as_f64(),
        }
    };

    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 30, "thin fixture: {} cases", cases.len());
    for c in cases {
        let spec = spec_of(c["args"].as_str().unwrap());
        let (w, h) = (
            c["width"].as_u64().unwrap() as usize,
            c["height"].as_u64().unwrap() as usize,
        );
        let (n_h, n_w, best_h, best_w) = plan_image_grid(&spec, w, h);
        let label = format!("{} {w}x{h}", c["args"].as_str().unwrap());
        assert_eq!(
            n_h,
            c["n_llm_h"].as_u64().unwrap() as usize,
            "{label}: n_llm_h"
        );
        assert_eq!(
            n_w,
            c["n_llm_w"].as_u64().unwrap() as usize,
            "{label}: n_llm_w"
        );
        assert_eq!(
            best_h,
            c["best_height"].as_u64().unwrap() as usize,
            "{label}: best_height"
        );
        assert_eq!(
            best_w,
            c["best_width"].as_u64().unwrap() as usize,
            "{label}: best_width"
        );
        assert_eq!(
            num_image_tokens(n_h, n_w),
            c["n_image_tokens"].as_u64().unwrap() as usize,
            "{label}: token count"
        );
        assert_eq!(
            best_h / spec.patch_size,
            c["n_vit_h"].as_u64().unwrap() as usize,
            "{label}: n_vit_h"
        );
    }

    // the span layout: START, then a newline after every row, then END
    let ids = &fx["type_ids"];
    let code = |t: ImageTokenType| -> u64 {
        match t {
            ImageTokenType::Start => ids["IMAGE_START"].as_u64().unwrap(),
            ImageTokenType::Image => ids["IMAGE"].as_u64().unwrap(),
            ImageTokenType::NewLine => ids["IMAGE_NEW_LINE"].as_u64().unwrap(),
            ImageTokenType::End => ids["IMAGE_END"].as_u64().unwrap(),
        }
    };
    for l in fx["layouts"].as_array().unwrap() {
        let (h, w) = (
            l["n_llm_h"].as_u64().unwrap() as usize,
            l["n_llm_w"].as_u64().unwrap() as usize,
        );
        let got: Vec<u64> = image_token_types(h, w).into_iter().map(code).collect();
        let want: Vec<u64> = l["types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(got, want, "layout {h}x{w}");
        assert_eq!(got.len(), num_image_tokens(h, w));
    }
}

/// The MoE on **real routed experts**, end to end: real gate, real routing
/// decisions, real FP4 expert bytes, through the paged path.
///
/// Every other MoE check here runs on synthetic weights. That covers the
/// arithmetic but not the thing a real load actually does: read the gate, decide
/// which of 384 experts a token wants, page exactly those in from quantized
/// bytes, and index them by slot. The fixture was produced by fetching the gate
/// first, running the reference routing to learn which experts were needed, and
/// range-fetching only those — 209 MB instead of the layer's 6.8 GB.
///
/// It checks the routing *and* the output, because agreeing on the numbers while
/// disagreeing on which experts to use would be a coincidence, not correctness.
#[test]
fn real_moe_matches_reference_on_routed_experts() {
    use rlx_models_core::dsv41_moe::{paged_names, route_on_host};
    use rlx_models_core::dsv41_pager::{ExpertPager, Proj};
    use rlx_models_core::dsv41_weights::{StreamingLoader, WeightIndex};
    use rlx_models_core::weight_loader::WeightLoader;

    let _graph = graph_guard();
    let Ok(dir) = rlx_ir::env::var_result("RLX_DSV41_WEIGHTS") else {
        eprintln!("set RLX_DSV41_WEIGHTS to run this");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let path = format!(
        "{}/tests/fixtures/dsv41_real_moe_ref.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(format!(
            "{}/tests/fixtures/dsv41_config.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap();
    let spec = DeepseekV41Spec::from_config(&cfg).unwrap();

    let block = fx["block"].as_u64().unwrap() as usize;
    let rows = fx["rows"].as_u64().unwrap() as usize;
    let top_k = fx["top_k"].as_u64().unwrap() as usize;
    let d = fx["dim"].as_u64().unwrap() as usize;
    assert_eq!(d, spec.dim, "fixture and config disagree on dim");
    let x = floats(&fx["x"]);

    let index = match WeightIndex::open(&dir) {
        Ok(i) if i.contains("layers.0.ffn.gate.weight") => i,
        _ => {
            eprintln!("the MoE subset is not in {dir:?}; see scripts/dsv41_ref/dump_real_moe.py");
            return;
        }
    };
    let mut loader = StreamingLoader::from_index(WeightIndex::open(&dir).expect("reopen"), block);

    // ── the routing, against the real gate ──
    let (gw, gshape) = loader
        .take("layers.0.ffn.gate.weight")
        .expect("gate weight");
    let (gb, _) = loader.take("layers.0.ffn.gate.bias").expect("gate bias");
    assert_eq!(gshape, vec![spec.n_routed_experts, d], "gate shape");
    let routing = route_on_host(&spec, &x, rows, &gw, &gb, None, &[], top_k).expect("routing");

    let want_idx: Vec<usize> = fx["top_idx"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
        })
        .collect();
    assert_eq!(
        routing.idx, want_idx,
        "the host router chose different experts than the reference"
    );
    let want_w = floats(
        &fx["top_w"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().clone())
            .collect::<Vec<_>>()
            .into(),
    );
    compare(&routing.w, &want_w, "routing weights", 2e-5);

    // ── the experts the routing named, paged in from real quantized bytes ──
    let bank = routing.distinct();
    let want_chosen: Vec<usize> = fx["chosen_experts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(bank, want_chosen, "the paged bank is not the routed set");

    let pager = ExpertPager::new(index, block, 4 << 30);
    let slots: Vec<f32> = routing.to_slots(&bank).iter().map(|&s| s as f32).collect();
    let b1 = pager
        .gather_bank(&spec, 0, Proj::W1, &bank)
        .expect("w1 bank");
    let b3 = pager
        .gather_bank(&spec, 0, Proj::W3, &bank)
        .expect("w3 bank");
    let b2 = pager
        .gather_bank(&spec, 0, Proj::W2, &bank)
        .expect("w2 bank");
    let inter = spec.moe_intermediate_size;
    assert_eq!(b1.len(), bank.len() * d * inter, "gathered w1 bank size");

    let mut packed = std::collections::HashMap::new();
    let (g, params) = rlx_models_core::dsv41_moe::build_paged_moe_graph(
        &spec,
        &mut loader,
        0,
        rows,
        bank.len(),
        top_k,
        &mut packed,
    )
    .expect("paged moe graph");
    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, v) in &params {
        sess.set_param(n, v);
    }
    let got = sess.run(&[
        ("x", x.as_slice()),
        (&paged_names::bank(0, "w1"), b1.as_slice()),
        (&paged_names::bank(0, "w3"), b3.as_slice()),
        (&paged_names::bank(0, "w2"), b2.as_slice()),
        (&paged_names::slots(0), slots.as_slice()),
        (&paged_names::weights(0), routing.w.as_slice()),
    ]);
    // FP4 weights and a 2304-wide reduction, so the tolerance is looser than the
    // dense paths': what is being checked is the layout and the routing, not the
    // last bit of a quantized GEMM.
    compare(&got[0], &floats(&fx["out"]), "real MoE output", 3e-3);
    println!(
        "real MoE matched on {rows} rows, {} routed experts of {} ({} paged reads)",
        bank.len(),
        spec.n_routed_experts,
        pager.stats().misses
    );
}
