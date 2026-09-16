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

use rlx_models_core::dsv41::DeepseekV41Spec;
use rlx_models_core::dsv41_engram::EngramHashPlan;
use rlx_models_core::dsv41_graph::{V41Inputs, build_deepseek_v41_prefill};
use rlx_models_core::weight_loader::WeightLoader;
use rlx_runtime::{Device, Session};
use serde_json::Value;
use std::collections::BTreeMap;

/// The name-keyed PRNG shared with the Python dumper: FNV-1a over the parameter
/// name seeds splitmix64, and each draw is uniform in `[-s, s)` with
/// `s = 1/sqrt(fan_in)` for matrices and `0.2` for vectors.
mod prng {
    pub fn fnv1a(name: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    pub fn param_values(name: &str, shape: &[usize]) -> Vec<f32> {
        let n: usize = shape.iter().product::<usize>().max(1);
        let scale = if shape.len() >= 2 {
            (1.0f64 / *shape.last().unwrap() as f64).sqrt()
        } else {
            0.2
        };
        let mut s = fnv1a(name);
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let f = (z >> 11) as f64 / (1u64 << 53) as f64;
                ((f - 0.5) * 2.0 * scale) as f32
            })
            .collect()
    }
}

/// Serves every parameter the builder asks for from the PRNG, using the shape
/// the reference model actually allocated. A key the reference does not have is
/// an error, which is what makes this also a naming test: the builder must ask
/// for exactly the checkpoint's tensor names.
struct RefLoader {
    shapes: BTreeMap<String, Vec<usize>>,
    asked: Vec<String>,
}

impl RefLoader {
    fn new(shapes: BTreeMap<String, Vec<usize>>) -> Self {
        Self {
            shapes,
            asked: Vec::new(),
        }
    }
}

impl WeightLoader for RefLoader {
    fn take(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let shape = self
            .shapes
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("reference checkpoint has no tensor `{key}`"))?
            .clone();
        self.asked.push(key.to_string());
        Ok((prng::param_values(key, &shape), shape))
    }

    fn take_transposed(&mut self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (d, s) = self.take(key)?;
        if s.len() != 2 {
            return Ok((d, s));
        }
        let (r, c) = (s[0], s[1]);
        let mut o = vec![0f32; d.len()];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = d[i * c + j];
            }
        }
        Ok((o, vec![c, r]))
    }

    fn len(&self) -> usize {
        self.shapes.len()
    }

    fn remaining_keys(&self) -> Vec<String> {
        self.shapes.keys().cloned().collect()
    }
}

fn fixture() -> Value {
    // An override points the test at a full (untrimmed) dump while iterating.
    let path = std::env::var("RLX_DSV41_REF").unwrap_or_else(|_| {
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

/// Max absolute and max relative deviation, the latter normalized by the
/// reference's own scale so a near-zero entry does not dominate.
fn compare(got: &[f32], want: &[f32], label: &str, tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
    let mut max_abs = 0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}: non-finite at {i}: {g}");
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
    }
    let rel = max_abs / scale;
    assert!(
        rel < tol,
        "{label}: max |Δ| {max_abs:e} at index {at} (got {}, want {}), scale {scale:e}, rel {rel:e} >= {tol:e}",
        got[at],
        want[at]
    );
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
            image_positions: Vec::new(),
            emit_main_hidden: false,
        },
    )
}

fn run_prefill(fx: &Value) -> (Vec<f32>, Vec<String>) {
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
    let mut loader = RefLoader::new(shapes);
    let mut packed = std::collections::HashMap::new();
    let (g, params) =
        build_deepseek_v41_prefill(&spec, &mut loader, ids.len(), &inputs, &mut packed)
            .expect("prefill graph builds");
    let asked = loader.asked.clone();

    let opts = rlx_models_core::flow_bridge::compile_options_for_packed_gguf_prefill_with_profile(
        &rlx_flow::CompileProfile::qwen3_prefill(),
        Device::Cpu,
    );
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
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
    let fx = fixture();
    let (_, asked) = run_prefill(&fx);
    let available: Vec<String> = fx["shapes"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
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
            spec.engram.as_ref().is_some_and(|e| e.layer_ids.contains(&il)),
            "layer {il}: engram placement"
        );
    }
}

/// A stage split must reproduce the single-shot prefill exactly — which it only
/// can if the boundary carries the Hyper-Connection pre-mix as well as the
/// hidden state.
#[test]
fn split_stages_reproduce_single_shot_prefill() {
    use rlx_models_core::dsv41_graph::build_deepseek_v41_stage;
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
    let mut l1 = RefLoader::new(shapes.clone());
    let mut packed = std::collections::HashMap::new();
    let (g1, p1) = build_deepseek_v41_stage(
        &spec, &mut l1, seq, 0..split, true, false, &inputs, &mut packed,
    )
    .expect("stage 1 builds");
    let mut s1 = Session::new(Device::Cpu).compile_with(g1, &opts);
    for (n, d) in &p1 {
        s1.set_param(n, d);
    }
    let mid = s1.run(&[("input_ids", ids_f32.as_slice())]);
    let (hidden, pre_mix) = (mid[0].clone(), mid[1].clone());
    assert_eq!(hidden.len(), seq * spec.hc_mult * spec.dim);
    assert_eq!(pre_mix.len(), seq * spec.hc_mult);

    let mut l2 = RefLoader::new(shapes);
    let mut packed2 = std::collections::HashMap::new();
    let (g2, p2) = build_deepseek_v41_stage(
        &spec,
        &mut l2,
        seq,
        split..spec.n_layers,
        false,
        true,
        &inputs,
        &mut packed2,
    )
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
    use rlx_models_core::dsv41::DeepseekV41Spec;
    use rlx_models_core::dsv41_vision::build_v41_vision;
    use rlx_ir::graph::Graph;
    use rlx_ir::{DType, Shape};

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
    let vs = spec.vision.as_ref().expect("vision config present");

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
    let mut loader = RefLoader::new(shapes);
    let mut g = Graph::new("v41_vision");
    let mut params = std::collections::HashMap::new();
    let n = n_h * n_w;
    let patch_in = 3 * vs.patch_size * vs.patch_size;
    let patches = g.input("patches", Shape::new(&[n, patch_in], DType::F32));
    let out = build_v41_vision(
        &mut g,
        &mut params,
        &mut loader,
        vs,
        spec.dim,
        patches,
        n_h,
        n_w,
    )
    .expect("vision graph builds");
    g.set_outputs(vec![out]);

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
    use rlx_models_core::dsv41_decode::{
        V41DecodeCache, V41DecodePlan, build_deepseek_v41_decode,
    };
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
            emit_main_hidden: false,
        };
        let mut loader = RefLoader::new(shapes.clone());
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
        let mut feed: Vec<(&str, &[f32])> = vec![("input_ids", id.as_slice())];
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
    assert_eq!(spec.moe_dims(spec.n_layers), (2, 2), "DSpark has its own bank");

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
    let mut loader = RefLoader::new(shapes.clone());
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
    let mut loader = RefLoader::new(shapes.clone());
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
    let mut loader = RefLoader::new(shapes);
    let (g, params) =
        build_v41_dspark_markov_step(&spec, &mut loader, block).expect("markov graph builds");
    let mut sess = Session::new(Device::Cpu).compile_with(g, &opts);
    for (n, d) in &params {
        sess.set_param(n, d);
    }
    // slot i is biased by the token that precedes it
    let toks: Vec<f32> = out_ids[..block].iter().map(|&i| i as f32).collect();
    let mk = sess.run(&[("token_ids", toks.as_slice()), ("hidden", out[1].as_slice())]);
    let biased: Vec<f32> = out[0]
        .iter()
        .zip(&mk[0])
        .map(|(a, b)| a + b)
        .collect();
    compare(&biased, &floats(&fx["biased_logits"]), "dspark biased logits", 2e-4);
    compare(&mk[1], &floats(&fx["confidence"]), "dspark confidence", 2e-4);

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
