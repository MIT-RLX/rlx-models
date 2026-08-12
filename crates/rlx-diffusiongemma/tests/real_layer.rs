// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! One text layer on **real** `google/diffusiongemma-26B-A4B-it` weights.
//!
//! This is the part most worth checking against trained weights: the 128-expert
//! MoE. A synthetic router with random weights spreads nearly uniformly, so it
//! never really tests top-k dispatch; a trained router is sharply peaked and
//! per-expert scaled, and routes each token to a specific 8 of 128 banks.
//!
//! Layer 0 is a *sliding* layer, so it also exercises the geometry that differs
//! from the global layers: a real `v_proj`, 16 heads × 256, 8 KV heads.
//!
//! ```sh
//! python3 scripts/diffusiongemma_fetch_subset.py /w/dg-layer0 --subset layer0
//! python3 scripts/diffusiongemma_real_layer.py /w/dg-layer0
//! RLX_DG_REAL_LAYER_DIR=/w/dg-layer0 \
//!     cargo test -p rlx-diffusiongemma --release --test real_layer -- --nocapture
//! ```
//!
//! Skips when the env var is unset. One layer is ~1.63 GB on disk and ~3.3 GB
//! as f32, which is why the test builds a single layer directly instead of
//! going through the full encoder flow.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_diffusiongemma::attention::AttnDims;
use rlx_diffusiongemma::layer::{LayerDims, emit_encoder_layer};
use rlx_diffusiongemma::moe::MoeDims;
use rlx_diffusiongemma::weights::prepare_layer_experts;
use rlx_diffusiongemma::{DiffusionGemmaConfig, enc_k_name, enc_v_name};
use rlx_flow::{CompileProfile, ModelFlow, WeightSource};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Shape};
use rlx_runtime::Device;

const LAYER: usize = 0;

fn dev() -> Device {
    std::env::var("RLX_TEST_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| rlx_cli::parse_device(&s).expect("bad RLX_TEST_DEVICE"))
        .unwrap_or(Device::Cpu)
}

fn dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("RLX_DG_REAL_LAYER_DIR").ok()?);
    assert!(
        d.join("layer_meta.json").is_file(),
        "RLX_DG_REAL_LAYER_DIR={d:?} has no layer_meta.json — \
         run scripts/diffusiongemma_real_layer.py first"
    );
    Some(d)
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let raw = std::fs::read(dir.join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("reading {name}.bin: {e}"));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn report(label: &str, got: &[f32], want: &[f32], rel_tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let cos = cosine(got, want);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let mad = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mean: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / got.len() as f64;
    println!(
        "  {label:<12} cos {cos:.8}  rel max {:.2e}  rel mean {:.2e}  (|x|max {scale:.4})",
        mad / scale,
        mean as f32 / scale
    );
    assert!(cos > 0.99999, "{label}: cosine {cos:.8}");
    assert!(
        mad / scale <= rel_tol,
        "{label}: rel max {:.2e}",
        mad / scale
    );
}

/// Non-destructive weight source (the crate's own is private).
struct Shared<'a>(&'a WeightMap);
impl WeightSource for Shared<'_> {
    fn take(&mut self, key: &str, transpose: bool) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .0
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing `{key}`"))?;
        if !transpose {
            return Ok((data.to_vec(), shape.to_vec()));
        }
        let (r, c) = (shape[0], shape[1]);
        let mut out = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = data[i * c + j];
            }
        }
        Ok((out, vec![c, r]))
    }
    fn has(&self, key: &str) -> bool {
        self.0.has(key)
    }
}

#[test]
fn text_layer_on_real_weights_matches_torch() {
    let Some(d) = dir() else {
        eprintln!("skipping: set RLX_DG_REAL_LAYER_DIR (see the module docs)");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("layer_meta.json")).unwrap()).unwrap();
    let seq = meta["seq"].as_u64().unwrap() as usize;
    let experts_hit = meta["experts_hit"].as_u64().unwrap() as usize;

    let cfg = DiffusionGemmaConfig::from_file(d.join("config.json")).expect("real config");
    let t = &cfg.text_config;
    // Guard the geometry this layer is supposed to have.
    assert!(!t.is_full(LAYER), "layer 0 is a sliding layer");
    assert_eq!(t.layer_head_dim(LAYER), 256);
    assert_eq!(t.layer_kv_heads(LAYER), 8);
    assert!(!t.layer_k_eq_v(LAYER), "sliding layers ship a real v_proj");
    assert_eq!((t.num_experts, t.top_k_experts), (128, 8));
    // The reference saw a genuinely distributed router, so this is a real test
    // of top-k dispatch rather than of a collapsed one.
    assert!(
        experts_hit > 16,
        "reference router only hit {experts_hit} experts; not a meaningful MoE test"
    );

    let mut wm = WeightMap::from_safetensors_dir(&d).expect("real layer weights");
    prepare_layer_experts(&cfg, &mut wm, LAYER).expect("pretranspose experts");
    println!("loaded {} real tensors for layer {LAYER}", wm.len());

    let f = DType::F32;
    let hidden = t.hidden_size;
    let dh = t.layer_head_dim(LAYER);
    let half = dh / 2;
    let dims = LayerDims {
        attn: AttnDims {
            hidden,
            num_heads: t.num_attention_heads,
            num_kv_heads: t.layer_kv_heads(LAYER),
            head_dim: dh,
            k_eq_v: t.layer_k_eq_v(LAYER),
            eps: t.rms_norm_eps,
            seq,
        },
        moe: MoeDims {
            hidden,
            moe_inter: t.moe_intermediate_size,
            num_experts: t.num_experts,
            top_k: t.top_k_experts,
            rows: seq,
            eps: t.rms_norm_eps,
            root_scale: t.router_root_scale(),
            experts_pretransposed: true,
        },
        intermediate: t.intermediate_size,
        hidden,
        eps: t.rms_norm_eps,
        seq,
        layer_scalar_key: format!("model.encoder.language_model.layers.{LAYER}.layer_scalar"),
    };

    let hs = Shape::new(&[1, seq, hidden], f);
    let prefix = format!("model.decoder.layers.{LAYER}");
    let window = t.sliding_window.saturating_sub(1);
    let flow = ModelFlow::new("dg_real_layer")
        .with_profile(CompileProfile::llama32_prefill())
        .input("hidden", hs.clone())
        .input("rope_cos", Shape::new(&[seq, half], f))
        .input("rope_sin", Shape::new(&[seq, half], f))
        .plugin_named("layer", move |emit, _prev| {
            let x = emit.flow_input("hidden")?.hir_id();
            let cos = emit.flow_input("rope_cos")?.hir_id();
            let sin = emit.flow_input("rope_sin")?.hir_id();
            let (out, tap) = emit_encoder_layer(
                emit,
                &prefix,
                x,
                &dims,
                cos,
                sin,
                MaskKind::SlidingWindow(window),
            )?;
            emit.state.side_outputs.push((enc_k_name(LAYER), tap.k));
            emit.state.side_outputs.push((enc_v_name(LAYER), tap.v));
            Ok(Some(emit.wrap(out, hs.clone())))
        })
        .output("out");

    let built = flow
        .build_with(&mut Shared(&wm), None)
        .expect("build real layer");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile real layer");

    let x = read_f32(&d, "layer_in");
    let (cos, sin) = t.rope_tables(LAYER, 0, seq);
    let outs = compiled.run(&[
        ("hidden", x.as_slice()),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ]);
    let by: HashMap<&str, &Vec<f32>> = names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();

    println!("real layer {LAYER}, seq {seq}, {experts_hit}/128 experts routed:");
    report(
        "K tap",
        by[enc_k_name(LAYER).as_str()],
        &read_f32(&d, "layer_k"),
        5e-3,
    );
    report(
        "V tap",
        by[enc_v_name(LAYER).as_str()],
        &read_f32(&d, "layer_v"),
        5e-3,
    );
    report("layer out", by["out"], &read_f32(&d, "layer_out"), 5e-3);

    let out = by["out"];
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(
        out.iter().fold(0f32, |m, v| m.max(v.abs())) > 1.0,
        "layer output looks collapsed"
    );
}
