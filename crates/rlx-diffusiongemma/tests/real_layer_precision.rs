// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: GPL-3.0-only
//
//! What each weight precision costs, measured on a **real** DiffusionGemma
//! layer.
//!
//! The routed experts are 761 M of this layer's 800 M parameters, and ~91 GB of
//! the model as f32 — so whether the whole model can run at all is really the
//! question "how much accuracy does quantizing the expert banks cost". This
//! sweeps the formats `rlx_ir::quant::QuantScheme` supports by round-tripping
//! the real weights through each one and re-running the layer, so the number
//! reported is the format's true numerical cost on trained weights.
//!
//! Round-tripping (quantize → dequantize → f32 graph) measures *accuracy*, not
//! speed or footprint: it deliberately does not exercise the packed
//! `Op::DequantGroupedMatMul` kernels, which is separate work.
//!
//! ```sh
//! RLX_DG_REAL_LAYER_DIR=/w/dg-layer0 \
//!     cargo test -p rlx-diffusiongemma --release --test real_layer_precision -- --nocapture
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rlx_core::flow_util::compile_built;
use rlx_core::weight_map::WeightMap;
use rlx_diffusiongemma::attention::AttnDims;
use rlx_diffusiongemma::layer::{LayerDims, emit_encoder_layer};
use rlx_diffusiongemma::moe::MoeDims;
use rlx_diffusiongemma::weights::{expert_bank_keys, prepare_layer_experts};
use rlx_diffusiongemma::{DiffusionGemmaConfig, TextConfig};
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
        "run the reference first"
    );
    Some(d)
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let raw = std::fs::read(dir.join(format!("{name}.bin"))).unwrap();
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

// ── Weight round-trips ────────────────────────────────────────────────────

/// IEEE binary16, round-to-nearest-even, with overflow to infinity clamped.
fn to_f16(x: f32) -> f32 {
    let h = half::f16::from_f32(x);
    let v = h.to_f32();
    if v.is_finite() {
        v
    } else {
        x.signum() * 65504.0
    }
}

/// bfloat16: keep the top 16 bits of the f32, round-to-nearest-even.
fn to_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// GGUF Q8_0: 32-element blocks, `d = absmax / 127`, signed 8-bit.
fn q8_0(block: &[f32], out: &mut [f32]) {
    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let dq = if d > 0.0 { d } else { 0.0 };
    for (o, &x) in out.iter_mut().zip(block) {
        let q = if dq > 0.0 {
            (x / dq).round().clamp(-127.0, 127.0)
        } else {
            0.0
        };
        *o = q * dq;
    }
}

/// GGUF Q4_0: 32-element blocks, `d = -absmax / 8`, nibbles offset by 8.
fn q4_0(block: &[f32], out: &mut [f32]) {
    // llama.cpp picks the extreme *signed* value, not the absolute max.
    let mut max = 0f32;
    let mut amax = 0f32;
    for &x in block {
        if x.abs() > amax {
            amax = x.abs();
            max = x;
        }
    }
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for (o, &x) in out.iter_mut().zip(block) {
        let q = ((x * id + 8.5).floor() as i32).clamp(0, 15);
        *o = (q - 8) as f32 * d;
    }
}

/// GGUF Q4_1: 32-element blocks, `min` + `d = (max - min) / 15`, unsigned.
fn q4_1(block: &[f32], out: &mut [f32]) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in block {
        min = min.min(x);
        max = max.max(x);
    }
    let d = (max - min) / 15.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for (o, &x) in out.iter_mut().zip(block) {
        let q = (((x - min) * id + 0.5).floor() as i32).clamp(0, 15);
        *o = q as f32 * d + min;
    }
}

#[derive(Clone, Copy)]
enum Fmt {
    Bf16,
    F16,
    Q8_0,
    Q4_1,
    Q4_0,
}

impl Fmt {
    fn name(self) -> &'static str {
        match self {
            Fmt::Bf16 => "bf16",
            Fmt::F16 => "f16",
            Fmt::Q8_0 => "Q8_0",
            Fmt::Q4_1 => "Q4_1",
            Fmt::Q4_0 => "Q4_0",
        }
    }
    /// Bits per weight, counting each block's scale/min overhead.
    fn bits(self) -> f32 {
        match self {
            Fmt::Bf16 | Fmt::F16 => 16.0,
            Fmt::Q8_0 => 8.0 + 16.0 / 32.0,
            // Q4_1 carries both an f16 scale and an f16 min per 32-elem block.
            Fmt::Q4_1 => 4.0 + 1.0,
            Fmt::Q4_0 => 4.0 + 16.0 / 32.0,
        }
    }
    fn apply(self, v: &mut [f32]) {
        match self {
            Fmt::Bf16 => v.iter_mut().for_each(|x| *x = to_bf16(*x)),
            Fmt::F16 => v.iter_mut().for_each(|x| *x = to_f16(*x)),
            Fmt::Q8_0 | Fmt::Q4_1 | Fmt::Q4_0 => {
                let f = match self {
                    Fmt::Q8_0 => q8_0,
                    Fmt::Q4_1 => q4_1,
                    _ => q4_0,
                };
                let mut buf = [0f32; 32];
                for chunk in v.chunks_mut(32) {
                    let n = chunk.len();
                    f(chunk, &mut buf[..n]);
                    chunk.copy_from_slice(&buf[..n]);
                }
            }
        }
    }
}

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

fn run_layer(cfg: &DiffusionGemmaConfig, wm: &WeightMap, seq: usize, x: &[f32]) -> Vec<f32> {
    let t: &TextConfig = &cfg.text_config;
    let f = DType::F32;
    let hidden = t.hidden_size;
    let dh = t.layer_head_dim(LAYER);
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
    let built = ModelFlow::new("dg_prec")
        .with_profile(CompileProfile::llama32_prefill())
        .input("hidden", hs.clone())
        .input("rope_cos", Shape::new(&[seq, dh / 2], f))
        .input("rope_sin", Shape::new(&[seq, dh / 2], f))
        .plugin_named("layer", move |emit, _prev| {
            let x = emit.flow_input("hidden")?.hir_id();
            let cos = emit.flow_input("rope_cos")?.hir_id();
            let sin = emit.flow_input("rope_sin")?.hir_id();
            let (out, _) = emit_encoder_layer(
                emit,
                &prefix,
                x,
                &dims,
                cos,
                sin,
                MaskKind::SlidingWindow(window),
            )?;
            Ok(Some(emit.wrap(out, hs.clone())))
        })
        .output("out")
        .build_with(&mut Shared(wm), None)
        .expect("build");
    let names: Vec<String> = built.output_names().to_vec();
    let mut compiled = compile_built(built, dev()).expect("compile");
    let (cos, sin) = t.rope_tables(LAYER, 0, seq);
    let outs = compiled.run(&[
        ("hidden", x),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ]);
    let by: HashMap<&str, &Vec<f32>> = names.iter().map(|s| s.as_str()).zip(outs.iter()).collect();
    by["out"].clone()
}

/// Apply `fmt` to a copy of the checkpoint. `experts_only` limits it to the
/// routed banks, which are 95% of the layer's parameters.
fn requantized(base: &WeightMap, fmt: Fmt, experts_only: bool) -> WeightMap {
    let (gu, dn) = expert_bank_keys(LAYER);
    let mut m: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for k in base.keys().map(|s| s.to_string()).collect::<Vec<_>>() {
        let (d, sh) = base.get(&k).unwrap();
        let mut data = d.to_vec();
        let is_expert = k == gu || k == dn;
        // Norms and the layer scalar stay full precision, as every real
        // quantized checkpoint does — they are tiny and sensitive.
        let is_norm = k.contains("norm") || k.ends_with("layer_scalar") || k.ends_with(".scale");
        if (is_expert || !experts_only) && !is_norm {
            fmt.apply(&mut data);
        }
        m.insert(k, (data, sh.to_vec()));
    }
    WeightMap::from_tensors(m)
}

#[test]
fn precision_sweep_on_a_real_layer() {
    let Some(d) = dir() else {
        eprintln!("skipping: set RLX_DG_REAL_LAYER_DIR (see the module docs)");
        return;
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("layer_meta.json")).unwrap()).unwrap();
    let seq = meta["seq"].as_u64().unwrap() as usize;
    let cfg = DiffusionGemmaConfig::from_file(d.join("config.json")).expect("config");
    let t = &cfg.text_config;

    let mut base = WeightMap::from_safetensors_dir(&d).expect("weights");
    prepare_layer_experts(&cfg, &mut base, LAYER).expect("pretranspose");
    let x = read_f32(&d, "layer_in");
    let torch_ref = read_f32(&d, "layer_out");

    // f32 baseline: the same graph, unquantized. Everything else is measured
    // against this so the numbers isolate the format, not the port.
    let f32_out = run_layer(&cfg, &base, seq, &x);
    let base_cos = cosine(&f32_out, &torch_ref);
    println!("\nreal layer {LAYER}, seq {seq} — f32 baseline vs torch: cos {base_cos:.8}\n");
    assert!(base_cos > 0.99999);

    let expert_params =
        t.num_experts * 3 * t.moe_intermediate_size * t.hidden_size * t.num_hidden_layers;
    println!(
        "{:<7} {:>6}  {:<22} {:<22} {:<13}",
        "format", "bits", "experts only (cos)", "all weights (cos)", "model experts",
    );
    for fmt in [Fmt::Bf16, Fmt::F16, Fmt::Q8_0, Fmt::Q4_1, Fmt::Q4_0] {
        let mut line = format!("{:<7} {:>6.2}  ", fmt.name(), fmt.bits());
        let mut cosines = Vec::new();
        for experts_only in [true, false] {
            let wm = requantized(&base, fmt, experts_only);
            let out = run_layer(&cfg, &wm, seq, &x);
            let c = cosine(&out, &f32_out);
            let rel: f32 = {
                let scale = f32_out.iter().fold(0f32, |m, v| m.max(v.abs()));
                out.iter()
                    .zip(&f32_out)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max)
                    / scale
            };
            line.push_str(&format!("{c:.8} (rel {rel:.1e})  "));
            cosines.push(c);
        }
        let gb = expert_params as f64 * fmt.bits() as f64 / 8.0 / 1e9;
        println!("{line}{gb:>6.1} GB");
        // bf16 must be a no-op: the checkpoint is already bf16.
        if matches!(fmt, Fmt::Bf16) {
            assert!(
                cosines.iter().all(|&c| c > 0.999999999),
                "bf16 round-trip should be exact on a bf16 checkpoint, got {cosines:?}"
            );
        }
        // Even 4-bit experts should stay recognisably the same layer.
        assert!(
            cosines[0] > 0.99,
            "{}: experts-only cosine {:.6} is implausibly low",
            fmt.name(),
            cosines[0]
        );
    }
    println!(
        "\n(f32 experts for the whole model: {:.1} GB)",
        expert_params as f64 * 4.0 / 1e9
    );
}
