//! ONNX-graph engine: import each TinyTTS subgraph into rlx-ir HIR, compile it
//! per `(component, device, length)` with on-disk + in-memory caching, run it,
//! and orchestrate the full VITS pipeline with the Rust [`crate::glue`] stage.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, DType, Device};

use crate::config::BundleConfig;
use crate::glue::{self, Rng};

/// Per-call synthesis controls.
#[derive(Debug, Clone)]
pub struct InferOpts {
    /// Duration scaling: `> 1.0` slows speech, `< 1.0` speeds it up (`1/speed`).
    pub length_scale: f32,
    /// Prior sampling temperature (`z_p = m_p + N(0,1)·exp(logs_p)·noise_scale`).
    pub noise_scale: f32,
    /// RNG seed for the latent sampling (reproducible synthesis).
    pub seed: u64,
}

impl InferOpts {
    pub fn from_config(cfg: &BundleConfig) -> Self {
        Self {
            length_scale: cfg.length_scale,
            noise_scale: cfg.noise_scale,
            seed: 1234,
        }
    }
}

/// Bump when the import/compile pipeline changes in a way that invalidates the
/// on-disk AOT cache for these graphs.
const CACHE_TAG: &str = "tiny_tts_v1";

pub struct TinyModel {
    onnx_dir: PathBuf,
    cfg: BundleConfig,
    /// Compiled graphs keyed by `(component, device, length)`.
    cache: Mutex<HashMap<(&'static str, Device, usize), CompiledGraph>>,
}

fn aot_root() -> PathBuf {
    if let Ok(p) = std::env::var("TINY_TTS_AOT_CACHE") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("rlx/tiny_tts_aot")
}

fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Decode an f32 graph output (panics-free: errors if the dtype is not F32).
fn as_f32((bytes, dt): &(Vec<u8>, DType)) -> Result<Vec<f32>> {
    anyhow::ensure!(*dt == DType::F32, "expected F32 output, got {dt:?}");
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl TinyModel {
    pub fn new(onnx_dir: PathBuf, cfg: BundleConfig) -> Self {
        Self {
            onnx_dir,
            cfg,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Import + compile one subgraph for a given symbolic length, with caching.
    pub fn graph(
        &self,
        component: &'static str,
        device: Device,
        length: usize,
    ) -> Result<CompiledGraph> {
        let key = (component, device, length);
        if let Some(g) = self.cache.lock().expect("graph cache").get(&key) {
            return Ok(g.clone());
        }
        let compiled = self.compile(component, device, length)?;
        self.cache
            .lock()
            .expect("graph cache")
            .insert(key, compiled.clone());
        Ok(compiled)
    }

    fn compile(&self, component: &str, device: Device, length: usize) -> Result<CompiledGraph> {
        let path = self.onnx_dir.join(format!("{component}.onnx"));
        anyhow::ensure!(path.is_file(), "missing graph {}", path.display());
        // Decompose every ConvTranspose into zero-insert + a regular Conv. The
        // decomposition is bit-exact against onnxruntime (verified on the decoder's
        // five HiFi-GAN upsamplers), whereas the backends' native transposed-conv
        // kernels (CPU/MLX) are numerically wrong here — so we route ALL devices
        // through it. It also unblocks wgpu/CoreML, which lack a native kernel.
        let _ = device;
        let decompose_ct = true;
        let (hir, params, report) = import_graph(&path, component, length, decompose_ct)?;
        if report.stubbed > 0 || !report.unsupported.is_empty() {
            eprintln!(
                "[tiny-tts] warn: {component} import stubbed={} unsupported={:?}",
                report.stubbed, report.unsupported
            );
        }

        let cache_key = format!("{CACHE_TAG}_{component}_{device:?}_s{length}");
        let cache = AotCache::new(aot_root());
        let mut compiled = cache
            .compile_hir_cached(&cache_key, device, hir, &CompileOptions::default())
            .map_err(|e| anyhow::anyhow!("compile {component}: {e}"))?;
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        compiled.finalize_params();
        Ok(compiled)
    }

    /// Full pipeline: phone/tone/lang ids → waveform (raw, pre-normalization).
    pub fn synthesize(
        &self,
        device: Device,
        phone: &[i64],
        tone: &[i64],
        lang: &[i64],
        speaker: i64,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        let t = phone.len();
        anyhow::ensure!(t > 0, "empty phoneme sequence");
        let c = self.cfg.inter_channels; // latent channels (== g/m_p channel width)

        // ── 1. Text encoder ──────────────────────────────────────────────
        let mut enc = self.graph("text_encoder", device, t)?;
        let phone_b = i64_bytes(phone);
        let tone_b = i64_bytes(tone);
        let lang_b = i64_bytes(lang);
        let len_b = i64_bytes(&[t as i64]);
        let sid_b = i64_bytes(&[speaker]);
        let bert_b = f32_bytes(&vec![0.0f32; 1024 * t]);
        let ja_bert_b = f32_bytes(&vec![0.0f32; 768 * t]);
        dbg_dump(
            "phone",
            &phone.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        );
        dbg_dump("tone", &tone.iter().map(|&x| x as f32).collect::<Vec<_>>());
        dbg_dump("lang", &lang.iter().map(|&x| x as f32).collect::<Vec<_>>());
        dbg_dump("sid", &[speaker as f32]);
        let enc_out = enc.run_typed(&[
            ("phone_ids", &phone_b, DType::I64),
            ("phone_lengths", &len_b, DType::I64),
            ("tone_ids", &tone_b, DType::I64),
            ("language_ids", &lang_b, DType::I64),
            ("bert", &bert_b, DType::F32),
            ("ja_bert", &ja_bert_b, DType::F32),
            ("speaker_id", &sid_b, DType::I64),
        ]);
        anyhow::ensure!(
            enc_out.len() >= 5,
            "text_encoder returned {} outputs",
            enc_out.len()
        );
        // Declared output order: x_enc, m_p, logs_p, x_mask, g.
        let x_enc = &enc_out[0].0; // raw bytes reused as duration-predictor input
        let m_p = as_f32(&enc_out[1])?;
        let logs_p = as_f32(&enc_out[2])?;
        let g_bytes = enc_out[4].0.clone();
        // We feed exactly `t` tokens with no padding, so the phone mask is all
        // ones. (The graph's own `x_mask` output is a degenerate length-1 tensor —
        // the importer collapses the Range/Less mask — but every graph broadcasts
        // it correctly internally, so only this host-side glue needs the real mask.)
        let x_mask = vec![1.0f32; t];

        // ── 2. Duration predictor ────────────────────────────────────────
        let mut dp = self.graph("duration_predictor", device, t)?;
        let x_mask_b = f32_bytes(&x_mask);
        let dp_out = dp.run_typed(&[
            ("x", x_enc, DType::F32),
            ("x_mask", &x_mask_b, DType::F32),
            ("g", &g_bytes, DType::F32),
        ]);
        anyhow::ensure!(!dp_out.is_empty(), "duration_predictor returned no output");
        let logw = as_f32(&dp_out[0])?; // [1,1,T] → T values
        dbg_mag("m_p", &m_p);
        dbg_mag("logs_p", &logs_p);
        dbg_mag("logw", &logw);

        // ── 3. Alignment + latent sampling (Rust glue) ───────────────────
        let (w_ceil, y_len) = glue::durations(&logw, &x_mask, opts.length_scale);
        if std::env::var("RLX_TTS_DBG").is_ok() {
            eprintln!("[dbg] w_ceil={w_ceil:?} y_len={y_len}");
        }
        let attn = glue::alignment_path(&w_ceil, y_len);
        let m_exp = glue::expand_prior(&attn, &m_p, c, t, y_len);
        let logs_exp = glue::expand_prior(&attn, &logs_p, c, t, y_len);
        let mut rng = Rng::new(opts.seed);
        let z_p = glue::sample_z_p(&m_exp, &logs_exp, opts.noise_scale, &mut rng);
        dbg_mag("z_p", &z_p);
        dbg_dump("z_p", &z_p);
        dbg_dump("y_mask", &vec![1.0f32; y_len]);
        dbg_dump("m_p", &m_p);
        dbg_dump("logs_p", &logs_p);
        dbg_dump("logw", &logw);
        let y_mask = vec![1.0f32; y_len]; // frame count == sum of durations → all ones

        // ── 4. Flow (reverse) ────────────────────────────────────────────
        let mut flow = self.graph("flow", device, y_len)?;
        let z_p_b = f32_bytes(&z_p);
        let y_mask_b = f32_bytes(&y_mask);
        let flow_out = flow.run_typed(&[
            ("z_p", &z_p_b, DType::F32),
            ("y_mask", &y_mask_b, DType::F32),
            ("g", &g_bytes, DType::F32),
        ]);
        anyhow::ensure!(!flow_out.is_empty(), "flow returned no output");
        let z = as_f32(&flow_out[0])?; // [1, c, y_len]
        dbg_mag("z(flow)", &z);
        dbg_dump("z", &z);
        dbg_dump("g", &as_f32(&enc_out[4]).unwrap_or_default());
        if let Ok(p) = std::env::var("RLX_TTS_DUMP") {
            std::fs::write(format!("{p}/dims.txt"), format!("c={c} y_len={y_len}\n")).ok();
        }

        // ── 5. Decoder (z·y_mask → waveform) ─────────────────────────────
        // y_mask is all ones here, so masking is the identity.
        let mut dec = self.graph("decoder", device, y_len)?;
        let z_b = f32_bytes(&z);
        let dec_out = dec.run_typed(&[("z", &z_b, DType::F32), ("g", &g_bytes, DType::F32)]);
        anyhow::ensure!(!dec_out.is_empty(), "decoder returned no output");
        let wav = as_f32(&dec_out[0])?; // [1, 1, samples]
        dbg_mag("dec_out", &wav);
        dbg_dump("dec_out", &wav);
        for (i, o) in dec_out.iter().enumerate() {
            if let Ok(v) = as_f32(o) {
                dbg_dump(&format!("dec_out_{i}"), &v);
            }
        }
        Ok(wav)
    }
}

/// Env-gated (`RLX_TTS_DUMP=<dir>`) raw-f32 dump for ORT cross-validation.
fn dbg_dump(name: &str, v: &[f32]) {
    if let Ok(dir) = std::env::var("RLX_TTS_DUMP") {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{name}.f32"), bytes).ok();
    }
}

/// Env-gated (`RLX_TTS_DBG`) magnitude print for cross-backend divergence triage.
fn dbg_mag(name: &str, v: &[f32]) {
    if std::env::var("RLX_TTS_DBG").is_err() {
        return;
    }
    let n = v.len().max(1);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut sa = 0.0f64;
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
        sa += x.abs() as f64;
    }
    eprintln!(
        "[dbg] {name:10} len={:6} min={lo:+.4e} max={hi:+.4e} mean|x|={:.4e}",
        v.len(),
        sa / n as f64
    );
}

/// Import one ONNX subgraph into HIR for a given symbolic length.
///
/// The decoder upsamples its length 512× through five `ConvTranspose` layers.
/// `prepare_onnx_file` runs the importer's shape heuristic (which assumes a
/// channel-first tensor's length equals `sequence_length`) and bakes the result
/// into every node's `output_meta`; for the decoder that collapses all upsampled
/// lengths back to the input length. We clear the decoder's `output_meta` so the
/// lowering recomputes each conv/transpose length from its actual input shape.
/// The other three graphs keep length `T` throughout, where the heuristic is
/// already correct.
pub fn import_graph(
    path: &Path,
    component: &str,
    length: usize,
    decompose_conv_transpose: bool,
) -> Result<(
    rlx_ir::hir::HirModule,
    HashMap<String, Vec<f32>>,
    rlx_onnx_import::ImportReport,
)> {
    use rlx_onnx_import::{
        ImportOptions, build_hir_from_parts, prepare_onnx_file, tensor_data::TypedParams,
    };

    let opts = ImportOptions {
        sequence_length: length,
        max_waveform_samples: (length * 1024).max(48_000),
        use_quantized_kernels: false,
        strict: false,
        dynamic_sequence: false,
        decompose_conv_transpose,
        ..ImportOptions::default()
    };

    let (manifest, mut nodes, params, i64_params, init_shapes) =
        prepare_onnx_file(path).with_context(|| format!("prepare {}", path.display()))?;
    // `prepare_onnx_file` propagates shapes with a fixed default `sequence_length`
    // (128), inconsistent with our per-call compile length. Reset each meta entry
    // to an empty placeholder (keeping one entry per output so re-propagation — which
    // skips nodes with too few meta entries — still visits every node), then
    // re-propagate at the real compile length.
    let _ = component;
    for node in &mut nodes {
        for meta in &mut node.output_meta {
            *meta = serde_json::json!({});
        }
    }
    rlx_onnx_import::shape_propagate::propagate_shapes(&mut nodes, &manifest, &init_shapes, &opts);
    let (hir, params, _typed, report) = build_hir_from_parts(
        &manifest,
        nodes,
        params,
        TypedParams::new(),
        i64_params,
        &init_shapes,
        opts,
    )
    .with_context(|| format!("lower {}", path.display()))?;
    Ok((hir, params, report))
}

/// Standalone keystone helper: import + compile one graph (used by the example).
pub fn compile_graph(
    onnx_dir: &Path,
    component: &'static str,
    device: Device,
    length: usize,
) -> Result<CompiledGraph> {
    let cfg = BundleConfig {
        model: String::new(),
        sample_rate: 44100,
        add_blank: true,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        length_scale: 1.0,
        inter_channels: 80,
        gin_channels: 80,
    };
    let m = TinyModel::new(onnx_dir.to_path_buf(), cfg);
    m.compile(component, device, length)
}
