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

/// Kernel-variant / numeric-precision policy for the compiled TTS graphs.
///
/// Mirrors the per-op kernel selection in `../rlx` (Metal `SgemmVariant`, CUDA
/// TF32, CPU conv), but exposes it as a single per-model / per-call knob instead
/// of raw `RLX_*` env vars. [`KernelVariant::apply`] installs the corresponding
/// `rlx_ir::env` **code overrides** — which take precedence over the process
/// environment and are read by the backends at dispatch time — so no
/// process-global `std::env` mutation is needed and the same compiled graph can
/// run fast or precise kernels without recompiling.
///
/// The knobs are read at dispatch, so applying a variant affects every
/// subsequent compile/run in the process (last-writer-wins across concurrent
/// models). Set it once per process for a consistent policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelVariant {
    /// Backend-default kernels — highest throughput. Metal picks its SIMD
    /// matmul (e.g. `simd4x4`) via the cost model; CPU uses the fast im2col
    /// conv; CUDA allows TF32. Best for production synthesis.
    #[default]
    Fast,
    /// Precision / parity kernels — leans bit-exact vs onnxruntime. Metal forces
    /// the scalar fp32 `naive` matmul (`RLX_METAL_PRECISE`); CPU uses the exact
    /// conv path; CUDA disables TF32 (`RLX_CUDA_NO_TF32` + `RLX_CUDA_PARITY`).
    /// Slower; for parity validation and precision-critical work.
    Precise,
    /// Leave all kernel-selection knobs untouched — honor the caller's own
    /// `RLX_*` env / `rlx_ir::env` overrides.
    Inherit,
}

impl KernelVariant {
    /// Parse `RLX_TTS_KERNEL` (`fast` | `precise` | `inherit`); `Fast` otherwise.
    pub fn from_env() -> Self {
        match std::env::var("RLX_TTS_KERNEL")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "precise" | "exact" | "parity" => Self::Precise,
            "inherit" | "env" | "none" => Self::Inherit,
            _ => Self::Fast,
        }
    }

    /// Install the backend kernel-selection overrides for this variant via
    /// `rlx_ir::env` (code-side, precedence over process env).
    pub fn apply(self) {
        use rlx_ir::env;
        match self {
            Self::Fast => {
                // Let the Metal cost model pick the fast SIMD matmul; ensure the
                // precise override isn't left on from a prior Precise run.
                env::set("RLX_METAL_PRECISE", "0");
                env::set("RLX_FAST_CONV", "1");
                env::set("RLX_FFT_FAST", "1");
                env::set("RLX_CUDA_CONV_TF32", "1");
                env::set("RLX_CUDA_NO_TF32", "0");
                env::set("RLX_CUDA_PARITY", "0");
                env::set("RLX_WGPU_MATMUL_F32_ONLY", "0");
                env::set("RLX_WGPU_NO_F16_MIRROR", "0");
            }
            Self::Precise => {
                env::set("RLX_METAL_PRECISE", "1");
                env::set("RLX_FAST_CONV", "0");
                env::set("RLX_FFT_FAST", "0");
                env::set("RLX_CUDA_CONV_TF32", "0");
                env::set("RLX_CUDA_NO_TF32", "1");
                env::set("RLX_CUDA_PARITY", "1");
                // wgpu: skip f16 weight mirrors / coop shortcuts for ODE-sensitive
                // graphs (F5 DiT). Still not Metal-class; prefer Metal DiT on Apple.
                env::set("RLX_WGPU_MATMUL_F32_ONLY", "1");
                env::set("RLX_WGPU_NO_F16_MIRROR", "1");
            }
            Self::Inherit => {}
        }
    }
}

/// Per-call synthesis controls.
#[derive(Debug, Clone)]
pub struct InferOpts {
    /// Duration scaling: `> 1.0` slows speech, `< 1.0` speeds it up (`1/speed`).
    pub length_scale: f32,
    /// Prior sampling temperature (`z_p = m_p + N(0,1)·exp(logs_p)·noise_scale`).
    pub noise_scale: f32,
    /// RNG seed for the latent sampling (reproducible synthesis).
    pub seed: u64,
    /// Kernel-variant / precision policy applied to the compiled graphs. Drives
    /// the `../rlx` backend kernel selection (see [`KernelVariant`]).
    pub kernel: KernelVariant,
}

impl InferOpts {
    pub fn from_config(cfg: &BundleConfig) -> Self {
        Self {
            length_scale: cfg.length_scale,
            noise_scale: cfg.noise_scale,
            seed: 1234,
            kernel: KernelVariant::from_env(),
        }
    }
}

/// Bump when the import/compile pipeline changes in a way that invalidates the
/// on-disk AOT cache for these graphs.
const CACHE_TAG: &str = "tiny_tts_v3_ct";

/// Whether to lower ONNX `ConvTranspose` as zero-insert + forward `Conv`.
///
/// Metal keeps the inflate+Conv form (proven F5 e2e). ANE has no native CT.
/// MLX / CPU / wgpu / CUDA emit [`rlx_ir::Op::ConvTranspose2d`] — MLX host-evals
/// that op (native MLX CT is wrong for Vocos ISTFT; the inflate+Conv form
/// forced a ~627 GB im2col). Zero-insert Constants were also replaced with
/// Expand(scalar) when decompose is still used.
fn should_decompose_conv_transpose(device: Device) -> bool {
    matches!(device, Device::Ane | Device::Metal)
}

pub struct TinyModel {
    onnx_dir: PathBuf,
    cfg: BundleConfig,
    /// Compiled graphs keyed by `(component, device, length)`.
    cache: Mutex<HashMap<(&'static str, Device, usize), CompiledGraph>>,
}

/// Tiny FNV-1a hash for cache-key disambiguation (tap set → stable short tag).
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn aot_root() -> PathBuf {
    if let Ok(p) = std::env::var("TINY_TTS_AOT_CACHE") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("rlx/tiny_tts_aot")
}

/// Short tag that changes whenever the running binary is relinked — e.g. after
/// any `../rlx` rebuild (importer / backend / memory-planner changes). Keyed on
/// the executable's mtime: a rebuild relinks the consumer and bumps mtime, so a
/// stale on-disk AOT graph compiled by an *older* rlx is never silently reused;
/// an unchanged binary keeps the disk cache warm. Prevents the class of bug
/// where an importer fix was masked by a cached pre-fix graph (which previously
/// required a manual `rm -rf ~/Library/Caches/rlx/tiny_tts_aot` after every rlx
/// change). Override / pin for reproducibility via `TINY_TTS_BUILD_TAG`.
fn build_tag() -> String {
    if let Ok(t) = std::env::var("TINY_TTS_BUILD_TAG") {
        return t;
    }
    let mtime = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{:08x}", fnv1a(&mtime.to_string()) & 0xffff_ffff)
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
        // TinyTTS is dominated by its HiFi-GAN decoder convolutions. On CPU the
        // rlx-cpu default is a scalar reference conv (~20× slower here); the
        // im2col+Accelerate path (`RLX_FAST_CONV`) takes the CPU decoder from
        // ~17 s to ~0.7 s (0.1× → ~2× RT). Default it on via the thread-safe
        // rlx-ir code override (no global `set_var`), only when the caller hasn't
        // pinned it in the process env — so `RLX_FAST_CONV=0` still forces the
        // bit-exact reference. Set at construction, before any conv runs (the
        // rlx-cpu flag is read + cached on first use).
        if std::env::var_os("RLX_FAST_CONV").is_none() {
            rlx_ir::env::set("RLX_FAST_CONV", "1");
        }
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

    /// Run one cached subgraph **in place** (no per-call clone): compile + cache
    /// on first use, then reuse the same `CompiledGraph`. This is the hot path —
    /// `graph()` hands out a `.clone()`, but on Metal `CompiledGraph::clone`
    /// re-compiles a fresh arena (~0.55 s/utterance for flow+decoder); reusing
    /// the instance eliminates that. MLX/CPU/wgpu clone cheaply but still benefit.
    /// The run holds the cache lock, so concurrent `synthesize` calls on one
    /// model serialize (each `run_typed` returns owned output bytes).
    pub fn run_graph(
        &self,
        component: &'static str,
        device: Device,
        length: usize,
        inputs: &[(&str, &[u8], DType)],
    ) -> Result<Vec<(Vec<u8>, DType)>> {
        let key = (component, device, length);
        // Compile outside the run lock so a cold compile doesn't block lookups.
        if !self.cache.lock().expect("graph cache").contains_key(&key) {
            let compiled = self.compile(component, device, length)?;
            self.cache
                .lock()
                .expect("graph cache")
                .insert(key, compiled);
        }
        let mut cache = self.cache.lock().expect("graph cache");
        let g = cache.get_mut(&key).expect("graph just cached");
        Ok(g.run_typed(inputs))
    }

    /// Compile one graph binding distinct ONNX `dim_param` names to concrete
    /// lengths (`named`). For cross-attention CFM/DiT decoders whose `text_length`
    /// and `latent_length` differ — a single `length` would collapse them. The
    /// returned graph is *not* cached in the in-memory tuple cache (its key can't
    /// express the named pair); the AOT disk cache still keys on the named tag, so
    /// repeated calls with the same names skip recompilation. Callers reuse the
    /// returned `CompiledGraph` across an ODE loop.
    pub fn compile_named(
        &self,
        component: &str,
        device: Device,
        length: usize,
        named: &[(&str, usize)],
    ) -> Result<CompiledGraph> {
        self.compile_named_with_options(component, device, length, named, CompileOptions::default())
    }

    /// Compile one graph with caller-selected runtime compile options.
    pub fn compile_named_with_options(
        &self,
        component: &str,
        device: Device,
        length: usize,
        named: &[(&str, usize)],
        mut copts: CompileOptions,
    ) -> Result<CompiledGraph> {
        if matches!(device, Device::Ane) {
            crate::coreml::ensure_coreml_units_for_tts();
        }
        let path = self.onnx_dir.join(format!("{component}.onnx"));
        anyhow::ensure!(path.is_file(), "missing graph {}", path.display());
        let _t_imp = std::time::Instant::now();
        let (hir, mut params, report) = import_graph_named(
            &path,
            component,
            length,
            should_decompose_conv_transpose(device),
            named,
        )?;
        if std::env::var_os("RLX_PHASE_TIMING").is_some() {
            eprintln!(
                "[phase] import({component}) = {}ms",
                _t_imp.elapsed().as_millis()
            );
        }
        if report.stubbed > 0 || !report.unsupported.is_empty() {
            eprintln!(
                "[tiny-tts] warn: {component} import stubbed={} unsupported={:?}",
                report.stubbed, report.unsupported
            );
        }
        let named_tag = named
            .iter()
            .map(|(k, v)| format!("{k}{v}"))
            .collect::<Vec<_>>()
            .join("_");
        // Fold RLX_ONNX_TAP into the key: taps append extra graph outputs, so a
        // cached graph with a different (or empty) tap set must not be reused.
        let tap_tag = std::env::var("RLX_ONNX_TAP")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| format!("_tap{:016x}", fnv1a(&s)))
            .unwrap_or_default();
        // RLX_NO_OPT disables DCE/constant-folding/fusion so a tapped graph
        // (extra outputs) compiles the SAME node set as the untapped graph —
        // makes RLX_ONNX_TAP intermediates a faithful mirror of the real run for
        // debugging (otherwise DCE/fusion differ per output set).
        let no_opt = std::env::var("RLX_NO_OPT").is_ok();
        if no_opt {
            copts.dce = false;
            copts.constant_folding = false;
            copts.fusion_target = None;
            copts.fusion_opts.skip_fusion = true;
        }
        // Bake uniform-fill ONNX initializers (affine-free LN γ=1/β=0, scalar 1
        // for adaLN `(1+scale)`) into Constants before fusion so FuseAdaLayerNorm
        // can match F5/FLUX DiT graphs. Real weights stay Params → set_param.
        let uniform = uniform_fill_bindings(&params);
        let uni_n = uniform.len();
        if uni_n > 0 {
            copts = copts.param_bindings(uniform);
        }
        // Cache key must reflect opt flags — otherwise AOT reuse silently
        // ignores RLX_BISECT_OPTS / skip_fusion / dce toggles.
        let opt_tag = format!(
            "_dce{}_fold{}_fuse{}_uni{uni_n}",
            copts.dce as u8,
            copts.constant_folding as u8,
            (!copts.fusion_opts.skip_fusion) as u8
        );
        let mps_tag = if copts.disable_mpsgraph { "_nomps" } else { "" };
        let cml_tag = if matches!(device, Device::Ane) {
            crate::coreml::coreml_units_cache_tag()
        } else {
            String::new()
        };
        let cache_key = format!(
            "{CACHE_TAG}_b{}_{component}_{device:?}_s{length}_{named_tag}{tap_tag}{opt_tag}{mps_tag}{cml_tag}{}",
            build_tag(),
            if no_opt { "_noopt" } else { "" }
        );
        let cache = AotCache::new(aot_root());
        let _t_cmp = std::time::Instant::now();
        let mut compiled = cache
            .compile_hir_cached(&cache_key, device, hir, &copts)
            .map_err(|e| anyhow::anyhow!("compile {component}: {e}"))?;
        if std::env::var_os("RLX_PHASE_TIMING").is_some() {
            eprintln!(
                "[phase] rlx_compile({component}) = {}ms",
                _t_cmp.elapsed().as_millis()
            );
        }
        // DRAIN (not borrow) so each weight Vec is freed the moment it is copied
        // into the compiled graph — peak memory is one arena copy, not the import
        // params map AND the arena copy held together. For a 664 MB f16 DiT (→~1.3 GB
        // f32) that halves peak (~2.6 → ~1.3 GB), the difference between OOM and not
        // on a RAM-pressured host (F5-TTS transformer).
        for (name, data) in params.drain() {
            compiled.set_param(&name, &data);
        }
        compiled.finalize_params();
        Ok(compiled)
    }

    fn compile(&self, component: &str, device: Device, length: usize) -> Result<CompiledGraph> {
        if matches!(device, Device::Ane) {
            crate::coreml::ensure_coreml_units_for_tts();
        }
        let path = self.onnx_dir.join(format!("{component}.onnx"));
        anyhow::ensure!(path.is_file(), "missing graph {}", path.display());
        // Metal/ANE: decompose CT. MLX/CPU/wgpu/CUDA: native Op::ConvTranspose2d
        // (MLX host-evals CT — see rlx-mlx; avoids ~627 GB im2col on Vocos ISTFT).
        let decompose_ct = should_decompose_conv_transpose(device);
        let (hir, mut params, report) = import_graph(&path, component, length, decompose_ct)?;
        if report.stubbed > 0 || !report.unsupported.is_empty() {
            eprintln!(
                "[tiny-tts] warn: {component} import stubbed={} unsupported={:?}",
                report.stubbed, report.unsupported
            );
        }

        let cml_tag = if matches!(device, Device::Ane) {
            crate::coreml::coreml_units_cache_tag()
        } else {
            String::new()
        };
        let cache_key = format!(
            "{CACHE_TAG}_b{}_{component}_{device:?}_s{length}{cml_tag}",
            build_tag()
        );
        let cache = AotCache::new(aot_root());
        let uniform = uniform_fill_bindings(&params);
        let mut copts = CompileOptions::default();
        if !uniform.is_empty() {
            copts = copts.param_bindings(uniform);
        }
        let mut compiled = cache
            .compile_hir_cached(&cache_key, device, hir, &copts)
            .map_err(|e| anyhow::anyhow!("compile {component}: {e}"))?;
        // Drain so each weight is freed as it's copied into the arena (see compile_named).
        for (name, data) in params.drain() {
            compiled.set_param(&name, &data);
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
        // Install the requested kernel-variant / precision policy before any
        // graph compiles or runs (backends read these knobs at dispatch).
        opts.kernel.apply();
        let c = self.cfg.inter_channels; // latent channels (== g/m_p channel width)

        // ── 1. Text encoder ──────────────────────────────────────────────
        let stage_dbg = std::env::var("RLX_TTS_STAGE").is_ok();
        let time_dbg = std::env::var("RLX_TTS_TIME").is_ok();
        macro_rules! tick {
            () => {
                std::time::Instant::now()
            };
        }
        macro_rules! tock {
            ($t:expr, $label:literal) => {
                if time_dbg {
                    eprintln!(
                        "[time] {:<22} {:>8.1} ms",
                        $label,
                        $t.elapsed().as_secs_f64() * 1e3
                    );
                }
            };
        }
        if stage_dbg {
            eprintln!("[stage] text_encoder t={t}");
        }
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
        let _t = tick!();
        let enc_out = self.run_graph(
            "text_encoder",
            device,
            t,
            &[
                ("phone_ids", &phone_b, DType::I64),
                ("phone_lengths", &len_b, DType::I64),
                ("tone_ids", &tone_b, DType::I64),
                ("language_ids", &lang_b, DType::I64),
                ("bert", &bert_b, DType::F32),
                ("ja_bert", &ja_bert_b, DType::F32),
                ("speaker_id", &sid_b, DType::I64),
            ],
        )?;
        tock!(_t, "text_encoder");
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
        if stage_dbg {
            eprintln!("[stage] duration_predictor t={t}");
        }
        let x_mask_b = f32_bytes(&x_mask);
        let _t = tick!();
        let dp_out = self.run_graph(
            "duration_predictor",
            device,
            t,
            &[
                ("x", x_enc, DType::F32),
                ("x_mask", &x_mask_b, DType::F32),
                ("g", &g_bytes, DType::F32),
            ],
        )?;
        tock!(_t, "dur_pred");
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
        if stage_dbg {
            eprintln!("[stage] flow y_len={y_len}");
        }
        let z_p_b = f32_bytes(&z_p);
        let y_mask_b = f32_bytes(&y_mask);
        let _t = tick!();
        let flow_out = self.run_graph(
            "flow",
            device,
            y_len,
            &[
                ("z_p", &z_p_b, DType::F32),
                ("y_mask", &y_mask_b, DType::F32),
                ("g", &g_bytes, DType::F32),
            ],
        )?;
        tock!(_t, "flow");
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
        if stage_dbg {
            eprintln!("[stage] decoder y_len={y_len}");
        }
        let z_b = f32_bytes(&z);
        let _t = tick!();
        let dec_out = self.run_graph(
            "decoder",
            device,
            y_len,
            &[("z", &z_b, DType::F32), ("g", &g_bytes, DType::F32)],
        )?;
        tock!(_t, "decoder");
        anyhow::ensure!(!dec_out.is_empty(), "decoder returned no output");
        let wav = as_f32(&dec_out[0])?; // [1, 1, samples]
        dbg_mag("dec_out", &wav);
        dbg_dump("dec_out", &wav);
        for (i, o) in dec_out.iter().enumerate() {
            if let Ok(v) = as_f32(o) {
                dbg_dump(&format!("dec_out_{i}"), &v);
            }
        }
        // Catch the MSI wgpu/vulkan failure mode: duration collapse → y_len=1 →
        // exactly 512 samples of near-silence after the 512× HiFi-GAN upsample.
        crate::audio::ensure_audible(&wav).with_context(|| {
            format!("device={device:?} y_len={y_len} (decoder upsample is 512× per frame)")
        })?;
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
    import_graph_named(path, component, length, decompose_conv_transpose, &[])
}

/// Like [`import_graph`], but binds specific ONNX `dim_param` names to distinct
/// concrete lengths (`named_lengths`). Lets a single graph carry two+ dynamic
/// lengths — e.g. a CFM cross-attention decoder whose `text_length` and
/// `latent_length` differ. Names not listed fall back to `length`.
pub fn import_graph_named(
    path: &Path,
    component: &str,
    length: usize,
    decompose_conv_transpose: bool,
    named_lengths: &[(&str, usize)],
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
        named_lengths: named_lengths
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(),
        max_waveform_samples: (length * 1024).max(48_000),
        use_quantized_kernels: false,
        strict: false,
        dynamic_sequence: false,
        decompose_conv_transpose,
        ..ImportOptions::default()
    };

    let (manifest, mut nodes, params, mut i64_params, init_shapes) =
        prepare_onnx_file(path).with_context(|| format!("prepare {}", path.display()))?;
    // Bind scalar-INPUT *values* from `named_lengths`. Some graphs (e.g. F5-TTS's
    // `F5_Preprocess`) build tensor SHAPES from a runtime scalar input —
    // `max_duration → Unsqueeze → Concat → ConstantOfShape / Range → noise/rope`.
    // The lowerer folds those chains via `eval_static_shape_vector`, which reads
    // `i64_params` first; without the value the fold defaults to a garbage length
    // and every derived tensor (noise, rope) collapses. When a named_length key is
    // also a scalar (rank-0 or all-1) graph input, seed its value here so the whole
    // shape-arithmetic subgraph resolves to the concrete compile-time length.
    for io in &manifest.inputs {
        if let Some(&v) = opts.named_lengths.get(&io.name) {
            let scalar = io.meta.shape.is_empty()
                || io
                    .meta
                    .shape
                    .iter()
                    .all(|d| d.as_i64() == Some(1) || d.as_u64() == Some(1));
            if scalar {
                i64_params
                    .entry(io.name.clone())
                    .or_insert_with(|| vec![v as i64]);
            }
        }
    }
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
    // Give shape inference the folded i64 constant VALUES (axes/shape tensors) so
    // opset-13+ Unsqueeze/Squeeze/Reduce with input-form axes resolve the new/removed
    // dim position correctly (else e.g. a `[-1]` mask unsqueeze mis-places to axis 1).
    rlx_onnx_import::shape_propagate::set_shape_i64_consts(i64_params.clone());
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

/// Params whose values are a single repeated fill (affine-free LN γ=1 / β=0,
/// scalar `1` for adaLN). Baking these to Constants before fusion unlocks
/// [`FuseAdaLayerNorm`] on ONNX DiT graphs (F5/FLUX).
fn uniform_fill_bindings(params: &HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
    params
        .iter()
        .filter_map(|(name, data)| {
            let first = *data.first()?;
            if data.iter().all(|&x| x == first) {
                Some((name.clone(), data.clone()))
            } else {
                None
            }
        })
        .collect()
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
