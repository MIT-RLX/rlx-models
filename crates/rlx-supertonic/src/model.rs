// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Supertonic-3 runner: four chained subgraphs + Rust glue, mirroring the
//! reference `TextToSpeech` (`py/helper.py`). **Runs natively on RLX** — each
//! subgraph is imported to rlx-ir via `rlx-onnx-import` and compiled per-device
//! through the shared [`TinyModel`] engine (no ONNX Runtime on the default
//! path; `ort` is an opt-in `onnx` feature for parity validation only).
//!
//! 1. `duration_predictor(text_ids, style_dp, text_mask)` → total duration (s)
//! 2. `text_encoder(text_ids, style_ttl, text_mask)` → text embedding `[1,256,T]`
//! 3. sample `noisy_latent ~ N(0,I)` of `[1,144,L]`, `L = ceil(dur·sr / chunk)`
//! 4. flow-matching ODE loop `total_step×`: `xt = vector_estimator(xt, …, step)`
//!    (the estimator integrates internally; the caller just feeds `xt` back)
//! 5. `vocoder(xt)` → waveform, trimmed to `dur·sr` samples.

use std::path::Path;

use anyhow::{Context, Result};
use rlx_runtime::{DType, Device};
use rlx_tiny_tts::BundleConfig;
use rlx_tiny_tts::model::TinyModel;

#[cfg(feature = "onnx")]
use ort::{session::Session, value::Tensor};
#[cfg(feature = "onnx")]
use std::sync::Mutex;

use crate::config::StConfig;
use crate::tokenize::UnicodeIndexer;
use crate::voices::Voice;

/// Default flow-matching denoising steps (reference default).
pub const DEFAULT_TOTAL_STEP: usize = 8;
/// Default speaking-rate multiplier (reference default; >1 = faster).
pub const DEFAULT_SPEED: f32 = 1.05;

/// Peak amplitude below this is treated as silent (failed) output.
pub const MIN_AUDIBLE_PEAK: f32 = 1e-3;

pub fn peak_amplitude(a: &[f32]) -> f32 {
    a.iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0, f32::max)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i64_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A throwaway [`TinyModel`] config: only its ONNX dir + graph cache are used
/// (we drive the subgraphs via `compile_named`/`run_typed`, never tiny-tts's
/// VITS synthesize glue), so these fields are irrelevant.
fn tiny_config(sample_rate: u32) -> BundleConfig {
    BundleConfig {
        model: String::new(),
        sample_rate,
        add_blank: true,
        language: "EN".into(),
        speakers: Default::default(),
        default_speaker: None,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        length_scale: 1.0,
        inter_channels: 144,
        gin_channels: 256,
    }
}

fn resolve_exec_device(requested: Device) -> Device {
    requested
}

/// Deterministic Gaussian source (xorshift128+ → Box–Muller).
pub struct Rng {
    s0: u64,
    s1: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s0: next() | 1,
            s1: next() | 1,
            spare: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        self.s0 = s0;
        s1 ^= s1 << 23;
        self.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.s1.wrapping_add(s0)
    }
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn randn(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let mut u1 = self.next_unit();
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_unit();
        }
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }
}

/// Per-call synthesis options.
#[derive(Debug, Clone, Copy)]
pub struct InferOpts {
    pub total_step: usize,
    pub speed: f32,
    pub seed: u64,
}

impl Default for InferOpts {
    fn default() -> Self {
        Self {
            total_step: DEFAULT_TOTAL_STEP,
            speed: DEFAULT_SPEED,
            seed: 0,
        }
    }
}

/// A loaded Supertonic-3 model. The four subgraphs run natively on RLX via
/// [`TinyModel`]; `ort` sessions are built only under the `onnx` feature for
/// parity validation (`synthesize_ort`).
pub struct Supertonic {
    device: Device,
    cfg: StConfig,
    indexer: UnicodeIndexer,
    /// Native rlx-ir runner for the four subgraphs (import → compile → run).
    model: TinyModel,
    #[cfg(feature = "onnx")]
    dp: Mutex<Session>,
    #[cfg(feature = "onnx")]
    text_enc: Mutex<Session>,
    #[cfg(feature = "onnx")]
    vector_est: Mutex<Session>,
    #[cfg(feature = "onnx")]
    vocoder: Mutex<Session>,
}

impl Supertonic {
    /// Load all four subgraphs from `<dir>/onnx/` on CPU (native RLX).
    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    /// Load on a specific device. The native path compiles the subgraphs on
    /// `device` through `TinyModel`; under the `onnx` feature, ONNX Runtime
    /// sessions are also built (CPU EP) for validation.
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let device = rlx_tiny_tts::resolve_tts_device(resolve_exec_device(device));
        let onnx_dir = if dir.join("onnx").is_dir() {
            dir.join("onnx")
        } else {
            dir.to_path_buf()
        };
        let cfg = StConfig::load(&onnx_dir)?;
        let indexer = UnicodeIndexer::load(&onnx_dir)?;
        let model = TinyModel::new(onnx_dir.clone(), tiny_config(cfg.sample_rate));

        #[cfg(not(feature = "onnx"))]
        {
            eprintln!("[supertonic] loaded 4 subgraphs on rlx-native/{device:?}");
            Ok(Self {
                device,
                cfg,
                indexer,
                model,
            })
        }
        #[cfg(feature = "onnx")]
        {
            let load = |name: &str| -> Result<(Session, String)> {
                let path = onnx_dir.join(name);
                // Dev/validation ORT parity path: CPU-only session. (The shared
                // rlx-kittentts execution-provider selector was removed along with
                // that crate's ONNX Runtime support; supertonic already carries its
                // own optional `ort` dependency for this validation-only path.)
                let session = Session::builder()
                    .context("ORT session builder")?
                    .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                    .context("ORT optimization level")?
                    .commit_from_file(&path)
                    .with_context(|| format!("build session {name}"))?;
                Ok((session, "cpu".to_string()))
            };
            let (dp, ep) = load("duration_predictor.onnx")?;
            let (text_enc, _) = load("text_encoder.onnx")?;
            let (vector_est, _) = load("vector_estimator.onnx")?;
            let (vocoder, _) = load("vocoder.onnx")?;
            eprintln!(
                "[supertonic] loaded 4 subgraphs on rlx-native/{device:?} (+ort {ep} for validation)"
            );
            Ok(Self {
                device,
                cfg,
                indexer,
                model,
                dp: Mutex::new(dp),
                text_enc: Mutex::new(text_enc),
                vector_est: Mutex::new(vector_est),
                vocoder: Mutex::new(vocoder),
            })
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }
    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }
    /// Backend label for CLI reporting (native rlx execution provider).
    pub fn ort_ep(&self) -> String {
        format!("rlx-native/{:?}", self.device)
    }

    /// Run one native subgraph (compile via the AOT-cached `TinyModel`, then
    /// execute), returning the first output as `f32`. `named` binds the graph's
    /// distinct symbolic length dims (e.g. `text_length`, `latent_length`).
    fn run1(
        &self,
        comp: &'static str,
        length: usize,
        named: &[(&str, usize)],
        inputs: &[(&str, &[u8], DType)],
    ) -> Result<Vec<f32>> {
        let out = self
            .model
            .run_named(comp, self.device, length, named, inputs)
            .map_err(|e| anyhow::anyhow!("run {comp}: {e:#}"))?;
        let (bytes, _dt) = out
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("{comp}: no output"))?;
        Ok(as_f32(&bytes))
    }

    /// Synthesize `text` (single utterance, no sentence chunking) with `voice`,
    /// natively on RLX.
    pub fn synthesize(
        &self,
        text: &str,
        lang: &str,
        voice: &Voice,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        let ids = self.indexer.encode(text, lang)?;
        anyhow::ensure!(!ids.is_empty(), "text tokenized to empty sequence");
        let t = ids.len();
        let text_mask = vec![1.0f32; t];
        let ids_b = i64_bytes(&ids);
        let tm_b = f32_bytes(&text_mask);

        // 1. duration predictor → total seconds (text_ids, style_dp, text_mask).
        let dp_out = self.run1(
            "duration_predictor",
            t,
            &[("text_length", t)],
            &[
                ("text_ids", &ids_b, DType::I64),
                ("style_dp", &f32_bytes(&voice.dp.data), DType::F32),
                ("text_mask", &tm_b, DType::F32),
            ],
        )?;
        anyhow::ensure!(!dp_out.is_empty(), "duration predictor returned no value");
        let duration = (dp_out[0] / opts.speed).max(0.05);

        // 2. text encoder → [1, 256, T] (text_ids, style_ttl, text_mask).
        let style_ttl_b = f32_bytes(&voice.ttl.data);
        let text_emb = self.run1(
            "text_encoder",
            t,
            &[("text_length", t)],
            &[
                ("text_ids", &ids_b, DType::I64),
                ("style_ttl", &style_ttl_b, DType::F32),
                ("text_mask", &tm_b, DType::F32),
            ],
        )?;

        // 3. sample noisy latent [1, 144, L] and its (all-ones) mask.
        let l = self.cfg.latent_len(duration);
        let ch = self.cfg.latent_channels();
        let mut rng = Rng::new(opts.seed);
        let mut xt: Vec<f32> = (0..ch * l).map(|_| rng.randn()).collect();
        let latent_mask = vec![1.0f32; l];
        let total = opts.total_step.max(1);

        // Parity dump (dev): write the exact subgraph inputs (incl. the sampled
        // noise) so a Python onnxruntime run can be compared bit-for-bit.
        if let Some(dir) = std::env::var_os("RLX_ST_PARITY_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            dump_f32(&dir.join("style_dp.f32"), &voice.dp.data);
            dump_f32(&dir.join("style_ttl.f32"), &voice.ttl.data);
            dump_f32(&dir.join("noise.f32"), &xt);
            dump_i64(&dir.join("ids.i64"), &ids);
            let meta = format!(
                "{{\"t\":{t},\"l\":{l},\"ch\":{ch},\"duration\":{duration},\"total\":{total},\"dp_rows\":{},\"dp_cols\":{},\"ttl_rows\":{},\"ttl_cols\":{}}}",
                voice.dp.rows, voice.dp.cols, voice.ttl.rows, voice.ttl.cols
            );
            let _ = std::fs::write(dir.join("meta.json"), meta);
        }

        // The duration predictor and text encoder are done — hand their idle
        // activation arenas back to the OS so they don't sit resident (alongside
        // the flow decoder's + vocoder's) for the rest of the utterance. Their
        // outputs were already copied out by `run1`.
        self.model.release_named_scratch(
            "duration_predictor",
            self.device,
            t,
            &[("text_length", t)],
        );
        self.model
            .release_named_scratch("text_encoder", self.device, t, &[("text_length", t)]);

        // 4. flow-matching ODE loop. The estimator integrates internally (it maps
        //    xt→x_{t+1}); the caller just feeds xt back. `run_named` builds the
        //    graph once (first step / first utterance) and reuses it in place for
        //    every subsequent step and utterance — no per-step compile or clone.
        let ve_named = [("text_length", t), ("latent_length", l)];
        let text_emb_b = f32_bytes(&text_emb);
        let lm_b = f32_bytes(&latent_mask);
        let ts_b = f32_bytes(&[total as f32]);
        for step in 0..total {
            let nl_b = f32_bytes(&xt);
            let cs_b = f32_bytes(&[step as f32]);
            let out = self
                .model
                .run_named(
                    "vector_estimator",
                    self.device,
                    l,
                    &ve_named,
                    &[
                        ("noisy_latent", &nl_b, DType::F32),
                        ("text_emb", &text_emb_b, DType::F32),
                        ("style_ttl", &style_ttl_b, DType::F32),
                        ("latent_mask", &lm_b, DType::F32),
                        ("text_mask", &tm_b, DType::F32),
                        ("current_step", &cs_b, DType::F32),
                        ("total_step", &ts_b, DType::F32),
                    ],
                )
                .map_err(|e| anyhow::anyhow!("run vector_estimator: {e:#}"))?;
            let (bytes, _dt) = out
                .into_iter()
                .next()
                .context("vector_estimator: no output")?;
            xt = as_f32(&bytes);
        }

        // The flow decoder is the single largest activation arena and is now done
        // (its result is in `xt`) — release it before the vocoder runs so the two
        // ~GB working sets never sit resident together.
        self.model
            .release_named_scratch("vector_estimator", self.device, l, &ve_named);

        // 5. vocoder → waveform, trim to dur·sr.
        let wav = self.run1(
            "vocoder",
            l,
            &[("latent_length", l)],
            &[("latent", &f32_bytes(&xt), DType::F32)],
        )?;
        // Vocoder done too — free its scratch so idle RSS between utterances stays
        // near the weight footprint, not the peak activation footprint.
        self.model
            .release_named_scratch("vocoder", self.device, l, &[("latent_length", l)]);
        let n = ((duration * self.cfg.sample_rate as f32) as usize).min(wav.len());
        let audio = wav[..n.max(1)].to_vec();

        if let Some(dir) = std::env::var_os("RLX_ST_PARITY_DUMP") {
            dump_f32(&std::path::PathBuf::from(dir).join("audio_rlx.f32"), &audio);
        }

        let peak = peak_amplitude(&audio);
        anyhow::ensure!(
            peak >= MIN_AUDIBLE_PEAK,
            "synthesized audio is silent (peak={peak:.2e})"
        );
        Ok(audio)
    }

    /// Validation-only reference path through ONNX Runtime (CPU EP). Mirrors
    /// `synthesize` op-for-op; used by parity tests to check the native path.
    #[cfg(feature = "onnx")]
    pub fn synthesize_ort(
        &self,
        text: &str,
        lang: &str,
        voice: &Voice,
        opts: &InferOpts,
    ) -> Result<Vec<f32>> {
        let ids = self.indexer.encode(text, lang)?;
        anyhow::ensure!(!ids.is_empty(), "text tokenized to empty sequence");
        let t = ids.len();
        let text_mask = vec![1.0f32; t];

        let (_ds, dp_out) = {
            let a = i64_t(&[1, t], ids.clone())?;
            let b = f32_t(&[1, voice.dp.rows, voice.dp.cols], voice.dp.data.clone())?;
            let c = f32_t(&[1, 1, t], text_mask.clone())?;
            let mut s = self.dp.lock().expect("ort poisoned");
            extract0(&s.run(ort::inputs![a, b, c]).context("duration_predictor")?)?
        };
        let duration = (dp_out[0] / opts.speed).max(0.05);

        let (emb_shape, text_emb) = {
            let a = i64_t(&[1, t], ids.clone())?;
            let b = f32_t(&[1, voice.ttl.rows, voice.ttl.cols], voice.ttl.data.clone())?;
            let c = f32_t(&[1, 1, t], text_mask.clone())?;
            let mut s = self.text_enc.lock().expect("ort poisoned");
            extract0(&s.run(ort::inputs![a, b, c]).context("text_encoder")?)?
        };
        let emb_shape = shape_usize(&emb_shape);

        let l = self.cfg.latent_len(duration);
        let ch = self.cfg.latent_channels();
        let mut rng = Rng::new(opts.seed);
        let mut xt: Vec<f32> = (0..ch * l).map(|_| rng.randn()).collect();
        let latent_mask = vec![1.0f32; l];
        let total = opts.total_step.max(1);
        for step in 0..total {
            let nl = f32_t(&[1, ch, l], xt)?;
            let te = f32_t(&emb_shape, text_emb.clone())?;
            let st = f32_t(&[1, voice.ttl.rows, voice.ttl.cols], voice.ttl.data.clone())?;
            let lm = f32_t(&[1, 1, l], latent_mask.clone())?;
            let tm = f32_t(&[1, 1, t], text_mask.clone())?;
            let cs = f32_t(&[1], vec![step as f32])?;
            let ts = f32_t(&[1], vec![total as f32])?;
            let mut s = self.vector_est.lock().expect("ort poisoned");
            let out = s
                .run(ort::inputs![nl, te, st, lm, tm, cs, ts])
                .context("vector_estimator")?;
            xt = extract0(&out)?.1;
        }

        let wav = {
            let a = f32_t(&[1, ch, l], xt)?;
            let mut s = self.vocoder.lock().expect("ort poisoned");
            extract0(&s.run(ort::inputs![a]).context("vocoder")?)?.1
        };
        let n = ((duration * self.cfg.sample_rate as f32) as usize).min(wav.len());
        Ok(wav[..n.max(1)].to_vec())
    }

    /// Validation-only: run each subgraph via BOTH native RLX and ONNX Runtime with
    /// identical (ort-reference) inputs, and print the per-subgraph cosine. Isolates
    /// which subgraph's native execution diverges from ort.
    #[cfg(feature = "onnx")]
    pub fn debug_subgraph_parity(
        &self,
        text: &str,
        lang: &str,
        voice: &Voice,
        opts: &InferOpts,
    ) -> Result<()> {
        fn cos(a: &[f32], b: &[f32]) -> f32 {
            let n = a.len().min(b.len());
            let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
            let na: f32 = a[..n].iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb: f32 = b[..n].iter().map(|v| v * v).sum::<f32>().sqrt();
            dot / (na * nb + 1e-12)
        }
        let ids = self.indexer.encode(text, lang)?;
        let t = ids.len();
        let text_mask = vec![1.0f32; t];
        let ids_b = i64_bytes(&ids);
        let tm_b = f32_bytes(&text_mask);
        let style_ttl_b = f32_bytes(&voice.ttl.data);

        // duration_predictor
        let dp_n = self.run1(
            "duration_predictor",
            t,
            &[("text_length", t)],
            &[
                ("text_ids", &ids_b, DType::I64),
                ("style_dp", &f32_bytes(&voice.dp.data), DType::F32),
                ("text_mask", &tm_b, DType::F32),
            ],
        )?;
        let dp_o = {
            let a = i64_t(&[1, t], ids.clone())?;
            let b = f32_t(&[1, voice.dp.rows, voice.dp.cols], voice.dp.data.clone())?;
            let c = f32_t(&[1, 1, t], text_mask.clone())?;
            let mut s = self.dp.lock().expect("ort");
            extract0(&s.run(ort::inputs![a, b, c])?)?.1
        };
        eprintln!(
            "[parity] duration_predictor cos={:.6} native={:?} ort={:?}",
            cos(&dp_n, &dp_o),
            dp_n.first(),
            dp_o.first()
        );

        // text_encoder
        let te_n = self.run1(
            "text_encoder",
            t,
            &[("text_length", t)],
            &[
                ("text_ids", &ids_b, DType::I64),
                ("style_ttl", &style_ttl_b, DType::F32),
                ("text_mask", &tm_b, DType::F32),
            ],
        )?;
        let (emb_shape_o, te_o) = {
            let a = i64_t(&[1, t], ids.clone())?;
            let b = f32_t(&[1, voice.ttl.rows, voice.ttl.cols], voice.ttl.data.clone())?;
            let c = f32_t(&[1, 1, t], text_mask.clone())?;
            let mut s = self.text_enc.lock().expect("ort");
            extract0(&s.run(ort::inputs![a, b, c])?)?
        };
        let emb_shape_o = shape_usize(&emb_shape_o);
        eprintln!(
            "[parity] text_encoder   cos={:.6} (len {} vs {}) shape={:?}",
            cos(&te_n, &te_o),
            te_n.len(),
            te_o.len(),
            emb_shape_o
        );
        if let Some(d) = std::env::var_os("RLX_ST_TE_DUMP") {
            let d = std::path::PathBuf::from(d);
            let _ = std::fs::create_dir_all(&d);
            dump_f32(&d.join("te_native.f32"), &te_n);
            dump_f32(&d.join("te_ort.f32"), &te_o);
            let _ = std::fs::write(d.join("te_shape.json"), format!("{:?}", emb_shape_o));
            // Dump the exact text_encoder inputs so an external ORT run can
            // reproduce the per-node reference on identical data.
            dump_i64(&d.join("in_ids.i64"), &ids);
            dump_f32(&d.join("in_style_ttl.f32"), &voice.ttl.data);
            dump_f32(&d.join("in_text_mask.f32"), &text_mask);
            let _ = std::fs::write(
                d.join("in_meta.json"),
                format!(
                    "{{\"t\":{t},\"ttl_rows\":{},\"ttl_cols\":{}}}",
                    voice.ttl.rows, voice.ttl.cols
                ),
            );
        }

        // vector_estimator (1 step, fed the SAME noise + ort text_emb to isolate)
        let duration = (dp_o[0] / opts.speed).max(0.05);
        let l = self.cfg.latent_len(duration);
        let ch = self.cfg.latent_channels();
        let mut rng = Rng::new(opts.seed);
        let xt: Vec<f32> = (0..ch * l).map(|_| rng.randn()).collect();
        let latent_mask = vec![1.0f32; l];
        let total = opts.total_step.max(1);
        let mut ve = self
            .model
            .compile_named(
                "vector_estimator",
                self.device,
                l,
                &[("text_length", t), ("latent_length", l)],
            )
            .map_err(|e| anyhow::anyhow!("{e:#}"))?;
        let ve_n = {
            let out = ve.run_typed(&[
                ("noisy_latent", &f32_bytes(&xt), DType::F32),
                ("text_emb", &f32_bytes(&te_o), DType::F32),
                ("style_ttl", &style_ttl_b, DType::F32),
                ("latent_mask", &f32_bytes(&latent_mask), DType::F32),
                ("text_mask", &tm_b, DType::F32),
                ("current_step", &f32_bytes(&[0.0]), DType::F32),
                ("total_step", &f32_bytes(&[total as f32]), DType::F32),
            ]);
            as_f32(&out.into_iter().next().context("ve no output")?.0)
        };
        let ve_o = {
            let nl = f32_t(&[1, ch, l], xt.clone())?;
            let te = f32_t(&emb_shape_o, te_o.clone())?;
            let st = f32_t(&[1, voice.ttl.rows, voice.ttl.cols], voice.ttl.data.clone())?;
            let lm = f32_t(&[1, 1, l], latent_mask.clone())?;
            let tm = f32_t(&[1, 1, t], text_mask.clone())?;
            let cs = f32_t(&[1], vec![0.0f32])?;
            let ts = f32_t(&[1], vec![total as f32])?;
            let mut s = self.vector_est.lock().expect("ort");
            extract0(&s.run(ort::inputs![nl, te, st, lm, tm, cs, ts])?)?.1
        };
        eprintln!(
            "[parity] vector_estimator cos={:.6} (len {} vs {})",
            cos(&ve_n, &ve_o),
            ve_n.len(),
            ve_o.len()
        );

        // vocoder (fed the SAME ort ve output to isolate)
        let voc_n = self.run1(
            "vocoder",
            l,
            &[("latent_length", l)],
            &[("latent", &f32_bytes(&ve_o), DType::F32)],
        )?;
        let voc_o = {
            let a = f32_t(&[1, ch, l], ve_o.clone())?;
            let mut s = self.vocoder.lock().expect("ort");
            extract0(&s.run(ort::inputs![a])?)?.1
        };
        eprintln!(
            "[parity] vocoder        cos={:.6} (len {} vs {})",
            cos(&voc_n, &voc_o),
            voc_n.len(),
            voc_o.len()
        );
        Ok(())
    }

    /// Write mono 16-bit PCM WAV at the model sample rate.
    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.cfg.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create WAV: {}", path.display()))?;
        for &s in audio {
            let v = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            w.write_sample(v).context("WAV write")?;
        }
        w.finalize().context("WAV finalize")?;
        Ok(())
    }
}

#[cfg(feature = "onnx")]
fn f32_t(shape: &[usize], data: Vec<f32>) -> Result<Tensor<f32>> {
    Tensor::<f32>::from_array((shape.to_vec(), data)).context("build f32 tensor")
}

#[cfg(feature = "onnx")]
fn i64_t(shape: &[usize], data: Vec<i64>) -> Result<Tensor<i64>> {
    Tensor::<i64>::from_array((shape.to_vec(), data)).context("build i64 tensor")
}

#[cfg(feature = "onnx")]
fn shape_usize(shape: &[i64]) -> Vec<usize> {
    shape.iter().map(|&d| d.max(1) as usize).collect()
}

fn dump_f32(path: &Path, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _ = std::fs::write(path, bytes);
}
fn dump_i64(path: &Path, data: &[i64]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _ = std::fs::write(path, bytes);
}

/// Extract the first output of an ORT run as `(shape, f32 data)`.
#[cfg(feature = "onnx")]
fn extract0(outputs: &ort::session::SessionOutputs) -> Result<(Vec<i64>, Vec<f32>)> {
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("extract f32 output")?;
    Ok((shape.to_vec(), data.to_vec()))
}
