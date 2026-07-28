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

//! Native (RLX, NO onnxruntime) ChatterBox. The four ONNX graphs are imported to
//! rlx-ir, compiled for the target backend (cpu/metal/mlx/wgpu/coreml), and run
//! via `run_typed`. The T3 language model is driven in the *re-prefill* style
//! (the GQA import decompose has no KV cache): each AR step re-runs the growing
//! sequence padded to a fixed compile length and reads the last real position —
//! the same playbook that made MOSS-TTS-Nano bit-exact across backends.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use rlx_core::WeightMap;
use rlx_ir::DType;
use rlx_llama32::{Llama32Config, Llama32Flow};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};
use rlx_runtime::{AotCache, CompileOptions, CompiledGraph, Device};
use tokenizers::Tokenizer;

use crate::common::{
    CFG_RATE, ISTFT_HOP, ISTFT_NFFT, N_FLOW_STEPS, N_LAYERS, Rng, SAMPLE_RATE, SPEECH_VOCAB,
    START_SPEECH, START_TEXT, STOP_TEXT, SynthOpts, cosine_t_span, is_eos, istft, polish_onset,
    resample, sample,
};

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i64_le(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn as_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Pad / truncate reference PCM so the speech_encoder sees a stable length.
/// The Whisper-style frontend is compiled per `num_samples`; keep a 2 s floor
/// so short clips still produce usable conditioning.
fn pad_ref_wav(mut pcm: Vec<f32>) -> Vec<f32> {
    const MIN: usize = 48_000;
    if pcm.len() < MIN {
        pcm.resize(MIN, 0.0);
    }
    pcm
}

/// Fetch a named f32 output from a run result.
fn hm_f32(hm: &HashMap<String, (Vec<u8>, DType)>, name: &str) -> Result<Vec<f32>> {
    Ok(as_f32(
        &hm.get(name).with_context(|| format!("output {name}"))?.0,
    ))
}

/// Fetch a named integer output (GPU backends may materialize it as f32).
fn hm_i64(hm: &HashMap<String, (Vec<u8>, DType)>, name: &str) -> Result<Vec<i64>> {
    let (b, d) = hm.get(name).with_context(|| format!("output {name}"))?;
    Ok(match d {
        DType::I64 => as_i64(b),
        DType::I32 => b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
            .collect(),
        _ => as_f32(b).into_iter().map(|x| x.round() as i64).collect(),
    })
}

/// A compiled graph + its ONNX output names (so `run_typed`'s positional outputs
/// can be looked up by name).
struct Graph {
    g: CompiledGraph,
    outputs: Vec<String>,
}

impl Graph {
    fn run(&mut self, inputs: &[(&str, &[u8], DType)]) -> Vec<(Vec<u8>, DType)> {
        self.g.run_typed(inputs)
    }
}

pub struct NativeChatterBox {
    dir: PathBuf,
    device: Device,
    cache: AotCache,
    /// Fallback CPU cache for graphs routed off the main GPU device (MLX can't
    /// compile the whisper-style `speech_encoder` frontend — moss/kokoro playbook).
    cpu_cache: AotCache,
    tokenizer: Tokenizer,
    /// Compiled + param-loaded graphs, keyed by `component`+lengths. Avoids
    /// re-importing the ONNX and re-copying weights on every call (the AR loop
    /// re-embeds a single token each step).
    graphs: Mutex<HashMap<String, Graph>>,
    /// Drive the T3 LM through the HAND-AUTHORED `rlx-llama32` graph (weights from
    /// `native/t3_lm.safetensors`) instead of the ONNX-imported `language_model`.
    /// Auto-on for Metal (ONNX LM emits zeros there); also via `RLX_CB_NATIVE_LM=1`.
    /// Force ONNX with `RLX_CB_ONNX_LM=1`.
    native_lm: bool,
    /// Use a true KV-cache (prefill once + O(1) decode steps) instead of the
    /// O(N²) re-prefill. Opt-in via `RLX_CB_NATIVE_LM_KV=1` (implies native_lm).
    native_lm_kv: bool,
}

impl NativeChatterBox {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        // The HiFT vocoder + CFM DiT are conv-heavy; opt into the CPU im2col+BLAS
        // fast-conv path (2.4× faster end-to-end, whisper-identical) unless the
        // caller pinned it. `fast_conv_enabled()` caches this in a OnceLock, so it
        // must be set before the first conv runs — here at construction is safe.
        if std::env::var_os("RLX_FAST_CONV").is_none() {
            // SAFETY: called at load time, before any rlx compute thread starts.
            unsafe {
                std::env::set_var("RLX_FAST_CONV", "1");
            }
        }
        // Same BNNS crash as TinyModel TTS graphs — pin CoreML via shared policy.
        let device = if matches!(device, Device::Ane) {
            // ChatterBox does not depend on rlx-tiny-tts; mirror resolve_tts_device.
            if std::env::var_os("RLX_COREML_UNITS").is_none() {
                unsafe {
                    std::env::set_var("RLX_COREML_UNITS", "gpu");
                }
                eprintln!(
                    "[chatterbox] CoreML: set RLX_COREML_UNITS=gpu \
                     (Neural-Engine BNNS crashes on large TTS MIL graphs)"
                );
            }
            device
        } else {
            device
        };
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))?;
        let cache =
            AotCache::new(std::env::temp_dir().join(format!("rlx_chatterbox_aot_{device:?}")));
        let cpu_cache = AotCache::new(std::env::temp_dir().join("rlx_chatterbox_aot_Cpu"));
        // Metal's f32-uniform arena corrupts the ONNX-imported T3 LM (all-zero
        // speech tokens → noise). The hand-authored rlx-llama32 graph is
        // bit-exact on Metal — auto-select it when native weights are present
        // unless the caller forced the ONNX LM (`RLX_CB_ONNX_LM=1`).
        let force_onnx_lm = std::env::var_os("RLX_CB_ONNX_LM").is_some();
        let want_native = std::env::var_os("RLX_CB_NATIVE_LM").is_some()
            || std::env::var_os("RLX_CB_NATIVE_LM_KV").is_some()
            || (matches!(device, Device::Metal | Device::Gpu) && !force_onnx_lm);
        let native_weights_ok = dir.join("native/t3_lm.safetensors").is_file();
        let native_lm = want_native && native_weights_ok;
        if matches!(device, Device::Metal | Device::Gpu) && want_native && !native_weights_ok {
            eprintln!(
                "[chatterbox] {device:?} needs native/t3_lm.safetensors (ONNX LM emits zeros / garbage)"
            );
        }
        if matches!(device, Device::Gpu) {
            eprintln!(
                "[chatterbox] wgpu graphs → Cpu (T3/CFM still diverge on Device::Gpu; session label stays Gpu)"
            );
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            device,
            cache,
            cpu_cache,
            tokenizer,
            graphs: Mutex::new(HashMap::new()),
            native_lm,
            native_lm_kv: std::env::var_os("RLX_CB_NATIVE_LM_KV").is_some(),
        })
    }

    /// Device to compile `component` on.
    ///
    /// - `speech_encoder`: CPU on all GPU devices (Whisper-style frontend —
    ///   STFT / mel ops fail to lower on CUDA/Vulkan the same way as Metal/MLX/wgpu).
    /// - Metal: also keep the S3Gen flow + HiFT stack on CPU — Metal's
    ///   f32-uniform path still corrupts CFM/HiFT (peak explosion, fox 0/6).
    ///   Embed + T3 LM stay on Metal. MLX/CUDA/Vulkan run the decoder on-device.
    /// - wgpu (`Device::Gpu`): full graph stack on CPU for now — native T3 + CFM
    ///   still diverge (fox 0/6, cos≈0) even with Metal's hybrid layout.
    fn component_device(&self, component: &str) -> Device {
        if matches!(self.device, Device::Gpu) {
            return Device::Cpu;
        }
        if component == "speech_encoder"
            && matches!(
                self.device,
                Device::Mlx | Device::Metal | Device::Cuda | Device::Vulkan | Device::Rocm
            )
        {
            return Device::Cpu;
        }
        // Metal: f32-uniform / arena paths still corrupt CFM/HiFT
        // (fox 0/6, cos≈0). Keep S3Gen flow + HiFT on CPU; embed + T3 LM stay
        // on-device. MLX/CUDA/Vulkan run the decoder on-device when available.
        if matches!(self.device, Device::Metal)
            && (component.starts_with("hift_")
                || component == "conditional_decoder"
                || component == "flow_encoder"
                || component == "cfm_estimator")
        {
            return Device::Cpu;
        }
        self.device
    }

    /// Compile `component` at `key` once (outside the cache lock), then run it.
    /// Returns the outputs zipped with their ONNX names for by-name lookup.
    fn run_cached(
        &self,
        key: &str,
        component: &str,
        seq: usize,
        named: &[(&str, usize)],
        max_wav: usize,
        inputs: &[(&str, &[u8], DType)],
    ) -> Result<HashMap<String, (Vec<u8>, DType)>> {
        if !self.graphs.lock().unwrap().contains_key(key) {
            let g = self.compile(component, seq, named, max_wav)?;
            self.graphs.lock().unwrap().insert(key.to_string(), g);
        }
        let mut cache = self.graphs.lock().unwrap();
        let g = cache.get_mut(key).expect("graph just cached");
        let names = g.outputs.clone();
        let outs = g.run(inputs);
        Ok(names.into_iter().zip(outs).collect())
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Drop every cached graph whose key starts with `prefix`, freeing its
    /// compiled arena (weights + intermediates). Opt-in via `RLX_CB_LOW_MEM`:
    /// the default KEEPS graphs so repeated `synthesize` calls skip `compile_lir`
    /// and `set_param` (fast steady state); low-mem trades that for a much lower
    /// PEAK RSS (the LM, estimator and vocoder never coexist).
    fn release(&self, prefix: &str) {
        if std::env::var_os("RLX_CB_LOW_MEM").is_some() {
            self.graphs
                .lock()
                .unwrap()
                .retain(|k, _| !k.starts_with(prefix));
        }
    }

    /// Fast pipeline smoke: import + compile + set_param + run the (small)
    /// `embed_tokens` graph on a tiny input; returns its hidden size (1024).
    pub fn smoke_embed(&self) -> Result<usize> {
        let e = self.embed(&[START_TEXT, 100, 200], 0, 0.5)?;
        Ok(e.len() / 3)
    }

    /// De-risk the re-exported CFM `cfm_estimator.onnx` (the flow-decoder loop
    /// body): import → compile → run one step on the target backend at batch 2,
    /// `T` frames. Returns the `dxdt` element count (should be `2*80*T`).
    pub fn smoke_estimator(&self, tf: usize) -> Result<usize> {
        let b = 2usize; // estimator is exported with a static CFG-doubled batch=2
        let z = vec![0.1f32; b * 80 * tf];
        let mask = vec![1.0f32; b * tf];
        let mu = vec![0.2f32; b * 80 * tf];
        let t = vec![0.3f32; b];
        let spks = vec![0.4f32; b * 80];
        let cond = vec![0.5f32; b * 80 * tf];
        let outs = self.run_cached(
            &format!("est_T{tf}"),
            "cfm_estimator",
            tf,
            &[("T", tf)],
            0,
            &[
                ("x", &f32_le(&z), DType::F32),
                ("mask", &f32_le(&mask), DType::F32),
                ("mu", &f32_le(&mu), DType::F32),
                ("t", &f32_le(&t), DType::F32),
                ("spks", &f32_le(&spks), DType::F32),
                ("cond", &f32_le(&cond), DType::F32),
            ],
        )?;
        Ok(hm_f32(&outs, "dxdt")?.len())
    }

    /// Debug helper: compile + run an arbitrary `component.onnx` (single f32
    /// input) and return each named f32 output — for per-stage rlx-vs-ort
    /// bisection of the vocoder import.
    pub fn debug_run(
        &self,
        component: &str,
        seq: usize,
        named: &[(&str, usize)],
        input_name: &str,
        data: &[f32],
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let outs = self.run_cached(
            &format!("dbg_{component}_{seq}"),
            component,
            seq,
            named,
            0,
            &[(input_name, &f32_le(data), DType::F32)],
        )?;
        Ok(outs
            .into_iter()
            .map(|(k, (b, _))| (k, as_f32(&b)))
            .collect())
    }

    /// Import + compile one of the four ONNX graphs, binding its dynamic length
    /// dims and copying the weights into the compiled arena.
    fn compile(
        &self,
        component: &str,
        seq: usize,
        named: &[(&str, usize)],
        max_wav: usize,
    ) -> Result<Graph> {
        let path = self.dir.join("onnx").join(format!("{component}.onnx"));
        let mut named_lengths: std::collections::HashMap<String, usize> =
            named.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        named_lengths.entry("sequence_length".into()).or_insert(seq);
        named_lengths
            .entry("num_speech_tokens".into())
            .or_insert(seq);
        named_lengths.entry("batch_size".into()).or_insert(1);
        named_lengths
            .entry("past_sequence_length".into())
            .or_insert(0);
        named_lengths
            .entry("total_sequence_length".into())
            .or_insert(seq);
        let opts = ImportOptions {
            sequence_length: seq,
            named_lengths,
            max_waveform_samples: max_wav,
            strict: false,
            // Decompose 1-D ConvTranspose (zero-insert + forward conv) — the native
            // ConvTranspose2d path mis-computes the HiFT vocoder's `ups` (k16/s8/p4)
            // → wrong magnitude/phase → noise. The decompose path is bit-accurate.
            decompose_conv_transpose: true,
            ..Default::default()
        };
        let prof = std::env::var_os("RLX_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let (hir, mut params, _report, manifest) =
            build_hir_from_onnx_file(&path, opts).with_context(|| format!("import {component}"))?;
        let t_import = t0.elapsed();
        let outputs: Vec<String> = manifest.outputs.iter().map(|o| o.name.clone()).collect();
        let device = self.component_device(component);
        let cache = if device == self.device {
            &self.cache
        } else {
            &self.cpu_cache
        };
        let key = format!("cb_{component}_{device:?}_s{seq}");
        let t1 = std::time::Instant::now();
        let mut g = cache
            .compile_hir_cached(&key, device, hir, &CompileOptions::default())
            .map_err(|e| anyhow::anyhow!("compile {component}: {e}"))?;
        let t_compile = t1.elapsed();
        let t2 = std::time::Instant::now();
        let (mut np, mut nbytes) = (0usize, 0usize);
        for (name, data) in params.drain() {
            nbytes += data.len();
            np += 1;
            g.set_param(&name, &data);
        }
        g.finalize_params();
        if prof {
            eprintln!(
                "[compile] {component}: import(read+parse)={t_import:?} compile_lir={t_compile:?} set_param={:?} ({np} params, {} MB)",
                t2.elapsed(),
                nbytes / 1_048_576
            );
        }
        Ok(Graph { g, outputs })
    }

    /// `embed_tokens(ids, position_ids, exaggeration)` → `[n, 1024]` flat.
    fn embed(&self, ids: &[i64], pos0: usize, exag: f32) -> Result<Vec<f32>> {
        let n = ids.len();
        let pos: Vec<i64> = (pos0..pos0 + n).map(|p| p as i64).collect();
        let outs = self.run_cached(
            &format!("embed_s{n}"),
            "embed_tokens",
            n,
            &[],
            48_000,
            &[
                ("input_ids", &i64_le(ids), DType::I64),
                ("position_ids", &i64_le(&pos), DType::I64),
                ("exaggeration", &f32_le(&[exag]), DType::F32),
            ],
        )?;
        // Single output → [1, n, 1024].
        Ok(as_f32(
            &outs.into_values().next().context("embed_tokens output")?.0,
        ))
    }

    /// Build the HAND-AUTHORED native T3 prefill graph (rlx-llama32, `inputs_embeds`
    /// entry, causal, full-sequence logits) from `native/t3_lm.safetensors` — NOT
    /// the ONNX `language_model` graph. Compiled once per `compile_len`, cached.
    fn compile_native_lm(&self, compile_len: usize) -> Result<Graph> {
        let ndir = self.dir.join("native");
        let cfg = Llama32Config::from_file(&ndir.join("t3_config.json"))
            .context("load native/t3_config.json")?;
        let st = ndir.join("t3_lm.safetensors");
        let mut wm = WeightMap::from_file(st.to_str().context("t3_lm path")?)
            .with_context(|| format!("load {}", st.display()))?;
        let t0 = std::time::Instant::now();
        let built = Llama32Flow::new(&cfg)
            .prefill()
            .batch(1)
            .seq(compile_len)
            .inputs_embeds()
            .lm_head()
            .build(&mut wm)
            .context("build native T3 prefill flow")?;
        let (hir, mut params) = built.into_parts()?;
        let lm_dev = self.component_device("language_model");
        let key = format!("cb_native_lm_{lm_dev:?}_s{compile_len}");
        let cache = if lm_dev == self.device {
            &self.cache
        } else {
            &self.cpu_cache
        };
        let mut g = cache
            .compile_hir_cached(&key, lm_dev, hir, &CompileOptions::default())
            .map_err(|e| anyhow::anyhow!("compile native LM: {e}"))?;
        for (name, data) in params.drain() {
            g.set_param(&name, &data);
        }
        g.finalize_params();
        if std::env::var_os("RLX_PROFILE").is_some() {
            eprintln!(
                "[compile] native T3 LM (rlx-llama32) s{compile_len} on {lm_dev:?}: {:?}",
                t0.elapsed()
            );
        }
        Ok(Graph {
            g,
            outputs: vec!["logits".into()],
        })
    }

    /// Native (rlx-llama32) counterpart of [`Self::lm_logits_last`]: pad the growing
    /// `[seq,1024]` embeds to `compile_len`, run the causal prefill, read the logits
    /// at the last real position. No `attention_mask`/`past_kv` — causal masking
    /// makes the read at position `seq-1` independent of the trailing padding.
    fn native_lm_logits_last(
        &self,
        embeds: &[f32],
        seq: usize,
        compile_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden = 1024;
        let mut emb = vec![0f32; compile_len * hidden];
        emb[..seq * hidden].copy_from_slice(&embeds[..seq * hidden]);
        let key = format!("lm_native_s{compile_len}");
        if !self.graphs.lock().unwrap().contains_key(&key) {
            let g = self.compile_native_lm(compile_len)?;
            self.graphs.lock().unwrap().insert(key.clone(), g);
        }
        let mut cache = self.graphs.lock().unwrap();
        let g = cache.get_mut(&key).expect("native lm graph just cached");
        let outs = g.run(&[("inputs_embeds", &f32_le(&emb), DType::F32)]);
        let logits = as_f32(&outs[0].0); // [compile_len, SPEECH_VOCAB]
        let start = (seq - 1) * SPEECH_VOCAB;
        anyhow::ensure!(
            logits.len() >= start + SPEECH_VOCAB,
            "native LM logits len {} < {}",
            logits.len(),
            start + SPEECH_VOCAB
        );
        Ok(logits[start..start + SPEECH_VOCAB].to_vec())
    }

    /// Build a hand-authored `rlx-llama32` graph (prefill or bucketed decode),
    /// compile once, cache under `key`. `f` receives a fresh `Llama32Flow`.
    fn compile_llama_graph(
        &self,
        key: &str,
        build: impl FnOnce(
            &Llama32Config,
        ) -> Result<(rlx_ir::hir::HirModule, HashMap<String, Vec<f32>>)>,
    ) -> Result<Graph> {
        let ndir = self.dir.join("native");
        let cfg = Llama32Config::from_file(&ndir.join("t3_config.json"))
            .context("load native/t3_config.json")?;
        let (hir, mut params) = build(&cfg)?;
        let mut g = self
            .cache
            .compile_hir_cached(key, self.device, hir, &CompileOptions::default())
            .map_err(|e| anyhow::anyhow!("compile {key}: {e}"))?;
        for (name, data) in params.drain() {
            g.set_param(&name, &data);
        }
        g.finalize_params();
        Ok(Graph {
            g,
            outputs: vec!["logits".into()],
        })
    }

    fn native_weights(&self) -> Result<WeightMap> {
        let st = self.dir.join("native").join("t3_lm.safetensors");
        WeightMap::from_file(st.to_str().context("t3_lm path")?)
            .with_context(|| format!("load {}", st.display()))
    }

    /// KV-cache AR loop: prefill the prompt once (export KV), then O(1) bucketed
    /// decode steps (mask out padding, append each new token's K/V) — replaces the
    /// O(N²) re-prefill. `upper` is the static decode bucket (= compile_len).
    /// Returns the generated speech tokens.
    fn ar_loop_kv(
        &self,
        embeds: &[f32],
        prompt_seq: usize,
        upper: usize,
        opts: &SynthOpts,
        rng: &mut Rng,
    ) -> Result<Vec<i64>> {
        // --- prefill: inputs_embeds[0..prompt_seq] → per-layer KV + last logits ---
        let pf_key = format!("lm_native_prefill_s{prompt_seq}");
        if !self.graphs.lock().unwrap().contains_key(&pf_key) {
            let g = self.compile_llama_graph(&pf_key, |cfg| {
                let mut wm = self.native_weights()?;
                let built = Llama32Flow::new(cfg)
                    .prefill()
                    .batch(1)
                    .seq(prompt_seq)
                    .inputs_embeds()
                    .lm_head()
                    .export_kv()
                    .build(&mut wm)?;
                built.into_parts()
            })?;
            self.graphs.lock().unwrap().insert(pf_key.clone(), g);
        }
        let hidden = 1024;
        let stride = hidden * 4; // bytes per KV row
        // Persistent padded KV byte-buffers per layer ([upper] rows). The real
        // prefix grows in place — no per-step realloc / f32↔bytes churn; padding
        // rows are masked out. Seeded from the prefill's exported K/V bytes.
        let mut kv_k: Vec<Vec<u8>> = Vec::with_capacity(N_LAYERS);
        let mut kv_v: Vec<Vec<u8>> = Vec::with_capacity(N_LAYERS);
        let mut logits: Vec<f32> = {
            let mut cache = self.graphs.lock().unwrap();
            let g = cache.get_mut(&pf_key).expect("prefill graph");
            let outs = g.run(&[(
                "inputs_embeds",
                &f32_le(&embeds[..prompt_seq * hidden]),
                DType::F32,
            )]);
            for l in 0..N_LAYERS {
                let mut kb = vec![0u8; upper * stride];
                let mut vb = vec![0u8; upper * stride];
                kb[..prompt_seq * stride]
                    .copy_from_slice(&outs[1 + 2 * l].0[..prompt_seq * stride]);
                vb[..prompt_seq * stride]
                    .copy_from_slice(&outs[2 + 2 * l].0[..prompt_seq * stride]);
                kv_k.push(kb);
                kv_v.push(vb);
            }
            let all = as_f32(&outs[0].0);
            all[(prompt_seq - 1) * SPEECH_VOCAB..prompt_seq * SPEECH_VOCAB].to_vec()
        };
        // Prefill graph is done for this utterance — free its ~2GB arena so it
        // doesn't coexist with the decode graph (LIR stays on disk; a warm
        // re-synthesize just re-`set_param`s). Holds LM peak RSS at ~one graph.
        self.graphs.lock().unwrap().remove(&pf_key);

        // --- decode: one static bucket graph reused every step ---
        let dec_key = format!("lm_native_decode_u{upper}");
        if !self.graphs.lock().unwrap().contains_key(&dec_key) {
            let g = self.compile_llama_graph(&dec_key, |cfg| {
                let mut wm = self.native_weights()?;
                let built = Llama32Flow::new(cfg)
                    .decode()
                    .batch(1)
                    .past(upper)
                    .custom_mask()
                    .inputs_embeds()
                    .export_kv()
                    .lm_head()
                    .build(&mut wm)?;
                built.into_parts()
            })?;
            self.graphs.lock().unwrap().insert(dec_key.clone(), g);
        }

        let k_names: Vec<String> = (0..N_LAYERS).map(|l| format!("past_k_{l}")).collect();
        let v_names: Vec<String> = (0..N_LAYERS).map(|l| format!("past_v_{l}")).collect();
        let mut generated: Vec<i64> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::new();
        for i in 0..opts.max_frames {
            let next = sample(&logits, &seen, opts, rng);
            if is_eos(next) {
                break;
            }
            generated.push(next);
            seen.insert(next);
            let pos = prompt_seq + i; // absolute position of `next`; == current KV length
            let emb_b = f32_le(&self.embed(&[next], pos, opts.exaggeration)?);
            let mask_b = f32_le(&rlx_runtime::attn_mask::bucket_decode_mask(pos, upper));
            let pos_b = f32_le(&[pos as f32]);
            // Feed the persistent KV buffers directly (no copy). `din` borrows
            // kv_k/kv_v immutably; it is dropped before we write the new row back.
            let douts = {
                let mut din: Vec<(&str, &[u8], DType)> = Vec::with_capacity(3 + 2 * N_LAYERS);
                din.push(("inputs_embeds", &emb_b, DType::F32));
                din.push(("mask", &mask_b, DType::F32));
                din.push(("position", &pos_b, DType::F32));
                for l in 0..N_LAYERS {
                    din.push((k_names[l].as_str(), &kv_k[l], DType::F32));
                    din.push((v_names[l].as_str(), &kv_v[l], DType::F32));
                }
                let mut cache = self.graphs.lock().unwrap();
                let g = cache.get_mut(&dec_key).expect("decode graph");
                g.run(&din)
            };
            logits = as_f32(&douts[0].0); // [1,1,VOCAB]
            // The new token's K/V come back at output row `upper` (full concat
            // export); write them into each layer's buffer at real position `pos`.
            let src = upper * stride;
            let dst = pos * stride;
            for l in 0..N_LAYERS {
                kv_k[l][dst..dst + stride].copy_from_slice(&douts[1 + 2 * l].0[src..src + stride]);
                kv_v[l][dst..dst + stride].copy_from_slice(&douts[2 + 2 * l].0[src..src + stride]);
            }
        }
        Ok(generated)
    }

    /// One `language_model` forward over `[1, seq, 1024]` embeds (padded to the
    /// compiled length). Returns the logits at the LAST real position `[SPEECH_VOCAB]`.
    fn lm_logits_last(&self, embeds: &[f32], seq: usize, compile_len: usize) -> Result<Vec<f32>> {
        if self.native_lm {
            self.native_lm_logits_last(embeds, seq, compile_len)
        } else {
            self.onnx_lm_logits_last(embeds, seq, compile_len)
        }
    }

    /// The ONNX-imported `language_model` re-prefill path (the parity oracle).
    fn onnx_lm_logits_last(
        &self,
        embeds: &[f32],
        seq: usize,
        compile_len: usize,
    ) -> Result<Vec<f32>> {
        // Pad embeds [seq,1024] → [compile_len,1024]; attention_mask 1s then 0s.
        let hidden = 1024;
        let mut emb = vec![0f32; compile_len * hidden];
        emb[..seq * hidden].copy_from_slice(&embeds[..seq * hidden]);
        let mut mask = vec![0i64; compile_len];
        for m in mask.iter_mut().take(seq) {
            *m = 1;
        }
        // Empty past-KV for every layer (re-prefill: the GQA decompose ignores them).
        let mut inputs: Vec<(String, Vec<u8>, DType)> = Vec::with_capacity(2 + 2 * N_LAYERS);
        inputs.push(("inputs_embeds".into(), f32_le(&emb), DType::F32));
        inputs.push(("attention_mask".into(), i64_le(&mask), DType::I64));
        let empty: Vec<u8> = Vec::new();
        for i in 0..N_LAYERS {
            inputs.push((
                format!("past_key_values.{i}.key"),
                empty.clone(),
                DType::F32,
            ));
            inputs.push((
                format!("past_key_values.{i}.value"),
                empty.clone(),
                DType::F32,
            ));
        }
        let refs: Vec<(&str, &[u8], DType)> = inputs
            .iter()
            .map(|(n, b, d)| (n.as_str(), b.as_slice(), *d))
            .collect();
        // Compiled once at `compile_len` and reused every AR step.
        let outs = self.run_cached(
            &format!("lm_s{compile_len}"),
            "language_model_fp16",
            compile_len,
            &[],
            48_000,
            &refs,
        )?;
        let logits = hm_f32(&outs, "logits")?; // [compile_len, SPEECH_VOCAB]
        let start = (seq - 1) * SPEECH_VOCAB;
        Ok(logits[start..start + SPEECH_VOCAB].to_vec())
    }

    /// Greedy token-parity of the HAND-AUTHORED native T3 LM vs the ONNX-imported
    /// LM in the real AR loop. Each step computes BOTH logits at the last real
    /// position, compares argmax + cosine, then advances with the ONNX (reference)
    /// token so both see the identical growing prefix. Returns
    /// `(agree, total, first_divergence, mean_cosine)`. Fast dev gate (both paths
    /// are ort-free; the ONNX path is the whisper-validated oracle).
    pub fn token_parity(
        &self,
        text: &str,
        reference: &[f32],
        ref_sr: u32,
        opts: &SynthOpts,
    ) -> Result<(usize, usize, Option<usize>, f64)> {
        let ref24 = pad_ref_wav(resample(reference, ref_sr, SAMPLE_RATE));
        let n_samp = ref24.len();
        let se_out = self.run_cached(
            &format!("enc_s{n_samp}"),
            "speech_encoder",
            100,
            &[],
            n_samp,
            &[("audio_values", &f32_le(&ref24), DType::F32)],
        )?;
        let audio_features = hm_f32(&se_out, "audio_features")?;
        let af_len = audio_features.len() / 1024;
        let text_ids: Vec<i64> = std::iter::once(START_TEXT)
            .chain(
                self.tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
                    .get_ids()
                    .iter()
                    .map(|&i| i as i64),
            )
            .chain(std::iter::once(STOP_TEXT))
            .collect();
        let text_embeds = self.embed(&text_ids, 0, opts.exaggeration)?;
        let start_embed = self.embed(&[START_SPEECH], text_ids.len(), opts.exaggeration)?;
        let mut embeds = audio_features;
        embeds.extend_from_slice(&text_embeds);
        embeds.extend_from_slice(&start_embed);
        let prompt_seq = af_len + text_ids.len() + 1;
        let compile_len = prompt_seq + opts.max_frames;

        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as i64
        };
        let cosine = |a: &[f32], b: &[f32]| {
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                d += x as f64 * y as f64;
                na += (x as f64).powi(2);
                nb += (y as f64).powi(2);
            }
            d / (na.sqrt() * nb.sqrt())
        };

        let (mut agree, mut total, mut first_div, mut cos_sum) = (0usize, 0usize, None, 0f64);
        for (step, seq) in (prompt_seq..).enumerate().take(opts.max_frames) {
            let onnx = self.onnx_lm_logits_last(&embeds, seq, compile_len)?;
            let nat = self.native_lm_logits_last(&embeds, seq, compile_len)?;
            total += 1;
            cos_sum += cosine(&nat, &onnx);
            let (to, tn) = (argmax(&onnx), argmax(&nat));
            if to == tn {
                agree += 1;
            } else if first_div.is_none() {
                first_div = Some(step);
            }
            if is_eos(to) {
                break;
            }
            let emb = self.embed(&[to], seq, opts.exaggeration)?;
            embeds.extend_from_slice(&emb);
        }
        Ok((agree, total, first_div, cos_sum / total.max(1) as f64))
    }

    /// Synthesize `text` in the voice of `reference` (PCM at `ref_sr`). 24 kHz PCM.
    pub fn synthesize(
        &self,
        text: &str,
        reference: &[f32],
        ref_sr: u32,
        opts: &SynthOpts,
    ) -> Result<Vec<f32>> {
        let prof = std::env::var_os("RLX_PROFILE").is_some();
        let mut t = std::time::Instant::now();
        let mut lap = |name: &str| {
            if prof {
                eprintln!("[prof] {name}: {:?}", t.elapsed());
                t = std::time::Instant::now();
            }
        };
        // 1) reference → speech_encoder conditioning.
        let ref24 = pad_ref_wav(resample(reference, ref_sr, SAMPLE_RATE));
        let n_samp = ref24.len();
        let se_out = self.run_cached(
            &format!("enc_s{n_samp}"),
            "speech_encoder",
            100,
            &[],
            n_samp,
            &[("audio_values", &f32_le(&ref24), DType::F32)],
        )?;
        let audio_features = hm_f32(&se_out, "audio_features")?;
        let audio_tokens = hm_i64(&se_out, "audio_tokens")?;
        let speaker_embeddings = hm_f32(&se_out, "speaker_embeddings")?;
        let speaker_features = hm_f32(&se_out, "speaker_features")?;
        let af_len = audio_features.len() / 1024;
        let mel_len = speaker_features.len() / 80;
        lap("speech_encoder");

        // 2) prompt embeds = cat(audio_features, text_embeds, start_speech_embed).
        let text_ids: Vec<i64> = std::iter::once(START_TEXT)
            .chain(
                self.tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
                    .get_ids()
                    .iter()
                    .map(|&i| i as i64),
            )
            .chain(std::iter::once(STOP_TEXT))
            .collect();
        let text_embeds = self.embed(&text_ids, 0, opts.exaggeration)?;
        let start_embed = self.embed(&[START_SPEECH], text_ids.len(), opts.exaggeration)?;
        let mut embeds = audio_features;
        embeds.extend_from_slice(&text_embeds);
        embeds.extend_from_slice(&start_embed);
        let prompt_seq = af_len + text_ids.len() + 1;

        // 3) prefill + AR loop. `native_lm_kv` = true KV-cache (prefill once +
        //    O(1) bucketed decode steps); else re-prefill (pad to a fixed length).
        let compile_len = prompt_seq + opts.max_frames;
        let mut rng = Rng::new(opts.seed);
        let generated: Vec<i64> = if self.native_lm_kv {
            self.ar_loop_kv(&embeds, prompt_seq, compile_len, opts, &mut rng)?
        } else {
            let mut generated: Vec<i64> = Vec::new();
            let mut seen: HashSet<i64> = HashSet::new();
            let mut embeds = embeds;
            for seq in (prompt_seq..).take(opts.max_frames) {
                let logits = self.lm_logits_last(&embeds, seq, compile_len)?;
                let next = sample(&logits, &seen, opts, &mut rng);
                if is_eos(next) {
                    break;
                }
                generated.push(next);
                seen.insert(next);
                let emb = self.embed(&[next], seq, opts.exaggeration)?;
                embeds.extend_from_slice(&emb);
            }
            generated
        };
        anyhow::ensure!(!generated.is_empty(), "no speech tokens generated");
        lap(&format!("LM AR loop ({} tokens)", generated.len()));
        // The LM / embed / speech-encoder graphs are done — free them (their
        // outputs are already owned Vecs) so they don't coexist with the decoder.
        self.release("lm_");
        self.release("embed_");
        self.release("enc_");

        // 4) S3Gen flow decoder (loop-based re-export — replaces the 24k-node
        //    unrolled `conditional_decoder`): flow_encoder → CFM Euler loop over
        //    `cfm_estimator` (run cond+uncond per step for CFG) → drop the prompt
        //    frames → hift head → Rust ISTFT.
        let prompt_len = (mel_len / 2).min(audio_tokens.len());
        let prompt_token = &audio_tokens[..prompt_len];
        let enc = self.run_cached(
            &format!(
                "flowenc_np{}_nt{}_lp{mel_len}",
                prompt_token.len(),
                generated.len()
            ),
            "flow_encoder",
            generated.len().max(1),
            &[
                ("Np", prompt_token.len()),
                ("Nt", generated.len()),
                ("Lp", mel_len),
            ],
            0,
            &[
                ("token", &i64_le(&generated), DType::I64),
                ("prompt_token", &i64_le(prompt_token), DType::I64),
                ("prompt_feat", &f32_le(&speaker_features), DType::F32),
                ("embedding", &f32_le(&speaker_embeddings), DType::F32),
            ],
        )?;
        let mu = hm_f32(&enc, "mu")?; // [1,80,T]
        let mask = hm_f32(&enc, "mask")?; // [1,1,T]
        let spks = hm_f32(&enc, "spks")?; // [1,80]
        let cond = hm_f32(&enc, "cond")?; // [1,80,T]
        let tf = mu.len() / 80;
        let plane = 80 * tf;
        lap("flow_encoder");
        let dbg = std::env::var_os("RLX_DBG").is_some();
        let stat = |n: &str, v: &[f32]| {
            let (mn, mx) = v
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
            let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
            eprintln!(
                "[dbg] {n}: len={} min={mn:.3} max={mx:.3} mean={mean:.4}",
                v.len()
            );
        };
        if dbg {
            eprintln!(
                "[dbg] generated {} tokens: {:?}",
                generated.len(),
                &generated[..generated.len().min(12)]
            );
            eprintln!("[dbg] mel_len={mel_len} prompt_len={prompt_len} tf={tf} af_len={af_len}");
            stat("mu", &mu);
            stat("spks", &spks);
            stat("cond", &cond);
        }

        // CFM Euler solver (`solve_euler`): z = randn; per step run the estimator
        // ONCE on the CFG-DOUBLED batch `[cond; uncond]` (static batch=2 — half the
        // invocations vs two batch-1 runs), then combine with CFG guidance.
        let mut x = vec![0f32; plane];
        for v in x.iter_mut() {
            *v = rng.normal();
        }
        // Static conditioning halves: [mu ; 0], [spks ; 0], [cond ; 0], mask twice.
        let cat0 = |a: &[f32]| {
            let mut v = Vec::with_capacity(2 * a.len());
            v.extend_from_slice(a);
            v.resize(2 * a.len(), 0.0);
            v
        };
        let mu2 = cat0(&mu);
        let spks2 = cat0(&spks);
        let cond2 = cat0(&cond);
        let mut mask2 = Vec::with_capacity(2 * tf);
        mask2.extend_from_slice(&mask);
        mask2.extend_from_slice(&mask);
        let n_steps = std::env::var("RLX_CB_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(N_FLOW_STEPS);
        let t_span = cosine_t_span(n_steps);
        for w in t_span.windows(2) {
            let (a, b) = (w[0], w[1]);
            let mut x2 = Vec::with_capacity(2 * plane);
            x2.extend_from_slice(&x);
            x2.extend_from_slice(&x);
            let outs = self.run_cached(
                &format!("est_T{tf}"),
                "cfm_estimator",
                tf,
                &[("T", tf)],
                0,
                &[
                    ("x", &f32_le(&x2), DType::F32),
                    ("mask", &f32_le(&mask2), DType::F32),
                    ("mu", &f32_le(&mu2), DType::F32),
                    ("t", &f32_le(&[a, a]), DType::F32),
                    ("spks", &f32_le(&spks2), DType::F32),
                    ("cond", &f32_le(&cond2), DType::F32),
                ],
            )?;
            let dxdt = hm_f32(&outs, "dxdt")?; // [2, 80, T] = [cond ; uncond]
            for i in 0..plane {
                let d = (1.0 + CFG_RATE) * dxdt[i] - CFG_RATE * dxdt[plane + i];
                x[i] += (b - a) * d;
            }
        }

        lap(&format!("CFM solver ({n_steps} steps, batch-2 CFG)"));
        self.release("est_"); // estimator done — free before the vocoder
        self.release("flowenc_");
        if dbg {
            stat("x_after_solver", &x);
        }
        // Drop the prompt region (first `mel_len` frames); x is [80, tf] row-major.
        let gen_frames = tf.saturating_sub(mel_len);
        anyhow::ensure!(gen_frames > 0, "flow decoder produced no generated frames");
        let mut mel = Vec::with_capacity(80 * gen_frames);
        for c in 0..80 {
            mel.extend_from_slice(&x[c * tf + mel_len..c * tf + tf]);
        }
        if dbg {
            stat("mel", &mel);
        }
        if let Some(p) = std::env::var_os("RLX_DUMP_MEL") {
            let bytes: Vec<u8> = mel.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&p, &bytes).ok();
            eprintln!("[dbg] dumped mel [80,{gen_frames}] to {:?}", p);
        }

        // HiFT spectral head → (magnitude, phase) → tiny n_fft=16 ISTFT in Rust.
        let hh = self.run_cached(
            &format!("hift_T{gen_frames}"),
            "hift_head",
            gen_frames,
            &[("T", gen_frames)],
            0,
            &[("speech_feat", &f32_le(&mel), DType::F32)],
        )?;
        let mag = hm_f32(&hh, "magnitude")?;
        let phase = hm_f32(&hh, "phase")?;
        let n_bins = ISTFT_NFFT / 2 + 1;
        let tw = mag.len() / n_bins;
        if dbg {
            stat("magnitude", &mag);
            stat("phase", &phase);
        }
        if let Some(p) = std::env::var_os("RLX_DUMP_MAG") {
            let mut b: Vec<u8> = (mag.len() as u32).to_le_bytes().to_vec();
            b.extend(mag.iter().flat_map(|x| x.to_le_bytes()));
            b.extend(phase.iter().flat_map(|x| x.to_le_bytes()));
            std::fs::write(&p, &b).ok();
        }
        let wav = istft(&mag, &phase, tw, ISTFT_NFFT, ISTFT_HOP);
        lap("vocoder (hift_head + istft)");
        let wav = polish_onset(&wav, SAMPLE_RATE);
        if dbg {
            stat("wav", &wav);
        }
        Ok(wav)
    }

    pub fn write_wav(&self, audio: &[f32], path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec)
            .with_context(|| format!("create {}", path.display()))?;
        for &s in audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        Ok(())
    }
}
