// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Measure DFlash acceptance rate against a **real** Qwen3 target.
//!
//! ## What this measures, and what it does not
//!
//! It measures **acceptance**: how many of the drafter's proposed tokens the
//! target would itself have emitted. That is the number that decides whether
//! speculation can pay, and it is a property of the two models, independent of
//! how the verification was scheduled.
//!
//! It does **not** measure speedup. The real win comes from verifying a whole
//! drafted block in one batched target forward; here the target re-decodes the
//! block one token at a time, because that needs no batched-prefill path and
//! keeps the KV bookkeeping obvious. Wall-clock here is therefore meaningless
//! and deliberately not reported.
//!
//! ## Wiring
//!
//! `Qwen3TapTarget` implements [`DflashTarget`] over a single decode graph
//! built with `tap_layers = drafter.target_layers`, so the residual streams the
//! drafter fuses come from the target's own forward — the whole point of an
//! Eagle-style head.
//!
//! Usage:
//! ```text
//! cargo run --release -p rlx-dflash --example dflash_measure -- \
//!     <drafter-dir> <target.gguf> [device] [n_tokens]
//! ```
//! `<drafter-dir>` holds the HF `config.json` + `model.safetensors`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use rlx_core::weight_loader::WeightLoader;
use rlx_dflash::{
    DflashConfig, DflashDrafter, DflashLoop, DflashTarget, DrafterOptions, TargetStep,
    hf_to_dflash_name, rope_tables,
};
use rlx_qwen3::Qwen3Config;
use rlx_qwen3::flow::{Qwen3DecodeOpts, build_qwen3_decode_built};
use rlx_runtime::spec_decode::SparseDist;
use rlx_runtime::{CompiledGraph, Device, Session};

/// Cache capacity. One decode graph, cache padded to this and masked.
const CAP: usize = 512;

fn device_from(name: &str) -> Device {
    match name {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" => Device::Rocm,
        "vulkan" => Device::Vulkan,
        "gpu" | "wgpu" => Device::Gpu,
        _ => Device::Cpu,
    }
}

/// Rename an HF DFlash checkpoint's tensors to the keys the builders load.
///
/// Unmapped names are dropped rather than passed through: a stray optimizer
/// tensor should not shadow a real weight, and a genuinely missing weight is
/// reported by the builder with the key it wanted.
struct RenamedLoader {
    inner: Box<dyn WeightLoader>,
    map: HashMap<String, String>,
}

impl RenamedLoader {
    fn new(inner: Box<dyn WeightLoader>) -> Self {
        let map = inner
            .remaining_keys()
            .into_iter()
            .filter_map(|k| hf_to_dflash_name(&k).map(|v| (v, k)))
            .collect();
        Self { inner, map }
    }
    fn src(&self, key: &str) -> Result<&str> {
        self.map
            .get(key)
            .map(|s| s.as_str())
            .with_context(|| format!("drafter checkpoint has no tensor for {key}"))
    }
}

impl WeightLoader for RenamedLoader {
    fn len(&self) -> usize {
        self.map.len()
    }
    fn remaining_keys(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
    fn take(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let s = self.src(key)?.to_string();
        self.inner.take(&s)
    }
    fn take_transposed(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let s = self.src(key)?.to_string();
        self.inner.take_transposed(&s)
    }
}

/// Qwen3 target that exports the residual streams a DFlash drafter fuses.
struct Qwen3TapTarget {
    cfg: Qwen3Config,
    dec: CompiledGraph,
    /// `[layer][CAP * kv_dim]`, zero-padded; `attn_mask` hides the slack.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    n_past: usize,
    n_taps: usize,
    /// Index of the first tap output.
    tap_base: usize,
    steps: usize,
    /// `n_past` at the start of the current verification step, so `rollback`
    /// can drop exactly the speculative tail.
    step_start: usize,
}

impl Qwen3TapTarget {
    fn new(
        cfg: Qwen3Config,
        weights: &mut dyn WeightLoader,
        tap_layers: &[usize],
        device: Device,
    ) -> Result<Self> {
        let opts = Qwen3DecodeOpts {
            batch: 1,
            past_seq: CAP,
            use_custom_mask: true,
            tap_layers: tap_layers.to_vec(),
            packed: true,
            ..Default::default()
        };
        let mut built = build_qwen3_decode_built(&cfg, weights, &opts)?;
        // `into_graph_parts()` returns only the F32 params and DROPS
        // `typed_params` — the packed U8 quant blobs every projection in a
        // K-quant checkpoint actually reads. Take them first, or the whole
        // model runs on zero weights: it still produces tokens, and they are
        // all noise.
        let typed = std::mem::take(&mut built.typed_params);
        let (graph, params) = built.into_graph_parts()?;
        let n_out = graph.outputs.len();
        let n_layers = cfg.num_hidden_layers;
        let n_taps = tap_layers.iter().filter(|i| **i < n_layers).count();
        if n_out != 1 + 2 * n_layers + n_taps {
            bail!(
                "target decode graph produced {n_out} outputs, expected {}",
                1 + 2 * n_layers + n_taps
            );
        }
        let mut dec = Session::new(device).compile(graph);
        for (k, v) in &params {
            dec.set_param(k, v);
        }
        if opts.packed && typed.is_empty() {
            bail!("packed decode requested but the build produced no packed blobs");
        }
        for (name, bytes, dt) in &typed {
            dec.set_param_typed(name, bytes, *dt);
        }
        println!(
            "  target weights: {} dense params, {} packed blobs",
            params.len(),
            typed.len()
        );
        let kv_dim = cfg.kv_proj_dim();
        Ok(Self {
            k: vec![vec![0f32; CAP * kv_dim]; n_layers],
            v: vec![vec![0f32; CAP * kv_dim]; n_layers],
            n_past: 0,
            n_taps,
            tap_base: 1 + 2 * n_layers,
            steps: 0,
            step_start: 0,
            cfg,
            dec,
        })
    }

    /// One token through the target. Returns `(top-1 dist, tap row)`.
    fn decode(&mut self, token: u32) -> Result<(SparseDist, Vec<f32>)> {
        if self.n_past >= CAP {
            bail!("target cache is full ({CAP}); shorten the run");
        }
        let dh = self.cfg.head_dim;
        let (cos, sin) = rope_tables(&[self.n_past], dh, self.cfg.rope_theta);
        let mut mask = vec![0f32; CAP + 1];
        mask[..self.n_past].fill(1.0);
        mask[CAP] = 1.0; // the token being decoded

        let ids = [token as f32];
        let names: Vec<String> = (0..self.cfg.num_hidden_layers)
            .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
            .collect();
        let mut inputs: Vec<(&str, &[f32])> = vec![
            ("input_ids", &ids),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
            ("mask", &mask),
        ];
        for (i, n) in names.iter().enumerate() {
            let layer = i / 2;
            inputs.push((
                n.as_str(),
                if i % 2 == 0 {
                    &self.k[layer]
                } else {
                    &self.v[layer]
                },
            ));
        }
        let out = self.dec.run(&inputs);
        self.steps += 1;

        // Append this token's K/V at the cache head.
        //
        // The decode graph returns `concat(past_k, k_new)` — the whole cache,
        // not the new row — so the new token's K/V is the LAST row, at index
        // CAP. Reading row 0 instead silently feeds the model a cache of
        // padding zeros: it still runs, still emits tokens, and every one of
        // them is garbage.
        let kv_dim = self.cfg.kv_proj_dim();
        let want = (CAP + 1) * kv_dim;
        if out[1].len() != want {
            bail!(
                "decode K output is {} values, expected {want} = (CAP+1) x kv_dim — \
                 the cache layout changed",
                out[1].len()
            );
        }
        let at = self.n_past * kv_dim;
        let last = CAP * kv_dim;
        for l in 0..self.cfg.num_hidden_layers {
            self.k[l][at..at + kv_dim].copy_from_slice(&out[1 + 2 * l][last..last + kv_dim]);
            self.v[l][at..at + kv_dim].copy_from_slice(&out[2 + 2 * l][last..last + kv_dim]);
        }
        // A cache row that is exactly zero means the append missed: the model
        // would then attend to nothing and emit noise, which reads as
        // "acceptance 0" rather than as a failure.
        if self.k[0][at..at + kv_dim].iter().all(|v| *v == 0.0) {
            bail!(
                "appended an all-zero K row at position {} — cache append is wrong",
                self.n_past
            );
        }
        self.n_past += 1;

        // Greedy target: a one-hot distribution over its argmax. That makes
        // acceptance exactly "would the target have emitted this token".
        let logits = &out[0];
        let best = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, x)| {
                if *x > a.1 { (i, *x) } else { a }
            })
            .0;
        let dist = SparseDist {
            ids: vec![best as u32],
            probs: vec![1.0],
        };

        // Taps, concatenated in ascending layer order — the order `fc` expects.
        let mut taps = Vec::with_capacity(self.n_taps * self.cfg.hidden_size);
        for t in 0..self.n_taps {
            taps.extend_from_slice(&out[self.tap_base + t]);
        }
        Ok((dist, taps))
    }
}

impl DflashTarget for Qwen3TapTarget {
    fn tap_dim(&self) -> usize {
        self.n_taps * self.cfg.hidden_size
    }

    fn prefill(&mut self, prompt: &[u32]) -> Result<TargetStep> {
        let mut taps = Vec::new();
        let mut last = SparseDist::default();
        for t in prompt {
            let (d, row) = self.decode(*t)?;
            taps.extend_from_slice(&row);
            last = d;
        }
        Ok(TargetStep {
            dists: vec![last],
            taps,
        })
    }

    fn step(&mut self, anchor: u32, draft: &[u32]) -> Result<TargetStep> {
        // Sequential, not one batched forward — see the module docs. Acceptance
        // is unaffected; wall-clock is, which is why it is not reported.
        self.step_start = self.n_past;
        let mut dists = Vec::with_capacity(draft.len() + 1);
        let mut taps = Vec::new();
        let (d, row) = self.decode(anchor)?;
        dists.push(d);
        taps.extend_from_slice(&row);
        for t in draft {
            let (d, row) = self.decode(*t)?;
            dists.push(d);
            taps.extend_from_slice(&row);
        }
        Ok(TargetStep { dists, taps })
    }

    fn rollback(&mut self, keep: usize) -> Result<()> {
        // The round decoded the anchor plus every drafted token, but only the
        // anchor and `keep` of the drafts survive. Dropping the rest is what
        // keeps the target's cache in step with the committed sequence; the
        // bonus token arrives as the next round's anchor.
        let keep_to = self.step_start + 1 + keep;
        if keep_to > self.n_past {
            bail!(
                "rollback(keep={keep}) wants {keep_to} cached tokens but only {} were decoded",
                self.n_past
            );
        }
        self.n_past = keep_to;
        Ok(())
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let drafter_dir = args
        .next()
        .context("usage: dflash_measure <drafter-dir> <target.gguf> [device] [n]")?;
    let target_path = args.next().context("missing <target.gguf>")?;
    let device = device_from(&args.next().unwrap_or_else(|| "cpu".into()));
    let n_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    // ── drafter ─────────────────────────────────────────────────────────
    let cfg_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{drafter_dir}/config.json"
    ))?)?;
    let dcfg = DflashConfig::from_hf_json(&cfg_json)?;
    println!(
        "drafter: {} layers, hidden {}, block {}, taps {:?}",
        dcfg.num_hidden_layers, dcfg.hidden_size, dcfg.block_size, dcfg.target_layers
    );

    // ── target ──────────────────────────────────────────────────────────
    let raw = rlx_gguf::GgufFile::from_path_mmap(&target_path)
        .or_else(|_| rlx_gguf::GgufFile::from_path(&target_path))?;
    let g = |k: &str| {
        raw.metadata
            .get(k)
            .and_then(rlx_gguf::MetaValue::as_u32)
            .map(|v| v as usize)
    };
    let gf = |k: &str| {
        raw.metadata.get(k).and_then(|v| match v {
            rlx_gguf::MetaValue::F32(x) => Some(*x as f64),
            _ => None,
        })
    };
    let nh = g("qwen3.attention.head_count").context("qwen3.attention.head_count")?;
    let tcfg = Qwen3Config {
        vocab_size: raw
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                rlx_gguf::MetaValue::Array(a) => Some(a.len()),
                _ => None,
            })
            .context("target GGUF has no tokenizer.ggml.tokens")?,
        hidden_size: g("qwen3.embedding_length").context("qwen3.embedding_length")?,
        intermediate_size: g("qwen3.feed_forward_length").context("qwen3.feed_forward_length")?,
        num_hidden_layers: g("qwen3.block_count").context("qwen3.block_count")?,
        num_attention_heads: nh,
        num_key_value_heads: g("qwen3.attention.head_count_kv").context("head_count_kv")?,
        head_dim: g("qwen3.attention.key_length").unwrap_or(0),
        max_position_embeddings: g("qwen3.context_length").unwrap_or(32768),
        rms_norm_eps: gf("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
        rope_theta: gf("qwen3.rope.freq_base").unwrap_or(1_000_000.0),
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        attention_bias: false,
        qk_norm: true,
        sliding_window: None,
        max_window_layers: usize::MAX,
        use_sliding_window: false,
        num_experts: 0,
        num_experts_used: 0,
        expert_ffn_size: 0,
        shared_expert_ffn_size: 0,
        expert_weights_scale: 1.0,
    };
    println!(
        "target: {} layers, hidden {}, vocab {}",
        tcfg.num_hidden_layers, tcfg.hidden_size, tcfg.vocab_size
    );
    if tcfg.hidden_size * dcfg.target_layers.len() != dcfg.fused_input_dim() {
        bail!("drafter/target hidden size mismatch");
    }

    // Which residual a `target_layer_ids` entry names is a convention, not a
    // fact: HF's `hidden_states[i]` is the INPUT of layer i (== output of i-1),
    // but an exporter may have meant the OUTPUT of layer i. `RLX_DFLASH_TAP_SHIFT=1`
    // tests the other reading; acceptance decides which is right.
    let shift: usize = std::env::var("RLX_DFLASH_TAP_SHIFT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let taps: Vec<usize> = dcfg.target_layers.iter().map(|l| l + shift).collect();
    if shift != 0 {
        println!("tap shift {shift}: {:?} -> {taps:?}", dcfg.target_layers);
    }
    let mut tload = rlx_core::weight_loader::load_from_path(&target_path)?;
    let target = Qwen3TapTarget::new(tcfg.clone(), tload.as_mut(), &taps, device)?;
    println!(
        "target decode graph compiled (cap {CAP}, taps {})",
        target.n_taps
    );

    // ── drafter graphs ──────────────────────────────────────────────────
    let dload =
        rlx_core::weight_loader::load_from_path(&format!("{drafter_dir}/model.safetensors"))?;
    let mut dw = RenamedLoader::new(dload);
    let mut drafter = DflashDrafter::new(
        dcfg.clone(),
        &mut dw,
        DrafterOptions {
            device,
            max_context: CAP,
            min_bucket: CAP,
            ..Default::default()
        },
    )?;

    // The drafter has no embedding and no LM head — both come from the target.
    for name in drafter.missing_shared().to_vec() {
        let hf = match name.as_str() {
            "token_embd.weight" => "model.embed_tokens.weight",
            "output.weight" => "lm_head.weight",
            other => bail!("unexpected shared param {other}"),
        };
        let mut l = rlx_core::weight_loader::load_from_path(&target_path)?;
        let (data, shape) = if name == "output.weight" {
            l.take_transposed(hf)?
        } else {
            l.take(hf)?
        };
        println!("  shared from target: {name} <- {hf} {shape:?}");
        drafter.set_shared_param(&name, &data)?;
    }

    // ── measure ─────────────────────────────────────────────────────────
    let mut l = DflashLoop::new(drafter, target, 0xd_f1a5)?;
    let prompt: Vec<u32> = vec![9707, 11, 847, 829, 374]; // arbitrary in-vocab ids
    let out = l.generate(&prompt, n_tokens, |_| false)?;

    println!("\ngenerated {} tokens", out.len());
    println!(
        "rounds {} | drafted {} | accepted {} | ACCEPTANCE {:.3} | tokens/target-forward {:.2}",
        l.stats.steps,
        l.stats.drafted,
        l.stats.accepted,
        l.stats.acceptance_rate(),
        l.stats.tokens_per_target_forward(),
    );
    println!("first 16 tokens: {:?}", &out[..16.min(out.len())]);
    Ok(())
}
