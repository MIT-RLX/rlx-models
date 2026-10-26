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

//! S2-Pro: the dual-AR text-to-semantic model.
//!
//! ```text
//!  row t ─ embed ─▶ slow backbone (36 × Qwen3 block, qk-norm, GQA 32/8,
//!                   interleaved RoPE) ─ final norm ─▶ h_t
//!     main token  ~ h_t · E[band]ᵀ          (semantic band + <|im_end|>)
//!     fast pos 0  ◀ h_t
//!     fast pos i  ◀ fast_embed[code i-1]  ─▶ code i   (i = 1..9)
//!  row t+1 = (E[main] + Σ_i C[i·4096 + code_i]) / √11
//! ```
//!
//! The slow stack runs as a compiled prefill graph over the whole prompt,
//! then one compiled decode step per frame against a KV cache kept on the
//! host (a fixed-capacity bucket, unused slots masked). The fast decoder is
//! a 4-layer step graph over a 10-slot cache that restarts every frame.
//!
//! The LM head is tied to the token embedding, but only its semantic band and
//! `<|im_end|>` are ever sampled, so the graph carries just those 4097 rows
//! instead of the 155 776-row matrix.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rlx_core::safetensors_checkpoint::SafetensorsCheckpoint;
use rlx_flow::storage::WeightStorage;
use rlx_ir::RopeStyle;
use rlx_ir::op::MaskKind;
use rlx_runtime::{CompiledGraph, Device};

use crate::config::{S2Config, StackConfig};
use crate::nn::{self, AttnSpec, BlockSpec, Builder, Keys};
use crate::prompt::{self, Row, Segment, TextEncoder};
use crate::sampling::{self, Flavor, Rng, SampleParams};

const SLOW: &str = "text_model.model.";
const FAST: &str = "audio_decoder.";
const EMBED: &str = "text_model.model.embeddings.weight";
const CODEBOOK_EMBED: &str = "audio_decoder.codebook_embeddings.weight";
const FAST_EMBED: &str = "audio_decoder.embeddings.weight";

/// Prompt lengths are rounded up to this for the prefill graph; the padding
/// sits after the real rows, which a causal mask keeps them from seeing.
const PREFILL_BUCKET: usize = 64;
/// Decode KV capacity granularity.
const DECODE_BUCKET: usize = 256;

/// Repetition-aware sampling (S2): when the main token repeats one of the last
/// `RAS_WINDOW` main tokens, a second draw at these settings replaces it.
const RAS_WINDOW: usize = 10;
const RAS_TEMPERATURE: f32 = 1.0;
const RAS_TOP_P: f32 = 0.9;

/// Default per-chunk frame budget (`DEFAULT_MAX_NEW_TOKENS`).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 1024;
/// Default UTF-8 byte budget of one text chunk (`DEFAULT_CHUNK_LENGTH`).
pub const DEFAULT_CHUNK_LENGTH: usize = 200;

/// How an S2 model is loaded.
#[derive(Clone, Debug, Default)]
pub struct S2Options {
    /// Run only the first `n` slow layers (tests against a truncated reference).
    pub layers: Option<usize>,
    /// Keep the prefill graph after a prompt instead of freeing its weights.
    /// On backends that share parameter storage between graphs (Metal) this
    /// is free; on the CPU it holds a second copy of the backbone.
    pub keep_prefill: bool,
    /// How matmul weights are stored in the graphs. `None` reads
    /// `RLX_WEIGHT_STORAGE`, else `Auto` (keep the checkpoint's bf16 —
    /// the same products as f32 at half the bytes).
    pub weight_storage: Option<WeightStorage>,
}

/// A loaded S2-Pro checkpoint.
pub struct S2Model {
    pub cfg: S2Config,
    dir: PathBuf,
    ckpt: SafetensorsCheckpoint,
    device: Device,
    layers: usize,
    keep_prefill: bool,
    matrix_dtype: rlx_ir::DType,
    /// LM head rows for the semantic band then `<|im_end|>`, as `[dim, band+1]`.
    pub(crate) band_head: Vec<f32>,
    prefill: Option<(usize, CompiledGraph)>,
    decode: Option<(usize, CompiledGraph)>,
    fast: Option<CompiledGraph>,
    /// The [`SlowState`] whose K/V the decode graph holds on device, if any.
    resident: Option<u64>,
}

/// The slow stack's KV cache and position after some rows.
pub struct SlowState {
    /// Rows already in the cache.
    pub len: usize,
    cap: usize,
    /// Identifies the cache whose rows the decode graph holds resident.
    id: u64,
    /// The host `k`/`v` stopped tracking the cache when it moved on device.
    host_stale: bool,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Final-norm hidden state of the newest row.
    pub hidden: Vec<f32>,
    /// Band logits (`band + <|im_end|>`) after the newest row.
    pub logits: Vec<f32>,
}

/// One generated frame: the main token and its `num_codebooks` codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub main: u32,
    pub codes: Vec<u32>,
}

impl S2Model {
    pub fn open(dir: &Path, device: Device, opts: S2Options) -> Result<Self> {
        let cfg = S2Config::from_dir(dir)?;
        let ckpt = SafetensorsCheckpoint::open(dir)?;
        let layers = opts
            .layers
            .unwrap_or(cfg.slow.n_layer)
            .min(cfg.slow.n_layer);
        let head_key = if cfg.tie_word_embeddings {
            EMBED.to_string()
        } else {
            format!("{SLOW}output.weight")
        };
        let mut ids: Vec<u32> = (cfg.semantic_begin..=cfg.semantic_end).collect();
        ids.push(cfg.eos);
        let rows = ckpt
            .load_tensor_rows_f32(&head_key, &ids, cfg.slow.dim)
            .context("LM head band")?;
        let nb = ids.len();
        let mut band_head = vec![0f32; cfg.slow.dim * nb];
        for (j, row) in rows.iter().enumerate() {
            for (i, &w) in row.iter().enumerate() {
                band_head[i * nb + j] = w;
            }
        }
        let storage = opts
            .weight_storage
            .or_else(WeightStorage::from_env)
            .unwrap_or_default();
        let probe = format!("{SLOW}layers.0.attention.wqkv.weight");
        let matrix_dtype = storage.resolve(stored_dtype(&ckpt, &probe));
        Ok(Self {
            cfg,
            dir: dir.to_path_buf(),
            ckpt,
            device,
            layers,
            // Metal maps one copy of each weight into every graph of the model, so
            // keeping the prefill graph alive costs activations, not weights — and
            // keeps those pages live for the decode graph to map instead of re-upload.
            keep_prefill: opts.keep_prefill || device == Device::Metal,
            matrix_dtype,
            band_head,
            prefill: None,
            decode: None,
            fast: None,
            resident: None,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn band_len(&self) -> usize {
        self.cfg.codebook_size + 1
    }

    /// Band logit index → token id.
    pub fn band_token(&self, idx: usize) -> u32 {
        if idx < self.cfg.codebook_size {
            self.cfg.semantic_begin + idx as u32
        } else {
            self.cfg.eos
        }
    }

    fn slow_spec(&self) -> BlockSpec {
        stack_spec(&self.cfg.slow, self.cfg.slow.ffn, self.cfg.norm_eps)
    }

    fn kv_width(s: &StackConfig) -> usize {
        s.n_kv * s.head_dim
    }

    // ── embeddings ──────────────────────────────────────────────────────

    /// Input embeddings `[rows, dim]`: the token embedding, plus — for audio
    /// rows — every code's codebook embedding, the sum scaled by `1/√(N+1)`.
    pub fn embed_rows(&self, rows: &[Row]) -> Result<Vec<f32>> {
        let dim = self.cfg.slow.dim;
        let nc = self.cfg.num_codebooks;
        let tokens: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let tok = self.ckpt.load_tensor_rows_f32(EMBED, &tokens, dim)?;
        let mut cb_ids = Vec::new();
        for r in rows.iter().filter(|r| self.cfg.is_semantic(r[0])) {
            ensure!(
                r.len() == 1 + nc,
                "row is {} wide, want {}",
                r.len(),
                1 + nc
            );
            for (i, &c) in r[1..].iter().enumerate() {
                cb_ids.push(i as u32 * self.cfg.codebook_size as u32 + c);
            }
        }
        let cb = if cb_ids.is_empty() {
            Vec::new()
        } else {
            self.ckpt
                .load_tensor_rows_f32(CODEBOOK_EMBED, &cb_ids, dim)?
        };
        let scale = 1.0 / ((nc + 1) as f32).sqrt();
        let mut out = vec![0f32; rows.len() * dim];
        let mut next_cb = cb.iter();
        for (i, (r, e)) in rows.iter().zip(tok).enumerate() {
            let slot = &mut out[i * dim..(i + 1) * dim];
            slot.copy_from_slice(&e);
            if self.cfg.is_semantic(r[0]) {
                for _ in 0..nc {
                    let c = next_cb.next().expect("codebook row");
                    for (s, v) in slot.iter_mut().zip(c) {
                        *s += v;
                    }
                }
                for s in slot.iter_mut() {
                    *s *= scale;
                }
            }
        }
        Ok(out)
    }

    // ── slow stack ──────────────────────────────────────────────────────

    fn build_prefill(&self, seq: usize) -> Result<CompiledGraph> {
        let (b, outs) = self.prefill_builder(seq);
        self.compile(b, outs, rlx_flow::CompileProfile::llama32_prefill())
    }

    /// The prefill graph over `seq` rows, uncompiled: the final norm's rows,
    /// then each layer's K and V.
    pub(crate) fn prefill_builder(&self, seq: usize) -> (Builder, Vec<rlx_ir::NodeId>) {
        let s = &self.cfg.slow;
        let spec = self.slow_spec();
        let mut b = Builder::with_matrix_dtype("fish_s2_prefill", self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let x = b.input("embeds", &[1, seq, s.dim]);
        let cos = b.input("rope_cos", &[seq, s.head_dim / 2]);
        let sin = b.input("rope_sin", &[seq, s.head_dim / 2]);
        let mut h = x;
        let mut outs = vec![];
        for l in 0..self.layers {
            let o = nn::block(
                &mut b,
                &spec,
                &format!("{SLOW}layers.{l}."),
                h,
                seq,
                cos,
                sin,
                Keys::Own(MaskKind::Causal),
            );
            h = o.x;
            outs.push(o.k);
            outs.push(o.v);
        }
        let hn = b.rms_norm(
            h,
            &format!("{SLOW}norm.weight"),
            s.dim,
            self.cfg.norm_eps,
            false,
        );
        outs.insert(0, hn);
        (b, outs)
    }

    fn build_decode(&self, cap: usize) -> Result<CompiledGraph> {
        let (b, outs) = self.decode_builder(cap);
        self.compile(b, outs, rlx_flow::CompileProfile::llama32_decode())
    }

    /// The decode step against a `cap`-row cache, uncompiled: the hidden
    /// state, the band logits, then each layer's new K and V row.
    pub(crate) fn decode_builder(&self, cap: usize) -> (Builder, Vec<rlx_ir::NodeId>) {
        let s = &self.cfg.slow;
        let spec = self.slow_spec();
        let kvw = Self::kv_width(s);
        let mut b = Builder::with_matrix_dtype("fish_s2_decode", self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let x = b.input("embeds", &[1, 1, s.dim]);
        let cos = b.input("rope_cos", &[1, s.head_dim / 2]);
        let sin = b.input("rope_sin", &[1, s.head_dim / 2]);
        let mask = b.input("mask", &[1, cap + 1]);
        let mut h = x;
        let mut kv = vec![];
        for l in 0..self.layers {
            let past_k = b.input(&format!("past_k_{l}"), &[1, cap, kvw]);
            let past_v = b.input(&format!("past_v_{l}"), &[1, cap, kvw]);
            let keys = Keys::Cached {
                past_k,
                past_v,
                past: cap,
                mask,
            };
            let o = nn::block(
                &mut b,
                &spec,
                &format!("{SLOW}layers.{l}."),
                h,
                1,
                cos,
                sin,
                keys,
            );
            h = o.x;
            kv.push(o.k);
            kv.push(o.v);
        }
        let hn = b.rms_norm(
            h,
            &format!("{SLOW}norm.weight"),
            s.dim,
            self.cfg.norm_eps,
            false,
        );
        let nb = self.band_len();
        let head = b.data("fish.band_head", self.band_head.clone(), &[s.dim, nb]);
        let logits = rlx_ir::infer::GraphExt::mm(&mut b.g, hn, head);
        let mut outs = vec![hn, logits];
        outs.extend(kv);
        (b, outs)
    }

    fn compile(
        &self,
        mut b: Builder,
        outs: Vec<rlx_ir::NodeId>,
        profile: rlx_flow::CompileProfile,
    ) -> Result<CompiledGraph> {
        b.g.set_outputs(outs);
        let label = format!("fish-s2:{}", self.dir.display());
        let device = self.device;
        let name = b.g.name.clone();
        let t = std::time::Instant::now();
        let mut g = rlx_runtime::with_param_sharing_scope(&label, || {
            rlx_core::flow_bridge::compile_graph_with_profile(device, b.g, &profile)
        })?;
        let t_compile = t.elapsed();
        let t = std::time::Instant::now();
        nn::upload(&mut g, b.params, &self.ckpt)?;
        if rlx_ir::env::flag("RLX_FISH_VERBOSE") {
            eprintln!(
                "s2: {name}: compile {:.1} s, weights {:.1} s",
                t_compile.as_secs_f32(),
                t.elapsed().as_secs_f32()
            );
        }
        Ok(g)
    }

    fn rope(&self, positions: &[usize], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        nn::rope_tables(positions, head_dim, self.cfg.rope_base, true)
    }

    /// Host matvec `hidden · band_head` (one row; the decode graph does this
    /// itself).
    fn band_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let nb = self.band_len();
        let mut out = vec![0f32; nb];
        for (i, &h) in hidden.iter().enumerate() {
            let row = &self.band_head[i * nb..(i + 1) * nb];
            for (o, w) in out.iter_mut().zip(row) {
                *o += h * w;
            }
        }
        out
    }

    /// Run `rows` from position 0 and return the cache, sized for `cap` rows.
    pub fn prefill(&mut self, rows: &[Row], cap: usize) -> Result<SlowState> {
        let t = rows.len();
        ensure!(t > 0, "empty prompt");
        ensure!(
            cap > t,
            "cache capacity {cap} does not exceed the prompt {t}"
        );
        let s = self.cfg.slow.clone();
        let seq = t.div_ceil(PREFILL_BUCKET) * PREFILL_BUCKET;
        if self.prefill.as_ref().map(|p| p.0) != Some(seq) {
            self.prefill = None;
            self.prefill = Some((seq, self.build_prefill(seq)?));
        }
        let mut embeds = self.embed_rows(rows)?;
        embeds.resize(seq * s.dim, 0.0);
        let positions: Vec<usize> = (0..seq).collect();
        let (cos, sin) = self.rope(&positions, s.head_dim);
        let outs = {
            let g = &mut self.prefill.as_mut().expect("prefill graph").1;
            g.run(&[("embeds", &embeds), ("rope_cos", &cos), ("rope_sin", &sin)])
        };
        if !self.keep_prefill {
            self.prefill = None;
        }
        let kvw = Self::kv_width(&s);
        let mut k = Vec::with_capacity(self.layers);
        let mut v = Vec::with_capacity(self.layers);
        for l in 0..self.layers {
            let mut kb = vec![0f32; cap * kvw];
            let mut vb = vec![0f32; cap * kvw];
            kb[..t * kvw].copy_from_slice(&outs[1 + 2 * l][..t * kvw]);
            vb[..t * kvw].copy_from_slice(&outs[2 + 2 * l][..t * kvw]);
            k.push(kb);
            v.push(vb);
        }
        let hidden = outs[0][(t - 1) * s.dim..t * s.dim].to_vec();
        let logits = self.band_logits(&hidden);
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Ok(SlowState {
            len: t,
            cap,
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            host_stale: false,
            k,
            v,
            hidden,
            logits,
        })
    }

    /// Append one row (given as its embedding) to `state`.
    ///
    /// Where the backend can hold inputs on the device, the cache is uploaded
    /// once and each step's new K/V row is copied into it there — only the
    /// embedding goes in and the hidden state and band logits come out. That
    /// keeps the per-step traffic flat as the context grows (the host path
    /// re-sends `2 · layers · cap · kv` floats every step).
    pub fn step(&mut self, state: &mut SlowState, embed: &[f32]) -> Result<()> {
        let s = self.cfg.slow.clone();
        ensure!(state.len < state.cap, "KV cache full ({} rows)", state.cap);
        if self.decode.as_ref().map(|d| d.0) != Some(state.cap) {
            self.decode = None;
            self.resident = None;
            self.decode = Some((state.cap, self.build_decode(state.cap)?));
        }
        let pos = state.len;
        let (cos, sin) = self.rope(&[pos], s.head_dim);
        let mut mask = vec![0f32; state.cap + 1];
        mask[..pos].fill(1.0);
        mask[state.cap] = 1.0;
        let kvw = Self::kv_width(&s);
        let names: Vec<(String, String)> = (0..self.layers)
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();
        if self.resident != Some(state.id)
            && !state.host_stale
            && rlx_core::gpu_kv::device_supports_resident_kv_row_feed(self.device)
        {
            let g = &mut self.decode.as_mut().expect("decode graph").1;
            let mut ok = true;
            for (l, (kn, vn)) in names.iter().enumerate() {
                ok &= g.bind_gpu_handle(kn, &state.k[l]) && g.register_kv_row_feed(kn, 2 + 2 * l);
                ok &= g.bind_gpu_handle(vn, &state.v[l]) && g.register_kv_row_feed(vn, 3 + 2 * l);
            }
            self.resident = ok.then_some(state.id);
        }
        let small: Vec<(&str, &[f32])> = vec![
            ("embeds", embed),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
            ("mask", &mask),
        ];
        if self.resident == Some(state.id) {
            let g = &mut self.decode.as_mut().expect("decode graph").1;
            let mut outs = g.run_read_outputs(&small, Some(&[0, 1]));
            ensure!(g.feed_kv_row(0, pos, kvw), "resident KV row feed failed");
            state.host_stale = true;
            state.logits = outs.swap_remove(1);
            state.hidden = outs.swap_remove(0);
            state.len += 1;
            return Ok(());
        }
        ensure!(
            !state.host_stale,
            "this cache now lives in another decode graph"
        );
        let mut inputs = small;
        for (l, (kn, vn)) in names.iter().enumerate() {
            inputs.push((kn, &state.k[l]));
            inputs.push((vn, &state.v[l]));
        }
        let outs = self.decode.as_mut().expect("decode graph").1.run(&inputs);
        for l in 0..self.layers {
            state.k[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[2 + 2 * l]);
            state.v[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[3 + 2 * l]);
        }
        state.hidden = outs[0].clone();
        state.logits = outs[1].clone();
        state.len += 1;
        Ok(())
    }

    // ── fast stack ──────────────────────────────────────────────────────

    fn build_fast(&self) -> Result<CompiledGraph> {
        let (b, outs) = self.fast_builder();
        self.compile(b, outs, rlx_flow::CompileProfile::llama32_decode())
    }

    /// The fast decoder's step, uncompiled: the logits, then each layer's
    /// new K and V row.
    pub(crate) fn fast_builder(&self) -> (Builder, Vec<rlx_ir::NodeId>) {
        let f = &self.cfg.fast;
        let cap = self.cfg.num_codebooks;
        let spec = stack_spec(f, f.ffn, self.cfg.norm_eps);
        let kvw = Self::kv_width(f);
        let mut b = Builder::with_matrix_dtype("fish_s2_fast", self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let x = b.input("x", &[1, 1, f.dim]);
        let cos = b.input("rope_cos", &[1, f.head_dim / 2]);
        let sin = b.input("rope_sin", &[1, f.head_dim / 2]);
        let mask = b.input("mask", &[1, cap + 1]);
        let mut h = x;
        let mut kv = vec![];
        for l in 0..f.n_layer {
            let past_k = b.input(&format!("past_k_{l}"), &[1, cap, kvw]);
            let past_v = b.input(&format!("past_v_{l}"), &[1, cap, kvw]);
            let keys = Keys::Cached {
                past_k,
                past_v,
                past: cap,
                mask,
            };
            let o = nn::block(
                &mut b,
                &spec,
                &format!("{FAST}layers.{l}."),
                h,
                1,
                cos,
                sin,
                keys,
            );
            h = o.x;
            kv.push(o.k);
            kv.push(o.v);
        }
        let hn = b.rms_norm(
            h,
            &format!("{FAST}norm.weight"),
            f.dim,
            self.cfg.norm_eps,
            false,
        );
        let logits = b.linear(
            hn,
            &format!("{FAST}output.weight"),
            f.dim,
            self.cfg.codebook_size,
        );
        let mut outs = vec![logits];
        outs.extend(kv);
        (b, outs)
    }

    /// Fill one frame's codes after the slow stack chose `main`: the fast
    /// decoder reads the slow hidden state at position 0 and each previous
    /// code's embedding after it. `on_logits` sees each codebook's logits
    /// before it is sampled (parity tests).
    pub fn fast_frame(
        &mut self,
        hidden: &[f32],
        main: u32,
        params: SampleParams,
        rng: &mut Rng,
        mut on_logits: impl FnMut(usize, &[f32]),
    ) -> Result<Vec<u32>> {
        let f = self.cfg.fast.clone();
        let nc = self.cfg.num_codebooks;
        if self.fast.is_none() {
            self.fast = Some(self.build_fast()?);
        }
        let kvw = Self::kv_width(&f);
        let mut k = vec![vec![0f32; nc * kvw]; f.n_layer];
        let mut v = vec![vec![0f32; nc * kvw]; f.n_layer];
        let names: Vec<(String, String)> = (0..f.n_layer)
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();
        let c0 = main
            .saturating_sub(self.cfg.semantic_begin)
            .min(self.cfg.codebook_size as u32 - 1);
        let mut codes = vec![c0];
        let mut x = hidden.to_vec();
        for pos in 0..nc {
            let (cos, sin) = nn::rope_tables(&[pos], f.head_dim, self.cfg.rope_base, true);
            let mut mask = vec![0f32; nc + 1];
            mask[..pos].fill(1.0);
            mask[nc] = 1.0;
            let mut inputs: Vec<(&str, &[f32])> = vec![
                ("x", &x),
                ("rope_cos", &cos),
                ("rope_sin", &sin),
                ("mask", &mask),
            ];
            for (l, (kn, vn)) in names.iter().enumerate() {
                inputs.push((kn, &k[l]));
                inputs.push((vn, &v[l]));
            }
            let outs = self.fast.as_mut().expect("fast graph").run(&inputs);
            for l in 0..f.n_layer {
                k[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[1 + 2 * l]);
                v[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[2 + 2 * l]);
            }
            if pos > 0 {
                on_logits(pos, &outs[0]);
                codes.push(sampling::sample(&outs[0], params, Flavor::S2, rng) as u32);
            }
            if pos + 1 < nc {
                let prev = *codes.last().expect("code");
                x = self
                    .ckpt
                    .load_tensor_rows_f32(FAST_EMBED, &[prev], f.dim)?
                    .swap_remove(0);
            }
        }
        Ok(codes)
    }

    // ── generation ──────────────────────────────────────────────────────

    /// Sample frames after `rows` until `<|im_end|>` or `max_new` frames.
    /// The stop frame is not returned.
    pub fn generate(
        &mut self,
        rows: &[Row],
        max_new: usize,
        params: SampleParams,
        rng: &mut Rng,
    ) -> Result<Vec<Frame>> {
        self.generate_with(rows, max_new, params, rng, |_| {})
    }

    /// [`Self::generate`] with a callback per finished frame.
    pub fn generate_with(
        &mut self,
        rows: &[Row],
        max_new: usize,
        params: SampleParams,
        rng: &mut Rng,
        mut on_frame: impl FnMut(&Frame),
    ) -> Result<Vec<Frame>> {
        let t = rows.len();
        if t + max_new > self.cfg.max_seq_len {
            bail!(
                "prompt {t} + {max_new} frames exceeds the {}-row context",
                self.cfg.max_seq_len
            );
        }
        let cap = (t + max_new).div_ceil(DECODE_BUCKET) * DECODE_BUCKET;
        let mut state = self.prefill(rows, cap.min(self.cfg.max_seq_len))?;
        let ras = SampleParams {
            temperature: RAS_TEMPERATURE,
            top_p: RAS_TOP_P,
            top_k: params.top_k,
        };
        // The reference's RAS window starts empty and only collects frames
        // sampled by the decode loop — not the first one, drawn at prefill.
        let mut window: Vec<u32> = Vec::with_capacity(RAS_WINDOW);
        let mut frames = Vec::new();
        let verbose = rlx_ir::env::flag("RLX_FISH_VERBOSE");
        let (mut t_slow, mut t_fast) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for step in 0..max_new {
            let first = self.band_token(sampling::sample(&state.logits, params, Flavor::S2, rng));
            let mut main = first;
            if step > 0 {
                let retry = self.band_token(sampling::sample(&state.logits, ras, Flavor::S2, rng));
                if self.cfg.is_semantic(first) && window.contains(&first) {
                    main = retry;
                }
            }
            if main == self.cfg.eos {
                break;
            }
            let t = std::time::Instant::now();
            let codes = self.fast_frame(&state.hidden, main, params, rng, |_, _| {})?;
            t_fast += t.elapsed();
            let frame = Frame { main, codes };
            on_frame(&frame);
            if step > 0 {
                if window.len() == RAS_WINDOW {
                    window.remove(0);
                }
                window.push(main);
            }
            if step + 1 < max_new {
                let mut row = vec![main];
                row.extend(&frame.codes);
                let t = std::time::Instant::now();
                let embed = self.embed_rows(&[row])?;
                self.step(&mut state, &embed)?;
                t_slow += t.elapsed();
            }
            frames.push(frame);
            if verbose && frames.len() % 16 == 0 {
                let n = frames.len() as f32;
                eprintln!(
                    "s2: {} frames, slow {:.0} ms/frame, fast {:.0} ms/frame",
                    frames.len(),
                    t_slow.as_secs_f32() * 1e3 / n,
                    t_fast.as_secs_f32() * 1e3 / n
                );
            }
        }
        Ok(frames)
    }

    /// Drop every compiled graph.
    pub fn release(&mut self) {
        self.prefill = None;
        self.decode = None;
        self.fast = None;
    }
}

/// The dtype a checkpoint tensor is stored in (`None` if unknown).
pub(crate) fn stored_dtype(ckpt: &SafetensorsCheckpoint, name: &str) -> Option<rlx_ir::DType> {
    match ckpt.stored_dtype(name)? {
        safetensors::Dtype::BF16 => Some(rlx_ir::DType::BF16),
        safetensors::Dtype::F16 => Some(rlx_ir::DType::F16),
        safetensors::Dtype::F32 => Some(rlx_ir::DType::F32),
        _ => None,
    }
}

fn stack_spec(s: &StackConfig, ffn: usize, eps: f32) -> BlockSpec {
    BlockSpec {
        dim: s.dim,
        ffn,
        eps,
        attn: AttnSpec {
            n_head: s.n_head,
            n_kv: s.n_kv,
            head_dim: s.head_dim,
            qk_norm: s.qk_norm,
            rope_dim: s.head_dim,
            rope_style: RopeStyle::GptJ,
            output_gate: false,
        },
        layer_scale: false,
        zero_centered: false,
    }
}

/// Frames → codes `[num_codebooks][frames]`.
pub fn frames_to_codes(frames: &[Frame], num_codebooks: usize) -> Vec<Vec<u32>> {
    (0..num_codebooks)
        .map(|i| frames.iter().map(|f| f.codes[i]).collect())
        .collect()
}

/// A reference voice: transcript + codec codes `[num_codebooks][frames]`.
#[derive(Clone, Debug)]
pub struct Reference {
    pub text: String,
    pub codes: Vec<Vec<u32>>,
}

/// S2's conversation for one utterance, as `generate_long` builds it.
pub struct Conversation {
    segments: Vec<Segment>,
}

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>\n";

impl Conversation {
    /// The system turn, with optional reference voices: their transcripts
    /// (each tagged `<|speaker:i|>` unless it carries a tag) joined by `\n`,
    /// and their codes concatenated in time.
    pub fn new(references: &[Reference]) -> Result<Self> {
        let mut segments = vec![Segment::Text(format!("{IM_START}system\n"))];
        if references.is_empty() {
            segments.push(Segment::Text("convert the provided text to speech".into()));
        } else {
            let texts: Vec<String> = references
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    if prompt::has_speaker_tag(&r.text) {
                        r.text.clone()
                    } else {
                        format!("<|speaker:{i}|>{}", r.text)
                    }
                })
                .collect();
            let nc = references[0].codes.len();
            let mut codes = vec![Vec::new(); nc];
            for r in references {
                ensure!(r.codes.len() == nc, "references disagree on codebook count");
                for (dst, src) in codes.iter_mut().zip(&r.codes) {
                    dst.extend(src);
                }
            }
            segments.push(Segment::Text(
                "convert the provided text to speech reference to the following:\n\nText:\n".into(),
            ));
            segments.push(Segment::Text(texts.join("\n")));
            segments.push(Segment::Text("\n\nSpeech:\n".into()));
            segments.push(Segment::Codes(codes));
        }
        segments.push(Segment::Text(IM_END.into()));
        Ok(Self { segments })
    }

    pub fn push_user(&mut self, text: &str) {
        self.segments
            .push(Segment::Text(format!("{IM_START}user\n")));
        self.segments.push(Segment::Text(text.into()));
        self.segments.push(Segment::Text(IM_END.into()));
    }

    /// Undo [`Self::push_user`].
    pub fn pop_user(&mut self) {
        let n = self.segments.len();
        self.segments.truncate(n - 3);
    }

    pub fn push_assistant(&mut self, codes: Vec<Vec<u32>>) {
        self.segments
            .push(Segment::Text(format!("{IM_START}assistant\n<|voice|>")));
        self.segments.push(Segment::Codes(codes));
        self.segments.push(Segment::Text(IM_END.into()));
    }

    /// The prompt for the next assistant turn (the conversation plus an open
    /// `<|im_start|>assistant\n<|voice|>`).
    pub fn prompt_rows(&self, enc: &dyn TextEncoder, cfg: &S2Config) -> Result<Vec<Row>> {
        let mut segs = self.segments.clone();
        segs.push(Segment::Text(format!("{IM_START}assistant\n<|voice|>")));
        prompt::rows(&segs, enc, cfg.num_codebooks, cfg.semantic_begin)
    }
}

/// Settings of a long-form synthesis.
#[derive(Clone, Debug)]
pub struct SynthOptions {
    pub params: SampleParams,
    pub seed: u64,
    pub max_new_tokens: usize,
    pub chunk_length: usize,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            params: SampleParams::default(),
            seed: 0,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            chunk_length: DEFAULT_CHUNK_LENGTH,
        }
    }
}

/// `generate_long`: split `text` by speaker into batches, generate each in a
/// growing conversation, and return all codes `[num_codebooks][frames]`.
pub fn synthesize_codes(
    model: &mut S2Model,
    enc: &dyn TextEncoder,
    text: &str,
    references: &[Reference],
    opts: &SynthOptions,
) -> Result<Vec<Vec<u32>>> {
    let turns = prompt::split_by_speaker(text);
    let batches = if turns.is_empty() {
        vec![text.to_string()]
    } else {
        prompt::group_turns(&turns, 5, opts.chunk_length)
    };
    let nc = model.cfg.num_codebooks;
    let mut conv = Conversation::new(references)?;
    let mut rng = Rng::new(opts.seed);
    let mut all = vec![Vec::new(); nc];
    for batch in batches {
        conv.push_user(&batch);
        let rows = conv.prompt_rows(enc, &model.cfg)?;
        if rows.len() + 2048 > model.cfg.max_seq_len {
            bail!("prompt is too long: {} rows", rows.len());
        }
        let frames = model.generate(&rows, opts.max_new_tokens, opts.params, &mut rng)?;
        if frames.is_empty() {
            conv.pop_user();
            continue;
        }
        let codes = frames_to_codes(&frames, nc);
        for (dst, src) in all.iter_mut().zip(&codes) {
            dst.extend(src);
        }
        conv.push_assistant(codes);
    }
    Ok(all)
}
