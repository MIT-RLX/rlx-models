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
//!
//! [`S2Model`] also runs Audio8, the same dual-AR design at 0.6 B with its own
//! tensor names, a `wqkv` bias on the slow stack and an unscaled codebook sum
//! ([`crate::config::TensorLayout`], [`S2Config`]); see [`crate::audio8`].

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

/// Prompt lengths are rounded up to this for the prefill graph; the padding
/// sits after the real rows, which a causal mask keeps them from seeing.
const PREFILL_BUCKET: usize = 64;
/// Decode KV capacity granularity.
pub(crate) const DECODE_BUCKET: usize = 256;

/// Default per-chunk frame budget (`DEFAULT_MAX_NEW_TOKENS`).
pub const DEFAULT_MAX_NEW_TOKENS: usize = 1024;
/// Default UTF-8 byte budget of one text chunk (`DEFAULT_CHUNK_LENGTH`).
pub const DEFAULT_CHUNK_LENGTH: usize = 200;

/// How an S2 model is loaded.
#[derive(Clone, Debug, Default)]
pub struct S2Options {
    /// Run only the first `n` slow layers (tests against a truncated reference).
    pub layers: Option<usize>,
    /// Run only some slow layers, as one stage of a pipeline (a browser
    /// splits the model across Web Workers). `None`: all of them.
    pub stage: Option<Stage>,
    /// Keep the prefill graph after a prompt instead of freeing its weights.
    /// On backends that share parameter storage between graphs (Metal) this
    /// is free; on the CPU it holds a second copy of the backbone.
    pub keep_prefill: bool,
    /// How matmul weights are stored in the graphs. `None` reads
    /// `RLX_WEIGHT_STORAGE`, else `Auto` (keep the checkpoint's bf16 —
    /// the same products as f32 at half the bytes).
    pub weight_storage: Option<WeightStorage>,
}

/// One stage of a layer pipeline: the slow layers it runs, and whether it is
/// the last (the one that applies the final norm and the band head). A stage
/// that is not the last hands its residual rows to the next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    pub layers: std::ops::Range<usize>,
    pub last: bool,
}

/// A loaded S2-Pro checkpoint.
pub struct S2Model {
    pub cfg: S2Config,
    dir: PathBuf,
    ckpt: SafetensorsCheckpoint,
    device: Device,
    /// The slow layers this model runs (all of them unless it is a stage).
    layers: std::ops::Range<usize>,
    /// Ends in the final norm and the band head.
    last: bool,
    keep_prefill: bool,
    matrix_dtype: rlx_ir::DType,
    /// LM head rows for the semantic band then `<|im_end|>`, as `[dim, band+1]`.
    pub(crate) band_head: Vec<f32>,
    prefill: Option<(usize, CompiledGraph)>,
    decode: Option<(usize, CompiledGraph)>,
    fast: Option<CompiledGraph>,
    /// The whole fast decoder over one frame, sampling included, for the
    /// sampling settings it was built for ([`Self::fast_frame_fused`]).
    fused: Option<(FusedKey, CompiledGraph)>,
    /// The batched decode step for `(cap, n)`, and whether its caches are
    /// bound on the device ([`Self::generate_batch`]).
    decode_batch: Option<((usize, usize), CompiledGraph)>,
    /// The batched fused fast decoder for its settings and batch size.
    fused_batch: Option<((FusedKey, usize), CompiledGraph)>,
    /// The [`BatchState`] whose caches the batched graph holds on device.
    batch_resident: Option<u64>,
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
    /// Final-norm hidden state of the newest row (a stage that is not the
    /// last: its residual row, the next stage's input).
    pub hidden: Vec<f32>,
    /// Band logits (`band + <|im_end|>`) after the newest row (empty on a
    /// stage that is not the last).
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
        Self::from_checkpoint(cfg, ckpt, dir.to_path_buf(), device, opts)
    }

    /// A model from a parsed config and an open checkpoint (one held in
    /// memory: [`SafetensorsCheckpoint::from_bytes`]). `dir` names it in logs
    /// and labels its compiled graphs.
    pub fn from_checkpoint(
        cfg: S2Config,
        ckpt: SafetensorsCheckpoint,
        dir: PathBuf,
        device: Device,
        opts: S2Options,
    ) -> Result<Self> {
        let n = cfg.slow.n_layer;
        let (layers, last) = match &opts.stage {
            Some(st) => {
                ensure!(
                    st.layers.start <= st.layers.end && st.layers.end <= n,
                    "stage layers {:?} outside 0..{n}",
                    st.layers
                );
                (st.layers.clone(), st.last)
            }
            None => (0..opts.layers.unwrap_or(n).min(n), true),
        };
        let slow = cfg.layout.slow;
        let band_head = if last {
            Self::load_band_head(&cfg, &ckpt)?
        } else {
            Vec::new()
        };
        let storage = opts
            .weight_storage
            .or_else(WeightStorage::from_env)
            .unwrap_or_default();
        // A stage's own first layer (another stage's are not in its tensors).
        let probe = if layers.is_empty() {
            format!("{}layers.0.attention.wqkv.weight", cfg.layout.fast)
        } else {
            format!("{slow}layers.{}.attention.wqkv.weight", layers.start)
        };
        let matrix_dtype = storage.resolve(stored_dtype(&ckpt, &probe));
        Ok(Self {
            cfg,
            dir,
            ckpt,
            device,
            layers,
            last,
            // Metal maps one copy of each weight into every graph of the model, so
            // keeping the prefill graph alive costs activations, not weights — and
            // keeps those pages live for the decode graph to map instead of re-upload.
            keep_prefill: opts.keep_prefill || device == Device::Metal,
            matrix_dtype,
            band_head,
            prefill: None,
            decode: None,
            fast: None,
            fused: None,
            decode_batch: None,
            fused_batch: None,
            batch_resident: None,
            resident: None,
        })
    }

    /// LM head rows for the semantic band then `<|im_end|>`, `[dim, band+1]`.
    fn load_band_head(cfg: &S2Config, ckpt: &SafetensorsCheckpoint) -> Result<Vec<f32>> {
        let slow = cfg.layout.slow;
        let head_key = if cfg.tie_word_embeddings {
            format!("{slow}embeddings.weight")
        } else {
            format!("{slow}output.weight")
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
        Ok(band_head)
    }

    /// The slow layers this model runs.
    pub fn layers(&self) -> std::ops::Range<usize> {
        self.layers.clone()
    }

    /// Whether this model ends in the final norm and the band head.
    pub fn is_last(&self) -> bool {
        self.last
    }

    /// The whole slow stack, ending in the band head: what `generate` needs.
    fn is_whole(&self) -> bool {
        self.last && self.layers == (0..self.cfg.slow.n_layer)
    }

    /// Where a decode step's K/V rows start in its outputs: after the hidden
    /// row, and the band logits on the last stage.
    fn kv_base(&self) -> usize {
        if self.last { 2 } else { 1 }
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

    /// The checkpoint tensors read on the host rather than through a graph.
    pub(crate) fn host_tensor_keys(&self) -> [String; 3] {
        let l = self.cfg.layout;
        [
            format!("{}embeddings.weight", l.slow),
            l.codebook_embed.to_string(),
            format!("{}embeddings.weight", l.fast),
        ]
    }

    /// Input embeddings `[rows, dim]`: the token embedding, plus — for audio
    /// rows — every code's codebook embedding, the sum scaled by `1/√(N+1)`
    /// where the checkpoint asks for it.
    pub fn embed_rows(&self, rows: &[Row]) -> Result<Vec<f32>> {
        let dim = self.cfg.slow.dim;
        let nc = self.cfg.num_codebooks;
        let tokens: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let [embed_key, codebook_key, _] = self.host_tensor_keys();
        let tok = self.ckpt.load_tensor_rows_f32(&embed_key, &tokens, dim)?;
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
                .load_tensor_rows_f32(&codebook_key, &cb_ids, dim)?
        };
        let scale = if self.cfg.scale_codebook_sum {
            1.0 / ((nc + 1) as f32).sqrt()
        } else {
            1.0
        };
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

    /// The prefill graph over `seq` rows, uncompiled: the final norm's rows
    /// (a stage that is not the last: its residual rows), then each layer's
    /// K and V.
    pub(crate) fn prefill_builder(&self, seq: usize) -> (Builder, Vec<rlx_ir::NodeId>) {
        let s = &self.cfg.slow;
        let slow = self.cfg.layout.slow;
        let spec = self.slow_spec();
        let mut b = Builder::with_matrix_dtype("fish_s2_prefill", self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let x = b.input("embeds", &[1, seq, s.dim]);
        let cos = b.input("rope_cos", &[seq, s.head_dim / 2]);
        let sin = b.input("rope_sin", &[seq, s.head_dim / 2]);
        let mut h = x;
        let mut outs = vec![];
        for l in self.layers.clone() {
            let o = nn::block(
                &mut b,
                &spec,
                &format!("{slow}layers.{l}."),
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
        let hn = if self.last {
            b.rms_norm(
                h,
                &format!("{slow}norm.weight"),
                s.dim,
                self.cfg.norm_eps,
                false,
            )
        } else {
            h
        };
        outs.insert(0, hn);
        (b, outs)
    }

    fn build_decode(&self, cap: usize) -> Result<CompiledGraph> {
        let (b, outs) = self.decode_builder(cap);
        self.compile(b, outs, rlx_flow::CompileProfile::llama32_decode())
    }

    /// The decode step against a `cap`-row cache, uncompiled: the hidden
    /// state, the band logits (on the last stage), then each layer's new K
    /// and V row. The cache inputs are numbered within the stage.
    pub(crate) fn decode_builder(&self, cap: usize) -> (Builder, Vec<rlx_ir::NodeId>) {
        let s = &self.cfg.slow;
        let slow = self.cfg.layout.slow;
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
        for (j, l) in self.layers.clone().enumerate() {
            let past_k = b.input(&format!("past_k_{j}"), &[1, cap, kvw]);
            let past_v = b.input(&format!("past_v_{j}"), &[1, cap, kvw]);
            let keys = Keys::Cached {
                past_k,
                past_v,
                past: cap,
                mask,
            };
            let o = nn::block(
                &mut b,
                &spec,
                &format!("{slow}layers.{l}."),
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
        if !self.last {
            let mut outs = vec![h];
            outs.extend(kv);
            return (b, outs);
        }
        let hn = b.rms_norm(
            h,
            &format!("{slow}norm.weight"),
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
        let t = crate::clock();
        let mut g = rlx_runtime::with_param_sharing_scope(&label, || {
            rlx_core::flow_bridge::compile_graph_with_profile(device, b.g, &profile)
        })?;
        let t_compile = crate::elapsed(t);
        let t = crate::clock();
        nn::upload(&mut g, b.params, &self.ckpt)?;
        if rlx_ir::env::flag("RLX_FISH_VERBOSE") {
            eprintln!(
                "s2: {name}: compile {:.1} s, weights {:.1} s",
                t_compile.as_secs_f32(),
                crate::elapsed(t).as_secs_f32()
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
        let embeds = self.embed_rows(rows)?;
        Ok(self.prefill_embeds(embeds, rows.len(), cap)?.0)
    }

    /// [`Self::prefill`] from the rows' input `[t, dim]`: their embeddings,
    /// or the previous pipeline stage's output. Also returns this stage's
    /// output rows `[t, dim]` — the next stage's input (on the last stage,
    /// the final-norm rows).
    pub fn prefill_embeds(
        &mut self,
        embeds: Vec<f32>,
        t: usize,
        cap: usize,
    ) -> Result<(SlowState, Vec<f32>)> {
        let job = self.prefill_prepare(embeds, t, cap)?;
        let outs = {
            let g = &mut self.prefill.as_mut().expect("prefill graph").1;
            g.run(&job.inputs())
        };
        Ok(self.prefill_finish(outs, t, cap))
    }

    /// [`Self::prefill_embeds`] with an async run (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn prefill_embeds_async(
        &mut self,
        embeds: Vec<f32>,
        t: usize,
        cap: usize,
    ) -> Result<(SlowState, Vec<f32>)> {
        let job = self.prefill_prepare(embeds, t, cap)?;
        let outs = {
            let g = &mut self.prefill.as_mut().expect("prefill graph").1;
            g.run_async(&job.inputs())
                .await
                .map_err(anyhow::Error::msg)?
        };
        Ok(self.prefill_finish(outs, t, cap))
    }

    /// The prefill graph for `t` rows, and its inputs.
    fn prefill_prepare(
        &mut self,
        mut embeds: Vec<f32>,
        t: usize,
        cap: usize,
    ) -> Result<PrefillJob> {
        ensure!(t > 0, "empty prompt");
        ensure!(
            cap > t,
            "cache capacity {cap} does not exceed the prompt {t}"
        );
        let s = self.cfg.slow.clone();
        ensure!(
            embeds.len() == t * s.dim,
            "{} input values for {t} rows of {}",
            embeds.len(),
            s.dim
        );
        let seq = t.div_ceil(PREFILL_BUCKET) * PREFILL_BUCKET;
        if self.prefill.as_ref().map(|p| p.0) != Some(seq) {
            self.prefill = None;
            self.prefill = Some((seq, self.build_prefill(seq)?));
        }
        embeds.resize(seq * s.dim, 0.0);
        let positions: Vec<usize> = (0..seq).collect();
        let (cos, sin) = self.rope(&positions, s.head_dim);
        Ok(PrefillJob { embeds, cos, sin })
    }

    /// The cache and output rows from a prefill run's outputs.
    fn prefill_finish(
        &mut self,
        outs: Vec<Vec<f32>>,
        t: usize,
        cap: usize,
    ) -> (SlowState, Vec<f32>) {
        if !self.keep_prefill {
            self.prefill = None;
        }
        let s = &self.cfg.slow;
        let kvw = Self::kv_width(s);
        let layers = self.layers.len();
        let mut k = Vec::with_capacity(layers);
        let mut v = Vec::with_capacity(layers);
        for l in 0..layers {
            let mut kb = vec![0f32; cap * kvw];
            let mut vb = vec![0f32; cap * kvw];
            kb[..t * kvw].copy_from_slice(&outs[1 + 2 * l][..t * kvw]);
            vb[..t * kvw].copy_from_slice(&outs[2 + 2 * l][..t * kvw]);
            k.push(kb);
            v.push(vb);
        }
        let mut rows_out = outs.into_iter().next().expect("prefill output");
        rows_out.truncate(t * s.dim);
        let hidden = rows_out[(t - 1) * s.dim..].to_vec();
        let logits = if self.last {
            self.band_logits(&hidden)
        } else {
            Vec::new()
        };
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let state = SlowState {
            len: t,
            cap,
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            host_stale: false,
            k,
            v,
            hidden,
            logits,
        };
        (state, rows_out)
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
        let (cos, sin, mask) = self.step_prepare(state)?;
        let pos = state.len;
        let kvw = Self::kv_width(&s);
        let names: Vec<(String, String)> = (0..self.layers.len())
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();
        self.make_resident(state, &names);
        let small: Vec<(&str, &[f32])> = vec![
            ("embeds", embed),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
            ("mask", &mask),
        ];
        if self.resident == Some(state.id) {
            let last = self.last;
            let g = &mut self.decode.as_mut().expect("decode graph").1;
            let read: &[usize] = if last { &[0, 1] } else { &[0] };
            let mut outs = g.run_read_outputs(&small, Some(read));
            ensure!(g.feed_kv_row(0, pos, kvw), "resident KV row feed failed");
            state.host_stale = true;
            state.logits = if last {
                outs.swap_remove(1)
            } else {
                Vec::new()
            };
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
        self.step_absorb(state, outs);
        Ok(())
    }

    /// [`Self::step`] with an async run (WebGPU in a browser). The cache
    /// stays on the host and goes in with every step.
    #[cfg(target_family = "wasm")]
    pub async fn step_async(&mut self, state: &mut SlowState, embed: &[f32]) -> Result<()> {
        let (cos, sin, mask) = self.step_prepare(state)?;
        let pos = state.len;
        let kvw = Self::kv_width(&self.cfg.slow);
        let names: Vec<(String, String)> = (0..self.layers.len())
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();
        // The cache on the device, as `step` keeps it: bound once, each new
        // row folded in there; only the hidden row and logits come back.
        self.make_resident(state, &names);
        let small: Vec<(&str, &[f32])> = vec![
            ("embeds", embed),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
            ("mask", &mask),
        ];
        if self.resident == Some(state.id) {
            let last = self.last;
            let g = &mut self.decode.as_mut().expect("decode graph").1;
            let read: &[usize] = if last { &[0, 1] } else { &[0] };
            let mut outs = g
                .run_read_outputs_async(&small, Some(read))
                .await
                .map_err(anyhow::Error::msg)?;
            ensure!(g.feed_kv_row(0, pos, kvw), "resident KV row feed failed");
            state.host_stale = true;
            state.logits = if last {
                outs.swap_remove(1)
            } else {
                Vec::new()
            };
            state.hidden = outs.swap_remove(0);
            state.len += 1;
            return Ok(());
        }
        ensure!(
            !state.host_stale,
            "this cache lives in another decode graph"
        );
        let mut inputs = small;
        for (l, (kn, vn)) in names.iter().enumerate() {
            inputs.push((kn, &state.k[l]));
            inputs.push((vn, &state.v[l]));
        }
        let outs = {
            let g = &mut self.decode.as_mut().expect("decode graph").1;
            g.run_async(&inputs).await.map_err(anyhow::Error::msg)?
        };
        self.step_absorb(state, outs);
        Ok(())
    }

    /// Bind `state`'s cache to the decode graph's inputs on the device, where
    /// the backend keeps inputs there (once per cache).
    fn make_resident(&mut self, state: &SlowState, names: &[(String, String)]) {
        if self.resident == Some(state.id)
            || state.host_stale
            || !rlx_core::gpu_kv::device_supports_resident_kv_row_feed(self.device)
        {
            return;
        }
        let base = self.kv_base();
        let g = &mut self.decode.as_mut().expect("decode graph").1;
        let mut ok = true;
        for (l, (kn, vn)) in names.iter().enumerate() {
            ok &= g.bind_gpu_handle(kn, &state.k[l]) && g.register_kv_row_feed(kn, base + 2 * l);
            ok &=
                g.bind_gpu_handle(vn, &state.v[l]) && g.register_kv_row_feed(vn, base + 1 + 2 * l);
        }
        self.resident = ok.then_some(state.id);
    }

    /// The decode graph for `state`'s capacity, and the step's RoPE row and
    /// mask.
    fn step_prepare(&mut self, state: &SlowState) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        ensure!(state.len < state.cap, "KV cache full ({} rows)", state.cap);
        if self.decode.as_ref().map(|d| d.0) != Some(state.cap) {
            self.decode = None;
            self.resident = None;
            self.decode = Some((state.cap, self.build_decode(state.cap)?));
        }
        let pos = state.len;
        let (cos, sin) = self.rope(&[pos], self.cfg.slow.head_dim);
        let mut mask = vec![0f32; state.cap + 1];
        mask[..pos].fill(1.0);
        mask[state.cap] = 1.0;
        Ok((cos, sin, mask))
    }

    /// A host-cache step's outputs into `state`.
    fn step_absorb(&self, state: &mut SlowState, outs: Vec<Vec<f32>>) {
        let kvw = Self::kv_width(&self.cfg.slow);
        let base = self.kv_base();
        let pos = state.len;
        for l in 0..self.layers.len() {
            state.k[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[base + 2 * l]);
            state.v[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[base + 1 + 2 * l]);
        }
        state.hidden = outs[0].clone();
        state.logits = if self.last {
            outs[1].clone()
        } else {
            Vec::new()
        };
        state.len += 1;
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
        let fast = self.cfg.layout.fast;
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
                &format!("{fast}layers.{l}."),
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
            &format!("{fast}norm.weight"),
            f.dim,
            self.cfg.norm_eps,
            false,
        );
        let logits = b.linear(
            hn,
            &format!("{fast}output.weight"),
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
        let mut run = self.fast_start(hidden, main)?;
        while !run.done() {
            let ins = run.inputs(&self.cfg);
            let refs: Vec<(&str, &[f32])> = ins
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let outs = self.fast.as_mut().expect("fast graph").run(&refs);
            run.absorb(self, outs, params, rng, &mut on_logits)?;
        }
        Ok(run.codes)
    }

    /// [`Self::fast_frame`] with async runs (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn fast_frame_async(
        &mut self,
        hidden: &[f32],
        main: u32,
        params: SampleParams,
        rng: &mut Rng,
    ) -> Result<Vec<u32>> {
        let mut run = self.fast_start(hidden, main)?;
        while !run.done() {
            let ins = run.inputs(&self.cfg);
            let refs: Vec<(&str, &[f32])> = ins
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let outs = {
                let g = self.fast.as_mut().expect("fast graph");
                g.run_async(&refs).await.map_err(anyhow::Error::msg)?
            };
            run.absorb(self, outs, params, rng, &mut |_, _| {})?;
        }
        Ok(run.codes)
    }

    /// [`Self::fast_frame`] as one graph run: the ten fast-decoder positions
    /// unrolled, each code sampled in the graph (top-k / top-p on the raw
    /// logits, the temperature after, the exponential race against draws the
    /// host makes with [`sampling::draws`]), each next position reading the
    /// sampled code's embedding. The same codes as `fast_frame` from the same
    /// RNG, in one submission instead of ten — what a GPU, where every run
    /// waits on a read-back, spends most of a frame on.
    pub fn fast_frame_fused(
        &mut self,
        hidden: &[f32],
        main: u32,
        params: SampleParams,
        rng: &mut Rng,
    ) -> Result<Vec<u32>> {
        let (inputs, c0) = self.fused_prepare(hidden, main, params, rng)?;
        let refs: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let outs = self.fused.as_mut().expect("fused graph").1.run(&refs);
        Ok(Self::fused_finish(c0, &outs[0]))
    }

    /// [`Self::fast_frame_fused`] with an async run (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn fast_frame_fused_async(
        &mut self,
        hidden: &[f32],
        main: u32,
        params: SampleParams,
        rng: &mut Rng,
    ) -> Result<Vec<u32>> {
        let (inputs, c0) = self.fused_prepare(hidden, main, params, rng)?;
        let refs: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let outs = {
            let g = &mut self.fused.as_mut().expect("fused graph").1;
            g.run_async(&refs).await.map_err(anyhow::Error::msg)?
        };
        Ok(Self::fused_finish(c0, &outs[0]))
    }

    fn fused_prepare(
        &mut self,
        hidden: &[f32],
        main: u32,
        params: SampleParams,
        rng: &mut Rng,
    ) -> Result<(Vec<(String, Vec<f32>)>, u32)> {
        let key = FusedKey::of(params);
        if self.fused.as_ref().map(|f| f.0) != Some(key) {
            self.fused = None;
            let (b, outs) = self.fused_builder(params);
            let g = self.compile(b, outs, rlx_flow::CompileProfile::llama32_decode())?;
            self.fused = Some((key, g));
        }
        let c0 = main
            .saturating_sub(self.cfg.semantic_begin)
            .min(self.cfg.codebook_size as u32 - 1);
        let mut inputs = vec![
            ("hidden".to_string(), hidden.to_vec()),
            ("c0".to_string(), vec![c0 as f32]),
        ];
        if params.top_k > 1 {
            // `-ln E` per code and rank, in the order `fast_frame` draws them.
            let mut noise = Vec::with_capacity((self.cfg.num_codebooks - 1) * params.top_k);
            for _ in 1..self.cfg.num_codebooks {
                noise.extend(
                    sampling::draws(params, rng)
                        .iter()
                        .map(|e| -(e.ln()) as f32),
                );
            }
            inputs.push(("noise".to_string(), noise));
        }
        Ok((inputs, c0))
    }

    fn fused_finish(c0: u32, codes: &[f32]) -> Vec<u32> {
        std::iter::once(c0)
            .chain(codes.iter().map(|&c| c as u32))
            .collect()
    }

    /// The fused frame graph for `params`: inputs `hidden` `[1, 1, dim]`,
    /// `c0` `[1]` and (sampling) `noise` `[num_codebooks - 1, top_k]`
    /// (`-ln` of the draws); output the codes after `c0`, `[num_codebooks - 1]`.
    pub(crate) fn fused_builder(&self, params: SampleParams) -> (Builder, Vec<rlx_ir::NodeId>) {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::op::CmpOp;
        use rlx_ir::{Op, Shape};
        let f32s = |d: &[usize]| Shape::new(d, rlx_ir::DType::F32);
        let f = &self.cfg.fast;
        let fast = self.cfg.layout.fast;
        let nc = self.cfg.num_codebooks;
        let book = self.cfg.codebook_size;
        let spec = stack_spec(f, f.ffn, self.cfg.norm_eps);
        let k = params.top_k.max(1);
        let mut b = Builder::with_matrix_dtype("fish_s2_fast_frame", self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let hidden = b.input("hidden", &[1, 1, f.dim]);
        let c0 = b.input("c0", &[1]);
        let noise = (k > 1).then(|| b.input("noise", &[nc - 1, k]));
        let [_, _, embed_key] = self.host_tensor_keys();
        let table = b.weight(&embed_key, &[book, f.dim], nn::TensorOp::Raw);
        let embed = |b: &mut Builder, code: rlx_ir::NodeId| {
            let row = b.g.gather_(table, code, 0);
            b.g.reshape_(row, vec![1, 1, f.dim as i64])
        };
        let mut x = hidden;
        let mut ks: Vec<Vec<rlx_ir::NodeId>> = vec![Vec::new(); f.n_layer];
        let mut vs: Vec<Vec<rlx_ir::NodeId>> = vec![Vec::new(); f.n_layer];
        let mut codes = Vec::new();
        let mut code = c0;
        for pos in 0..nc {
            if pos > 0 {
                x = embed(&mut b, code);
            }
            let (cos, sin) = nn::rope_tables(&[pos], f.head_dim, self.cfg.rope_base, true);
            let cos = b.data(&format!("fused.rope_cos.{pos}"), cos, &[1, f.head_dim / 2]);
            let sin = b.data(&format!("fused.rope_sin.{pos}"), sin, &[1, f.head_dim / 2]);
            let mut h = x;
            for l in 0..f.n_layer {
                let keys = if pos == 0 {
                    Keys::Own(MaskKind::Causal)
                } else {
                    let past_k = b.g.concat_(ks[l].clone(), 1);
                    let past_v = b.g.concat_(vs[l].clone(), 1);
                    let mask = b.data(
                        &format!("fused.mask.{pos}"),
                        vec![1.0; pos + 1],
                        &[1, pos + 1],
                    );
                    Keys::Cached {
                        past_k,
                        past_v,
                        past: pos,
                        mask,
                    }
                };
                let o = nn::block(
                    &mut b,
                    &spec,
                    &format!("{fast}layers.{l}."),
                    h,
                    1,
                    cos,
                    sin,
                    keys,
                );
                h = o.x;
                ks[l].push(o.k);
                vs[l].push(o.v);
            }
            if pos == 0 {
                continue;
            }
            let hn = b.rms_norm(
                h,
                &format!("{fast}norm.weight"),
                f.dim,
                self.cfg.norm_eps,
                false,
            );
            let logits = b.linear(hn, &format!("{fast}output.weight"), f.dim, book);
            let logits = b.g.reshape_(logits, vec![1, book as i64]);
            // Candidates by rank (ties to the lower index, as the host sorts).
            let idx = b.g.add_node(Op::TopK { k }, vec![logits], f32s(&[1, k]));
            let pick = if k == 1 {
                b.g.reshape_(idx, vec![1])
            } else {
                // The top-p cut on the raw logits' softmax (S2), the first kept
                // always; then the race: argmax of v / t − ln E over the kept.
                let probs =
                    b.g.add_node(Op::Softmax { axis: -1 }, vec![logits], f32s(&[1, book]));
                let pk = b.g.gather_(probs, idx, 1);
                let pk = b.g.reshape_(pk, vec![1, k as i64]);
                let cum = b.g.add_node(
                    Op::Cumsum {
                        axis: 1,
                        exclusive: false,
                    },
                    vec![pk],
                    f32s(&[1, k]),
                );
                let top_p = b.data(&format!("fused.top_p.{pos}"), vec![params.top_p], &[1]);
                let le = b.g.add_node(
                    Op::Compare(CmpOp::Le),
                    vec![cum, top_p],
                    Shape::new(&[1, k], rlx_ir::DType::Bool),
                );
                let le = b.g.cast(le, rlx_ir::DType::F32);
                let mut first = vec![0f32; k];
                first[0] = 1.0;
                let first = b.data(&format!("fused.first.{pos}"), first, &[1, k]);
                // keep = le OR first (both 0/1).
                let both = b.g.mul(le, first);
                let either = b.g.add(le, first);
                let keep = b.g.sub(either, both);
                let v = b.g.gather_(logits, idx, 1);
                let v = b.g.reshape_(v, vec![1, k as i64]);
                let inv_t = b.data(
                    &format!("fused.inv_t.{pos}"),
                    vec![1.0 / params.temperature.max(1e-5)],
                    &[1],
                );
                let scaled = b.g.mul(v, inv_t);
                let row = b.g.narrow_(noise.expect("noise input"), 0, pos - 1, 1);
                let score = b.g.add(scaled, row);
                // Dropped candidates far below any kept one.
                let big = b.data(&format!("fused.big.{pos}"), vec![1e30], &[1]);
                let one = b.data(&format!("fused.one.{pos}"), vec![1.0], &[1]);
                let off = b.g.sub(keep, one);
                let off = b.g.mul(off, big);
                let masked = b.g.add(score, off);
                let j = b.g.add_node(
                    Op::ArgMax {
                        axis: 1,
                        keep_dim: false,
                    },
                    vec![masked],
                    f32s(&[1]),
                );
                let flat = b.g.reshape_(idx, vec![k as i64]);
                b.g.gather_(flat, j, 0)
            };
            code = b.g.reshape_(pick, vec![1]);
            codes.push(code);
        }
        let out = b.g.concat_(codes, 0);
        (b, vec![out])
    }

    fn fast_start(&mut self, hidden: &[f32], main: u32) -> Result<FastRun> {
        if self.fast.is_none() {
            self.fast = Some(self.build_fast()?);
        }
        let f = &self.cfg.fast;
        let nc = self.cfg.num_codebooks;
        let kvw = Self::kv_width(f);
        let c0 = main
            .saturating_sub(self.cfg.semantic_begin)
            .min(self.cfg.codebook_size as u32 - 1);
        Ok(FastRun {
            pos: 0,
            nc,
            k: vec![vec![0f32; nc * kvw]; f.n_layer],
            v: vec![vec![0f32; nc * kvw]; f.n_layer],
            x: hidden.to_vec(),
            codes: vec![c0],
        })
    }

    // ── a batch of sequences ────────────────────────────────────────────
    //
    // Several utterances decode together: every matmul takes one row per
    // sequence, so a step reads each weight once for the batch (decode is
    // bound by reading weights, not by arithmetic). Attention stays per
    // sequence. The caches share one slot layout: each prompt's prefill is
    // placed to end at the same slot, and every step writes all sequences'
    // new rows at the same slot; RoPE uses each sequence's own position.

    /// The batched decode step for `n` sequences against `cap`-slot caches:
    /// inputs `embeds` `[1, n, dim]` (this stage's input rows), `rope_cos` /
    /// `rope_sin` `[n, half]`, `mask` `[n, cap + 1]`, `past_{k,v}_{l}`
    /// `[n, cap, kv]`; outputs the stage's rows `[1, n, dim]` (final-norm on
    /// the last stage), the band logits `[n, band]` (last stage only), then
    /// each layer's new K and V rows, `[1, n, kv]` (row `i` = sequence `i`).
    pub(crate) fn decode_batch_builder(
        &self,
        cap: usize,
        n: usize,
    ) -> (Builder, Vec<rlx_ir::NodeId>) {
        use rlx_ir::infer::GraphExt;
        let s = &self.cfg.slow;
        let slow = self.cfg.layout.slow;
        let spec = self.slow_spec();
        let kvw = Self::kv_width(s);
        let mut b = Builder::with_matrix_dtype("fish_s2_decode_batch", self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let x = b.input("embeds", &[1, n, s.dim]);
        let cos = b.input("rope_cos", &[n, s.head_dim / 2]);
        let sin = b.input("rope_sin", &[n, s.head_dim / 2]);
        let mask = b.input("mask", &[n, cap + 1]);
        let mut h = x;
        let mut kv = vec![];
        for (j, l) in self.layers.clone().enumerate() {
            let keys = Keys::Batched {
                n,
                keys: Box::new(Keys::Cached {
                    past_k: b.input(&format!("past_k_{j}"), &[n, cap, kvw]),
                    past_v: b.input(&format!("past_v_{j}"), &[n, cap, kvw]),
                    past: cap,
                    mask,
                }),
            };
            let o = nn::block(
                &mut b,
                &spec,
                &format!("{slow}layers.{l}."),
                h,
                n,
                cos,
                sin,
                keys,
            );
            h = o.x;
            kv.push(o.k);
            kv.push(o.v);
        }
        if !self.last {
            let mut outs = vec![h];
            outs.extend(kv);
            return (b, outs);
        }
        let hn = b.rms_norm(
            h,
            &format!("{slow}norm.weight"),
            s.dim,
            self.cfg.norm_eps,
            false,
        );
        let nb = self.band_len();
        let head = b.data("fish.band_head", self.band_head.clone(), &[s.dim, nb]);
        let flat = b.g.reshape_(hn, vec![n as i64, s.dim as i64]);
        let logits = b.g.mm(flat, head);
        let mut outs = vec![hn, logits];
        outs.extend(kv);
        (b, outs)
    }

    /// Decode the sequences of `states` (each prefilled alone, same capacity)
    /// together from here on: their caches placed in one `[n, cap, kv]`
    /// layout per layer, every prompt ending at the same slot.
    pub fn batch_join(&mut self, states: Vec<SlowState>) -> Result<BatchState> {
        let n = states.len();
        ensure!(n > 0, "a batch of no sequences");
        let cap = states[0].cap;
        ensure!(
            states.iter().all(|s| s.cap == cap && !s.host_stale),
            "batched caches need one capacity and their rows on the host"
        );
        let kvw = Self::kv_width(&self.cfg.slow);
        let lens: Vec<usize> = states.iter().map(|s| s.len).collect();
        let top = *lens.iter().max().expect("a sequence");
        ensure!(top < cap, "a prompt fills the {cap}-row cache");
        let layers = self.layers.len();
        let mut k = vec![vec![0f32; n * cap * kvw]; layers];
        let mut v = vec![vec![0f32; n * cap * kvw]; layers];
        let (mut hidden, mut logits) = (Vec::new(), Vec::new());
        for (i, st) in states.iter().enumerate() {
            let at = (i * cap + top - st.len) * kvw;
            let len = st.len * kvw;
            for l in 0..layers {
                k[l][at..at + len].copy_from_slice(&st.k[l][..len]);
                v[l][at..at + len].copy_from_slice(&st.v[l][..len]);
            }
            hidden.extend_from_slice(&st.hidden);
            logits.extend_from_slice(&st.logits);
        }
        let key = (cap, n);
        if self.decode_batch.as_ref().map(|d| d.0) != Some(key) {
            // The one-sequence step graphs are not needed while batching;
            // their activations are the difference between the footprints.
            self.decode = None;
            self.fused = None;
            self.resident = None;
            self.decode_batch = None;
            let (b, outs) = self.decode_batch_builder(cap, n);
            let g = self.compile(b, outs, rlx_flow::CompileProfile::llama32_decode())?;
            self.decode_batch = Some((key, g));
        }
        // The caches bound on the device; each step folds row `i` of the new
        // K/V outputs into slot `i · cap + slot` there.
        let base = self.kv_base();
        let on_device = rlx_core::gpu_kv::device_supports_resident_kv_row_feed(self.device) && {
            let g = &mut self.decode_batch.as_mut().expect("batch graph").1;
            let mut ok = true;
            for l in 0..layers {
                let (kn, vn) = (format!("past_k_{l}"), format!("past_v_{l}"));
                ok &= g.bind_gpu_handle(&kn, &k[l]) && g.register_kv_row_feed(&kn, base + 2 * l);
                ok &=
                    g.bind_gpu_handle(&vn, &v[l]) && g.register_kv_row_feed(&vn, base + 1 + 2 * l);
            }
            ok
        };
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.batch_resident = on_device.then_some(id);
        Ok(BatchState {
            n,
            cap,
            top,
            lens,
            steps: 0,
            id,
            k,
            v,
            hidden,
            logits,
        })
    }

    /// The step's RoPE rows and masks.
    fn batch_prepare(&self, st: &BatchState) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let slot = st.top + st.steps;
        ensure!(slot < st.cap, "KV cache full ({} rows)", st.cap);
        let positions: Vec<usize> = st.lens.iter().map(|&t| t + st.steps).collect();
        let (cos, sin) = self.rope(&positions, self.cfg.slow.head_dim);
        let mut mask = vec![0f32; st.n * (st.cap + 1)];
        for (m, &t) in mask.chunks_exact_mut(st.cap + 1).zip(&st.lens) {
            m[st.top - t..slot].fill(1.0);
            m[st.cap] = 1.0;
        }
        Ok((cos, sin, mask))
    }

    fn batch_check(&self, st: &BatchState, x: &[f32]) -> Result<bool> {
        ensure!(
            x.len() == st.n * self.cfg.slow.dim,
            "a batched step takes one row per sequence"
        );
        ensure!(
            self.decode_batch.as_ref().map(|d| d.0) == Some((st.cap, st.n)),
            "this batch's graph was dropped"
        );
        Ok(self.batch_resident == Some(st.id))
    }

    /// One row more for every sequence of `st`: `x` is this stage's input
    /// `[n, dim]` (the rows' embeddings, or the previous stage's output).
    pub fn batch_step(&mut self, st: &mut BatchState, x: &[f32]) -> Result<()> {
        let resident = self.batch_check(st, x)?;
        let (cos, sin, mask) = self.batch_prepare(st)?;
        let read: &[usize] = if self.last { &[0, 1] } else { &[0] };
        let mut feeds: Vec<(&str, &[f32])> = vec![
            ("embeds", x),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
            ("mask", &mask),
        ];
        let names = self.batch_kv_names();
        let g = &mut self.decode_batch.as_mut().expect("batch graph").1;
        let outs = if resident {
            g.run_read_outputs(&feeds, Some(read))
        } else {
            for ((kn, vn), (k, v)) in names.iter().zip(st.k.iter().zip(&st.v)) {
                feeds.push((kn, k));
                feeds.push((vn, v));
            }
            g.run(&feeds)
        };
        self.batch_absorb(st, outs, resident)
    }

    /// [`Self::batch_step`] with an async run (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn batch_step_async(&mut self, st: &mut BatchState, x: &[f32]) -> Result<()> {
        let resident = self.batch_check(st, x)?;
        let (cos, sin, mask) = self.batch_prepare(st)?;
        let read: &[usize] = if self.last { &[0, 1] } else { &[0] };
        let mut feeds: Vec<(&str, &[f32])> = vec![
            ("embeds", x),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
            ("mask", &mask),
        ];
        let names = self.batch_kv_names();
        let g = &mut self.decode_batch.as_mut().expect("batch graph").1;
        let outs = if resident {
            g.run_read_outputs_async(&feeds, Some(read))
                .await
                .map_err(anyhow::Error::msg)?
        } else {
            for ((kn, vn), (k, v)) in names.iter().zip(st.k.iter().zip(&st.v)) {
                feeds.push((kn, k));
                feeds.push((vn, v));
            }
            g.run_async(&feeds).await.map_err(anyhow::Error::msg)?
        };
        self.batch_absorb(st, outs, resident)
    }

    fn batch_kv_names(&self) -> Vec<(String, String)> {
        (0..self.layers.len())
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect()
    }

    fn batch_absorb(
        &mut self,
        st: &mut BatchState,
        mut outs: Vec<Vec<f32>>,
        resident: bool,
    ) -> Result<()> {
        let kvw = Self::kv_width(&self.cfg.slow);
        let slot = st.top + st.steps;
        let base = self.kv_base();
        if resident {
            let g = &mut self.decode_batch.as_mut().expect("batch graph").1;
            for i in 0..st.n {
                ensure!(
                    g.feed_kv_row(i, i * st.cap + slot, kvw),
                    "resident KV row feed failed"
                );
            }
        } else {
            for (l, (k, v)) in st.k.iter_mut().zip(st.v.iter_mut()).enumerate() {
                for i in 0..st.n {
                    let at = (i * st.cap + slot) * kvw;
                    k[at..at + kvw].copy_from_slice(&outs[base + 2 * l][i * kvw..(i + 1) * kvw]);
                    v[at..at + kvw]
                        .copy_from_slice(&outs[base + 1 + 2 * l][i * kvw..(i + 1) * kvw]);
                }
            }
        }
        outs.truncate(base);
        st.logits = if self.last {
            outs.swap_remove(1)
        } else {
            Vec::new()
        };
        st.hidden = outs.swap_remove(0);
        st.steps += 1;
        Ok(())
    }

    /// The fused fast decoder ([`Self::fused_builder`]) for `n` sequences at
    /// once: inputs `hidden` `[1, n, dim]`, `c0` `[n]` and (sampling) `noise`
    /// `[num_codebooks - 1, n, top_k]`; output the codes after `c0`, position
    /// by position, `[(num_codebooks - 1) · n]`.
    pub(crate) fn fused_batch_builder(
        &self,
        params: SampleParams,
        n: usize,
    ) -> (Builder, Vec<rlx_ir::NodeId>) {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::op::CmpOp;
        use rlx_ir::{Op, Shape};
        let f32s = |d: &[usize]| Shape::new(d, rlx_ir::DType::F32);
        let f = &self.cfg.fast;
        let fast = self.cfg.layout.fast;
        let nc = self.cfg.num_codebooks;
        let book = self.cfg.codebook_size;
        let spec = stack_spec(f, f.ffn, self.cfg.norm_eps);
        let k = params.top_k.max(1);
        let mut b = Builder::with_matrix_dtype("fish_s2_fast_frame_batch", self.matrix_dtype);
        b.native_gqa = nn::native_gqa(self.device);
        let hidden = b.input("hidden", &[1, n, f.dim]);
        let c0 = b.input("c0", &[n]);
        let noise = (k > 1).then(|| b.input("noise", &[nc - 1, n, k]));
        let [_, _, embed_key] = self.host_tensor_keys();
        let table = b.weight(&embed_key, &[book, f.dim], nn::TensorOp::Raw);
        // Row `i`'s offset into a flattened `[n, w]`.
        let offsets = |b: &mut Builder, w: usize, name: &str| {
            b.data(name, (0..n).map(|i| (i * w) as f32).collect(), &[n, 1])
        };
        let book_off = offsets(&mut b, book, "fused.batch.book_off");
        let k_off = offsets(&mut b, k, "fused.batch.k_off");
        let mut x = hidden;
        // Per layer, per row: that row's K/V rows so far.
        let kvw = Self::kv_width(f) as i64;
        // Per layer: the K/V rows so far, `[n, 1, kv]` each.
        let mut ks: Vec<Vec<rlx_ir::NodeId>> = vec![Vec::new(); f.n_layer];
        let mut vs: Vec<Vec<rlx_ir::NodeId>> = vec![Vec::new(); f.n_layer];
        let mut codes = Vec::new();
        let mut code = c0;
        for pos in 0..nc {
            if pos > 0 {
                let rows = b.g.gather_(table, code, 0);
                x = b.g.reshape_(rows, vec![1, n as i64, f.dim as i64]);
            }
            let (cos, sin) = nn::rope_tables(&vec![pos; n], f.head_dim, self.cfg.rope_base, true);
            let cos = b.data(
                &format!("fused.batch.rope_cos.{pos}"),
                cos,
                &[n, f.head_dim / 2],
            );
            let sin = b.data(
                &format!("fused.batch.rope_sin.{pos}"),
                sin,
                &[n, f.head_dim / 2],
            );
            let mut h = x;
            for l in 0..f.n_layer {
                let keys = if pos == 0 {
                    Keys::Own(MaskKind::Causal)
                } else {
                    let past_k = b.g.concat_(ks[l].clone(), 1);
                    let past_v = b.g.concat_(vs[l].clone(), 1);
                    let mask = b.data(
                        &format!("fused.batch.mask.{pos}"),
                        vec![1.0; n * (pos + 1)],
                        &[n, pos + 1],
                    );
                    Keys::Cached {
                        past_k,
                        past_v,
                        past: pos,
                        mask,
                    }
                };
                let keys = Keys::Batched {
                    n,
                    keys: Box::new(keys),
                };
                let o = nn::block(
                    &mut b,
                    &spec,
                    &format!("{fast}layers.{l}."),
                    h,
                    n,
                    cos,
                    sin,
                    keys,
                );
                h = o.x;
                ks[l].push(b.g.reshape_(o.k, vec![n as i64, 1, kvw]));
                vs[l].push(b.g.reshape_(o.v, vec![n as i64, 1, kvw]));
            }
            if pos == 0 {
                continue;
            }
            let hn = b.rms_norm(
                h,
                &format!("{fast}norm.weight"),
                f.dim,
                self.cfg.norm_eps,
                false,
            );
            let logits = b.linear(hn, &format!("{fast}output.weight"), f.dim, book);
            let logits = b.g.reshape_(logits, vec![n as i64, book as i64]);
            let idx = b.g.add_node(Op::TopK { k }, vec![logits], f32s(&[n, k]));
            // Flat indices of each row's candidates, for gathers over `[n·book]`.
            let flat_idx = b.g.add(idx, book_off);
            let flat_idx = b.g.reshape_(flat_idx, vec![(n * k) as i64]);
            let pick = if k == 1 {
                b.g.reshape_(idx, vec![n as i64])
            } else {
                let probs =
                    b.g.add_node(Op::Softmax { axis: -1 }, vec![logits], f32s(&[n, book]));
                let probs = b.g.reshape_(probs, vec![(n * book) as i64]);
                let pk = b.g.gather_(probs, flat_idx, 0);
                let pk = b.g.reshape_(pk, vec![n as i64, k as i64]);
                let cum = b.g.add_node(
                    Op::Cumsum {
                        axis: 1,
                        exclusive: false,
                    },
                    vec![pk],
                    f32s(&[n, k]),
                );
                let top_p = b.data(&format!("fused.top_p.{pos}"), vec![params.top_p], &[1]);
                let le = b.g.add_node(
                    Op::Compare(CmpOp::Le),
                    vec![cum, top_p],
                    Shape::new(&[n, k], rlx_ir::DType::Bool),
                );
                let le = b.g.cast(le, rlx_ir::DType::F32);
                let mut first = vec![0f32; k];
                first[0] = 1.0;
                let first = b.data(&format!("fused.first.{pos}"), first, &[1, k]);
                let both = b.g.mul(le, first);
                let either = b.g.add(le, first);
                let keep = b.g.sub(either, both);
                let flat_logits = b.g.reshape_(logits, vec![(n * book) as i64]);
                let v = b.g.gather_(flat_logits, flat_idx, 0);
                let v = b.g.reshape_(v, vec![n as i64, k as i64]);
                let inv_t = b.data(
                    &format!("fused.inv_t.{pos}"),
                    vec![1.0 / params.temperature.max(1e-5)],
                    &[1],
                );
                let scaled = b.g.mul(v, inv_t);
                let row = b.g.narrow_(noise.expect("noise input"), 0, pos - 1, 1);
                let row = b.g.reshape_(row, vec![n as i64, k as i64]);
                let score = b.g.add(scaled, row);
                let big = b.data(&format!("fused.big.{pos}"), vec![1e30], &[1]);
                let one = b.data(&format!("fused.one.{pos}"), vec![1.0], &[1]);
                let off = b.g.sub(keep, one);
                let off = b.g.mul(off, big);
                let masked = b.g.add(score, off);
                let j = b.g.add_node(
                    Op::ArgMax {
                        axis: 1,
                        keep_dim: false,
                    },
                    vec![masked],
                    f32s(&[n]),
                );
                let j = b.g.reshape_(j, vec![n as i64, 1]);
                let j = b.g.add(j, k_off);
                let j = b.g.reshape_(j, vec![n as i64]);
                let all = b.g.reshape_(idx, vec![(n * k) as i64]);
                b.g.gather_(all, j, 0)
            };
            code = b.g.reshape_(pick, vec![n as i64]);
            codes.push(code);
        }
        let out = b.g.concat_(codes, 0);
        (b, vec![out])
    }

    /// The codes of a frame for several sequences at once (the batched
    /// [`Self::fast_frame_fused`]): `hidden` `[n, dim]`, each sequence's
    /// main token, its RNG. Sequences not `active` ride along without
    /// drawing; their codes are meaningless.
    pub fn fast_frames_fused(
        &mut self,
        hidden: &[f32],
        mains: &[u32],
        active: &[bool],
        params: SampleParams,
        rngs: &mut [Rng],
    ) -> Result<Vec<Vec<u32>>> {
        let (inputs, c0) = self.fused_batch_prepare(hidden, mains, active, params, rngs)?;
        let refs: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_slice()))
            .collect();
        let out = self.fused_batch.as_mut().expect("fused batch").1.run(&refs);
        Ok(self.fused_batch_finish(&c0, &out[0]))
    }

    /// [`Self::fast_frames_fused`] with an async run (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn fast_frames_fused_async(
        &mut self,
        hidden: &[f32],
        mains: &[u32],
        active: &[bool],
        params: SampleParams,
        rngs: &mut [Rng],
    ) -> Result<Vec<Vec<u32>>> {
        let (inputs, c0) = self.fused_batch_prepare(hidden, mains, active, params, rngs)?;
        let refs: Vec<(&str, &[f32])> = inputs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_slice()))
            .collect();
        let out = {
            let g = &mut self.fused_batch.as_mut().expect("fused batch").1;
            g.run_async(&refs).await.map_err(anyhow::Error::msg)?
        };
        Ok(self.fused_batch_finish(&c0, &out[0]))
    }

    #[allow(clippy::type_complexity)]
    fn fused_batch_prepare(
        &mut self,
        hidden: &[f32],
        mains: &[u32],
        active: &[bool],
        params: SampleParams,
        rngs: &mut [Rng],
    ) -> Result<(Vec<(String, Vec<f32>)>, Vec<u32>)> {
        let n = mains.len();
        ensure!(
            active.len() == n && rngs.len() == n && hidden.len() == n * self.cfg.fast.dim,
            "one hidden row, flag and RNG per sequence"
        );
        let key = (FusedKey::of(params), n);
        if self.fused_batch.as_ref().map(|f| f.0) != Some(key) {
            self.fused_batch = None;
            let (b, outs) = self.fused_batch_builder(params, n);
            let g = self.compile(b, outs, rlx_flow::CompileProfile::llama32_decode())?;
            self.fused_batch = Some((key, g));
        }
        let c0: Vec<u32> = mains
            .iter()
            .map(|&m| {
                m.saturating_sub(self.cfg.semantic_begin)
                    .min(self.cfg.codebook_size as u32 - 1)
            })
            .collect();
        let mut inputs = vec![
            ("hidden".to_string(), hidden.to_vec()),
            ("c0".to_string(), c0.iter().map(|&c| c as f32).collect()),
        ];
        if params.top_k > 1 {
            // Each sequence's draws in the order `fast_frame` makes them.
            let (k, nc) = (params.top_k, self.cfg.num_codebooks);
            let mut noise = vec![0f32; (nc - 1) * n * k];
            for (i, rng) in rngs.iter_mut().enumerate().filter(|(i, _)| active[*i]) {
                for p in 0..nc - 1 {
                    for (r, e) in sampling::draws(params, rng).iter().enumerate() {
                        noise[(p * n + i) * k + r] = -(e.ln()) as f32;
                    }
                }
            }
            inputs.push(("noise".to_string(), noise));
        }
        Ok((inputs, c0))
    }

    fn fused_batch_finish(&self, c0: &[u32], out: &[f32]) -> Vec<Vec<u32>> {
        let n = c0.len();
        (0..n)
            .map(|i| {
                std::iter::once(c0[i])
                    .chain((0..self.cfg.num_codebooks - 1).map(|p| out[p * n + i] as u32))
                    .collect()
            })
            .collect()
    }

    /// Frames for several prompts at once, each sampled as [`Self::generate`]
    /// would sample it alone (its own RNG, same draws in the same order):
    /// one batched decode step and one batched fast-decoder run per frame for
    /// the whole batch. Sequences that end early ride along until the last
    /// one does.
    pub fn generate_batch(
        &mut self,
        prompts: &[Vec<Row>],
        max_new: usize,
        params: SampleParams,
        seeds: &[u64],
    ) -> Result<Vec<Vec<Frame>>> {
        ensure!(self.is_whole(), "generate_batch runs the whole model");
        let n = prompts.len();
        ensure!(n > 0 && seeds.len() == n, "one seed per prompt");
        if n == 1 {
            let mut rng = Rng::new(seeds[0]);
            return Ok(vec![self.generate(
                &prompts[0],
                max_new,
                params,
                &mut rng,
            )?]);
        }
        let top = prompts.iter().map(Vec::len).max().expect("prompts");
        ensure!(
            top + max_new < self.cfg.max_seq_len,
            "prompt {top} + {max_new} frames exceeds the context"
        );
        let cap = crate::audio8::cache_capacity(&self.cfg, top, max_new);
        let states = prompts
            .iter()
            .map(|rows| self.prefill(rows, cap))
            .collect::<Result<_>>()?;
        let mut st = self.batch_join(states)?;
        let mut sampler = BatchSampler::new(&self.cfg, params, seeds);
        let mut frames: Vec<Vec<Frame>> = vec![Vec::new(); n];
        let verbose = rlx_ir::env::flag("RLX_FISH_VERBOSE");
        let (mut t_slow, mut t_fast) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for step in 0..max_new {
            let t = crate::clock();
            let Some(rows) = sampler.frame(self, &st.hidden, &st.logits)? else {
                break;
            };
            t_fast += crate::elapsed(t);
            for (i, row) in rows.iter().enumerate().filter(|(i, _)| !sampler.ended(*i)) {
                frames[i].push(Frame {
                    main: row[0],
                    codes: row[1..].to_vec(),
                });
            }
            sampler.close();
            if step + 1 == max_new {
                break;
            }
            let t = crate::clock();
            let embeds = self.embed_rows(&rows)?;
            self.batch_step(&mut st, &embeds)?;
            t_slow += crate::elapsed(t);
        }
        if verbose {
            let steps = st.steps.max(1) as f32;
            eprintln!(
                "s2: batch of {n}: slow {:.0} ms/step, fast {:.0} ms/step (cache on device: {})",
                t_slow.as_secs_f32() * 1e3 / steps,
                t_fast.as_secs_f32() * 1e3 / steps,
                self.batch_resident == Some(st.id)
            );
        }
        Ok(frames)
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
        ensure!(
            self.is_whole(),
            "generate runs the whole model; this one is the pipeline stage {:?}",
            self.layers
        );
        let t = rows.len();
        if t + max_new > self.cfg.max_seq_len {
            bail!(
                "prompt {t} + {max_new} frames exceeds the {}-row context",
                self.cfg.max_seq_len
            );
        }
        let cap = crate::audio8::cache_capacity(&self.cfg, t, max_new);
        let mut state = self.prefill(rows, cap)?;
        let mut sampler = MainSampler::new(&self.cfg, params);
        let mut frames = Vec::new();
        let verbose = rlx_ir::env::flag("RLX_FISH_VERBOSE");
        let (mut t_slow, mut t_fast) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for step in 0..max_new {
            let main = sampler.choose(&state.logits, rng);
            if main == self.cfg.eos {
                break;
            }
            let t = crate::clock();
            let codes = self.fast_frame_fused(&state.hidden, main, params, rng)?;
            t_fast += crate::elapsed(t);
            let frame = Frame { main, codes };
            on_frame(&frame);
            sampler.accept(main);
            if step + 1 < max_new {
                let mut row = vec![main];
                row.extend(&frame.codes);
                let t = crate::clock();
                let embed = self.embed_rows(&[row])?;
                self.step(&mut state, &embed)?;
                t_slow += crate::elapsed(t);
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
        self.fused = None;
        self.decode_batch = None;
        self.fused_batch = None;
        self.batch_resident = None;
    }
}

/// The KV caches of `n` sequences decoded together ([`S2Model::batch_step`]):
/// per layer one `[n, cap, kv]` cache, sequence `i`'s prompt placed to end at
/// slot `top` of its block, so every step writes all sequences' rows at one
/// slot while each keeps its own positions.
pub struct BatchState {
    pub n: usize,
    cap: usize,
    top: usize,
    lens: Vec<usize>,
    /// Steps taken since the join.
    pub steps: usize,
    /// Identifies the batch whose caches the graph holds on device.
    id: u64,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// This stage's newest rows, `[n, dim]` (final-norm on the last stage).
    pub hidden: Vec<f32>,
    /// The newest band logits, `[n, band]` (last stage; empty elsewhere).
    pub logits: Vec<f32>,
}

/// The main-token samplers and RNGs of a batch ([`S2Model::generate_batch`]),
/// each drawing what [`S2Model::generate`] draws for its sequence alone.
pub struct BatchSampler {
    params: SampleParams,
    samplers: Vec<MainSampler>,
    rngs: Vec<Rng>,
    /// Sequences that chose `<|im_end|>`.
    ended: Vec<bool>,
    /// The frame's main tokens, accepted by [`Self::close`].
    pending: Vec<Option<u32>>,
}

impl BatchSampler {
    pub fn new(cfg: &S2Config, params: SampleParams, seeds: &[u64]) -> Self {
        Self {
            params,
            samplers: seeds
                .iter()
                .map(|_| MainSampler::new(cfg, params))
                .collect(),
            rngs: seeds.iter().map(|&s| Rng::new(s)).collect(),
            ended: vec![false; seeds.len()],
            pending: vec![None; seeds.len()],
        }
    }

    /// Whether sequence `i` has ended (its rows from then on are filler).
    pub fn ended(&self, i: usize) -> bool {
        self.ended[i]
    }

    pub fn all_ended(&self) -> bool {
        self.ended.iter().all(|&e| e)
    }

    /// Every sequence's next row `[main, code 0, …]` from the last stage's
    /// rows `[n, dim]` and band logits `[n, band]`, run on `model`'s fast
    /// decoder; `None` once every sequence has ended. An ended sequence gets
    /// a filler row (the batch still steps it). Call [`Self::close`] after.
    pub fn frame(
        &mut self,
        model: &mut S2Model,
        hidden: &[f32],
        logits: &[f32],
    ) -> Result<Option<Vec<Row>>> {
        let Some(mains) = self.mains(model, logits) else {
            return Ok(None);
        };
        let active: Vec<bool> = self.ended.iter().map(|e| !e).collect();
        let codes =
            model.fast_frames_fused(hidden, &mains, &active, self.params, &mut self.rngs)?;
        Ok(Some(rows_of(&mains, codes)))
    }

    /// [`Self::frame`] with an async run (WebGPU in a browser).
    #[cfg(target_family = "wasm")]
    pub async fn frame_async(
        &mut self,
        model: &mut S2Model,
        hidden: &[f32],
        logits: &[f32],
    ) -> Result<Option<Vec<Row>>> {
        let Some(mains) = self.mains(model, logits) else {
            return Ok(None);
        };
        let active: Vec<bool> = self.ended.iter().map(|e| !e).collect();
        let codes = model
            .fast_frames_fused_async(hidden, &mains, &active, self.params, &mut self.rngs)
            .await?;
        Ok(Some(rows_of(&mains, codes)))
    }

    fn mains(&mut self, model: &S2Model, logits: &[f32]) -> Option<Vec<u32>> {
        let n = self.samplers.len();
        let nb = logits.len() / n;
        let mut mains = vec![model.cfg.semantic_begin; n];
        for i in 0..n {
            self.pending[i] = None;
            if self.ended[i] {
                continue;
            }
            let m = self.samplers[i].choose(&logits[i * nb..(i + 1) * nb], &mut self.rngs[i]);
            if m == model.cfg.eos {
                self.ended[i] = true;
            } else {
                mains[i] = m;
                self.pending[i] = Some(m);
            }
        }
        (!self.all_ended()).then_some(mains)
    }

    /// Accept the frame's main tokens (the repetition window sees them).
    pub fn close(&mut self) {
        for (s, m) in self.samplers.iter_mut().zip(&self.pending) {
            if let Some(m) = m {
                s.accept(*m);
            }
        }
    }
}

fn rows_of(mains: &[u32], codes: Vec<Vec<u32>>) -> Vec<Row> {
    mains
        .iter()
        .zip(codes)
        .map(|(&m, c)| std::iter::once(m).chain(c).collect())
        .collect()
}

/// The sampling settings a fused frame graph was built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FusedKey(u32, u32, usize);

impl FusedKey {
    fn of(p: SampleParams) -> Self {
        Self(p.temperature.to_bits(), p.top_p.to_bits(), p.top_k)
    }
}

/// A prefill run's inputs.
struct PrefillJob {
    embeds: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl PrefillJob {
    fn inputs(&self) -> [(&str, &[f32]); 3] {
        [
            ("embeds", &self.embeds),
            ("rope_cos", &self.cos),
            ("rope_sin", &self.sin),
        ]
    }
}

/// The fast decoder over one frame: position 0 reads the slow hidden state,
/// position `i` the embedding of code `i - 1`, and positions `1..` sample a
/// code each. [`S2Model::fast_frame`] runs it.
struct FastRun {
    pos: usize,
    /// Positions in a frame: one per codebook.
    nc: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    x: Vec<f32>,
    codes: Vec<u32>,
}

impl FastRun {
    fn done(&self) -> bool {
        self.pos == self.nc
    }

    /// This position's graph inputs.
    fn inputs(&self, cfg: &S2Config) -> Vec<(String, Vec<f32>)> {
        let (nc, pos) = (cfg.num_codebooks, self.pos);
        let (cos, sin) = nn::rope_tables(&[pos], cfg.fast.head_dim, cfg.rope_base, true);
        let mut mask = vec![0f32; nc + 1];
        mask[..pos].fill(1.0);
        mask[nc] = 1.0;
        let mut ins = vec![
            ("x".to_string(), self.x.clone()),
            ("rope_cos".to_string(), cos),
            ("rope_sin".to_string(), sin),
            ("mask".to_string(), mask),
        ];
        for (l, (k, v)) in self.k.iter().zip(&self.v).enumerate() {
            ins.push((format!("past_k_{l}"), k.clone()));
            ins.push((format!("past_v_{l}"), v.clone()));
        }
        ins
    }

    /// This position's outputs: its K/V rows in, its code sampled (from
    /// position 1), the next position's input embedding read.
    fn absorb(
        &mut self,
        m: &S2Model,
        outs: Vec<Vec<f32>>,
        params: SampleParams,
        rng: &mut Rng,
        on_logits: &mut impl FnMut(usize, &[f32]),
    ) -> Result<()> {
        let (nc, pos) = (m.cfg.num_codebooks, self.pos);
        let kvw = S2Model::kv_width(&m.cfg.fast);
        for l in 0..self.k.len() {
            self.k[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[1 + 2 * l]);
            self.v[l][pos * kvw..(pos + 1) * kvw].copy_from_slice(&outs[2 + 2 * l]);
        }
        if pos > 0 {
            on_logits(pos, &outs[0]);
            // `top_k` draws a code, whatever the cut keeps: the fused frame
            // graph takes them as an input and draws the same codes.
            let noise = sampling::draws(params, rng);
            self.codes
                .push(sampling::sample_with(&outs[0], params, Flavor::S2, &noise) as u32);
        }
        if pos + 1 < nc {
            let prev = *self.codes.last().expect("code");
            let [_, _, fast_embed] = m.host_tensor_keys();
            self.x = m
                .ckpt
                .load_tensor_rows_f32(&fast_embed, &[prev], m.cfg.fast.dim)?
                .swap_remove(0);
        }
        self.pos += 1;
        Ok(())
    }
}

/// How a frame's main token is drawn: from the band logits at `params`, and
/// — from the second frame on — drawn again at the repetition-aware settings,
/// the redraw taken when the first repeats one of the last `ras.window` main
/// tokens. The reference's window starts empty and collects only frames the
/// decode loop sampled, not the first one, drawn at prefill.
///
/// [`S2Model::generate_with`] runs on it, and so does a pipeline that drives
/// the stages itself, so both draw the same tokens from the same seed.
pub struct MainSampler {
    params: SampleParams,
    ras: SampleParams,
    window: Vec<u32>,
    size: usize,
    step: usize,
    semantic: std::ops::RangeInclusive<u32>,
    codebook_size: usize,
    eos: u32,
}

impl MainSampler {
    pub fn new(cfg: &S2Config, params: SampleParams) -> Self {
        Self {
            params,
            ras: SampleParams {
                temperature: cfg.ras.temperature,
                top_p: cfg.ras.top_p,
                top_k: params.top_k,
            },
            window: Vec::with_capacity(cfg.ras.window),
            size: cfg.ras.window,
            step: 0,
            semantic: cfg.semantic_begin..=cfg.semantic_end,
            codebook_size: cfg.codebook_size,
            eos: cfg.eos,
        }
    }

    fn token(&self, idx: usize) -> u32 {
        if idx < self.codebook_size {
            self.semantic.start() + idx as u32
        } else {
            self.eos
        }
    }

    /// The main token after band logits `logits` (`<|im_end|>` ends the
    /// utterance).
    pub fn choose(&mut self, logits: &[f32], rng: &mut Rng) -> u32 {
        let first = self.token(sampling::sample(logits, self.params, Flavor::S2, rng));
        if self.step == 0 {
            return first;
        }
        let retry = self.token(sampling::sample(logits, self.ras, Flavor::S2, rng));
        if self.semantic.contains(&first) && self.window.contains(&first) {
            retry
        } else {
            first
        }
    }

    /// Record a frame that was kept, chosen by [`Self::choose`].
    pub fn accept(&mut self, main: u32) {
        if self.step > 0 {
            if self.window.len() == self.size {
                self.window.remove(0);
            }
            self.window.push(main);
        }
        self.step += 1;
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
            qkv_bias: s.qkv_bias,
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
