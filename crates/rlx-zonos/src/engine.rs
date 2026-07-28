// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Device-backed Zonos backbone — CFG batch=2, cached prefill, GPU-resident KV.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rlx_runtime::{CompiledGraph, Device};

use crate::compile_opts::{metal_compile_guard, metal_run_guard};
use crate::config::ZonosFileConfig;
use crate::flow::{
    CFG_BATCH, ZonosDims, bucket_decode_mask_cfg2, compile_decode, compile_prefill, graph_params,
    rope_cos_sin, rope_tables,
};
use crate::weights::WeightMap;

/// Prefill length buckets (pad up; causal keeps real prefix bit-exact).
const PREFILL_BUCKETS: &[usize] = &[64, 80, 96, 112, 128, 160, 192, 256];

fn device_supports_gpu_kv(device: Device) -> bool {
    if let Some(v) = rlx_ir::env::var("RLX_DISABLE_GPU_KV") {
        let v = v.trim().to_ascii_lowercase();
        let name = match device {
            Device::Gpu => "wgpu",
            Device::Cuda => "cuda",
            Device::Vulkan => "vulkan",
            Device::Metal => "metal",
            Device::Mlx => "mlx",
            Device::Rocm => "rocm",
            _ => "",
        };
        if v == "1" || v == "true" || v == name {
            return false;
        }
    }
    // CUDA/ROCm: `feed_kv_batch_major` currently fails mid-decode on the
    // CFG batch=2 path; host-pad KV is correct. Opt back in with
    // `RLX_ENABLE_GPU_KV=1` once upstream feed is fixed.
    if matches!(device, Device::Cuda | Device::Rocm)
        && rlx_ir::env::var("RLX_ENABLE_GPU_KV").as_deref() != Some("1")
    {
        return false;
    }
    matches!(
        device,
        Device::Mlx | Device::Metal | Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
    )
}

/// Host KV: time-major compact `[t][B][kv_w]` growing; pad scratch `[B][U][kv_w]`.
struct KvState {
    past_kv: Vec<(Vec<f32>, Vec<f32>)>,
    past_seq: usize,
}

impl KvState {
    fn new(n_layers: usize) -> Self {
        Self {
            past_kv: (0..n_layers).map(|_| (Vec::new(), Vec::new())).collect(),
            past_seq: 0,
        }
    }

    fn append_decode(&mut self, new_kv: Vec<(Vec<f32>, Vec<f32>)>, kv_w: usize) {
        let chunk = CFG_BATCH * kv_w;
        for (li, (nk, nv)) in new_kv.into_iter().enumerate() {
            debug_assert_eq!(nk.len(), chunk);
            self.past_kv[li].0.extend_from_slice(&nk);
            self.past_kv[li].1.extend_from_slice(&nv);
        }
        self.past_seq += 1;
    }

    fn seed_prefill(&mut self, full_kv: Vec<(Vec<f32>, Vec<f32>)>, seq: usize, kv_w: usize) {
        for (li, (k_bt, v_bt)) in full_kv.into_iter().enumerate() {
            let mut k_tb = vec![0.0f32; seq * CFG_BATCH * kv_w];
            let mut v_tb = vec![0.0f32; seq * CFG_BATCH * kv_w];
            for b in 0..CFG_BATCH {
                for t in 0..seq {
                    let src = (b * seq + t) * kv_w;
                    let dst = (t * CFG_BATCH + b) * kv_w;
                    k_tb[dst..dst + kv_w].copy_from_slice(&k_bt[src..src + kv_w]);
                    v_tb[dst..dst + kv_w].copy_from_slice(&v_bt[src..src + kv_w]);
                }
            }
            self.past_kv[li] = (k_tb, v_tb);
        }
        self.past_seq = seq;
    }

    fn seed_prefill_cropped(
        &mut self,
        full_kv: Vec<(Vec<f32>, Vec<f32>)>,
        bucket: usize,
        seq: usize,
        kv_w: usize,
    ) {
        for (li, (k_bt, v_bt)) in full_kv.into_iter().enumerate() {
            let mut k_tb = vec![0.0f32; seq * CFG_BATCH * kv_w];
            let mut v_tb = vec![0.0f32; seq * CFG_BATCH * kv_w];
            for b in 0..CFG_BATCH {
                for t in 0..seq {
                    let src = (b * bucket + t) * kv_w;
                    let dst = (t * CFG_BATCH + b) * kv_w;
                    k_tb[dst..dst + kv_w].copy_from_slice(&k_bt[src..src + kv_w]);
                    v_tb[dst..dst + kv_w].copy_from_slice(&v_bt[src..src + kv_w]);
                }
            }
            self.past_kv[li] = (k_tb, v_tb);
        }
        self.past_seq = seq;
    }

    /// Fill batch-major pad `[B, U, kv_w]` once (zeros after `past_seq`).
    /// Compact host KV stays time-major `[t][B][kv_w]` for cheap append.
    fn fill_batch_major_pad(&self, upper: usize, kv_w: usize, out: &mut Vec<(Vec<f32>, Vec<f32>)>) {
        let pad_len = CFG_BATCH * upper * kv_w;
        out.resize_with(self.past_kv.len(), || (Vec::new(), Vec::new()));
        for (li, (rk, rv)) in self.past_kv.iter().enumerate() {
            let (pk, pv) = &mut out[li];
            if pk.len() != pad_len {
                *pk = vec![0.0; pad_len];
                *pv = vec![0.0; pad_len];
            } else {
                pk.fill(0.0);
                pv.fill(0.0);
            }
            for t in 0..self.past_seq {
                for b in 0..CFG_BATCH {
                    let src = (t * CFG_BATCH + b) * kv_w;
                    let dst = (b * upper + t) * kv_w;
                    pk[dst..dst + kv_w].copy_from_slice(&rk[src..src + kv_w]);
                    pv[dst..dst + kv_w].copy_from_slice(&rv[src..src + kv_w]);
                }
            }
        }
    }

    /// Write one new decode token into the batch-major pad at `past_seq`.
    /// `new_kv` rows are `[B, kv_w]` packed.
    fn write_new_into_pad(
        &self,
        upper: usize,
        kv_w: usize,
        new_kv: &[(Vec<f32>, Vec<f32>)],
        out: &mut [(Vec<f32>, Vec<f32>)],
    ) {
        let t = self.past_seq;
        debug_assert!(t < upper);
        for (li, (nk, nv)) in new_kv.iter().enumerate() {
            for b in 0..CFG_BATCH {
                let src = b * kv_w;
                let dst = (b * upper + t) * kv_w;
                out[li].0[dst..dst + kv_w].copy_from_slice(&nk[src..src + kv_w]);
                out[li].1[dst..dst + kv_w].copy_from_slice(&nv[src..src + kv_w]);
            }
        }
    }
}

fn prefill_bucket(seq: usize) -> usize {
    PREFILL_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= seq)
        .unwrap_or_else(|| seq.div_ceil(32) * 32)
}

pub struct BackboneEngine {
    dims: ZonosDims,
    device: Device,
    upper: usize,
    params: HashMap<String, Vec<f32>>,
    compiled: CompiledGraph,
    prefill: HashMap<usize, CompiledGraph>,
    kv: KvState,
    /// Batch-major `[B,U,kv_w]` pads — updated in place across decode steps.
    pad_scratch: Vec<(Vec<f32>, Vec<f32>)>,
    embeds_scratch: Vec<f32>,
    mask_scratch: Vec<f32>,
    cos_scratch: Vec<f32>,
    sin_scratch: Vec<f32>,
    /// Named host inputs reused every step (no per-step String/Vec alloc storm).
    input_names: Vec<String>,
    gpu_kv: bool,
    gpu_kv_bound: bool,
}

impl BackboneEngine {
    pub fn open(
        cfg: &ZonosFileConfig,
        weights: &WeightMap,
        device: Device,
        max_seq: usize,
    ) -> Result<Self> {
        let dims = ZonosDims::from_cfg(cfg);
        let upper = max_seq.max(8);
        let params = graph_params(&dims, weights).context("remap Zonos weights for graph")?;
        let t0 = Instant::now();
        let gpu_kv = device_supports_gpu_kv(device);
        eprintln!(
            "zonos: compiling CFG batch={} decode device={device:?} layers={} upper={upper} gpu_kv={gpu_kv} …",
            CFG_BATCH, dims.n_layers
        );
        let mut compiled =
            metal_compile_guard(device, || compile_decode(&dims, &params, upper, device))?;
        if gpu_kv {
            for li in 0..dims.n_layers {
                let k_name = format!("past_k_{li}");
                let v_name = format!("past_v_{li}");
                // new_k / new_v are outputs 1+2*li / 2+2*li (0 = hidden).
                compiled.register_kv_row_feed(&k_name, 1 + 2 * li);
                compiled.register_kv_row_feed(&v_name, 2 + 2 * li);
            }
        }
        eprintln!(
            "zonos: decode compile done in {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        let mut input_names = vec![
            "inputs_embeds".into(),
            "rope_cos".into(),
            "rope_sin".into(),
            "attn_mask".into(),
        ];
        for li in 0..dims.n_layers {
            input_names.push(format!("past_k_{li}"));
            input_names.push(format!("past_v_{li}"));
        }

        Ok(Self {
            dims,
            device,
            upper,
            params,
            compiled,
            prefill: HashMap::new(),
            kv: KvState::new(dims.n_layers),
            pad_scratch: Vec::new(),
            embeds_scratch: Vec::new(),
            mask_scratch: Vec::new(),
            cos_scratch: Vec::new(),
            sin_scratch: Vec::new(),
            input_names,
            gpu_kv,
            gpu_kv_bound: false,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dims(&self) -> &ZonosDims {
        &self.dims
    }

    pub fn reset(&mut self) {
        self.kv = KvState::new(self.dims.n_layers);
        self.gpu_kv_bound = false;
    }

    fn ensure_prefill(&mut self, bucket: usize) -> Result<()> {
        if self.prefill.contains_key(&bucket) {
            return Ok(());
        }
        let t0 = Instant::now();
        eprintln!(
            "zonos: compiling CFG prefill bucket={bucket} device={:?} …",
            self.device
        );
        let g = metal_compile_guard(self.device, || {
            compile_prefill(&self.dims, &self.params, bucket, self.device)
        })?;
        eprintln!(
            "zonos: prefill compile done in {:.1}s (cached)",
            t0.elapsed().as_secs_f64()
        );
        self.prefill.insert(bucket, g);
        Ok(())
    }

    fn bind_gpu_kv_from_pad(&mut self) -> Result<()> {
        for li in 0..self.dims.n_layers {
            let k_name = format!("past_k_{li}");
            let v_name = format!("past_v_{li}");
            let (pk, pv) = &self.pad_scratch[li];
            anyhow::ensure!(
                self.compiled.bind_gpu_handle(&k_name, pk),
                "bind_gpu_handle {k_name}"
            );
            anyhow::ensure!(
                self.compiled.bind_gpu_handle(&v_name, pv),
                "bind_gpu_handle {v_name}"
            );
        }
        self.gpu_kv_bound = true;
        Ok(())
    }

    pub fn step_cfg(&mut self, emb_cond: &[f32], emb_uncond: &[f32]) -> Result<Vec<f32>> {
        let d = self.dims.d_model;
        anyhow::ensure!(emb_cond.len() == d && emb_uncond.len() == d);
        let past_seq = self.kv.past_seq;
        if past_seq >= self.upper {
            bail!(
                "KV past_seq {past_seq} >= compile upper {} — raise max_seq",
                self.upper
            );
        }

        self.embeds_scratch.clear();
        self.embeds_scratch.extend_from_slice(emb_cond);
        self.embeds_scratch.extend_from_slice(emb_uncond);

        let (cos, sin) = rope_cos_sin(&self.dims, past_seq);
        self.cos_scratch = cos;
        self.sin_scratch = sin;
        self.mask_scratch = bucket_decode_mask_cfg2(past_seq, self.upper);
        let kv_w = self.dims.kv_width();

        if self.pad_scratch.is_empty() || !self.gpu_kv_bound {
            self.kv
                .fill_batch_major_pad(self.upper, kv_w, &mut self.pad_scratch);
            if self.gpu_kv {
                self.bind_gpu_kv_from_pad()?;
            }
        }

        let hidden = if self.gpu_kv && self.gpu_kv_bound {
            let refs: Vec<(&str, &[f32])> = vec![
                ("inputs_embeds", self.embeds_scratch.as_slice()),
                ("rope_cos", self.cos_scratch.as_slice()),
                ("rope_sin", self.sin_scratch.as_slice()),
                ("attn_mask", self.mask_scratch.as_slice()),
            ];
            let outs = metal_run_guard(self.device, || {
                self.compiled.run_read_outputs(&refs, Some(&[0]))
            });
            // Fold new-token into resident past[b, past_seq, :] for each CFG batch.
            anyhow::ensure!(
                self.compiled
                    .feed_kv_batch_major(past_seq, CFG_BATCH, self.upper, kv_w),
                "feed_kv_batch_major failed at past_seq={past_seq}"
            );
            let hidden = outs.into_iter().next().context("missing hidden")?;
            anyhow::ensure!(hidden.len() == CFG_BATCH * d);
            // Keep host compact KV in sync for reset / fallback (row only).
            // Cheap: one row per layer from pad after feed… pad isn't updated on
            // GPU path; extend compact from a zeroed placeholder — past_seq bump
            // alone is enough while gpu_kv_bound. Mirror length for safety.
            let row = CFG_BATCH * kv_w;
            for li in 0..self.dims.n_layers {
                self.kv.past_kv[li].0.extend(std::iter::repeat_n(0.0, row));
                self.kv.past_kv[li].1.extend(std::iter::repeat_n(0.0, row));
            }
            self.kv.past_seq += 1;
            hidden
        } else {
            // Host path: pass batch-major pads without cloning.
            let mut refs: Vec<(&str, &[f32])> = Vec::with_capacity(4 + 2 * self.dims.n_layers);
            refs.push((self.input_names[0].as_str(), self.embeds_scratch.as_slice()));
            refs.push((self.input_names[1].as_str(), self.cos_scratch.as_slice()));
            refs.push((self.input_names[2].as_str(), self.sin_scratch.as_slice()));
            refs.push((self.input_names[3].as_str(), self.mask_scratch.as_slice()));
            for li in 0..self.dims.n_layers {
                let (pk, pv) = &self.pad_scratch[li];
                refs.push((self.input_names[4 + 2 * li].as_str(), pk.as_slice()));
                refs.push((self.input_names[5 + 2 * li].as_str(), pv.as_slice()));
            }
            let outs = metal_run_guard(self.device, || self.compiled.run(&refs));
            let mut it = outs.into_iter();
            let hidden = it.next().context("missing hidden")?;
            anyhow::ensure!(hidden.len() == CFG_BATCH * d);
            let chunk = CFG_BATCH * kv_w;
            let mut new_kv = Vec::with_capacity(self.dims.n_layers);
            for _ in 0..self.dims.n_layers {
                let k = it.next().context("missing new_k")?;
                let v = it.next().context("missing new_v")?;
                anyhow::ensure!(k.len() == chunk && v.len() == chunk);
                new_kv.push((k, v));
            }
            self.kv
                .write_new_into_pad(self.upper, kv_w, &new_kv, &mut self.pad_scratch);
            self.kv.append_decode(new_kv, kv_w);
            hidden
        };
        Ok(hidden)
    }

    pub fn step_shared(&mut self, embed_d: &[f32]) -> Result<Vec<f32>> {
        self.step_cfg(embed_d, embed_d)
    }

    pub fn step_pair(&mut self, emb_cond: &[f32], emb_uncond: &[f32]) -> Result<Vec<f32>> {
        self.step_cfg(emb_cond, emb_uncond)
    }

    /// Causal prefill; pads to a cached bucket (≥ seq). Returns last-token `[2*d]`.
    pub fn prefill_pair(&mut self, cond: &[f32], uncond: &[f32], seq: usize) -> Result<Vec<f32>> {
        let d = self.dims.d_model;
        anyhow::ensure!(cond.len() == seq * d && uncond.len() == seq * d);
        anyhow::ensure!(seq > 0);
        if seq > self.upper {
            bail!("prefill seq {seq} > decode upper {}", self.upper);
        }
        self.gpu_kv_bound = false;

        if seq <= 8 {
            let mut last = vec![0.0; CFG_BATCH * d];
            for t in 0..seq {
                last = self.step_cfg(&cond[t * d..(t + 1) * d], &uncond[t * d..(t + 1) * d])?;
            }
            return Ok(last);
        }

        let bucket = prefill_bucket(seq);
        self.ensure_prefill(bucket)?;

        let mut embeds = vec![0.0f32; CFG_BATCH * bucket * d];
        embeds[..seq * d].copy_from_slice(cond);
        embeds[bucket * d..bucket * d + seq * d].copy_from_slice(uncond);

        let (cos, sin) = rope_tables(&self.dims, bucket);
        let refs = [
            ("inputs_embeds", embeds.as_slice()),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ];

        let outs = {
            let g = self.prefill.get_mut(&bucket).context("prefill missing")?;
            metal_run_guard(self.device, || g.run(&refs))
        };
        let mut it = outs.into_iter();
        let hidden_full = it.next().context("prefill missing hidden")?;
        anyhow::ensure!(hidden_full.len() == CFG_BATCH * bucket * d);
        let mut last = vec![0.0f32; CFG_BATCH * d];
        for b in 0..CFG_BATCH {
            let src = (b * bucket + (seq - 1)) * d;
            last[b * d..(b + 1) * d].copy_from_slice(&hidden_full[src..src + d]);
        }

        let kv_w = self.dims.kv_width();
        let expect = CFG_BATCH * bucket * kv_w;
        let mut full_kv = Vec::with_capacity(self.dims.n_layers);
        for _ in 0..self.dims.n_layers {
            let k = it.next().context("prefill missing k")?;
            let v = it.next().context("prefill missing v")?;
            anyhow::ensure!(k.len() == expect && v.len() == expect);
            full_kv.push((k, v));
        }
        if bucket == seq {
            self.kv.seed_prefill(full_kv, seq, kv_w);
        } else {
            self.kv.seed_prefill_cropped(full_kv, bucket, seq, kv_w);
        }
        // Materialize batch-major pad + bind GPU handles for the decode loop.
        self.kv
            .fill_batch_major_pad(self.upper, kv_w, &mut self.pad_scratch);
        if self.gpu_kv {
            self.bind_gpu_kv_from_pad()?;
        }
        Ok(last)
    }
}

pub fn prefer_eager() -> bool {
    matches!(
        std::env::var("RLX_ZONOS_EAGER").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

pub fn default_max_seq(max_new_tokens: usize, prefix_len: usize) -> usize {
    let need = prefix_len + max_new_tokens + 16;
    let rounded = need.div_ceil(32) * 32;
    rounded.max(64)
}

/// Suggest AR budget from phoneme count + speaking-rate conditioner dial.
///
/// Upper bound is ~24 s of speech (`86×24` frames) by default — enough for the
/// bench long paragraph without compiling a CFG×2 decode graph that jetsams
/// Metal/MLX (~3 GiB weights + large `upper`). Override ceiling with
/// `RLX_ZONOS_MAX_TOKENS`.
pub fn suggest_max_tokens(phoneme_len: usize, speaking_rate: f32) -> usize {
    let rate = speaking_rate.clamp(5.0, 40.0);
    let frames = ((phoneme_len as f32) * (86.0 / rate) * 1.25).ceil() as usize + 64;
    let ceiling = std::env::var("RLX_ZONOS_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(86 * 24);
    frames.clamp(128, ceiling.max(128))
}

/// Compile-seq upper for a device. GPU backends keep a leaner decode graph so
/// short→long upgrade in one process does not OOM.
pub fn compile_upper_cap(device: Device, need: usize) -> usize {
    let env_cap = std::env::var("RLX_ZONOS_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok());
    let default_cap = match device {
        Device::Cpu => 86 * 32 + 64, // ~32 s
        _ => 86 * 20 + 64,           // ~20 s on Metal/MLX/CUDA
    };
    let cap = env_cap.unwrap_or(default_cap).max(256);
    need.min(cap)
}
