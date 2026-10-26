// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! chatterbox-turbo S3Gen: S3 codec tokens → 24 kHz audio.
//!
//! [`encoder`] maps the reference's prompt tokens plus the generated ones to
//! `mu`; two meanflow Euler steps of the [`estimator`] (`t: 0 → ½ → 1`) carry
//! noise to a mel, with the reference mel as the in-context condition over the
//! prompt frames; [`hift`] vocodes the generated frames. Weights come from
//! `ResembleAI/chatterbox-turbo`'s `s3gen_meanflow.safetensors` (MIT).
//!
//! Every graph is specialized on its sequence length; the compiled graph of
//! the most recent length is kept per stage, since consecutive utterances
//! rarely share one.

pub mod encoder;
pub mod estimator;
pub mod hift;
pub mod nn;
pub mod student;

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rlx_ir::hir::{HirModule, HirMut};
use rlx_ir::{Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::sampling::Rng;
use crate::voices::S3Conditioning;
use nn::{Ctx, F32, S3GenWeights};

pub use hift::SourceNoise;

/// Codec vocabulary (3⁸ FSQ codes).
pub const SPEECH_VOCAB: u32 = 6561;
/// The silence token S3Gen pads with.
pub const SILENCE_TOKEN: u32 = 4299;
/// Mel frames per codec token.
pub const TOKEN_MEL_RATIO: usize = 2;
/// The released meanflow decoder takes two Euler steps.
pub const CFM_STEPS: usize = 2;
const MEL: usize = 80;

/// One compiled graph and the length it was specialized on.
struct Slot {
    len: usize,
    graph: CompiledGraph,
}

/// The S3Gen decoder on one device.
pub struct S3Gen {
    w: Arc<S3GenWeights>,
    device: Device,
    /// Meanflow Euler steps: 2 for the released flow, 1 for the student.
    steps: usize,
    encoder: RefCell<Option<Slot>>,
    estimator: RefCell<Option<Slot>>,
    f0: RefCell<Option<Slot>>,
    decode: RefCell<Option<Slot>>,
}

/// Stage outputs of [`S3Gen::flow_with_noise`], kept for parity checks.
#[derive(Debug, Clone)]
pub struct FlowTrace {
    /// `[80, L]` encoder output over prompt + generated frames.
    pub mu: Vec<f32>,
    /// Velocities of the two Euler steps, `[80, L]` each.
    pub velocities: Vec<Vec<f32>>,
    /// `[80, L − prompt_frames]` — the generated part of the mel.
    pub mel: Vec<f32>,
}

fn compile(
    device: Device,
    hir: HirModule,
    params: Vec<(String, Vec<f32>)>,
    what: &str,
) -> Result<CompiledGraph> {
    let graph = Graph::from_hir(hir).map_err(|e| anyhow::anyhow!("{what}: HIR lowering: {e}"))?;
    let mut g = Session::new(device).compile(graph);
    for (n, d) in &params {
        g.set_param(n, d);
    }
    g.finalize_params();
    Ok(g)
}

impl S3Gen {
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        Ok(Self::from_weights(
            Arc::new(S3GenWeights::open(path)?),
            device,
        ))
    }

    /// The distilled 4-bit student flow (`decoders/student_w4.pt`) with the
    /// teacher's vocoder from `s3gen_meanflow.safetensors`. One flow step.
    pub fn open_student(student: &Path, teacher: &Path, device: Device) -> Result<Self> {
        let mut g = Self::from_weights(Arc::new(student::open(student, teacher)?), device);
        g.steps = 1;
        Ok(g)
    }

    /// True for the distilled student decoder.
    pub fn is_student(&self) -> bool {
        self.steps == 1
    }

    pub fn from_weights(w: Arc<S3GenWeights>, device: Device) -> Self {
        Self {
            w,
            device,
            steps: CFM_STEPS,
            encoder: RefCell::new(None),
            estimator: RefCell::new(None),
            f0: RefCell::new(None),
            decode: RefCell::new(None),
        }
    }

    pub fn weights(&self) -> &S3GenWeights {
        &self.w
    }

    /// Run the slot's graph, (re)compiling it if `len` changed.
    fn run_slot(
        &self,
        slot: &RefCell<Option<Slot>>,
        len: usize,
        build: impl FnOnce() -> Result<CompiledGraph>,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>> {
        let mut s = slot.borrow_mut();
        if s.as_ref().is_none_or(|s| s.len != len) {
            *s = None;
            *s = Some(Slot {
                len,
                graph: build()?,
            });
        }
        Ok(s.as_mut().unwrap().graph.run(inputs))
    }

    /// Encoder: all tokens (prompt + generated) → `mu` `[80, 2T]`.
    pub fn encode(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let t = tokens.len();
        ensure!(t > 0, "no tokens to encode");
        let table = self.w.get("flow.input_embedding.weight")?;
        let d = self.w.shape("flow.input_embedding.weight")?[1];
        let mut x = Vec::with_capacity(t * d);
        for &tok in tokens {
            ensure!(tok < SPEECH_VOCAB, "codec token {tok} out of range");
            x.extend_from_slice(&table[tok as usize * d..(tok as usize + 1) * d]);
        }
        let w = self.w.clone();
        let device = self.device;
        let out = self.run_slot(
            &self.encoder,
            t,
            move || {
                let mut hir = HirModule::new("s3gen_encoder");
                let mut g = HirMut::new(&mut hir);
                let mut ctx = Ctx::new(&mut g);
                let xi = ctx.g.input("x", Shape::new(&[t, d], F32));
                let mu = encoder::build(&mut ctx, &w, xi)?;
                let params = ctx.into_params();
                hir.set_outputs(vec![mu]);
                compile(device, hir, params, "s3gen encoder")
            },
            &[("x", &x)],
        )?;
        out.into_iter().next().context("encoder produced no output")
    }

    /// One estimator evaluation: the velocity at `x` for a step `t → r`.
    pub fn velocity(
        &self,
        x: &[f32],
        mu: &[f32],
        spks: &[f32],
        cond: &[f32],
        t: f32,
        r: f32,
    ) -> Result<Vec<f32>> {
        let l = x.len() / MEL;
        let temb = estimator::time_embedding(&self.w, t, r)?;
        let w = self.w.clone();
        let device = self.device;
        let out = self.run_slot(
            &self.estimator,
            l,
            move || {
                let mut hir = HirModule::new("s3gen_estimator");
                let mut g = HirMut::new(&mut hir);
                let mut ctx = Ctx::new(&mut g);
                let s = |c: usize| Shape::new(&[1, c, l, 1], F32);
                let xi = ctx.g.input("x", s(MEL));
                let mi = ctx.g.input("mu", s(MEL));
                let ci = ctx.g.input("cond", s(MEL));
                let si = ctx.g.input("spks", Shape::new(&[1, MEL, 1, 1], F32));
                let ti = ctx
                    .g
                    .input("temb", Shape::new(&[1, estimator::time_dim(&w)?], F32));
                let v = estimator::build(&mut ctx, &w, xi, mi, si, ci, ti)?;
                let params = ctx.into_params();
                hir.set_outputs(vec![v]);
                compile(device, hir, params, "s3gen estimator")
            },
            &[
                ("x", x),
                ("mu", mu),
                ("cond", cond),
                ("spks", spks),
                ("temb", &temb),
            ],
        )?;
        out.into_iter()
            .next()
            .context("estimator produced no output")
    }

    /// `spk_embed_affine_layer(normalize(xvector))`, `[80]`.
    pub fn speaker(&self, xvector: &[f32]) -> Result<Vec<f32>> {
        let n = xvector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let e: Vec<f32> = xvector.iter().map(|v| v / n).collect();
        let wt = self.w.get("flow.spk_embed_affine_layer.weight")?;
        let b = self.w.get("flow.spk_embed_affine_layer.bias")?;
        Ok(wt
            .chunks_exact(e.len())
            .zip(b)
            .map(|(row, b)| row.iter().zip(&e).map(|(a, x)| a * x).sum::<f32>() + b)
            .collect())
    }

    /// Flow with caller-supplied initial noise `[80, L]` (`L` = 2 × all tokens).
    pub fn flow_with_noise(
        &self,
        generated: &[u32],
        cond: &S3Conditioning,
        noise: &[f32],
    ) -> Result<FlowTrace> {
        let mut tokens = cond.prompt_token.clone();
        tokens.extend_from_slice(generated);
        let mu = self.encode(&tokens)?;
        let l = mu.len() / MEL;
        ensure!(
            noise.len() == MEL * l,
            "noise {} for {l} frames",
            noise.len()
        );
        let p = cond.prompt_frames;
        ensure!(
            p < l,
            "prompt mel ({p} frames) covers the whole sequence ({l})"
        );
        let mut c = vec![0f32; MEL * l];
        for f in 0..p {
            for m in 0..MEL {
                c[m * l + f] = cond.prompt_feat[f * MEL + m];
            }
        }
        let spks = self.speaker(&cond.xvector)?;
        let mut x = noise.to_vec();
        let mut velocities = Vec::with_capacity(self.steps);
        for step in 0..self.steps {
            let t = step as f32 / self.steps as f32;
            let r = (step + 1) as f32 / self.steps as f32;
            let v = self.velocity(&x, &mu, &spks, &c, t, r)?;
            for (xi, vi) in x.iter_mut().zip(&v) {
                *xi += (r - t) * vi;
            }
            velocities.push(v);
        }
        let mel: Vec<f32> = (0..MEL)
            .flat_map(|m| x[m * l + p..(m + 1) * l].to_vec())
            .collect();
        Ok(FlowTrace {
            mu,
            velocities,
            mel,
        })
    }

    /// HiFT f0 for a mel `[80, L]`.
    pub fn f0(&self, mel: &[f32]) -> Result<Vec<f32>> {
        let l = mel.len() / MEL;
        let w = self.w.clone();
        let device = self.device;
        let out = self.run_slot(
            &self.f0,
            l,
            move || {
                let mut hir = HirModule::new("hift_f0");
                let mut g = HirMut::new(&mut hir);
                let mut ctx = Ctx::new(&mut g);
                let mi = ctx.g.input("mel", Shape::new(&[1, MEL, l, 1], F32));
                let f = hift::build_f0(&mut ctx, &w, mi)?;
                let params = ctx.into_params();
                hir.set_outputs(vec![f]);
                compile(device, hir, params, "hift f0")
            },
            &[("mel", mel)],
        )?;
        out.into_iter().next().context("f0 produced no output")
    }

    /// HiFT decode from a mel and an explicit source waveform (`[480 L]`).
    pub fn decode_with_source(&self, mel: &[f32], source: &[f32]) -> Result<Vec<f32>> {
        let l = mel.len() / MEL;
        ensure!(
            source.len() == l * hift::UPSAMPLE,
            "source {} for {l} frames",
            source.len()
        );
        let s_stft = hift::stft(source);
        let frames = source.len() / hift::HOP + 1;
        let w = self.w.clone();
        let device = self.device;
        let out = self.run_slot(
            &self.decode,
            l,
            move || {
                let mut hir = HirModule::new("hift_decode");
                let mut g = HirMut::new(&mut hir);
                let mut ctx = Ctx::new(&mut g);
                let mi = ctx.g.input("mel", Shape::new(&[1, MEL, l, 1], F32));
                let si = ctx.g.input(
                    "s_stft",
                    Shape::new(&[1, 2 * (hift::N_FFT / 2 + 1), frames, 1], F32),
                );
                let y = hift::build_decode(&mut ctx, &w, mi, si)?;
                let params = ctx.into_params();
                hir.set_outputs(vec![y]);
                compile(device, hir, params, "hift decode")
            },
            &[("mel", mel), ("s_stft", &s_stft)],
        )?;
        let head = out.into_iter().next().context("hift produced no output")?;
        Ok(hift::istft_head(&head, frames))
    }

    /// Mel `[80, L]` → waveform (`480 L` samples), no fade.
    pub fn vocode(&self, mel: &[f32], noise: SourceNoise<'_>) -> Result<Vec<f32>> {
        let f0 = self.f0(mel)?;
        let source = hift::sine_source(&self.w, &f0, noise)?;
        self.decode_with_source(mel, &source)
    }

    /// Generated codec tokens → 24 kHz audio, the upstream `S3Codec.decode`:
    /// out-of-range ids dropped, three silence tokens appended (S3Gen eats
    /// that much lookahead), Gaussian flow noise, fade-in.
    pub fn synthesize(
        &self,
        audio: &[u32],
        cond: &S3Conditioning,
        rng: &RefCell<Rng>,
    ) -> Result<Option<Vec<f32>>> {
        let mut tokens: Vec<u32> = audio
            .iter()
            .copied()
            .filter(|&t| t < SPEECH_VOCAB)
            .collect();
        if tokens.is_empty() {
            return Ok(None);
        }
        tokens.extend([SILENCE_TOKEN; 3]);
        let l = TOKEN_MEL_RATIO * (cond.prompt_token.len() + tokens.len());
        let noise: Vec<f32> = {
            let mut r = rng.borrow_mut();
            (0..MEL * l).map(|_| r.normal()).collect()
        };
        let t0 = std::time::Instant::now();
        let trace = self.flow_with_noise(&tokens, cond, &noise)?;
        let t1 = std::time::Instant::now();
        let mut wav = self.vocode(&trace.mel, SourceNoise::Sampled(rng))?;
        if rlx_ir::env::flag("RLX_KITTENTTS2_TIMING") {
            eprintln!(
                "[kittentts2] timing: flow {:.2}s ({l} frames), vocode {:.2}s",
                (t1 - t0).as_secs_f32(),
                t1.elapsed().as_secs_f32()
            );
        }
        hift::apply_trim_fade(&mut wav);
        Ok(Some(wav))
    }
}
