// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! GLARE continual pre-training driver.
//!
//! Only the adapter + cross-attention + head params train (the backbone is
//! frozen); the teacher is their EMA. Each step: forward the teacher (clean
//! view) → centered+sharpened targets, one student backward step (blurred view)
//! via `rlx-tune`'s `Trainer`, then EMA-update the teacher.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use rlx_tune::{DpConfig, ParamSlot, Trainer};

use crate::dino::Rng;
use crate::dino::head::{DinoHeadConfig, init_head_params};
use crate::dino::teacher::{Center, ema_update, teacher_targets};
use crate::snapvit::CalibImage;
use crate::vit::config::VitConfig;
use crate::vit::preprocess::{PreprocessWeights, assemble_hidden, rgb_u8_to_imagenet_nchw};
use crate::vit::weights::LoadedVit;

use super::adapter::{AdapterConfig, init_adapter_params};
use super::cross_attn::init_cross_attention_params;
use super::losses::{GlareWeights, build_glare_core, build_glare_student};
use super::patch_aug::strong_blur_patches;
use super::regions::RegionLayout;

/// GLARE hyperparameters.
#[derive(Clone)]
pub struct GlareConfig {
    pub adapter: AdapterConfig,
    pub head: DinoHeadConfig,
    pub n_regions: usize,
    pub tau: f32,
    pub temp_s: f32,
    pub temp_t: f32,
    pub ema_momentum: f32,
    pub center_momentum: f32,
    pub weights: GlareWeights,
    pub blur_frac: f32,
    pub lr: f32,
    pub seed: u64,
}

impl GlareConfig {
    /// Paper-scale config for a backbone of width `hidden` (K=8192 head).
    pub fn new(hidden: usize) -> Self {
        Self {
            adapter: AdapterConfig::default(),
            head: DinoHeadConfig::dino(hidden, 8192),
            n_regions: 8,
            tau: 1.0 / (hidden as f32).sqrt(),
            temp_s: 0.1,
            temp_t: 0.04,
            ema_momentum: 0.996,
            center_momentum: 0.9,
            weights: GlareWeights::default(),
            blur_frac: 0.3,
            lr: 1e-3,
            seed: 0x61A2E,
        }
    }
    /// A tiny config for tests (small head, few regions).
    pub fn small(hidden: usize) -> Self {
        Self {
            adapter: AdapterConfig {
                rank: 8,
                scale: 0.1,
            },
            head: DinoHeadConfig::small(hidden, 24),
            n_regions: 4,
            ..Self::new(hidden)
        }
    }
}

/// A running GLARE trainer (student `Trainer` + EMA teacher).
pub struct GlareTrainer<'a> {
    student: Trainer<'a>,
    teacher: CompiledGraph,
    teacher_params: HashMap<String, Vec<f32>>,
    trainable_names: Vec<String>,
    center_cls: Center,
    center_patch: Center,
    center_reg: Center,
    preprocess: PreprocessWeights,
    ones_head: Vec<f32>,
    ones_ffn: Vec<f32>,
    cfg: VitConfig,
    gc: GlareConfig,
    k: usize,
    n_patch: usize,
    n_regions: usize,
    rng: Rng,
}

impl<'a> GlareTrainer<'a> {
    /// Build the student + teacher graphs, initialize the trainable params, and
    /// upload the frozen backbone.
    pub fn new(
        cfg: &VitConfig,
        loaded: &LoadedVit,
        gc: &GlareConfig,
        total_steps: usize,
        device: Device,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let n_patch = cfg.num_patches();
        let region = RegionLayout::new(n_patch, gc.n_regions);
        // GLARE is a training loop → runs on CPU (Metal/GPU autodiff NaN in the
        // transpose/narrow backward kernels the DINO head uses).
        let device = crate::snapvit::local::backward_device(device, "glare");

        // Student loss graph.
        let core = build_glare_core(cfg, &gc.head, &gc.adapter, &region, gc.tau);
        let student_graph = build_glare_student(core, gc.temp_s, gc.weights);
        let k = student_graph.k;
        let n_regions = student_graph.n_regions;

        // Trainable init (adapter + cross-attention + head).
        let mut trainable: HashMap<String, Vec<f32>> = HashMap::new();
        trainable.extend(init_adapter_params(cfg, &gc.adapter, 1));
        trainable.extend(init_cross_attention_params(h, "glare.ca", 2));
        trainable.extend(init_head_params(&gc.head, "glare.head", 3));
        let trainable_names: Vec<String> = student_graph
            .trainable_params
            .iter()
            .map(|p| p.name.clone())
            .collect();

        // Full param map for the student graph: frozen backbone + region pool +
        // trainable. (Trainer sets all once; re-sets only wrt each step.)
        let mut params: HashMap<String, Vec<f32>> = loaded.params.clone();
        params.insert(student_graph.region_pool.name.clone(), region.pool_matrix());
        for (name, val) in &trainable {
            params.insert(name.clone(), val.clone());
        }
        let wrt: Vec<ParamSlot> = student_graph
            .trainable_params
            .iter()
            .map(|p| ParamSlot {
                name: p.name.clone(),
                node: p.node,
            })
            .collect();

        let dp = DpConfig::new(gc.lr).device(device);
        let student = Trainer::new(student_graph.graph, &wrt, &params, total_steps, None, &dp)?;

        // Teacher forward graph (outputs all_logits), same param names.
        let tcore = build_glare_core(cfg, &gc.head, &gc.adapter, &region, gc.tau);
        let mut teacher = Session::new(device).compile_with(tcore.graph, &CompileOptions::new());
        for p in &tcore.backbone_params {
            teacher.set_param(
                &p.name,
                loaded
                    .params
                    .get(&p.name)
                    .ok_or_else(|| anyhow!("missing backbone param {}", p.name))?,
            );
        }
        teacher.set_param(&tcore.region_pool.name, &region.pool_matrix());

        // Teacher trainable = a copy of the student's initial trainable params.
        let teacher_params = trainable.clone();

        Ok(Self {
            student,
            teacher,
            teacher_params,
            trainable_names,
            center_cls: Center::new(k, gc.center_momentum),
            center_patch: Center::new(k, gc.center_momentum),
            center_reg: Center::new(k, gc.center_momentum),
            preprocess: loaded.preprocess.clone(),
            ones_head: vec![1.0; cfg.num_hidden_layers * h],
            ones_ffn: vec![1.0; cfg.num_hidden_layers * cfg.ffn_inner()],
            cfg: cfg.clone(),
            gc: gc.clone(),
            k,
            n_patch,
            n_regions,
            rng: Rng::new(gc.seed),
        })
    }

    /// Aligned clean (teacher) + strong-blurred (student) views of one image.
    fn make_views(&mut self, img: &CalibImage) -> Result<(Vec<f32>, Vec<f32>)> {
        let sz = self.cfg.img_size;
        let clean = rgb_u8_to_imagenet_nchw(&img.rgb, img.h, img.w, sz);
        let blurred = strong_blur_patches(
            &clean,
            sz,
            self.cfg.patch_size,
            self.gc.blur_frac,
            &mut self.rng,
        );
        let hidden_teacher = assemble_hidden(&self.preprocess, &clean, 1)?;
        let hidden_student = assemble_hidden(&self.preprocess, &blurred, 1)?;
        Ok((hidden_teacher, hidden_student))
    }

    /// One GLARE step over `img`; returns the student total loss.
    pub fn step(&mut self, img: &CalibImage) -> Result<f32> {
        let (hidden_teacher, hidden_student) = self.make_views(img)?;
        let k = self.k;

        // ---- teacher forward (clean view) → targets ----
        for name in &self.trainable_names {
            self.teacher.set_param(name, &self.teacher_params[name]);
        }
        let t_out = self
            .teacher
            .run(&[
                ("hidden", hidden_teacher.as_slice()),
                ("head_mask", self.ones_head.as_slice()),
                ("ffn_mask", self.ones_ffn.as_slice()),
            ])
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("teacher forward produced no output"))?;

        let cls_l = &t_out[0..k];
        let patch_l = &t_out[k..k * (1 + self.n_patch)];
        let reg_l = &t_out[k * (1 + self.n_patch)..k * (1 + self.n_patch + self.n_regions)];
        self.center_cls.update(cls_l, 1, k);
        self.center_patch.update(patch_l, self.n_patch, k);
        self.center_reg.update(reg_l, self.n_regions, k);
        let cls_t = teacher_targets(cls_l, 1, k, self.gc.temp_t, &self.center_cls.c);
        let patch_t = teacher_targets(
            patch_l,
            self.n_patch,
            k,
            self.gc.temp_t,
            &self.center_patch.c,
        );
        let reg_t = teacher_targets(reg_l, self.n_regions, k, self.gc.temp_t, &self.center_reg.c);

        // ---- student backward step (blurred view) ----
        let mut next = |_s: usize, _m: usize| {
            vec![
                ("hidden".to_string(), hidden_student.clone()),
                ("head_mask".to_string(), self.ones_head.clone()),
                ("ffn_mask".to_string(), self.ones_ffn.clone()),
                ("cls_target".to_string(), cls_t.clone()),
                ("patch_target".to_string(), patch_t.clone()),
                ("reg_target".to_string(), reg_t.clone()),
            ]
        };
        let m = self.student.step(&mut next)?;

        // ---- EMA teacher ← student ----
        let student_now = self.student.params();
        ema_update(&mut self.teacher_params, &student_now, self.gc.ema_momentum);

        Ok(m.loss)
    }

    /// Train over `images` for `total_steps` (cycling the data), returning the
    /// per-step loss.
    pub fn train(&mut self, images: &[CalibImage], total_steps: usize) -> Result<Vec<f32>> {
        let mut losses = Vec::with_capacity(total_steps);
        for s in 0..total_steps {
            let img = &images[s % images.len()];
            losses.push(self.step(img)?);
        }
        Ok(losses)
    }

    /// The trained adapter/cross-attention/head params.
    pub fn trained_params(&self) -> HashMap<String, Vec<f32>> {
        self.student.params()
    }

    /// The current EMA teacher trainable params.
    pub fn teacher_params(&self) -> &HashMap<String, Vec<f32>> {
        &self.teacher_params
    }
}
