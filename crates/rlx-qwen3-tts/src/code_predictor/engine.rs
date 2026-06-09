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

//! Code-predictor AR (compiled GPU/MLX/Metal when available, else CPU eager).

use crate::code_predictor::compiled::CpCompiledEngine;
use crate::code_predictor::eager::CpEagerModel;
use crate::config::CodePredictorConfig;
use crate::load::Qwen3TtsWeightStore;
use anyhow::{Context, Result, ensure};
use ndarray::{Array2, ArrayView1};
use rlx_runtime::Device;

fn cp_force_eager() -> bool {
    std::env::var("RLX_QWEN3_TTS_CP_EAGER").ok().as_deref() == Some("1")
}

/// GPU sessions use compiled CP on Metal/CUDA/ROCm. CPU eager: `RLX_QWEN3_TTS_CP_EAGER=1`.
pub fn cp_use_compiled_for_device(talker_device: Device) -> bool {
    if cp_force_eager() {
        return false;
    }
    if std::env::var("RLX_QWEN3_TTS_CP_COMPILED").ok().as_deref() == Some("1") {
        return true;
    }
    if crate::gpu_pipeline::gpu_session_enabled(talker_device) {
        return crate::gpu_pipeline::cp_use_gpu_on_device(talker_device);
    }
    talker_device != Device::Cpu && talker_device != Device::Metal
}

fn cp_execution_device(talker_device: Device) -> Device {
    if !cp_use_compiled_for_device(talker_device) {
        Device::Cpu
    } else {
        crate::compile_opts::cp_compile_device(talker_device)
    }
}

enum CpBackend {
    Eager(CpEagerModel),
    Compiled(CpCompiledEngine),
}

pub struct CodePredictorEngine {
    talker_device: Device,
    cp_device: Device,
    backend: CpBackend,
    talker_codec: Array2<f32>,
    talker_codec_flat: Vec<f32>,
    group_embeds: Vec<Array2<f32>>,
    group_embed_flat: Vec<Vec<f32>>,
    lm_heads: Vec<Array2<f32>>,
    lm_head_flat: Vec<Vec<f32>>,
    lm_head_vocab: Vec<usize>,
    hidden: usize,
}

impl CodePredictorEngine {
    pub fn open(
        store: &Qwen3TtsWeightStore,
        cp: &CodePredictorConfig,
        device: Device,
    ) -> Result<Self> {
        let talker_snap = store.tensor_snapshot(&["talker.model.codec_embedding.weight"])?;
        let (tc_data, tc_shape) = talker_snap
            .get("talker.model.codec_embedding.weight")
            .context("talker codec_embedding")?;
        let talker_codec_flat = tc_data.clone();
        let talker_codec =
            Array2::from_shape_vec((tc_shape[0], tc_shape[1]), talker_codec_flat.clone())?;

        let mut group_embeds = Vec::with_capacity(cp.num_code_groups - 1);
        let mut group_embed_flat = Vec::with_capacity(cp.num_code_groups - 1);
        for i in 0..cp.num_code_groups - 1 {
            let key = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
            let (data, shape) = store.tensor_snapshot(&[&key])?[&key].clone();
            group_embeds.push(Array2::from_shape_vec((shape[0], shape[1]), data.clone())?);
            group_embed_flat.push(data);
        }
        let mut lm_heads = Vec::with_capacity(cp.num_code_groups - 1);
        let mut lm_head_flat = Vec::with_capacity(cp.num_code_groups - 1);
        let mut lm_head_vocab = Vec::with_capacity(cp.num_code_groups - 1);
        for i in 0..cp.num_code_groups - 1 {
            let key = format!("talker.code_predictor.lm_head.{i}.weight");
            let (data, shape) = store.tensor_snapshot(&[&key])?[&key].clone();
            lm_head_vocab.push(shape[0]);
            lm_head_flat.push(data.clone());
            lm_heads.push(Array2::from_shape_vec((shape[0], shape[1]), data)?);
        }

        let cp_device = cp_execution_device(device);
        let backend = if cp_use_compiled_for_device(device) {
            CpBackend::Compiled(CpCompiledEngine::open(
                store.model_dir(),
                store,
                cp,
                cp_device,
            )?)
        } else {
            CpBackend::Eager(CpEagerModel::open(store, cp)?)
        };

        Ok(Self {
            talker_device: device,
            cp_device,
            backend,
            talker_codec,
            talker_codec_flat,
            group_embeds,
            group_embed_flat,
            lm_heads,
            lm_head_flat,
            lm_head_vocab,
            hidden: cp.hidden_size,
        })
    }

    pub fn is_eager(&self) -> bool {
        matches!(self.backend, CpBackend::Eager(_))
    }

    /// Talker codec embedding table flat (row-major `[codec_vocab, hidden]`).
    /// Used by the speculative path to do cheap g0 group-embedding swaps when
    /// synthesising verifier inputs from drafted g0 proposals.
    pub fn talker_codec_flat(&self) -> (&[f32], usize) {
        (&self.talker_codec_flat, self.hidden)
    }

    pub fn device(&self) -> Device {
        self.cp_device
    }

    pub fn cp_backend_label(&self) -> String {
        match &self.backend {
            CpBackend::Eager(_) => "CPU eager".into(),
            CpBackend::Compiled(_) if self.cp_device != self.talker_device => {
                format!("compiled (CPU, talker {:?})", self.talker_device)
            }
            CpBackend::Compiled(_) => format!("compiled ({:?})", self.cp_device),
        }
    }

    pub fn warmup(&mut self, max_frames: usize) -> Result<()> {
        match &mut self.backend {
            CpBackend::Eager(e) => {
                let mut hidden = vec![0f32; self.hidden];
                for (i, v) in hidden.iter_mut().enumerate() {
                    *v = ((i % 17) as f32) * 1e-5;
                }
                let _ = e.predict_groups(
                    &self.talker_codec,
                    &self.group_embeds,
                    &self.lm_heads,
                    ArrayView1::from(&hidden),
                    1995,
                )?;
                Ok(())
            }
            CpBackend::Compiled(c) => c.warmup(max_frames),
        }
    }

    pub fn predict_groups_slice(&mut self, talker_hidden: &[f32], group0: u32) -> Result<Vec<u32>> {
        self.predict_groups(ArrayView1::from(talker_hidden), group0)
    }

    /// CP predict + codec embed sum + pad (fused on eager).
    pub fn predict_groups_fill_emb(
        &mut self,
        talker_hidden: &[f32],
        group0: u32,
        pad: &[f32],
        codec_emb: &mut [f32],
    ) -> Result<Vec<u32>> {
        ensure!(codec_emb.len() == self.hidden);
        match &mut self.backend {
            CpBackend::Eager(e) => e.predict_groups_fill_emb_flat(
                &self.talker_codec_flat,
                &self.group_embed_flat,
                &self.lm_head_flat,
                &self.lm_head_vocab,
                ArrayView1::from(talker_hidden),
                group0,
                pad,
                codec_emb,
                self.hidden,
            ),
            CpBackend::Compiled(c) => {
                let groups = c.predict_groups(
                    &self.talker_codec,
                    &self.group_embeds,
                    &self.lm_heads,
                    ArrayView1::from(talker_hidden),
                    group0,
                )?;
                codec_emb.fill(0.0);
                self.sum_codec_groups_into(&groups, codec_emb)?;
                for (j, v) in pad.iter().enumerate() {
                    codec_emb[j] += *v;
                }
                Ok(groups)
            }
        }
    }

    pub fn predict_groups(
        &mut self,
        talker_hidden: ArrayView1<f32>,
        group0: u32,
    ) -> Result<Vec<u32>> {
        ensure!(talker_hidden.len() == self.hidden);
        match &mut self.backend {
            CpBackend::Eager(e) => e.predict_groups(
                &self.talker_codec,
                &self.group_embeds,
                &self.lm_heads,
                talker_hidden,
                group0,
            ),
            CpBackend::Compiled(c) => c.predict_groups(
                &self.talker_codec,
                &self.group_embeds,
                &self.lm_heads,
                talker_hidden,
                group0,
            ),
        }
    }

    /// Sum codec group embeddings into `out` (group 0 = talker table).
    pub fn sum_codec_groups_into(&self, groups: &[u32], out: &mut [f32]) -> Result<()> {
        ensure!(out.len() == self.hidden, "codec emb buffer len mismatch");
        out.fill(0.0);
        for (gi, &tok) in groups.iter().enumerate() {
            if gi == 0 {
                ensure!(
                    (tok as usize) < self.talker_codec.nrows(),
                    "group0 token {tok} oob"
                );
                for (j, v) in self.talker_codec.row(tok as usize).iter().enumerate() {
                    out[j] += *v;
                }
            } else {
                let table = &self.group_embeds[gi - 1];
                ensure!(
                    (tok as usize) < table.nrows(),
                    "token {tok} oob for group {gi}"
                );
                for (j, v) in table.row(tok as usize).iter().enumerate() {
                    out[j] += *v;
                }
            }
        }
        Ok(())
    }

    pub fn sum_codec_groups(&self, groups: &[u32]) -> Result<Vec<f32>> {
        let mut emb = vec![0f32; self.hidden];
        self.sum_codec_groups_into(groups, &mut emb)?;
        Ok(emb)
    }

    /// Per-step codec embeds for fused backbone (`groups[1..]` → `cp_step_embed_{i}`).
    pub fn cp_step_embeds_from_groups(&self, groups: &[u32]) -> Result<Vec<Vec<f32>>> {
        use crate::cp_frame::CP_DECODE_BACKBONE_STEPS;
        ensure!(
            groups.len() > CP_DECODE_BACKBONE_STEPS,
            "groups len {} < {}",
            groups.len(),
            1 + CP_DECODE_BACKBONE_STEPS
        );
        let mut out = Vec::with_capacity(CP_DECODE_BACKBONE_STEPS);
        for step in 0..CP_DECODE_BACKBONE_STEPS {
            out.push(self.codec_embed_row(step + 1, groups[step + 1])?);
        }
        Ok(out)
    }

    pub fn codec_embed_row(&self, group_idx: usize, token: u32) -> Result<Vec<f32>> {
        if group_idx == 0 {
            ensure!(
                (token as usize) < self.talker_codec.nrows(),
                "group0 token {token} oob"
            );
            return Ok(self.talker_codec.row(token as usize).to_vec());
        }
        let gi = group_idx - 1;
        ensure!(gi < self.group_embeds.len(), "group_idx {group_idx} oob");
        let table = &self.group_embeds[gi];
        ensure!(
            (token as usize) < table.nrows(),
            "token {token} oob for group {group_idx}"
        );
        Ok(table.row(token as usize).to_vec())
    }
}
