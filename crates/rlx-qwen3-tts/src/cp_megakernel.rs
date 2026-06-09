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

//! Per-frame CP megakernel: host lm_head + CPU eager 5-layer micro-kernel (~58 ms/frame on Metal).
//!
//! Production path for codec-frame fusion when compiled CP is slower than eager on 0.6B.

use crate::code_predictor::CpEagerModel;
use crate::config::CodePredictorConfig;
use crate::load::Qwen3TtsWeightStore;
use anyhow::{Context, Result, ensure};
use ndarray::ArrayView1;
use rlx_runtime::Device;

/// Greedy CP codec AR — host lm_head + unrolled CPU eager backbone (codec-frame megakernel).
pub struct CpMegakernelGreedy {
    model: CpEagerModel,
    talker_codec_flat: Vec<f32>,
    group_embed_flat: Vec<Vec<f32>>,
    lm_head_flat: Vec<Vec<f32>>,
    lm_head_vocab: Vec<usize>,
    hidden: usize,
}

impl CpMegakernelGreedy {
    pub fn open(store: &Qwen3TtsWeightStore, cp: &CodePredictorConfig) -> Result<Self> {
        let n_groups = cp.num_code_groups - 1;
        let mut keys: Vec<String> = Vec::with_capacity(1 + 2 * n_groups);
        keys.push("talker.model.codec_embedding.weight".into());
        for i in 0..n_groups {
            keys.push(format!(
                "talker.code_predictor.model.codec_embedding.{i}.weight"
            ));
            keys.push(format!("talker.code_predictor.lm_head.{i}.weight"));
        }
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let snap = store.tensor_snapshot(&key_refs)?;
        let (tc_data, _) = snap
            .get("talker.model.codec_embedding.weight")
            .context("talker codec_embedding")?;
        let talker_codec_flat = tc_data.clone();

        let mut group_embed_flat = Vec::with_capacity(n_groups);
        let mut lm_head_flat = Vec::with_capacity(n_groups);
        let mut lm_head_vocab = Vec::with_capacity(n_groups);
        for i in 0..n_groups {
            let emb_key = format!("talker.code_predictor.model.codec_embedding.{i}.weight");
            let (data, _) = snap[&emb_key].clone();
            group_embed_flat.push(data);
            let lh_key = format!("talker.code_predictor.lm_head.{i}.weight");
            let (lh_data, lh_shape) = snap[&lh_key].clone();
            lm_head_vocab.push(lh_shape[0]);
            lm_head_flat.push(lh_data);
        }

        let model = CpEagerModel::open(store, cp)?;
        Ok(Self {
            model,
            talker_codec_flat,
            group_embed_flat,
            lm_head_flat,
            lm_head_vocab,
            hidden: cp.hidden_size,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Exercise full 2-token prefill + 15 AR lm_head steps (JIT / buffer warmup).
    pub fn warmup(&mut self) -> Result<()> {
        let mut hidden = vec![0f32; self.hidden];
        for (i, v) in hidden.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 1e-5;
        }
        let mut codec_emb = vec![0.0; self.hidden];
        let _ = self.predict_groups_fill_emb(&hidden, 1995, &[], &mut codec_emb)?;
        Ok(())
    }

    pub fn predict_groups_fill_emb(
        &mut self,
        talker_hidden: &[f32],
        group0: u32,
        pad: &[f32],
        codec_emb: &mut [f32],
    ) -> Result<Vec<u32>> {
        ensure!(talker_hidden.len() == self.hidden);
        ensure!(codec_emb.len() == self.hidden);
        self.model.predict_groups_fill_emb_flat(
            &self.talker_codec_flat,
            &self.group_embed_flat,
            &self.lm_head_flat,
            &self.lm_head_vocab,
            ArrayView1::from(talker_hidden),
            group0,
            pad,
            codec_emb,
            self.hidden,
        )
    }

    pub fn codec_embed_row(&self, group_idx: usize, token: u32) -> Result<Vec<f32>> {
        let h = self.hidden;
        if group_idx == 0 {
            let rows = self.talker_codec_flat.len() / h;
            ensure!((token as usize) < rows, "group0 token {token} oob");
            let off = token as usize * h;
            return Ok(self.talker_codec_flat[off..off + h].to_vec());
        }
        let gi = group_idx - 1;
        ensure!(
            gi < self.group_embed_flat.len(),
            "group_idx {group_idx} oob"
        );
        let table = &self.group_embed_flat[gi];
        let rows = table.len() / h;
        ensure!(
            (token as usize) < rows,
            "token {token} oob for group {group_idx}"
        );
        let off = token as usize * h;
        Ok(table[off..off + h].to_vec())
    }

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
}

/// CPU eager CP megakernel for codec-frame fusion (default on GPU sessions).
pub fn cp_megakernel_enabled(device: Device) -> bool {
    match std::env::var("RLX_QWEN3_TTS_CP_MEGAKERNEL").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => {
            crate::gpu_pipeline::gpu_session_enabled(device)
                && !crate::code_predictor::engine::cp_use_compiled_for_device(device)
        }
    }
}
