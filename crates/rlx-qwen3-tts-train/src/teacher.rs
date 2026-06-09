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

//! Teacher hidden states for LoRA distillation (GPU fused prefill when available).

use anyhow::{Result, ensure};
use ndarray::ArrayView2;
use rlx_qwen3_tts::config::TalkerConfig;
use rlx_qwen3_tts::load::Qwen3TtsWeightStore;
use rlx_qwen3_tts::talker::eager::TalkerEagerModel;
use rlx_qwen3_tts::talker::engine::TalkerEngine;
use rlx_runtime::Device;

pub struct TalkerTeacher {
    hidden: usize,
    eager: TalkerEagerModel,
    gpu: Option<TalkerEngine>,
}

impl TalkerTeacher {
    pub fn open(
        store: &Qwen3TtsWeightStore,
        talker: &TalkerConfig,
        device: Device,
    ) -> Result<Self> {
        let hidden = talker.hidden_size;
        let eager = TalkerEagerModel::open(store, talker)?;
        let gpu = if use_gpu_teacher(device) {
            match TalkerEngine::open(store, talker, device) {
                Ok(mut eng) => {
                    eng.warmup(32)?;
                    Some(eng)
                }
                Err(e) => {
                    eprintln!("[jfk-lora] GPU teacher unavailable ({e}) — CPU eager");
                    None
                }
            }
        } else {
            None
        };
        Ok(Self { hidden, eager, gpu })
    }

    pub fn prefill_hidden(&mut self, embeds: ArrayView2<f32>) -> Result<Vec<f32>> {
        let (_seq, h) = embeds.dim();
        ensure!(h == self.hidden);
        let out = if let Some(eng) = &mut self.gpu {
            eng.reset_kv();
            let y = eng.prefill(embeds)?;
            y.iter().copied().collect()
        } else {
            self.eager.reset_kv();
            let y = self.eager.prefill(embeds)?;
            y.iter().copied().collect()
        };
        Ok(out)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }
}

fn use_gpu_teacher(device: Device) -> bool {
    if std::env::var("RLX_QWEN3_TTS_TRAIN_EAGER")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
    {
        return false;
    }
    matches!(
        device,
        Device::Metal | Device::Mlx | Device::Cuda | Device::Gpu
    ) && rlx_runtime::is_available(device)
}
