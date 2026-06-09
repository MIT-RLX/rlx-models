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

//! Embedded Silero VAD 16 kHz weights (`silero_vad_16k.safetensors` from official ONNX).

use anyhow::{Context, Result};
use rlx_core::embedded_safetensors::EmbeddedSafetensors;
use std::path::Path;
use std::sync::OnceLock;

const SAFETENSORS: &[u8] = include_bytes!("../../weights/silero_vad_16k.safetensors");

static PARSED: OnceLock<super::weights::SileroWeights> = OnceLock::new();

pub fn embedded() -> &'static super::weights::SileroWeights {
    PARSED.get_or_init(|| parse_bytes(SAFETENSORS).expect("embedded silero safetensors"))
}

pub fn load_file(path: &Path) -> Result<super::weights::SileroWeights> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_bytes(&bytes)
}

fn parse_bytes(bytes: &[u8]) -> Result<super::weights::SileroWeights> {
    let st = EmbeddedSafetensors::parse(bytes)?;
    let final_w = st.tensor_f32("final_conv.weight")?;
    Ok(super::weights::SileroWeights {
        stft_conv: st.tensor_f32("stft_conv.weight")?,
        conv1_w: st.tensor_f32("conv1.weight")?,
        conv1_b: st.tensor_f32("conv1.bias")?,
        conv2_w: st.tensor_f32("conv2.weight")?,
        conv2_b: st.tensor_f32("conv2.bias")?,
        conv3_w: st.tensor_f32("conv3.weight")?,
        conv3_b: st.tensor_f32("conv3.bias")?,
        conv4_w: st.tensor_f32("conv4.weight")?,
        conv4_b: st.tensor_f32("conv4.bias")?,
        lstm_w_ih: st.tensor_f32("lstm_cell.weight_ih")?,
        lstm_w_hh: st.tensor_f32("lstm_cell.weight_hh")?,
        lstm_b_ih: st.tensor_f32("lstm_cell.bias_ih")?,
        lstm_b_hh: st.tensor_f32("lstm_cell.bias_hh")?,
        final_w,
        final_b: st.tensor_f32("final_conv.bias")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_safetensors_parses() {
        let w = embedded();
        assert_eq!(w.stft_conv.len(), 130 * 128);
        assert_eq!(w.conv1_w.len(), 128 * 65 * 3);
        assert_eq!(w.lstm_w_ih.len(), 512 * 128);
        assert_eq!(w.final_w.len(), 128);
    }
}
