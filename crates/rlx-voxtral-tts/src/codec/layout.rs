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

//! Codec encoder/decoder block index layout (vLLM-Omni `VoxtralTTSAudioTokenizer`).

use crate::config::CodecArgs;
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderExecBlock {
    Transformer {
        block_idx: usize,
        n_layers: usize,
        window: usize,
    },
    ConvDown {
        block_idx: usize,
        kernel: usize,
        stride: usize,
        in_dim: usize,
        out_dim: usize,
    },
}

/// Execution order for `_forward_encoder` (transformer before downsample conv per stage).
pub fn encoder_execution_plan(cfg: &CodecArgs) -> Result<Vec<EncoderExecBlock>> {
    let enc_kernels = cfg.encoder_convs_kernels();
    let enc_strides = cfg.encoder_convs_strides();
    let enc_lens = cfg.encoder_transformer_lengths();
    ensure!(
        enc_kernels.len() == enc_strides.len() && enc_kernels.len() == enc_lens.len(),
        "encoder conv config mismatch"
    );

    let latent = cfg.latent_dim();
    let mut window = cfg.attn_sliding_window_size;
    let mut block_idx = 0usize;
    let mut plan = Vec::new();

    for (stage, &n_layers) in enc_lens.iter().enumerate() {
        plan.push(EncoderExecBlock::Transformer {
            block_idx,
            n_layers,
            window,
        });
        block_idx += 1;

        let is_last = stage + 1 == enc_lens.len();
        let k = enc_kernels[stage];
        let st = enc_strides[stage];
        if k != 1 || st != 1 || is_last {
            let out_dim = if is_last { latent } else { cfg.dim };
            plan.push(EncoderExecBlock::ConvDown {
                block_idx,
                kernel: k,
                stride: st,
                in_dim: cfg.dim,
                out_dim,
            });
            block_idx += 1;
            if st > 1 {
                window = (window / 2).max(1);
            }
        }
    }
    Ok(plan)
}

/// Decoder transformer block index for encoder stage `stage` (mirror init).
pub fn decoder_transformer_block_for_encoder_stage(stage: usize) -> usize {
    stage * 2 + 1
}

/// Decoder conv block index paired with encoder downsample conv after stage `stage`.
pub fn decoder_conv_block_for_encoder_downsample(stage: usize) -> usize {
    (stage + 1) * 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CodecArgs;

    fn voxtral_codec_args() -> CodecArgs {
        CodecArgs {
            channels: 1,
            sampling_rate: 24000,
            pretransform_patch_size: 240,
            patch_proj_kernel_size: 7,
            semantic_codebook_size: 8192,
            semantic_dim: 256,
            acoustic_codebook_size: 21,
            acoustic_dim: 36,
            dim: 1024,
            hidden_dim: 4096,
            head_dim: 128,
            n_heads: 8,
            n_kv_heads: 8,
            attn_sliding_window_size: 16,
            encoder_transformer_lengths_str: "2,2,2,2".into(),
            encoder_convs_kernels_str: "4,4,4,3".into(),
            encoder_convs_strides_str: "2,2,2,1".into(),
            decoder_transformer_lengths_str: "2,2,2,2".into(),
            decoder_convs_kernels_str: "3,4,4,4".into(),
            decoder_convs_strides_str: "1,2,2,2".into(),
        }
    }

    #[test]
    fn encoder_plan_matches_vllm_block_indices() {
        let cfg = voxtral_codec_args();
        let plan = encoder_execution_plan(&cfg).unwrap();
        let blocks: Vec<_> = plan
            .iter()
            .map(|b| match b {
                EncoderExecBlock::Transformer { block_idx, .. } => (*block_idx, "T"),
                EncoderExecBlock::ConvDown {
                    block_idx, out_dim, ..
                } => (*block_idx, if *out_dim == 292 { "C*" } else { "C" }),
            })
            .collect();
        assert_eq!(
            blocks,
            [
                (0, "T"),
                (1, "C"),
                (2, "T"),
                (3, "C"),
                (4, "T"),
                (5, "C"),
                (6, "T"),
                (7, "C*"),
            ]
        );
    }

    #[test]
    fn decoder_mirror_indices() {
        assert_eq!(decoder_transformer_block_for_encoder_stage(0), 1);
        assert_eq!(decoder_transformer_block_for_encoder_stage(3), 7);
        assert_eq!(decoder_conv_block_for_encoder_downsample(0), 2);
        assert_eq!(decoder_conv_block_for_encoder_downsample(2), 6);
    }
}
