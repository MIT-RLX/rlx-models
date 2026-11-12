// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Student shape. Fixed by `rlx-kroko`'s exporter and the teacher: the conv
//! embed (192 wide), relative positions (48, 4 per head), the 45/32-frame
//! chunk, the 512-wide joiner space.

/// A streaming Zipformer2 the MCU core can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StudentConfig {
    /// Width per stack.
    pub dims: Vec<usize>,
    /// Layers per stack.
    pub layers: Vec<usize>,
    /// Frame-rate divisor per stack.
    pub downs: Vec<usize>,
    /// Attention heads per stack.
    pub heads: Vec<usize>,
    /// Conv kernel per stack.
    pub kernels: Vec<usize>,
    /// Feed-forward widths per stack (three modules).
    pub ff: Vec<[usize; 3]>,
    /// Query/key width per head.
    pub qdim: usize,
    /// Value width per head.
    pub vdim: usize,
    /// Quantization-aware training: the layer matrices are rounded in the
    /// forward pass as the MCU export stores them at `Some(bits)` (4: one
    /// scale per 32 values, else per row), gradients pass straight through.
    pub qat_bits: Option<u8>,
}

/// Fbank frames per chunk, with the 13-frame lookahead.
pub const CHUNK_SIZE: usize = 45;
/// Fbank frames a chunk advances (320 ms).
pub const CHUNK_SHIFT: usize = 32;
/// Zipformer frames per chunk (after the 2× embed).
pub const FRAMES: usize = CHUNK_SHIFT / 2;
/// Encoder output frames per chunk (after the final 2× downsample).
pub const OUT_FRAMES: usize = FRAMES / 2;
/// Fbank bins.
pub const N_MELS: usize = 80;
/// Conv embed output width (fixed by the exporter).
pub const EMBED_DIM: usize = 192;
/// Relative position encoding width.
pub const POS_DIM: usize = 48;
/// Position query width per head.
pub const POS_HEAD: usize = 4;
/// Left context of a stack at full rate (Zipformer frames).
pub const LEFT_BASE: usize = 64;

impl StudentConfig {
    /// The ~10M-parameter encoder: widths 192,160,192,256,192,160, layers
    /// 1,1,2,2,2,1, the teacher's rates, kernels and head shapes.
    pub fn small10m() -> Self {
        let dims = vec![192, 160, 192, 256, 192, 160];
        let ff = dims
            .iter()
            .map(|&d| {
                let r = |m: f64| ((d as f64 * m / 32.0).round() as usize) * 32;
                [r(2.0), r(8.0 / 3.0), r(10.0 / 3.0)]
            })
            .collect();
        Self {
            dims,
            layers: vec![1, 1, 2, 2, 2, 1],
            downs: vec![1, 2, 4, 8, 4, 2],
            heads: vec![4; 6],
            kernels: vec![31, 31, 15, 15, 15, 31],
            ff,
            qdim: 32,
            vdim: 12,
            qat_bits: None,
        }
    }

    /// Stacks.
    pub fn n_stacks(&self) -> usize {
        self.dims.len()
    }

    /// Left context of stack `s` (frames at its own rate).
    pub fn left(&self, s: usize) -> usize {
        LEFT_BASE / self.downs[s]
    }

    /// Widest stack (the final concat width).
    pub fn max_dim(&self) -> usize {
        *self.dims.iter().max().unwrap_or(&0)
    }

    /// Fbank frames of a `k`-chunk segment.
    pub fn segment_frames(k: usize) -> usize {
        (k - 1) * CHUNK_SHIFT + CHUNK_SIZE
    }

    /// Native tensor-name prefix of layer `l` of stack `s`.
    pub fn layer_prefix(s: usize, l: usize) -> String {
        if s == 0 {
            format!("encoder.encoder.encoders.0.layers.{l}.")
        } else {
            format!("encoder.encoder.encoders.{s}.encoder.layers.{l}.")
        }
    }

    /// The `config.json` `rlx-kroko`'s exporter reads, with the teacher's
    /// decoder settings.
    pub fn to_json(&self, vocab: usize, context: usize) -> serde_json::Value {
        serde_json::json!({
            "model_type": "zipformer2",
            "sample_rate": 16000,
            "feature_dim": N_MELS,
            "chunk_size": CHUNK_SIZE,
            "chunk_shift": CHUNK_SHIFT,
            "subsampling_factor": 4,
            "encoder_dims": self.dims,
            "query_head_dims": vec![self.qdim; self.n_stacks()],
            "value_head_dims": vec![self.vdim; self.n_stacks()],
            "num_heads": self.heads,
            "num_encoder_layers": self.layers,
            "cnn_module_kernels": self.kernels,
            "left_context_len": (0..self.n_stacks()).map(|s| self.left(s)).collect::<Vec<_>>(),
            "downsampling_factors": self.downs,
            "vocab_size": vocab,
            "context_size": context,
            "blank_id": 0,
            "unk_id": 2,
            "variant": "rlx-zipformer-distill student",
        })
    }
}
