// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// Licensed under GNU GPL v3. See top-level LICENSE.

//! Mimi codec — decoder path used by Pocket TTS.
//!
//! Pocket TTS uses a modified Mimi where the "quantizer" is a `DummyQuantizer`
//! (i.e. a Conv1d 32→512 k=1 projecting continuous FlowLM latents up to the
//! codec's internal dim). The decode path is:
//!
//! ```text
//! latent[B, 32, T]
//!   → quantizer.output_proj  Conv1d(32→512, k=1)
//!   → upsample (depthwise ConvTranspose1d, k=32, stride=16, groups=512)
//!   → decoder_transformer (2-layer, d=512, sliding window=250, layer_scale=0.01)
//!   → SEANetDecoder (ratios=[6,5,4], n_filters=64)
//!   → audio[B, 1, T*1920] at 24 kHz
//! ```

pub mod conv;
pub mod seanet;

use anyhow::Result;
use ndarray::{Array2, Array3};

use crate::config::PocketTtsConfig;
use crate::flow_lm::transformer::StreamingTransformer;
use crate::ops::{PadMode, causal_conv_transpose1d, causal_conv1d};
use crate::weights::WeightFile;

use seanet::SeanetDecoder;

pub struct MimiDecoder {
    cfg: PocketTtsConfig,

    /// `quantizer.output_proj`: Conv1d(inner_dim → outer_dim, k=1, no bias).
    /// Shape `[outer, inner, 1]`.
    output_proj_w: Array3<f32>,

    /// Frame-rate upsample (depthwise ConvTranspose1d).
    /// Shape `[outer, 1, K]`.
    upsample_w: Array3<f32>,

    decoder_transformer: StreamingTransformer,
    decoder: SeanetDecoder,
}

impl MimiDecoder {
    pub fn load(wf: &WeightFile, cfg: PocketTtsConfig) -> Result<Self> {
        let output_proj_w = wf.get_3d("mimi.quantizer.output_proj.weight")?;
        let upsample_w = wf.get_3d("mimi.upsample.convtr.convtr.weight")?;
        let decoder_transformer = StreamingTransformer::load(
            wf,
            "mimi.decoder_transformer.transformer",
            &cfg.mimi.decoder_transformer,
            1e-5,
        )?;
        let decoder = SeanetDecoder::load(wf, "mimi.decoder", &cfg.mimi)?;
        Ok(Self {
            cfg,
            output_proj_w,
            upsample_w,
            decoder_transformer,
            decoder,
        })
    }

    /// Decode a sequence of `[T_lat, ldim]` un-normalized latents (un-normalized
    /// = `latent * emb_std + emb_mean`) into audio samples at 24 kHz.
    pub fn decode_latents(&self, latents: &Array2<f32>) -> Vec<f32> {
        let (t_lat, ldim) = latents.dim();
        let outer = self.cfg.mimi.outer_dim;
        let down_s = self.cfg.mimi.downsample_stride;

        // ── 1) quantizer.output_proj: Conv1d(in=32, out=512, k=1, groups=1) ──
        // Input layout: [ldim, T_lat]; output: [outer, T_lat].
        let mut emb_in = Array2::<f32>::zeros((ldim, t_lat));
        for t in 0..t_lat {
            for c in 0..ldim {
                emb_in[[c, t]] = latents[[t, c]];
            }
        }
        let emb = causal_conv1d(
            emb_in.view(),
            self.output_proj_w.view(),
            None,
            1,
            1,
            PadMode::Constant,
            1,
        );

        // ── 2) upsample: depthwise ConvTranspose1d (groups=outer) ──
        // `emb: [outer, T_lat]` → `[outer, T_lat * down_s]`.
        let emb_up =
            causal_conv_transpose1d(emb.view(), self.upsample_w.view(), None, down_s, outer);

        // ── 3) decoder_transformer: ProjectedTransformer (input_proj=Identity) ──
        // The transformer expects `[T, d_model]` layout. Transpose emb_up.
        let (_, t_up) = emb_up.dim();
        let mut tx_in = Array2::<f32>::zeros((t_up, outer));
        for t in 0..t_up {
            for c in 0..outer {
                tx_in[[t, c]] = emb_up[[c, t]];
            }
        }
        let mut cache = self.decoder_transformer.make_cache();
        let tx_out = self.decoder_transformer.forward(tx_in, &mut cache);

        // Back to `[outer, T]` for the SEANet stack.
        let mut feat = Array2::<f32>::zeros((outer, t_up));
        for t in 0..t_up {
            for c in 0..outer {
                feat[[c, t]] = tx_out[[t, c]];
            }
        }

        // ── 4) SEANet decoder → [1, T_audio] ──
        let audio = self.decoder.forward(feat.view());
        let (channels, t_audio) = audio.dim();
        debug_assert!(channels >= 1);
        let mut out = Vec::with_capacity(t_audio);
        for t in 0..t_audio {
            out.push(audio[[0, t]]);
        }
        out
    }
}
