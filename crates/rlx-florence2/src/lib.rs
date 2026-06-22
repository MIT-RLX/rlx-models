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

//! Microsoft Florence-2 vision-language model for RLX.
//!
//! Florence-2 pairs a **DaViT** vision backbone with a **BART** encoder-decoder
//! language model. An image is encoded to 577 visual tokens, concatenated with
//! the task-prompt text embeddings, run through the BART encoder, and the BART
//! decoder autoregressively emits task tokens that a post-processor turns into
//! captions, boxes, or polygons.
//!
//! Weight keys match [`microsoft/Florence-2-large`](https://huggingface.co/microsoft/Florence-2-large).

mod builder;
pub mod cli;
pub mod config;
pub mod flow;
pub mod generate;
mod language;
pub mod postprocess;
pub mod preprocess;
pub mod runner;
pub mod session;
pub mod tokenizer;
mod vision;
mod weight_source;
pub mod weights;

pub use config::{Florence2Config, Florence2TextConfig, Florence2VisionConfig};
pub use generate::GenerateConfig;
pub use postprocess::{BBoxInstance, Florence2Result, PolygonInstance, QuadInstance};
pub use runner::Florence2Model;
pub use session::Florence2Session;
pub use tokenizer::Florence2Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_preset_dims() {
        let cfg = Florence2Config::large();
        assert_eq!(cfg.vision.dim_embed, vec![256, 512, 1024, 2048]);
        assert_eq!(cfg.vision.depths, vec![1, 1, 9, 1]);
        assert_eq!(cfg.vision.window_size, 12);
        assert_eq!(cfg.text.d_model, 1024);
        assert_eq!(cfg.text.decoder_layers, 12);
        assert_eq!(cfg.vocab_size, 51289);
        assert_eq!(cfg.text.decoder_start_token_id, 2);
        assert_eq!(cfg.text.forced_bos_token_id, 0);
    }

    #[test]
    fn task_prompt_expansion() {
        assert_eq!(
            config::construct_prompt("<CAPTION>"),
            "What does the image describe?"
        );
        assert_eq!(
            config::construct_prompt("<CAPTION_TO_PHRASE_GROUNDING>a cat"),
            "Locate the phrases in the caption: a cat"
        );
        assert_eq!(
            config::task_post_processing_type("<OD>"),
            "description_with_bboxes"
        );
        assert_eq!(
            config::task_post_processing_type("<OCR_WITH_REGION>"),
            "ocr"
        );
    }

    #[test]
    fn preprocess_shape_and_range() {
        // 4×4 grey image → 8×8 pixel_values, normalized.
        let rgb = vec![128u8; 4 * 4 * 3];
        let px = preprocess::preprocess_rgb(&rgb, 4, 4, 8);
        assert_eq!(px.len(), 3 * 8 * 8);
        // Constant input → constant normalized output per channel.
        let c0 = px[0];
        assert!(px[..64].iter().all(|&v| (v - c0).abs() < 1e-4));
        assert!(px.iter().all(|v| v.is_finite()));
    }
}
