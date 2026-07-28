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

//! Qwen3.5 VLM vision encoder — mmproj load, CPU forward, multimodal helpers.

mod builder;
mod config;
mod encoder;
mod flow;
mod multimodal;
mod preprocess;
mod weights;

pub use builder::{build_qwen35_vision_graph, build_qwen35_vision_hir};
pub use config::MmProjConfig;
pub use encoder::{Qwen35VisionEncoder, VisionEncodeOutput, load_vision_encoder};
pub use flow::{Qwen35VisionFlow, build_qwen35_vision_built};
pub use multimodal::{
    MEDIA_MARKER, MultimodalPrefill, MultimodalPrompt, VISION_END, VISION_START,
    build_multimodal_mrope_sections, image_chunk_n_pos, image_decoder_pos,
    merge_text_and_vision_embd,
};
pub use preprocess::{build_vision_positions, preprocess_rgb};
pub use weights::MmProjWeights;

#[cfg(feature = "qwen35-vlm")]
#[allow(unused_imports)] // public API re-exports
pub use preprocess::{resize_rgb_nearest, rgb_to_nchw_f32, smart_resize};
#[cfg(feature = "qwen35-vlm")]
#[allow(unused_imports)] // public API re-exports
pub use weights::{DeepstackWeights, VisionBlockWeights};

#[cfg(feature = "qwen35-vlm")]
pub use encoder::encode_image_file;
#[cfg(feature = "qwen35-vlm")]
pub use preprocess::load_rgb_image;

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::{Device, Session};

    fn tiny_cfg() -> MmProjConfig {
        MmProjConfig {
            patch_size: 2,
            n_embd: 16,
            n_head: 2,
            n_layer: 1,
            image_size: 4,
            image_min_pixels: 16,
            image_max_pixels: 256,
            n_merge: 2,
            eps: 1e-6,
            projector_type: "qwen3vl".into(),
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            spatial_merge_size: 2,
            llm_hidden_size: 32,
            n_ff: 32,
            deepstack_layers: vec![],
        }
    }

    fn ramp_rgb(w: usize, h: usize) -> Vec<u8> {
        (0..(w * h * 3)).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn synthetic_vision_forward_produces_finite_embeddings() {
        let cfg = tiny_cfg();
        let weights = MmProjWeights::synthetic(&cfg);
        let img_w = 4;
        let img_h = 4;

        let (graph, params) =
            build_qwen35_vision_graph(&cfg, &weights, img_w, img_h).expect("build graph");
        let opts = rlx_core::flow_bridge::compile_options_for_profile(
            &rlx_flow::CompileProfile::qwen35_prefill(),
            Device::Cpu,
        );
        let mut compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
        for (k, v) in &params {
            compiled.set_param(k, v);
        }

        let rgb = ramp_rgb(img_w, img_h);
        let (nchw, tw, th) = preprocess_rgb(&rgb, img_w, img_h, &cfg);
        assert_eq!((tw, th), (4, 4));

        let position_hw = preprocess::build_vision_position_hw(tw, th, &cfg);
        let (rope_cos, rope_sin) =
            preprocess::vision_rope_feeds(&position_hw, cfg.n_embd / cfg.n_head);
        let outs = compiled.run(&[
            ("image", &nchw),
            ("vision_rope_cos", &rope_cos),
            ("vision_rope_sin", &rope_sin),
        ]);
        let emb = &outs[0];
        assert_eq!(emb.len(), cfg.n_out_tokens(tw, th) * cfg.llm_hidden_size);
        assert!(emb.iter().all(|v| v.is_finite()), "non-finite embedding");

        let mut enc =
            Qwen35VisionEncoder::from_parts(cfg.clone(), weights, img_w, img_h).expect("encoder");
        let out = enc.encode_rgb(&rgb, img_w, img_h).expect("encode");
        assert_eq!(out.n_tokens, 1);
        assert_eq!(out.grid_x, 1);
        assert_eq!(out.grid_y, 1);
        assert!(out.embeddings.iter().all(|v| v.is_finite()));
    }
}
