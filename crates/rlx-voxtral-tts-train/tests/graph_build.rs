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

use rlx_runtime::{Device, Session};
use rlx_voxtral_tts::config::CodecArgs;
use rlx_voxtral_tts_train::codec_graph::{CodecGraphLayout, build_codec_forward_graph};
use rlx_voxtral_tts_train::config::latent_frames;

fn sample_codec() -> CodecArgs {
    CodecArgs {
        channels: 1,
        sampling_rate: 24000,
        pretransform_patch_size: 240,
        patch_proj_kernel_size: 7,
        semantic_codebook_size: 128,
        semantic_dim: 256,
        acoustic_codebook_size: 21,
        acoustic_dim: 36,
        dim: 1024,
        hidden_dim: 4096,
        head_dim: 128,
        n_heads: 8,
        n_kv_heads: 8,
        attn_sliding_window_size: 16,
        encoder_transformer_lengths_str: "1,1".into(),
        encoder_convs_kernels_str: "4,3".into(),
        encoder_convs_strides_str: "2,1".into(),
        decoder_transformer_lengths_str: "1,1".into(),
        decoder_convs_kernels_str: "3,4".into(),
        decoder_convs_strides_str: "1,2".into(),
    }
}

#[test]
fn encoder_forward_graph_builds_and_compiles() {
    let cfg = sample_codec();
    let layout = CodecGraphLayout::new(&cfg, 16);
    assert_eq!(layout.latent_t, latent_frames(&cfg, 16));
    let fwd = build_codec_forward_graph(&cfg, &layout).expect("forward graph");
    assert!(fwd.params.iter().any(|p| p.trainable));

    let session = Session::new(Device::Cpu);
    let _ = session.compile(fwd.graph);
}
