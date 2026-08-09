//! Bisect the DiT graph: run with 0/1/2 refiner and block layers and report
//! where the first non-finite value appears.

use rlx_minimax_h3::config::{H3TransformerConfig, Modality};
use rlx_minimax_h3::layout::{H3Geometry, build_packed_sequence, build_row_timesteps};
use rlx_minimax_h3::rope::RopeTables;
use rlx_minimax_h3::transformer::{H3DitInputs, H3DitLayout, compile_dit};
use rlx_minimax_h3::weights::synthetic_dit_weights;
use rlx_runtime::Device;

fn cfg(refiner: usize, layers: usize) -> H3TransformerConfig {
    H3TransformerConfig {
        num_attention_heads: 2,
        attention_head_dim: 16,
        hidden_size: 24,
        num_layers: layers,
        num_refiner_layers: refiner,
        ffn_dim: 32,
        in_channels: 4,
        audio_in_channels: 6,
        patch_size: [1, 2, 2],
        text_dim: 8,
        freq_dim: 16,
        time_embed_hidden_dim: 24,
        time_embed_dim: 12,
        rope_freq_dim: 2,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        qk_norm_eps: 1e-5,
        final_norm_eps: 1e-5,
    }
}

fn main() -> anyhow::Result<()> {
    let geometry = H3Geometry {
        height: 64,
        width: 64,
        num_frames: 39,
        num_latent_frames: 3,
        latent_height: 4,
        latent_width: 4,
        num_audio_latents: 3,
    };
    let layout = build_packed_sequence(&[Modality::Text.tag(); 5], &geometry, [1, 2, 2], &[])?;
    let nv = layout.video_indices.len();
    let na = layout.audio_indices.len();
    let nt = layout.text_indices.len();
    println!(
        "seq={} text={nt} video={nv} audio={na}",
        layout.sequence_length()
    );

    for (refiner, layers) in [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)] {
        let c = cfg(refiner, layers);
        let mut w = synthetic_dit_weights(&c, 11);
        let mut dit = compile_dit(
            &c,
            &mut w,
            Device::Cpu,
            layout.sequence_length(),
            nt,
            nv,
            na,
        )?;
        let rows = build_row_timesteps(&layout, 0.3, 0.4, 0.999, 1.0)?;
        let dl = H3DitLayout::new(&layout, &rows, &c)?;
        let tables = RopeTables::build(&layout.flat_position_ids(), c.rope_freq_dim, c.rope_theta)?;
        let video: Vec<f32> = (0..nv * c.video_patch_dim())
            .map(|i| (i % 17) as f32 / 17.0 - 0.5)
            .collect();
        let audio: Vec<f32> = (0..na * c.audio_in_channels)
            .map(|i| (i % 17) as f32 / 17.0 - 0.5)
            .collect();
        let text: Vec<f32> = (0..nt * c.text_dim)
            .map(|i| (i % 17) as f32 / 17.0 - 0.5)
            .collect();
        let out = dit.forward(&H3DitInputs {
            video_rows: &video,
            audio_rows: &audio,
            text_rows: &text,
            cos: &tables.cos,
            sin: &tables.sin,
            layout: &dl,
        })?;
        let vb = out.video.iter().filter(|x| !x.is_finite()).count();
        let ab = out.audio.iter().filter(|x| !x.is_finite()).count();
        println!(
            "refiner={refiner} layers={layers}: video nonfinite {vb}/{} audio nonfinite {ab}/{}  v[0]={:?} a[0]={:?}",
            out.video.len(),
            out.audio.len(),
            out.video.first(),
            out.audio.first()
        );
    }
    Ok(())
}
