//! Bisect a CPU/accelerator disagreement in the DiT by layer count.
//!
//! `refiner=0 layers=0` exercises only the projections, the packed gather and
//! the two output heads; adding blocks brings in the AdaLN table gather, the
//! gated residuals and partial RoPE.

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

fn run(device: Device, refiner: usize, layers: usize) -> (Vec<f32>, Vec<f32>) {
    let c = cfg(refiner, layers);
    let g = H3Geometry {
        height: 64,
        width: 64,
        num_frames: 39,
        num_latent_frames: 3,
        latent_height: 4,
        latent_width: 4,
        num_audio_latents: 3,
    };
    let layout = build_packed_sequence(&[Modality::Text.tag(); 5], &g, c.patch_size, &[]).unwrap();
    let mut w = synthetic_dit_weights(&c, 31);
    let mut dit = compile_dit(
        &c,
        &mut w,
        device,
        layout.sequence_length(),
        layout.text_indices.len(),
        layout.video_indices.len(),
        layout.audio_indices.len(),
    )
    .unwrap();
    let rows = build_row_timesteps(&layout, 0.3, 0.4, 0.999, 1.0).unwrap();
    let dl = H3DitLayout::new(&layout, &rows, &c).unwrap();
    let t = RopeTables::build(&layout.flat_position_ids(), c.rope_freq_dim, c.rope_theta).unwrap();
    let ramp = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 17) as f32 / 17.0) - 0.5).collect() };
    let video = ramp(layout.video_indices.len() * c.video_patch_dim());
    let audio = ramp(layout.audio_indices.len() * c.audio_in_channels);
    let text = ramp(layout.text_indices.len() * c.text_dim);
    let out = dit
        .forward(&H3DitInputs {
            video_rows: &video,
            audio_rows: &audio,
            text_rows: &text,
            cos: &t.cos,
            sin: &t.sin,
            layout: &dl,
        })
        .unwrap();
    (out.video, out.audio)
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let s = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / s))
}

fn main() {
    #[allow(unused_mut)]
    let mut devices: Vec<(&str, Device)> = Vec::new();
    #[cfg(feature = "metal")]
    devices.push(("metal", Device::Metal));
    #[cfg(feature = "mlx")]
    devices.push(("mlx", Device::Mlx));
    #[cfg(feature = "gpu")]
    devices.push(("wgpu", Device::Gpu));

    for (refiner, layers) in [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)] {
        let (rv, ra) = run(Device::Cpu, refiner, layers);
        let mut line = format!("refiner={refiner} layers={layers}:");
        for (name, d) in &devices {
            let (v, a) = run(*d, refiner, layers);
            line.push_str(&format!(
                "  {name} v={:.2e} a={:.2e}",
                rel(&rv, &v),
                rel(&ra, &a)
            ));
        }
        println!("{line}");
    }
}
