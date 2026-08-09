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

//! `rlx-minimax-h3` command line.
//!
//! ```text
//! rlx-minimax-h3 inspect  --weights <root>
//! rlx-minimax-h3 plan     --weights <root> [--task t2va] [--height H --width W]
//!                         [--num-frames N] [--steps N] [--tokens N]
//! rlx-minimax-h3 decode-audio --weights <root> --latents <file.safetensors>
//!                         [--out out.wav]
//! rlx-minimax-h3 decode-video --weights <root> --latents <file.safetensors>
//!                         [--out frames/]
//! ```
//!
//! `inspect` reads the component configs and reports the architecture. `plan`
//! resolves a request's geometry and prints the packed layout it implies —
//! sequence length, per-modality row counts and the schedule — which is what
//! decides whether a clip fits in memory. `decode-audio` runs the audio VAE on a
//! latent dump.

use crate::config::H3Config;
use crate::layout::{self, AUDIO_CHANNELS, H3Geometry, KeyframeAnchor};
use crate::pipeline::{H3Request, H3Task};
use crate::scheduler::H3Scheduler;
use crate::text_encoder::placeholder_conditioning;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

const USAGE: &str = "\
rlx-minimax-h3 — MiniMax-H3 (Hailuo 3.0) omni-modal video+audio generation

USAGE:
  rlx-minimax-h3 <COMMAND> --weights <CHECKPOINT_ROOT> [OPTIONS]

COMMANDS:
  inspect        Report the architecture of a checkpoint
  plan           Resolve a request and print the packed layout it implies
  decode-audio   Decode an audio latent dump to a WAV file
  decode-video   Decode a video latent dump to PNG frames

OPTIONS:
  --weights <PATH>     MiniMax-H3 checkpoint root (holds model_index.json)
  --task <NAME>        t2va | i2va | fl2va | ref2va          [default: t2va]
  --height <N>         Output height, a multiple of 32
  --width <N>          Output width, a multiple of 32
  --num-frames <N>     Frames at 24 fps, snapped up to 17*n+5  [default: 124]
  --steps <N>          Sampling steps                          [default: 32]
  --tokens <N>         Text conditioning rows to assume        [default: 64]
  --latents <PATH>     Latents (.safetensors) for decode-audio / decode-video
  --out <PATH>         Output file or directory                [default: out.wav]
  -h, --help           Show this message
";

/// Parsed command-line arguments.
struct Args {
    command: String,
    weights: Option<PathBuf>,
    task: H3Task,
    height: Option<usize>,
    width: Option<usize>,
    num_frames: usize,
    steps: usize,
    tokens: usize,
    latents: Option<PathBuf>,
    out: PathBuf,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut out = Args {
        command: String::new(),
        weights: None,
        task: H3Task::T2VA,
        height: None,
        width: None,
        num_frames: 124,
        steps: 32,
        tokens: 64,
        latents: None,
        out: PathBuf::from("out.wav"),
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let want = |name: &str| -> Result<String> {
            args.get(i + 1)
                .cloned()
                .with_context(|| format!("{name} needs a value"))
        };
        match a {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--weights" => {
                out.weights = Some(PathBuf::from(want("--weights")?));
                i += 2;
            }
            "--task" => {
                out.task = H3Task::parse(&want("--task")?)?;
                i += 2;
            }
            "--height" => {
                out.height = Some(want("--height")?.parse().context("--height")?);
                i += 2;
            }
            "--width" => {
                out.width = Some(want("--width")?.parse().context("--width")?);
                i += 2;
            }
            "--num-frames" => {
                out.num_frames = want("--num-frames")?.parse().context("--num-frames")?;
                i += 2;
            }
            "--steps" => {
                out.steps = want("--steps")?.parse().context("--steps")?;
                i += 2;
            }
            "--tokens" => {
                out.tokens = want("--tokens")?.parse().context("--tokens")?;
                i += 2;
            }
            "--latents" => {
                out.latents = Some(PathBuf::from(want("--latents")?));
                i += 2;
            }
            "--out" => {
                out.out = PathBuf::from(want("--out")?);
                i += 2;
            }
            other if other.starts_with('-') => bail!("unknown option `{other}`\n\n{USAGE}"),
            other => {
                if out.command.is_empty() {
                    out.command = other.to_string();
                } else {
                    bail!("unexpected argument `{other}`\n\n{USAGE}");
                }
                i += 1;
            }
        }
    }
    Ok(out)
}

pub fn run(args: &[String]) -> Result<()> {
    if args.is_empty() {
        println!("{USAGE}");
        return Ok(());
    }
    let a = parse(args)?;
    let root = a
        .weights
        .clone()
        .context("--weights is required (the checkpoint root holding model_index.json)")?;

    match a.command.as_str() {
        "inspect" => inspect(&root),
        "plan" => plan(&root, &a),
        "decode-audio" => decode_audio(&root, &a),
        "decode-video" => decode_video(&root, &a),
        "" => {
            println!("{USAGE}");
            Ok(())
        }
        other => bail!("unknown command `{other}`\n\n{USAGE}"),
    }
}

fn inspect(root: &Path) -> Result<()> {
    if !crate::is_minimax_h3_checkpoint(root) {
        eprintln!(
            "warning: {} does not look like a MiniMax-H3 checkpoint (no model_index.json)",
            root.display()
        );
    }
    let cfg = H3Config::from_root(root)?;
    let t = &cfg.transformer;
    println!("MiniMax-H3 checkpoint: {}", root.display());
    println!("\ntransformer (joint video+audio DiT)");
    println!("  layers            {}", t.num_layers);
    println!("  hidden            {}", t.hidden_size);
    println!(
        "  heads             {} x {}  (inner {})",
        t.num_attention_heads,
        t.attention_head_dim,
        t.inner_dim()
    );
    println!("  ffn (SwiGLU)      {}", t.ffn_dim);
    println!("  refiner layers    {}", t.num_refiner_layers);
    println!("  patch (t,h,w)     {:?}", t.patch_size);
    println!(
        "  video / audio ch  {} / {}",
        t.in_channels, t.audio_in_channels
    );
    println!("  text_dim          {}", t.text_dim);
    println!(
        "  rope              {} freq/axis, {} of {} channels rotated, theta {}",
        t.rope_freq_dim,
        t.rope_rotary_dim(),
        t.attention_head_dim,
        t.rope_theta
    );
    println!(
        "  adaln table       {} rows/timestep x {} wide",
        t.adaln_rows_per_timestep(),
        6 * t.hidden_size
    );
    let params: usize = crate::weights::dit_parameter_keys(t)
        .iter()
        .filter_map(|k| crate::weights::dit_parameter_shape(t, k))
        .map(|s| s.iter().product::<usize>())
        .sum();
    let adaln: usize = crate::weights::dit_parameter_keys(t)
        .iter()
        .filter(|k| k.contains("adaln_proj"))
        .filter_map(|k| crate::weights::dit_parameter_shape(t, k))
        .map(|s| s.iter().product::<usize>())
        .sum();
    println!(
        "  parameters        {:.2}B ({:.2}B in the AdaLN branches)",
        params as f64 / 1e9,
        adaln as f64 / 1e9
    );
    println!(
        "  ref2va partition  {}",
        if cfg.transformer_ref.is_some() {
            "present"
        } else {
            "absent"
        }
    );

    println!("\nvideo VAE");
    println!(
        "  compression       {}x spatial, {}x temporal",
        cfg.vae.spatial_compression(),
        cfg.vae.temporal_compression()
    );
    println!("  latent channels   {}", cfg.vae.latent_channels);
    println!(
        "  ViT decoder       {} layers, {} wide, {} register tokens",
        cfg.vae.decoder_num_layers,
        cfg.vae.decoder_hidden_size(),
        cfg.vae.decoder_num_register_tokens
    );

    println!("\naudio VAE");
    println!("  sampling rate     {} Hz", cfg.audio_vae.sampling_rate);
    println!(
        "  hop               {} samples ({} latents/s)",
        cfg.audio_vae.decoder_hop(),
        cfg.audio_vae.sampling_rate / cfg.audio_vae.decoder_hop().max(1)
    );
    println!("  latent channels   {}", cfg.audio_vae.latent_channels);

    println!("\ntext encoder (Qwen3-VL)");
    println!("  layers            {}", cfg.text_encoder.num_hidden_layers);
    println!("  hidden            {}", cfg.text_encoder.hidden_size);
    println!(
        "  conditioning tap  layer {} (unnormalized)",
        crate::config::H3TextEncoderConfig::TAP_LAYER
    );

    println!("\nschedulers");
    println!("  video shift       {}", cfg.scheduler.shift);
    println!("  audio shift       {}", cfg.audio_scheduler.shift);
    Ok(())
}

fn plan(root: &Path, a: &Args) -> Result<()> {
    let cfg = H3Config::from_root(root)?;
    let t = &cfg.transformer;
    let multiple = cfg.vae.spatial_compression() * t.patch_size[2];

    let (height, width) = match (a.height, a.width) {
        (Some(h), Some(w)) => (h, w),
        (None, None) => layout::resolve_canvas_size(
            16.0,
            9.0,
            multiple,
            layout::CANVAS_SHORT_EDGE,
            layout::CANVAS_MAX_PIXELS,
        )?,
        _ => bail!("--height and --width must be given together, or neither"),
    };

    let geometry = H3Geometry::resolve(
        height,
        width,
        a.num_frames,
        cfg.vae.spatial_compression(),
        t.patch_size[2],
    )?;
    let anchors: Vec<KeyframeAnchor> = match a.task {
        H3Task::I2VA => vec![KeyframeAnchor::First],
        H3Task::FL2VA => vec![KeyframeAnchor::First, KeyframeAnchor::Last],
        _ => Vec::new(),
    };
    let mut request = H3Request::t2va(geometry, a.steps);
    request.task = a.task;
    request.keyframe_anchors = anchors;
    if a.task == H3Task::Ref2VA {
        bail!("`plan` does not model ref2va reference blocks; use the library API");
    }

    let conditioning = placeholder_conditioning(a.tokens, t.text_dim);
    let l = request.build_layout(&conditioning, t.patch_size)?;

    println!("task              {}", a.task.as_str());
    println!("canvas            {width}x{height} (multiple of {multiple})");
    println!(
        "frames            {} at {} fps = {:.3} s",
        geometry.num_frames,
        layout::FPS,
        geometry.num_frames as f64 / layout::FPS
    );
    println!(
        "latents           {} frames x {}x{}",
        geometry.num_latent_frames, geometry.latent_height, geometry.latent_width
    );
    println!(
        "audio latents     {} per channel x {AUDIO_CHANNELS} channels",
        geometry.num_audio_latents
    );
    println!("\npacked sequence   {} rows", l.sequence_length());
    println!("  text            {}", l.text_indices.len());
    println!(
        "  video           {} ({} conditioning)",
        l.video_indices.len(),
        l.num_condition_video_rows
    );
    println!(
        "  audio           {} ({} reference)",
        l.audio_indices.len(),
        l.num_condition_audio_rows
    );

    let mut vs = H3Scheduler::new(cfg.scheduler.shift)?;
    let mut as_ = H3Scheduler::new(cfg.audio_scheduler.shift)?;
    vs.set_timesteps(a.steps)?;
    as_.set_timesteps(a.steps)?;
    println!(
        "\nschedules         video {} steps (shift {}), audio {} steps (shift {})",
        vs.num_inference_steps(),
        vs.shift(),
        as_.num_inference_steps(),
        as_.shift()
    );

    // The residual stream is what actually decides whether this fits.
    let stream = l.sequence_length() * t.hidden_size * 4;
    println!(
        "\nresidual stream   {:.2} GiB per activation ({} rows x {} x f32)",
        stream as f64 / (1u64 << 30) as f64,
        l.sequence_length(),
        t.hidden_size
    );
    Ok(())
}

fn decode_audio(root: &Path, a: &Args) -> Result<()> {
    use crate::vae_audio::{H3AudioDecoder, Signal, denormalize_latents};
    let latents_path = a
        .latents
        .clone()
        .context("decode-audio needs --latents <file.safetensors>")?;
    let cfg = crate::config::H3AudioVaeConfig::from_dir(&root.join("audio_vae"))?;
    let weights = crate::weights::load_audio_vae(root)?;
    let decoder = H3AudioDecoder::load(&cfg, &weights)?;

    let bytes =
        std::fs::read(&latents_path).with_context(|| format!("read {}", latents_path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse {}", latents_path.display()))?;
    let t = st
        .tensor("latents")
        .context("the latent file needs a `latents` tensor of [channels, frames]")?;
    let shape = t.shape();
    if shape.len() != 2 {
        bail!("`latents` must be [channels, frames], got {shape:?}");
    }
    let data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut sig = Signal::from_data(shape[0], data)?;
    denormalize_latents(&mut sig, &cfg)?;

    let wave = decoder.decode_mono(&sig)?;
    write_wav(&a.out, &wave, cfg.sampling_rate as u32)?;
    println!(
        "wrote {} ({} samples, {:.2} s at {} Hz)",
        a.out.display(),
        wave.len(),
        wave.len() as f64 / cfg.sampling_rate as f64,
        cfg.sampling_rate
    );
    Ok(())
}

fn decode_video(root: &Path, a: &Args) -> Result<()> {
    use crate::vae_video::{compile_video_decoder, denormalize_latents, to_display_range};
    let latents_path = a
        .latents
        .clone()
        .context("decode-video needs --latents <file.safetensors>")?;
    let cfg = crate::config::H3VideoVaeConfig::from_dir(&root.join("vae"))?;

    let bytes =
        std::fs::read(&latents_path).with_context(|| format!("read {}", latents_path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse {}", latents_path.display()))?;
    let t = st
        .tensor("latents")
        .context("the latent file needs a `latents` tensor of [channels, frames, height, width]")?;
    let shape = t.shape();
    if shape.len() != 4 {
        bail!("`latents` must be [channels, frames, height, width], got {shape:?}");
    }
    let (c, frames, lh, lw) = (shape[0], shape[1], shape[2], shape[3]);
    if c != cfg.latent_channels {
        bail!(
            "`latents` has {c} channels, the VAE expects {}",
            cfg.latent_channels
        );
    }
    let mut data: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let g = cfg.chunk_geometry(frames);
    if !g.is_decodable() {
        bail!(
            "a clip of {frames} latent frames is shorter than one {}-frame decode window",
            g.window_frames
        );
    }
    denormalize_latents(&mut data, &cfg, frames * lh * lw)?;

    // Tiling is on by default in the released pipeline; a canvas that fits in
    // one tile decodes as a single window.
    let ratio = cfg.spatial_compression();
    let tile_px = crate::vae_video::TILE_MIN_SIZE;
    let overlap_px = crate::vae_video::TILE_MIN_OVERLAP;
    let y = crate::vae_video::split_tiles(lh * ratio, tile_px, overlap_px, ratio)?;
    let x = crate::vae_video::split_tiles(lw * ratio, tile_px, overlap_px, ratio)?;
    let (tile_lh, tile_lw) = (y.length / ratio, x.length / ratio);

    let mut weights = crate::weights::load_video_vae_decoder(root, &cfg)?;
    let mut dec = compile_video_decoder(
        &cfg,
        &mut weights,
        rlx_runtime::Device::Cpu,
        g.window_frames,
        tile_lh,
        tile_lw,
    )?;
    println!(
        "decoding {}x{} in a {}x{} tile grid ({} latent frames -> {} pixel frames)",
        lh * ratio,
        lw * ratio,
        y.count(),
        x.count(),
        frames,
        g.num_pixel_frames
    );
    let volume =
        crate::vae_video::decode_clip_tiled(&mut dec, &data, frames, lh, lw, tile_px, overlap_px)?;
    let (oh, ow) = (volume.height, volume.width);
    let mut pixels = volume.data;
    to_display_range(&mut pixels, cfg.out_channels)?;

    let frame_px = cfg.out_channels * oh * ow;
    std::fs::create_dir_all(&a.out).with_context(|| format!("create {}", a.out.display()))?;
    for f in 0..g.num_pixel_frames {
        let frame = &pixels[f * frame_px..(f + 1) * frame_px];
        let mut img = image::RgbImage::new(ow as u32, oh as u32);
        for (y, x, px) in img
            .enumerate_pixels_mut()
            .map(|(x, y, p)| (y as usize, x as usize, p))
        {
            let at = |ch: usize| {
                (frame[(ch * oh + y) * ow + x] * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8
            };
            *px = image::Rgb([at(0), at(1), at(2)]);
        }
        let path = a.out.join(format!("frame_{f:05}.png"));
        img.save(&path)
            .with_context(|| format!("write {}", path.display()))?;
    }
    println!(
        "wrote {} frames of {ow}x{oh} to {}",
        g.num_pixel_frames,
        a.out.display()
    );
    Ok(())
}

/// Write a mono 16-bit PCM WAV.
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_full_command_line() {
        let args: Vec<String> = [
            "plan",
            "--weights",
            "/tmp/h3",
            "--task",
            "fl2va",
            "--height",
            "768",
            "--width",
            "1344",
            "--num-frames",
            "124",
            "--steps",
            "16",
            "--tokens",
            "32",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let a = parse(&args).unwrap();
        assert_eq!(a.command, "plan");
        assert_eq!(a.weights, Some(PathBuf::from("/tmp/h3")));
        assert_eq!(a.task, H3Task::FL2VA);
        assert_eq!((a.height, a.width), (Some(768), Some(1344)));
        assert_eq!(a.steps, 16);
        assert_eq!(a.tokens, 32);
    }

    #[test]
    fn rejects_unknown_options_and_tasks() {
        assert!(parse(&["--nope".to_string()]).is_err());
        assert!(
            parse(&["--task".to_string(), "nonsense".to_string()]).is_err(),
            "an unknown task must be rejected"
        );
    }

    #[test]
    fn missing_values_are_reported() {
        assert!(parse(&["--weights".to_string()]).is_err());
        assert!(parse(&["--steps".to_string()]).is_err());
    }

    #[test]
    fn defaults_are_the_released_settings() {
        let a = parse(&["inspect".to_string()]).unwrap();
        assert_eq!(a.task, H3Task::T2VA);
        assert_eq!(a.num_frames, 124);
        assert_eq!(a.steps, 32);
        assert!(a.height.is_none() && a.width.is_none());
    }

    #[test]
    fn wav_header_is_well_formed() {
        let dir = std::env::temp_dir().join("rlx_h3_wav_check");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        write_wav(&path, &[0.0, 0.5, -0.5, 1.0], 32_000).unwrap();
        let b = std::fs::read(&path).unwrap();
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(&b[8..12], b"WAVE");
        assert_eq!(&b[36..40], b"data");
        assert_eq!(b.len(), 44 + 4 * 2);
        // 32 kHz, mono, 16-bit.
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 32_000);
        assert_eq!(u16::from_le_bytes(b[22..24].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(b[34..36].try_into().unwrap()), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wav_clamps_out_of_range_samples() {
        let dir = std::env::temp_dir().join("rlx_h3_wav_clamp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        write_wav(&path, &[9.0, -9.0], 32_000).unwrap();
        let b = std::fs::read(&path).unwrap();
        assert_eq!(i16::from_le_bytes(b[44..46].try_into().unwrap()), i16::MAX);
        assert_eq!(i16::from_le_bytes(b[46..48].try_into().unwrap()), -i16::MAX);
        std::fs::remove_file(&path).ok();
    }
}
