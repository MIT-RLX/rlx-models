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

#[allow(unused_imports)]
use crate::{
    BluenessReward, DiamondGuidanceParams, DiamondMethod, Flux2ReferenceConditioning, Flux2Runner,
    Flux2RunnerBuilder, Flux2SampleParams, download_flux2_repo, flow_match_init_timestep,
    flux2_latent_geometry, generate_to_rgb, init_latent_noise, load_rgb_planar, parse_lora_scale,
    sample_rectified_flow, sample_rectified_flow_diamond, write_ppm,
};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::{parse_standard_device, req};
use rlx_runtime::Device;
use std::path::PathBuf;

const FLUX2_USAGE: &str = "(see README — flux2 flags)";

#[derive(Debug, Clone)]
struct Flux2Cli {
    weights: Option<PathBuf>,
    hf_repo: Option<String>,
    config: Option<PathBuf>,
    text_encoder_dir: Option<PathBuf>,
    vae_dir: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    prompt: Option<String>,
    negative_prompt: Option<String>,
    cfg_scale: Option<f32>,
    device: Device,
    compiled_text_encoder: bool,
    compiled_vae: bool,
    compiled_denoiser: bool,
    skip_text_encoder: bool,
    aot_cache: Option<PathBuf>,
    lora: Option<PathBuf>,
    lora_scale: f32,
    use_flow_api: bool,
    reuse_session: bool,
    image_path: Option<PathBuf>,
    image_paths: Vec<PathBuf>,
    image_strength: f32,
    pixel_width: Option<usize>,
    pixel_height: Option<usize>,
    batch: usize,
    img_seq: Option<usize>,
    latent_h: Option<usize>,
    latent_w: Option<usize>,
    txt_seq: usize,
    steps: usize,
    seed: u64,
    output: Option<PathBuf>,
    nvfp4: Option<bool>,
    dry: bool,
    diamond_guidance: bool,
    diamond_method: String,
    diamond_reward: String,
    diamond_mc_samples: usize,
    diamond_inner_steps: usize,
    diamond_guidance_steps: usize,
    diamond_reward_scale: f32,
    diamond_snr_factor: f32,
    diamond_decode_reward: bool,
    diamond_no_flow_map: bool,
    diamond_theorem_weights: bool,
    diamond_no_likelihood: bool,
    diamond_no_score: bool,
    dual_time_embedder: bool,
}

impl Flux2Cli {
    fn parse(args: &[String]) -> Result<Option<Self>> {
        let mut cli = Flux2Cli {
            txt_seq: 128,
            seed: 42,
            lora_scale: 1.0,
            image_strength: 0.75,
            batch: 1,
            device: Device::Cpu,
            weights: None,
            hf_repo: None,
            config: None,
            text_encoder_dir: None,
            vae_dir: None,
            tokenizer_path: None,
            prompt: None,
            negative_prompt: None,
            cfg_scale: None,
            compiled_text_encoder: false,
            compiled_vae: false,
            compiled_denoiser: false,
            skip_text_encoder: false,
            aot_cache: None,
            lora: None,
            use_flow_api: false,
            reuse_session: false,
            image_path: None,
            image_paths: Vec::new(),
            pixel_width: None,
            pixel_height: None,
            img_seq: None,
            latent_h: None,
            latent_w: None,
            steps: 0,
            output: None,
            nvfp4: None,
            dry: false,
            diamond_guidance: false,
            diamond_method: "glass".to_string(),
            diamond_reward: "blueness".to_string(),
            diamond_mc_samples: 4,
            diamond_inner_steps: 10,
            diamond_guidance_steps: 5,
            diamond_reward_scale: 1.0,
            diamond_snr_factor: 5.0,
            diamond_decode_reward: false,
            diamond_no_flow_map: false,
            diamond_theorem_weights: false,
            diamond_no_likelihood: false,
            diamond_no_score: false,
            dual_time_embedder: false,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--weights" => cli.weights = Some(req(args, &mut i)?.into()),
                "--hf-repo" => cli.hf_repo = Some(req(args, &mut i)?),
                "--config" => cli.config = Some(req(args, &mut i)?.into()),
                "--text-encoder" => cli.text_encoder_dir = Some(req(args, &mut i)?.into()),
                "--vae" => cli.vae_dir = Some(req(args, &mut i)?.into()),
                "--tokenizer" => cli.tokenizer_path = Some(req(args, &mut i)?.into()),
                "--prompt" => cli.prompt = Some(req(args, &mut i)?),
                "--negative-prompt" => cli.negative_prompt = Some(req(args, &mut i)?),
                "--cfg-scale" => {
                    cli.cfg_scale = Some(req(args, &mut i)?.parse().context("--cfg-scale: f32")?)
                }
                "--device" => {
                    cli.device = parse_standard_device("flux2", &req(args, &mut i)?)?;
                }
                "--compiled-text-encoder" => {
                    cli.compiled_text_encoder = true;
                    i += 1;
                }
                "--compiled-denoiser" => {
                    cli.compiled_denoiser = true;
                    i += 1;
                }
                "--compiled-vae" => {
                    cli.compiled_vae = true;
                    i += 1;
                }
                "--skip-text-encoder" => {
                    cli.skip_text_encoder = true;
                    i += 1;
                }
                "--aot-cache" => cli.aot_cache = Some(req(args, &mut i)?.into()),
                "--lora" => cli.lora = Some(req(args, &mut i)?.into()),
                "--lora-scale" => cli.lora_scale = crate::parse_lora_scale(&req(args, &mut i)?)?,
                "--use-flow-api" => {
                    cli.use_flow_api = true;
                    i += 1;
                }
                "--reuse-session" => {
                    cli.reuse_session = true;
                    i += 1;
                }
                "--image-path" => cli.image_path = Some(req(args, &mut i)?.into()),
                "--image-paths" => {
                    let raw = req(args, &mut i)?;
                    cli.image_paths = raw.split(',').map(|s| PathBuf::from(s.trim())).collect();
                }
                "--image-strength" => {
                    cli.image_strength = req(args, &mut i)?
                        .parse()
                        .context("--image-strength: f32")?
                }
                "--pixel-width" => {
                    cli.pixel_width =
                        Some(req(args, &mut i)?.parse().context("--pixel-width: usize")?)
                }
                "--pixel-height" => {
                    cli.pixel_height = Some(
                        req(args, &mut i)?
                            .parse()
                            .context("--pixel-height: usize")?,
                    )
                }
                "--batch" => cli.batch = req(args, &mut i)?.parse().context("--batch: usize")?,
                "--img-seq" => {
                    cli.img_seq = Some(req(args, &mut i)?.parse().context("--img-seq: usize")?)
                }
                "--height" => {
                    cli.latent_h = Some(req(args, &mut i)?.parse().context("--height: usize")?)
                }
                "--width" => {
                    cli.latent_w = Some(req(args, &mut i)?.parse().context("--width: usize")?)
                }
                "--txt-seq" => {
                    cli.txt_seq = req(args, &mut i)?.parse().context("--txt-seq: usize")?
                }
                "--steps" => cli.steps = req(args, &mut i)?.parse().context("--steps: usize")?,
                "--seed" => cli.seed = req(args, &mut i)?.parse().context("--seed: u64")?,
                "--output" => cli.output = Some(req(args, &mut i)?.into()),
                "--packed" => cli.nvfp4 = Some(true),
                "--no-nvfp4" => cli.nvfp4 = Some(false),
                "--dry" => {
                    cli.dry = true;
                    i += 1;
                }
                "--diamond-guidance" => {
                    cli.diamond_guidance = true;
                    i += 1;
                }
                "--diamond-method" => cli.diamond_method = req(args, &mut i)?,
                "--diamond-reward" => cli.diamond_reward = req(args, &mut i)?,
                "--diamond-mc-samples" => {
                    cli.diamond_mc_samples = req(args, &mut i)?
                        .parse()
                        .context("--diamond-mc-samples: usize")?
                }
                "--diamond-inner-steps" => {
                    cli.diamond_inner_steps = req(args, &mut i)?
                        .parse()
                        .context("--diamond-inner-steps: usize")?
                }
                "--diamond-guidance-steps" => {
                    cli.diamond_guidance_steps = req(args, &mut i)?
                        .parse()
                        .context("--diamond-guidance-steps: usize")?
                }
                "--diamond-reward-scale" => {
                    cli.diamond_reward_scale = req(args, &mut i)?
                        .parse()
                        .context("--diamond-reward-scale: f32")?
                }
                "--diamond-snr-factor" => {
                    cli.diamond_snr_factor = req(args, &mut i)?
                        .parse()
                        .context("--diamond-snr-factor: f32")?
                }
                "--diamond-decode-reward" => {
                    cli.diamond_decode_reward = true;
                    i += 1;
                }
                "--diamond-no-flow-map" => {
                    cli.diamond_no_flow_map = true;
                    i += 1;
                }
                "--diamond-theorem-weights" => {
                    cli.diamond_theorem_weights = true;
                    i += 1;
                }
                "--diamond-no-likelihood" => {
                    cli.diamond_no_likelihood = true;
                    i += 1;
                }
                "--diamond-no-score" => {
                    cli.diamond_no_score = true;
                    i += 1;
                }
                "--dual-time-embedder" => {
                    cli.dual_time_embedder = true;
                    i += 1;
                }
                "--help" | "-h" => {
                    eprintln!("{FLUX2_USAGE}");
                    return Ok(None);
                }
                other => bail!("unknown flag: {other}"),
            }
        }
        Ok(Some(cli))
    }

    fn resolve_weights(&self) -> Result<PathBuf> {
        if let Some(w) = &self.weights {
            return Ok(w.clone());
        }
        if let Some(repo) = &self.hf_repo {
            let ckpt = crate::download_flux2_repo(repo)?;
            return Ok(ckpt.transformer_weights);
        }
        bail!("--weights or --hf-repo is required")
    }

    fn resolve_latent_grid(&self) -> Result<(usize, usize, usize)> {
        let resolved_img_seq = match (self.img_seq, self.latent_h, self.latent_w) {
            (Some(s), _, _) => s,
            (_, Some(h), Some(w)) => h * w,
            _ => 256,
        };
        let (latent_h, latent_w) = match (self.latent_h, self.latent_w) {
            (Some(h), Some(w)) => (h, w),
            _ => {
                let s = resolved_img_seq.isqrt();
                if s * s != resolved_img_seq {
                    bail!(
                        "latent grid: pass --height and --width, or use --img-seq that is a perfect square (got {resolved_img_seq})"
                    );
                }
                (s, s)
            }
        };
        if latent_h * latent_w != resolved_img_seq {
            bail!("--height * --width must equal latent patch count ({resolved_img_seq})");
        }
        Ok((resolved_img_seq, latent_h, latent_w))
    }

    fn open_session(&self) -> Result<crate::Flux2Session> {
        flux2_guard_compiled_te(self.device, self.compiled_text_encoder)?;
        let weights = self.resolve_weights()?;
        let (img_seq, _, _) = self.resolve_latent_grid()?;
        let mut builder = Flux2Runner::builder()
            .weights(&weights)
            .batch(self.batch)
            .img_seq(img_seq)
            .txt_seq(self.txt_seq)
            .device(self.device);
        if self.compiled_denoiser {
            builder = builder.compiled_denoiser(true);
        }
        if self.compiled_text_encoder {
            builder = builder.compiled_text_encoder(true);
        }
        if self.compiled_vae {
            builder = builder.compiled_vae(true);
        }
        if self.skip_text_encoder {
            builder = builder.skip_text_encoder(true);
        }
        if matches!(
            self.device,
            Device::Cuda | Device::Rocm | Device::Gpu | Device::Vulkan
        ) {
            builder = builder
                .compiled_text_encoder(false)
                .drop_text_encoder_after_encode(true);
        }
        if let Some(use_nvfp4) = self.nvfp4 {
            builder = builder.nvfp4(use_nvfp4);
        }
        if let Some(cfg) = &self.config {
            builder = builder.config_path(cfg);
        }
        if let Some(te) = &self.text_encoder_dir {
            builder = builder.text_encoder_dir(te);
        }
        if let Some(vae) = &self.vae_dir {
            builder = builder.vae_dir(vae);
        }
        if let Some(tok) = &self.tokenizer_path {
            builder = builder.tokenizer_path(tok);
        }
        if let Some(aot) = &self.aot_cache {
            builder = builder.aot_cache_dir(aot);
        }
        if let Some(lora) = &self.lora {
            builder = builder.lora(lora, self.lora_scale);
        }
        if self.dual_time_embedder {
            builder = builder.dual_time_embedder(true);
        }
        if self.use_flow_api {
            builder = builder.use_flow_api(true);
        }
        if self.reuse_session {
            crate::Flux2SessionCache::global().get_or_open(builder)
        } else {
            crate::Flux2Session::open(builder)
        }
    }
}

fn flux2_guard_compiled_te(device: Device, explicit: bool) -> Result<()> {
    if explicit && !matches!(device, Device::Metal | Device::Mlx) {
        bail!(
            "--compiled-text-encoder on {device:?} can take hours and exhaust VRAM; \
             native CPU TE is the default on CUDA/ROCm/wgpu/Vulkan"
        );
    }
    Ok(())
}

fn flux2_log_loaded(runner: &Flux2Runner, device: Device) {
    let cfg = runner.config();
    eprintln!(
        "[rlx-flux2] loaded — inner_dim={} double_layers={} single_layers={} guidance={} \
         text_encoder={} vae={} nvfp4={} denoiser={} te={} vae_dec={} device={:?}",
        cfg.inner_dim(),
        cfg.num_layers,
        cfg.num_single_layers,
        cfg.guidance_embeds,
        runner.has_text_encoder(),
        runner.has_vae(),
        runner.uses_nvfp4(),
        if runner.uses_compiled_denoiser() {
            "compiled"
        } else {
            "native-cpu"
        },
        if runner.uses_compiled_text_encoder() {
            "compiled"
        } else {
            "native-cpu"
        },
        if runner.has_vae() {
            if runner.uses_compiled_vae() {
                "compiled"
            } else {
                "native-cpu"
            }
        } else {
            "none"
        },
        device,
    );
}

fn flux2_execute(runner: &Flux2Runner, cli: &Flux2Cli) -> Result<()> {
    use crate::{
        Flux2ReferenceConditioning, Flux2SampleParams, flow_match_init_timestep,
        flux2_latent_geometry, generate_to_rgb, init_latent_noise, load_rgb_planar,
        sample_rectified_flow, write_ppm,
    };

    let cfg = runner.config();
    let batch = cli.batch;
    let (resolved_img_seq, latent_h, latent_w) = cli.resolve_latent_grid()?;
    let pixel_w = cli.pixel_width.unwrap_or(latent_w * 16);
    let pixel_h = cli.pixel_height.unwrap_or(latent_h * 16);
    let (_, _, eff_h, eff_w) = flux2_latent_geometry(pixel_h, pixel_w);

    if cli.dry {
        eprintln!("[rlx-flux2] --dry set; skipping forward pass");
        return Ok(());
    }

    let (pos_encoder, pos_txt_ids, neg_encoder, neg_txt_ids) = if let Some(p) = &cli.prompt {
        eprintln!(
            "[rlx-flux2] encoding prompt ({txt_seq} tokens)…",
            txt_seq = cli.txt_seq
        );
        let neg = cli.negative_prompt.as_deref().or_else(|| {
            if cli.cfg_scale.is_some_and(|s| s > 1.0) {
                Some("")
            } else {
                None
            }
        });
        runner.encode_prompt_pair(p, neg)?
    } else {
        (
            vec![0.0f32; batch * cli.txt_seq * cfg.joint_attention_dim],
            vec![0.0f32; cli.txt_seq * 4],
            None,
            None,
        )
    };

    let mut reference: Option<Flux2ReferenceConditioning> = None;
    if !cli.image_paths.is_empty() {
        if !runner.has_vae() {
            bail!("--image-paths (edit) requires VAE weights");
        }
        let mut refs = Vec::with_capacity(cli.image_paths.len());
        for p in &cli.image_paths {
            let rgb = load_rgb_planar(p, pixel_w, pixel_h)?;
            refs.push((rgb, pixel_h, pixel_w));
        }
        let slice: Vec<(&[f32], usize, usize)> = refs
            .iter()
            .map(|(rgb, h, w)| (rgb.as_slice(), *h, *w))
            .collect();
        reference =
            Some(runner.prepare_edit_conditioning(&slice, eff_h, eff_w, latent_h, latent_w)?);
    }

    let mut initial_latents: Option<Vec<f32>> = None;
    let mut init_timestep = 0usize;
    if let Some(img_path) = &cli.image_path {
        if !runner.has_vae() {
            bail!("--image-path (img2img) requires VAE weights");
        }
        let rgb = load_rgb_planar(img_path, pixel_w, pixel_h)?;
        let noise = init_latent_noise(batch, resolved_img_seq, cfg.in_channels, cli.seed);
        initial_latents = Some(runner.prepare_img2img_packed(
            &rgb,
            pixel_h,
            pixel_w,
            latent_h,
            latent_w,
            eff_h,
            eff_w,
            &noise,
            cli.image_strength,
            cli.steps.max(1),
        )?);
        init_timestep = flow_match_init_timestep(cli.image_strength, cli.steps.max(1));
    }

    if cli.steps > 0 {
        let scale = cli.cfg_scale.unwrap_or(1.0);
        let guidance = vec![3.5f32; batch];
        let init_slice = initial_latents.as_deref();
        let sample_params = Flux2SampleParams {
            encoder_hidden_states: &pos_encoder,
            encoder_negative: neg_encoder.as_deref(),
            txt_ids: &pos_txt_ids,
            neg_txt_ids: neg_txt_ids.as_deref(),
            num_inference_steps: cli.steps,
            cfg_scale: scale,
            guidance: Some(&guidance),
            latent_h,
            latent_w,
            seed: cli.seed,
            init_timestep,
            initial_latents: init_slice,
            reference: reference.as_ref(),
        };
        let t0 = std::time::Instant::now();
        let method = DiamondMethod::parse(&cli.diamond_method)
            .ok_or_else(|| anyhow!("unknown --diamond-method: {}", cli.diamond_method))?;
        let diamond_params = DiamondGuidanceParams {
            method,
            mc_samples: cli.diamond_mc_samples,
            inner_steps: cli.diamond_inner_steps,
            guidance_steps: cli.diamond_guidance_steps,
            reward_scale: cli.diamond_reward_scale,
            snr_factor: cli.diamond_snr_factor,
            decode_reward: cli.diamond_decode_reward,
            use_flow_map: !cli.diamond_no_flow_map,
            include_likelihood: !cli.diamond_no_likelihood,
            include_score: !cli.diamond_no_score,
            include_weights: cli.diamond_theorem_weights,
            seed: cli.seed,
            ..DiamondGuidanceParams::default()
        };
        let blueness = BluenessReward { scale: 1.0 };
        if cli.diamond_guidance && cli.diamond_reward != "blueness" {
            bail!("only --diamond-reward blueness is supported currently");
        }
        if cli.diamond_guidance {
            eprintln!(
                "[rlx-flux2] diamond guidance: method={:?} mc={} inner={} steps={}",
                diamond_params.method,
                diamond_params.mc_samples,
                diamond_params.inner_steps,
                diamond_params.guidance_steps
            );
            if diamond_params.method == DiamondMethod::Weighted
                && diamond_params.use_flow_map
                && cli.lora.is_none()
            {
                eprintln!(
                    "[rlx-flux2] weighted diamond: for flow-map LoRA use --lora (safetensors base); HF default repo {}",
                    crate::diamond::FLOW_MAP_LORA_HF_REPO
                );
            }
        }
        if let Some(out_path) = &cli.output {
            if !runner.has_vae() {
                bail!("--output requires VAE weights (--vae or vae/ next to weights)");
            }
            let (rgb, w_px, h_px) = if cli.diamond_guidance {
                let sample = sample_rectified_flow_diamond(
                    runner,
                    &sample_params,
                    &diamond_params,
                    &blueness,
                )?;
                runner.decode_to_rgb(&sample.latents, &sample.img_ids, latent_h, latent_w)?
            } else {
                generate_to_rgb(runner, &sample_params)?
            };
            write_ppm(out_path, &rgb, w_px, h_px)?;
            eprintln!(
                "[rlx-flux2] flux2 sample+decode in {:?} — wrote {}x{} to {out_path:?}",
                t0.elapsed(),
                w_px,
                h_px
            );
        } else {
            let sample = if cli.diamond_guidance {
                sample_rectified_flow_diamond(runner, &sample_params, &diamond_params, &blueness)?
            } else {
                sample_rectified_flow(runner, &sample_params)?
            };
            let norm: f32 = sample.latents.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!(
                "[rlx-flux2] flux2 sample ({steps} steps) in {:?} — latents len={} ||lat||₂={norm:.4}",
                t0.elapsed(),
                sample.latents.len(),
                steps = cli.steps
            );
        }
        return Ok(());
    }

    let hidden = vec![0.0f32; batch * resolved_img_seq * cfg.in_channels];
    let timestep = vec![0.5f32; batch];
    let guidance = vec![3.5f32; batch];
    let img_ids = crate::prepare_latent_ids(batch, latent_h, latent_w);

    let t0 = std::time::Instant::now();
    let out = if let (Some(scale), Some(neg_e), Some(neg_ids)) =
        (cli.cfg_scale, neg_encoder.as_ref(), neg_txt_ids.as_ref())
    {
        if scale <= 1.0 {
            runner.forward(
                &hidden,
                &pos_encoder,
                &timestep,
                Some(&guidance),
                &img_ids,
                &pos_txt_ids,
            )?
        } else {
            eprintln!("[rlx-flux2] CFG forward (scale={scale})…");
            runner.forward_cfg(
                &hidden,
                &pos_encoder,
                neg_e,
                &timestep,
                Some(&guidance),
                &img_ids,
                &pos_txt_ids,
                neg_ids,
                scale,
            )?
        }
    } else {
        runner.forward(
            &hidden,
            &pos_encoder,
            &timestep,
            Some(&guidance),
            &img_ids,
            &pos_txt_ids,
        )?
    };
    let dt = t0.elapsed();
    let norm: f32 = out.noise_pred.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!(
        "[rlx-flux2] flux2 forward in {dt:?} — noise_pred len={} ||pred||₂={norm:.4}",
        out.noise_pred.len()
    );
    Ok(())
}

pub fn run(args: &[String]) -> Result<()> {
    let Some(cli) = Flux2Cli::parse(args)? else {
        return Ok(());
    };
    let (resolved_img_seq, latent_h, latent_w) = cli.resolve_latent_grid()?;
    eprintln!(
        "[rlx-flux2] flux2: batch={} img_seq={resolved_img_seq} ({latent_h}x{latent_w}) \
         txt_seq={} steps={} prompt={:?}",
        cli.batch, cli.txt_seq, cli.steps, cli.prompt
    );
    let session = cli.open_session()?;
    flux2_log_loaded(session.runner(), cli.device);
    flux2_execute(session.runner(), &cli)
}

pub fn run_serve(args: &[String]) -> Result<()> {
    let Some(mut cli) = Flux2Cli::parse(args)? else {
        return Ok(());
    };
    cli.reuse_session = true;
    eprintln!("[rlx-flux2] flux2-serve: loading model (reuse-session)…");
    let session = cli.open_session()?;
    flux2_log_loaded(session.runner(), cli.device);
    eprintln!("[rlx-flux2] flux2-serve: ready — one JSON object per stdin line");
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        let n = handle.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: serde_json::Value =
            serde_json::from_str(trimmed).with_context(|| format!("invalid JSON: {trimmed}"))?;
        let mut req_cli = cli.clone();
        if let Some(p) = req.get("prompt").and_then(|v| v.as_str()) {
            req_cli.prompt = Some(p.to_string());
        }
        if let Some(p) = req.get("negative_prompt").and_then(|v| v.as_str()) {
            req_cli.negative_prompt = Some(p.to_string());
        }
        if let Some(v) = req.get("cfg_scale").and_then(|v| v.as_f64()) {
            req_cli.cfg_scale = Some(v as f32);
        }
        if let Some(v) = req.get("steps").and_then(|v| v.as_u64()) {
            req_cli.steps = v as usize;
        }
        if let Some(v) = req.get("height").and_then(|v| v.as_u64()) {
            req_cli.latent_h = Some(v as usize);
        }
        if let Some(v) = req.get("width").and_then(|v| v.as_u64()) {
            req_cli.latent_w = Some(v as usize);
        }
        if let Some(v) = req.get("seed").and_then(|v| v.as_u64()) {
            req_cli.seed = v;
        }
        if let Some(p) = req.get("output").and_then(|v| v.as_str()) {
            req_cli.output = Some(PathBuf::from(p));
        }
        if let Some(p) = req.get("image_path").and_then(|v| v.as_str()) {
            req_cli.image_path = Some(PathBuf::from(p));
        }
        if let Some(v) = req.get("image_strength").and_then(|v| v.as_f64()) {
            req_cli.image_strength = v as f32;
        }
        if let Some(arr) = req.get("image_paths").and_then(|v| v.as_array()) {
            req_cli.image_paths = arr
                .iter()
                .filter_map(|v| v.as_str().map(PathBuf::from))
                .collect();
        }
        if let Some(v) = req.get("pixel_width").and_then(|v| v.as_u64()) {
            req_cli.pixel_width = Some(v as usize);
        }
        if let Some(v) = req.get("pixel_height").and_then(|v| v.as_u64()) {
            req_cli.pixel_height = Some(v as usize);
        }
        let t0 = std::time::Instant::now();
        match flux2_execute(session.runner(), &req_cli) {
            Ok(()) => {
                println!(
                    "{{\"ok\":true,\"elapsed_ms\":{}}}",
                    t0.elapsed().as_millis()
                );
            }
            Err(e) => {
                println!(
                    "{{\"ok\":false,\"error\":{},\"elapsed_ms\":{}}}",
                    serde_json::to_string(&format!("{e:#}")).unwrap_or_else(|_| "\"error\"".into()),
                    t0.elapsed().as_millis()
                );
            }
        }
    }
    Ok(())
}
