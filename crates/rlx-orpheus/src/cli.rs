// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use crate::backbone::BackboneLoadOptions;
use crate::download::{
    DEFAULT_ORPHEUS_QUANT, default_hf_cache_dir, default_orpheus_dir, default_snac_dir,
    fetch_default, fetch_orpheus_gguf, fetch_snac_raw, print_snac_export_hint,
};
use crate::{GenerationConfig, OrpheusTts, VOICES, resolve_orpheus_device};
use anyhow::{Context, Result, anyhow, bail};
use rlx_cli::req;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut snac: Option<PathBuf> = None;
    let mut device = "auto".to_string();
    let mut text: Option<String> = None;
    let mut voice: Option<String> = None;
    let mut out_wav: Option<PathBuf> = None;
    let mut max_tokens = 1200u32;
    let mut temperature: Option<f32> = None;
    let mut top_p: Option<f32> = None;
    let mut top_k: Option<usize> = None;
    let mut repetition_penalty: Option<f32> = None;
    let mut seed: Option<u64> = None;
    let mut greedy = false;
    let mut metal_prefill: Option<rlx_llama32::MetalGgufPrefillMode> = None;
    let mut download_all = false;
    let mut download_orpheus = false;
    let mut download_snac = false;
    let mut download_quant = DEFAULT_ORPHEUS_QUANT.to_string();
    let mut download_orpheus_dir: Option<PathBuf> = None;
    let mut download_snac_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = Some(req(args, &mut i)?.into()),
            "--snac" => snac = Some(req(args, &mut i)?.into()),
            "--device" => device = req(args, &mut i)?,
            "--text" => text = Some(req(args, &mut i)?),
            "--voice" => voice = Some(req(args, &mut i)?),
            "--out" | "--wav" => out_wav = Some(req(args, &mut i)?.into()),
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?.parse().context("--max-tokens: u32")?;
            }
            "--temperature" => {
                temperature = Some(req(args, &mut i)?.parse().context("--temperature: f32")?);
            }
            "--top-p" => {
                top_p = Some(req(args, &mut i)?.parse().context("--top-p: f32")?);
            }
            "--top-k" => {
                top_k = Some(req(args, &mut i)?.parse().context("--top-k: usize")?);
            }
            "--repetition-penalty" => {
                repetition_penalty = Some(
                    req(args, &mut i)?
                        .parse()
                        .context("--repetition-penalty: f32")?,
                );
            }
            "--seed" => {
                seed = Some(req(args, &mut i)?.parse().context("--seed: u64")?);
            }
            "--greedy" => greedy = true,
            "--metal-prefill" => {
                let s = req(args, &mut i)?;
                metal_prefill =
                    Some(rlx_llama32::MetalGgufPrefillMode::parse(&s).ok_or_else(|| {
                        anyhow!("--metal-prefill: expected auto|cpu|packed|metal")
                    })?);
            }
            "--download" | "--fetch" => download_all = true,
            "--download-orpheus" | "--fetch-orpheus" => download_orpheus = true,
            "--download-snac" | "--fetch-snac" => download_snac = true,
            "--quant" => download_quant = req(args, &mut i)?,
            "--download-dir" | "--orpheus-dir" => {
                download_orpheus_dir = Some(req(args, &mut i)?.into());
            }
            "--snac-dir" => download_snac_dir = Some(req(args, &mut i)?.into()),
            "--list-voices" => {
                for v in VOICES {
                    println!("{v}");
                }
                return Ok(());
            }
            "--help" | "-h" => {
                eprintln!(
                    "rlx-orpheus — Orpheus TTS (Llama GGUF + SNAC)\n\
                     \n\
                     --weights PATH     Orpheus GGUF (e.g. orpheus-3b-0.1-ft-Q4_K_M.gguf)\n\
                     --snac PATH        SNAC decoder safetensors (or set ORPHEUS_SNAC_PATH)\n\
                     --text TEXT        Text to synthesize\n\
                     --voice NAME       Voice prefix ({VOICES:?})\n\
                     --out PATH         Output WAV path\n\
                     --device auto|cpu|metal|gpu|wgpu|vulkan|coreml|mlx|…  \
                       Default auto (Metal on Apple Silicon, else CUDA/ROCm/wgpu). \
                       coreml = Metal LM + SNAC on native RLX CoreML (Device::Ane)\n\
                     --max-tokens N     Default 1200\n\
                     --temperature F    Default 0.6\n\
                     --top-p F          Nucleus sampling (default 0.8)\n\
                     --top-k N          Top-k filter (0 = disabled)\n\
                     --repetition-penalty F  Default 1.3\n\
                     --seed N           RNG seed (default 42)\n\
                     --greedy           Greedy argmax (overrides temperature)\n\
                     --metal-prefill MODE  auto|cpu|packed|metal (Metal GGUF prefill)\n\
                     --download         Fetch Orpheus GGUF + SNAC raw weights from HF\n\
                     --download-orpheus Download finetune GGUF only\n\
                     --download-snac    Download SNAC PyTorch weights only\n\
                     --quant Q4_K_M     GGUF quant for --download-orpheus (default Q4_K_M)\n\
                     --download-dir DIR Orpheus GGUF destination (default /tmp/rlx-weights/orpheus)\n\
                     --snac-dir DIR     SNAC raw weights destination (default /tmp/rlx-weights/snac)\n\
                     --list-voices"
                );
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    if download_all || download_orpheus || download_snac {
        return run_download(
            download_all,
            download_orpheus,
            download_snac,
            &download_quant,
            download_orpheus_dir.as_deref(),
            download_snac_dir.as_deref(),
        );
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let text = text.ok_or_else(|| anyhow!("--text is required"))?;
    let out_wav = out_wav.unwrap_or_else(|| PathBuf::from("orpheus-out.wav"));

    if let Some(ref v) = voice {
        if !VOICES.contains(&v.as_str()) {
            eprintln!("warning: voice `{v}` is not in the built-in list {VOICES:?}");
        }
    }

    let runtime = resolve_orpheus_device(&device)?;
    eprintln!(
        "[orpheus] lm={:?} snac_coreml={}",
        runtime.lm, runtime.snac_coreml
    );

    let mut backbone_opts = BackboneLoadOptions::for_device(runtime.lm);
    if let Some(m) = metal_prefill {
        if m == rlx_llama32::MetalGgufPrefillMode::PackedGguf {
            backbone_opts = BackboneLoadOptions::fast_prefill();
        } else if m == rlx_llama32::MetalGgufPrefillMode::CpuF32 {
            eprintln!(
                "warning: --metal-prefill cpu drains the full 3B model to host F32 (~12 GiB); \
                 use default synthesis on memory-constrained hosts (set ORPHEUS_LOW_MEM=1)"
            );
            backbone_opts = BackboneLoadOptions::reference_parity().with_metal_prefill(m);
        } else {
            backbone_opts.metal_prefill = m;
        }
    }

    let mut session = if let Some(snac_path) = snac {
        OrpheusTts::load_on_with(&weights, &snac_path, runtime, backbone_opts)?
    } else {
        let snac_path = crate::decoder::decoder_weights_path()?;
        OrpheusTts::load_on_with(&weights, &snac_path, runtime, backbone_opts)?
    };
    let mut cfg = GenerationConfig {
        max_new_tokens: max_tokens,
        ..GenerationConfig::default()
    };
    if let Some(t) = temperature {
        cfg.temperature = t;
    }
    if let Some(p) = top_p {
        cfg.top_p = p;
    }
    if let Some(k) = top_k {
        cfg.top_k = k;
    }
    if let Some(r) = repetition_penalty {
        cfg.repetition_penalty = r;
    }
    if let Some(s) = seed {
        cfg.seed = s;
    }
    cfg.greedy = greedy;
    session.config = cfg;

    eprintln!("[orpheus] synthesizing voice={:?} text={text:?}", voice);
    let result = session.synthesize(&text, voice.as_deref())?;
    eprintln!(
        "[orpheus] generated {} codes -> {} samples ({:.2}s)",
        result.code_count,
        result.samples.len(),
        result.samples.len() as f64 / result.sample_rate as f64
    );

    write_wav(&out_wav, &result.samples, result.sample_rate)?;
    eprintln!("[orpheus] wrote {}", out_wav.display());
    Ok(())
}

fn run_download(
    all: bool,
    orpheus: bool,
    snac: bool,
    quant: &str,
    orpheus_dir: Option<&std::path::Path>,
    snac_dir: Option<&std::path::Path>,
) -> Result<()> {
    let fetch_orpheus = all || orpheus;
    let fetch_snac = all || snac;
    if fetch_orpheus && fetch_snac {
        let (gguf, snac_path) = fetch_default()?;
        eprintln!("\nOrpheus GGUF: {}", gguf.display());
        eprintln!("SNAC raw:      {}", snac_path.display());
        return Ok(());
    }
    let cache = default_hf_cache_dir();
    if fetch_orpheus {
        let dest = orpheus_dir
            .map(PathBuf::from)
            .unwrap_or_else(default_orpheus_dir);
        let gguf = fetch_orpheus_gguf(&cache, &dest, quant)?;
        eprintln!("\nOrpheus GGUF: {}", gguf.display());
    }
    if fetch_snac {
        let dest = snac_dir.map(PathBuf::from).unwrap_or_else(default_snac_dir);
        let raw = fetch_snac_raw(&cache, &dest)?;
        print_snac_export_hint(&raw);
    }
    Ok(())
}

fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create wav {}", path.display()))?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let int_s = (clamped * 32767.0).round() as i16;
        writer.write_sample(int_s)?;
    }
    writer.finalize()?;
    Ok(())
}
