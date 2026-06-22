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

//! Prefill + decode parity: CPU reference vs Metal / MLX / wgpu with shared enc/cross.
//!
//! ```sh
//! cargo test -p rlx-models --test whisper_backend_parity \
//!   --features "metal,mlx,gpu" --release
//! ```

#![cfg(any(feature = "metal", feature = "mlx", feature = "gpu"))]

use anyhow::Result;
use rlx_models::whisper::{WhisperRunner, jfk_wav_path, pcm_to_mel};
use rlx_runtime::Device;
use std::path::PathBuf;

fn cache_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache")
}

fn tiny_dir() -> PathBuf {
    std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| cache_root().join("whisper-tiny"))
}

fn jfk_wav() -> PathBuf {
    jfk_wav_path()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn parity_backends() -> Vec<Device> {
    let mut out = Vec::new();
    #[cfg(feature = "metal")]
    if rlx_runtime::is_available(Device::Metal) {
        out.push(Device::Metal);
    }
    #[cfg(feature = "mlx")]
    if rlx_runtime::is_available(Device::Mlx) {
        out.push(Device::Mlx);
    }
    #[cfg(feature = "gpu")]
    if rlx_runtime::is_available(Device::Gpu) {
        out.push(Device::Gpu);
    }
    out
}

fn device_label(d: Device) -> &'static str {
    match d {
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Gpu => "wgpu",
        _ => "other",
    }
}

fn shared_state() -> Result<(
    PathBuf,
    Vec<f32>,
    rlx_models::whisper::WhisperCrossCache,
    Vec<u32>,
)> {
    let dir = tiny_dir();
    let weights = dir.join("model.safetensors");
    if !weights.is_file() {
        anyhow::bail!("skip: no weights at {weights:?}");
    }
    let wav = jfk_wav();
    if !wav.is_file() {
        anyhow::bail!("skip: wav not found {wav:?}");
    }

    let pcm = rlx_models::whisper::load_wav_mono_f32(&wav)?;
    let mut cpu = WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let mel = pcm_to_mel(cpu.config(), &pcm);
    let enc = cpu.encode_mel(&mel)?;
    let cross = cpu.cross_cache_batch(&enc, 1)?;
    let prompt = cpu.build_prompt()?;
    Ok((dir, pcm, cross, prompt))
}

#[test]
fn whisper_prefill_logits_match_gpu_backends() -> Result<()> {
    let backends = parity_backends();
    if backends.is_empty() {
        eprintln!("skip: no Metal/MLX/wgpu backend available");
        return Ok(());
    }

    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    rlx_ir::env::set("OPENBLAS_NUM_THREADS", "1");

    let (dir, _pcm, cross, prompt) = match shared_state() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return Ok(());
        }
    };
    let weights = dir.join("model.safetensors");

    let mut cpu = WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let (cpu_logits, _) = cpu.prefill_prompt(&cross, &prompt, 1)?;
    let vocab = cpu.config().vocab_size.max(1);
    let last = prompt.len().saturating_sub(1) * vocab;
    let cpu_last = &cpu_logits[last..last + vocab];

    for dev in backends {
        let mut runner = WhisperRunner::builder()
            .weights(&weights)
            .config_path(dir.join("config.json"))
            .device(dev)
            .language("en")
            .build()?;
        let (dev_logits, _) = runner.prefill_prompt(&cross, &prompt, 1)?;
        let mx = max_abs(cpu_last, &dev_logits[last..last + vocab]);
        eprintln!("prefill {}/cpu last-row max_abs={mx:.6}", device_label(dev));
        assert!(
            mx < 1e-4,
            "{} prefill last-row logits diverge (max_abs={mx})",
            device_label(dev)
        );
    }
    Ok(())
}

#[test]
fn whisper_decode_step_matches_gpu_backends() -> Result<()> {
    let backends = parity_backends();
    if backends.is_empty() {
        eprintln!("skip: no Metal/MLX/wgpu backend available");
        return Ok(());
    }

    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    rlx_ir::env::set("OPENBLAS_NUM_THREADS", "1");

    let (dir, _pcm, cross, prompt) = match shared_state() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return Ok(());
        }
    };
    let weights = dir.join("model.safetensors");
    let tokenizer = dir.join("tokenizer.json");

    let mut cpu = WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .tokenizer_path(&tokenizer)
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let (prefill_logits, cache) = cpu.prefill_prompt(&cross, &prompt, 1)?;
    let (_, cpu_text) = cpu.bench_greedy_decode_from_state(
        &cross,
        &prompt,
        prefill_logits.clone(),
        cache.clone(),
        1,
    )?;

    for dev in backends {
        let mut runner = WhisperRunner::builder()
            .weights(&weights)
            .config_path(dir.join("config.json"))
            .tokenizer_path(&tokenizer)
            .device(dev)
            .language("en")
            .build()?;
        let share_cpu_decode = runner.decode_device() == Device::Cpu;
        if share_cpu_decode {
            cpu.swap_decode_cache(&mut runner);
        }
        let (_, dev_text) = runner.bench_greedy_decode_from_state(
            &cross,
            &prompt,
            prefill_logits.clone(),
            cache.clone(),
            1,
        )?;
        if share_cpu_decode {
            cpu.swap_decode_cache(&mut runner);
        }
        eprintln!(
            "decode step-1 {} vs cpu: {:?} vs {:?}",
            device_label(dev),
            dev_text,
            cpu_text
        );
        assert_eq!(
            dev_text,
            cpu_text,
            "{} greedy step-1 transcript",
            device_label(dev)
        );
    }
    Ok(())
}

#[test]
fn whisper_full_greedy_matches_cpu_on_gpu_backends() -> Result<()> {
    let backends = parity_backends();
    if backends.is_empty() {
        eprintln!("skip: no Metal/MLX/wgpu backend available");
        return Ok(());
    }

    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    rlx_ir::env::set("OPENBLAS_NUM_THREADS", "1");

    let (dir, _pcm, cross, prompt) = match shared_state() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return Ok(());
        }
    };
    let weights = dir.join("model.safetensors");
    let tokenizer = dir.join("tokenizer.json");

    let mut cpu = WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .tokenizer_path(&tokenizer)
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let (prefill_logits, cache) = cpu.prefill_prompt(&cross, &prompt, 1)?;
    let steps = cpu
        .config()
        .max_target_positions
        .saturating_sub(prompt.len());
    let (_, cpu_text) = cpu.bench_greedy_decode_from_state(
        &cross,
        &prompt,
        prefill_logits.clone(),
        cache.clone(),
        steps,
    )?;

    for dev in backends {
        let mut runner = WhisperRunner::builder()
            .weights(&weights)
            .config_path(dir.join("config.json"))
            .tokenizer_path(&tokenizer)
            .device(dev)
            .language("en")
            .build()?;
        let share_cpu_decode = runner.decode_device() == Device::Cpu;
        if share_cpu_decode {
            cpu.swap_decode_cache(&mut runner);
        }
        let (_, dev_text) = runner.bench_greedy_decode_from_state(
            &cross,
            &prompt,
            prefill_logits.clone(),
            cache.clone(),
            steps,
        )?;
        if share_cpu_decode {
            cpu.swap_decode_cache(&mut runner);
        }
        eprintln!(
            "greedy decode {} vs cpu (shared cross): {:?} / {:?}",
            device_label(dev),
            dev_text,
            cpu_text
        );
        assert_eq!(
            dev_text,
            cpu_text,
            "{} greedy decode with CPU cross",
            device_label(dev)
        );
    }
    Ok(())
}
