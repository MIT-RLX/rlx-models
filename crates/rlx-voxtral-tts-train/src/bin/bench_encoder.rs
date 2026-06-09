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

//! Encoder forward / backward matrix bench (compile + timed runs).

use anyhow::{Context, Result};
use rlx_runtime::{Device, Session, is_available};
use rlx_voxtral_tts::config::VoxtralTtsConfig;
use rlx_voxtral_tts_train::codec_graph::{CodecGraphLayout, build_codec_recon_graph};
use rlx_voxtral_tts_train::compile::compile_train_backward;
use rlx_voxtral_tts_train::config::patch_count;
use rlx_voxtral_tts_train::dataset::WavDataset;
use rlx_voxtral_tts_train::encoder_loss::build_encoder_train_graph;
use rlx_voxtral_tts_train::weights::{WeightStore, fit_params_to_graph, load_codec_weights};
use std::env;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_MODEL: &str = ".cache/voxtral/Voxtral-4B-TTS-2603";
const DEFAULT_WAVS: &str = ".cache/voxtral/bench-wavs";

#[derive(Clone, Copy, Debug)]
enum HybridMode {
    Native,
    Hybrid,
}

impl HybridMode {
    fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Clone, Debug)]
struct BenchRow {
    device: String,
    hybrid: Option<String>,
    phase: String,
    available: bool,
    ok: bool,
    compile_ms: f64,
    run_ms: f64,
    run_per_iter_ms: f64,
    active_device: String,
    note: String,
}

fn main() -> Result<()> {
    let model_dir = env::var("BENCH_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_MODEL));
    let wav_dir = env::var("BENCH_WAV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_WAVS));
    let runs: usize = env::var("BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let profile = env::var("BENCH_PROFILE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let warmup: usize = env::var("BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let cfg = VoxtralTtsConfig::from_model_dir(&model_dir).context("load model config")?;
    let codec = &cfg.audio_config.codec_args;
    let n_patches = patch_count(codec, 4.0);
    let layout = CodecGraphLayout::new(codec, n_patches);
    let train = build_encoder_train_graph(codec, &layout, 1.0, 1.0, 0.1, 0.1, 0.0, 0.0);
    let fwd = build_codec_recon_graph(codec, &layout).context("recon graph")?;

    let (enc, dec) = load_codec_weights(&model_dir, true, codec).context("codec weights")?;
    let mut weights = enc;
    weights.merge(&dec);
    fit_params_to_graph(&mut weights, &fwd.params).context("fit params")?;

    let batch = WavDataset::from_dir(&wav_dir, codec, 4.0)
        .context("wav dir")?
        .sample_batch()
        .context("sample batch")?;
    let audio = WavDataset::patches_to_ncl(&batch.pcm, codec.pretransform_patch_size);
    let mut target = vec![0f32; layout.patch_size * layout.wav_t];
    let copy = target.len().min(batch.pcm.len());
    target[..copy].copy_from_slice(&batch.pcm[..copy]);
    let mel = vec![0.001f32; 64 * layout.wav_t.max(1)];
    let stft = vec![0.001f32; 128 * layout.wav_t.max(1)];

    let devices = [
        ("cpu", Device::Cpu),
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("vulkan", Device::Vulkan),
    ];

    let mut rows = Vec::new();

    println!(
        "encoder bench — model={} runs={} warmup={} profile={}\n",
        model_dir.display(),
        runs,
        warmup,
        profile
    );

    if profile {
        if !is_available(Device::Metal) {
            anyhow::bail!("BENCH_PROFILE=1 requires Metal");
        }
        return run_metal_profile(&train, &weights, &audio, &target, &mel, &stft);
    }

    for (name, device) in devices {
        if !is_available(device) {
            rows.push(BenchRow {
                device: name.into(),
                hybrid: None,
                phase: "forward".into(),
                available: false,
                ok: false,
                compile_ms: 0.0,
                run_ms: 0.0,
                run_per_iter_ms: 0.0,
                active_device: "-".into(),
                note: "backend unavailable".into(),
            });
            rows.push(BenchRow {
                device: name.into(),
                hybrid: Some("native".into()),
                phase: "backward".into(),
                available: false,
                ok: false,
                compile_ms: 0.0,
                run_ms: 0.0,
                run_per_iter_ms: 0.0,
                active_device: "-".into(),
                note: "backend unavailable".into(),
            });
            rows.push(BenchRow {
                device: name.into(),
                hybrid: Some("hybrid".into()),
                phase: "backward".into(),
                available: false,
                ok: false,
                compile_ms: 0.0,
                run_ms: 0.0,
                run_per_iter_ms: 0.0,
                active_device: "-".into(),
                note: "backend unavailable".into(),
            });
            continue;
        }

        rows.push(bench_forward(
            name, device, &fwd, &weights, &audio, runs, warmup,
        ));

        if device == Device::Cpu {
            rows.push(bench_backward(
                name, device, None, &train, &weights, &audio, &target, &mel, &stft, runs, warmup,
            ));
        } else {
            rows.push(bench_backward(
                name,
                device,
                Some(HybridMode::Native),
                &train,
                &weights,
                &audio,
                &target,
                &mel,
                &stft,
                runs,
                warmup,
            ));
            rows.push(bench_backward(
                name,
                device,
                Some(HybridMode::Hybrid),
                &train,
                &weights,
                &audio,
                &target,
                &mel,
                &stft,
                runs,
                warmup,
            ));
        }
    }

    print_table(&rows);
    write_summary(&rows)?;
    Ok(())
}

fn bench_forward(
    name: &str,
    device: Device,
    fwd: &rlx_voxtral_tts_train::codec_graph::CodecForwardGraph,
    weights: &WeightStore,
    audio: &[f32],
    runs: usize,
    warmup: usize,
) -> BenchRow {
    let mut row = BenchRow {
        device: name.into(),
        hybrid: None,
        phase: "forward".into(),
        available: true,
        ok: false,
        compile_ms: 0.0,
        run_ms: 0.0,
        run_per_iter_ms: 0.0,
        active_device: format!("{device:?}"),
        note: String::new(),
    };

    let result = catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<(Duration, Duration)> {
            let mut graph = fwd.graph.clone();
            graph.set_outputs(vec![fwd.recon_wav]);
            let t0 = Instant::now();
            let mut exec = Session::new(device).compile(graph);
            let compile = t0.elapsed();
            for (k, v) in &weights.0 {
                exec.set_param(k, v);
            }
            let mut run = Duration::ZERO;
            for i in 0..(warmup + runs) {
                let t1 = Instant::now();
                let _ = exec.run(&[("audio", audio)]);
                let dt = t1.elapsed();
                if i >= warmup {
                    run += dt;
                }
            }
            Ok((compile, run))
        },
    ));

    match result {
        Ok(Ok((compile, run))) => {
            row.ok = true;
            row.compile_ms = ms(compile);
            row.run_ms = ms(run);
            row.run_per_iter_ms = row.run_ms / runs.max(1) as f64;
        }
        Ok(Err(e)) => row.note = truncate_err(&e.to_string()),
        Err(_) => row.note = "panic during compile/run".into(),
    }
    row
}

fn bench_backward(
    name: &str,
    device: Device,
    hybrid: Option<HybridMode>,
    train: &rlx_voxtral_tts_train::encoder_loss::EncoderTrainGraph,
    weights: &WeightStore,
    audio: &[f32],
    target: &[f32],
    mel: &[f32],
    stft: &[f32],
    runs: usize,
    warmup: usize,
) -> BenchRow {
    let hybrid_label = hybrid.map(HybridMode::label).map(str::to_string);
    let mut row = BenchRow {
        device: name.into(),
        hybrid: hybrid_label.clone(),
        phase: "backward".into(),
        available: true,
        ok: false,
        compile_ms: 0.0,
        run_ms: 0.0,
        run_per_iter_ms: 0.0,
        active_device: "-".into(),
        note: String::new(),
    };

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        bench_backward_inner(
            device, hybrid, train, weights, audio, target, mel, stft, runs, warmup,
        )
    }));

    match result {
        Ok(inner) => match inner {
            Ok((compile, run, active, note)) => {
                row.ok = true;
                row.compile_ms = ms(compile);
                row.run_ms = ms(run);
                row.run_per_iter_ms = row.run_ms / runs.max(1) as f64;
                row.active_device = format!("{active:?}");
                row.note = note;
            }
            Err(e) => row.note = truncate_err(&e.to_string()),
        },
        Err(_) => row.note = "panic during compile/run".into(),
    }
    row
}

fn bench_backward_inner(
    device: Device,
    hybrid: Option<HybridMode>,
    train: &rlx_voxtral_tts_train::encoder_loss::EncoderTrainGraph,
    weights: &WeightStore,
    audio: &[f32],
    target: &[f32],
    mel: &[f32],
    stft: &[f32],
    runs: usize,
    warmup: usize,
) -> Result<(Duration, Duration, Device, String)> {
    unsafe {
        env::remove_var("RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU");
        env::remove_var("RLX_VOXTRAL_TTS_TRAIN_NATIVE_BACKWARD");
        env::remove_var("RLX_VOXTRAL_TTS_TRAIN_MLX_NATIVE_BACKWARD");
    }
    match hybrid {
        Some(HybridMode::Hybrid) => unsafe {
            env::set_var("RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU", "1");
        },
        Some(HybridMode::Native) if device != Device::Cpu => unsafe {
            env::set_var("RLX_VOXTRAL_TTS_TRAIN_NATIVE_BACKWARD", "1");
        },
        _ => {}
    }

    let t0 = Instant::now();
    let (active, mut exec) =
        compile_train_backward(device, train.backward.clone(), "bench").context("compile")?;
    let compile = t0.elapsed();
    for (k, v) in &weights.0 {
        exec.set_param(k, v);
    }
    let mut run = Duration::ZERO;
    for i in 0..(warmup + runs) {
        let t1 = Instant::now();
        let _ = exec.run(&[
            ("audio", audio),
            ("target_wav", target),
            ("mel_basis", mel),
            ("stft_basis", stft),
            ("d_fake", &[0.0f32]),
            ("asr_mse", &[0.0f32]),
            ("d_output", &[1.0f32]),
        ]);
        let dt = t1.elapsed();
        if i >= warmup {
            run += dt;
        }
    }
    let note = if device != active {
        format!("hybrid requested={device:?} active={active:?}")
    } else {
        String::new()
    };
    Ok((compile, run, active, note))
}

fn run_metal_profile(
    train: &rlx_voxtral_tts_train::encoder_loss::EncoderTrainGraph,
    weights: &WeightStore,
    audio: &[f32],
    target: &[f32],
    mel: &[f32],
    stft: &[f32],
) -> Result<()> {
    unsafe {
        env::remove_var("RLX_VOXTRAL_TTS_TRAIN_BACKWARD_CPU");
        env::set_var("RLX_VOXTRAL_TTS_TRAIN_NATIVE_BACKWARD", "1");
        env::set_var("RLX_METAL_THUNK_PROFILE", "1");
    }
    let (active, mut exec) =
        compile_train_backward(Device::Metal, train.backward.clone(), "bench-profile")
            .context("compile profile")?;
    for (k, v) in &weights.0 {
        exec.set_param(k, v);
    }
    eprintln!("Metal backward profile — active={active:?}\n");
    let _ = exec.run(&[
        ("audio", audio),
        ("target_wav", target),
        ("mel_basis", mel),
        ("stft_basis", stft),
        ("d_fake", &[0.0f32]),
        ("asr_mse", &[0.0f32]),
        ("d_output", &[1.0f32]),
    ]);
    Ok(())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn truncate_err(s: &str) -> String {
    const MAX: usize = 120;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}…", &s[..MAX])
    }
}

fn print_table(rows: &[BenchRow]) {
    println!(
        "{:<8} {:<8} {:<9} {:>10} {:>10} {:>10} {:>12} status",
        "device", "hybrid", "phase", "compile_ms", "run_ms", "ms/iter", "active"
    );
    println!("{}", "-".repeat(96));
    for r in rows {
        let hybrid = r.hybrid.as_deref().unwrap_or("-");
        let status = if !r.available {
            "n/a"
        } else if r.ok {
            "ok"
        } else {
            "FAIL"
        };
        println!(
            "{:<8} {:<8} {:<9} {:>10.1} {:>10.1} {:>10.1} {:>12} {}{}",
            r.device,
            hybrid,
            r.phase,
            r.compile_ms,
            r.run_ms,
            r.run_per_iter_ms,
            r.active_device,
            status,
            if r.note.is_empty() {
                String::new()
            } else {
                format!(" ({})", r.note)
            }
        );
    }
}

fn write_summary(rows: &[BenchRow]) -> Result<()> {
    let out = Path::new(".cache/voxtral/bench-out/encoder-matrix.json");
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).context("mkdir bench-out")?;
    }
    let json = serde_json::to_string_pretty(rows).context("json")?;
    std::fs::write(out, json).context("write json")?;
    println!("\nwrote {}", out.display());
    Ok(())
}

impl serde::Serialize for BenchRow {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("BenchRow", 10)?;
        s.serialize_field("device", &self.device)?;
        s.serialize_field("hybrid", &self.hybrid)?;
        s.serialize_field("phase", &self.phase)?;
        s.serialize_field("available", &self.available)?;
        s.serialize_field("ok", &self.ok)?;
        s.serialize_field("compile_ms", &self.compile_ms)?;
        s.serialize_field("run_ms", &self.run_ms)?;
        s.serialize_field("run_per_iter_ms", &self.run_per_iter_ms)?;
        s.serialize_field("active_device", &self.active_device)?;
        s.serialize_field("note", &self.note)?;
        s.end()
    }
}
