//! Native (ort-free) ChatterBox smoke: import + compile + run `embed_tokens` on
//! the target backend. `cargo run -p rlx-chatterbox --features native --example
//! native_smoke` (add `native-metal`/`native-mlx`/… + `--device` for GPU).

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("RLX_CHATTERBOX_DIR")
        .unwrap_or_else(|_| "weights/tts/chatterbox".to_string());
    let dev = std::env::var("RLX_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let device = rlx_chatterbox::parse_device(&dev).unwrap_or(rlx_chatterbox::Device::Cpu);
    let path = std::path::Path::new(&dir);
    if !path.join("onnx/embed_tokens.onnx").exists() {
        eprintln!("skip: no weights at {dir} (set RLX_CHATTERBOX_DIR)");
        return Ok(());
    }
    let cb = rlx_chatterbox::NativeChatterBox::load_on(path, device)?;
    let hidden = cb.smoke_embed()?;
    println!("[native_smoke] device={device:?} embed_tokens OK, hidden={hidden}");
    assert_eq!(hidden, 1024, "expected hidden 1024");

    // De-risk the re-exported CFM estimator (the loop body) if present.
    if std::env::var_os("RLX_EST").is_some() && path.join("onnx/cfm_estimator.onnx").exists() {
        let t = std::time::Instant::now();
        let b: usize = std::env::var("RLX_EST_B")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let n = cb.smoke_estimator(160)?;
        println!(
            "[native_smoke] device={device:?} cfm_estimator OK: {n} dxdt elems (want {}), {:?}",
            b * 80 * 160,
            t.elapsed()
        );
        assert_eq!(n, b * 80 * 160, "estimator output size");
    }

    // Vocoder-stage bisection: run hift_f0 / hift_src on a dumped mel.
    if let Some(melp) = std::env::var_os("RLX_BISECT") {
        let bytes = std::fs::read(&melp)?;
        let mel: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let t = mel.len() / 80;
        let mx = |v: &[f32]| v.iter().fold(0f32, |m, &x| m.max(x.abs()));
        for comp in ["hift_f0", "hift_src", "hift_stages"] {
            if !path.join(format!("onnx/{comp}.onnx")).exists() {
                continue;
            }
            let outs = cb.debug_run(comp, t, &[("T", t)], "speech_feat", &mel)?;
            for (name, v) in &outs {
                println!(
                    "[bisect] {comp}/{name}: len={} max|x|={:.4}",
                    v.len(),
                    mx(v)
                );
            }
        }
        return Ok(());
    }

    // Greedy token-parity of the native (rlx-llama32) T3 LM vs the ONNX-imported
    // LM in the real AR loop. `RLX_PARITY=1 RLX_FRAMES=N`.
    if std::env::var_os("RLX_PARITY").is_some() {
        let (reference, ref_sr) = load_reference(path);
        let max_frames: usize = std::env::var("RLX_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(24);
        let text = std::env::var("RLX_TEXT")
            .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog.".to_string());
        let opts = rlx_chatterbox::SynthOpts {
            max_frames,
            greedy: std::env::var_os("RLX_CB_GREEDY").is_some(),
            ..Default::default()
        };
        let (agree, total, first_div, mean_cos) =
            cb.token_parity(&text, &reference, ref_sr, &opts)?;
        println!(
            "[parity] native-vs-ONNX T3 LM: argmax_agree={agree}/{total}  mean_cos={mean_cos:.6}  first_divergence={first_div:?}"
        );
        if agree == total {
            println!("[parity] ✅ native T3 LM token-exact vs ONNX in the real AR loop");
        }
        return Ok(());
    }

    // Full end-to-end synthesize (all graphs + AR loop) when RLX_FULL is set.
    if std::env::var_os("RLX_FULL").is_some() {
        // Real reference voice if present (better speaker conditioning + a
        // whisper-checkable result); else a 3 s 220 Hz tone.
        let (reference, ref_sr) = load_reference(path);
        let max_frames: usize = std::env::var("RLX_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let text = std::env::var("RLX_TEXT")
            .unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog.".to_string());
        let opts = rlx_chatterbox::SynthOpts {
            max_frames,
            greedy: std::env::var_os("RLX_CB_GREEDY").is_some(),
            ..Default::default()
        };
        // `RLX_WARM=N` runs synthesize N times (in-process) to show the warm
        // steady-state (2nd+ calls reuse compiled graphs).
        let runs: usize = std::env::var("RLX_WARM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let mut audio = Vec::new();
        for r in 0..runs.max(1) {
            let t = std::time::Instant::now();
            audio = cb.synthesize(&text, &reference, ref_sr, &opts)?;
            println!(
                "[native_smoke] device={device:?} run {r} synthesize OK: {} samples ({:.2}s), peak={:.3}, {:?}",
                audio.len(),
                audio.len() as f32 / 24_000.0,
                rlx_chatterbox::peak_amplitude(&audio),
                t.elapsed(),
            );
        }
        let out = std::env::var("RLX_OUT").unwrap_or_else(|_| "cb_native.wav".to_string());
        cb.write_wav(&audio, std::path::Path::new(&out))?;
        println!("[native_smoke] wrote {out}");

        // End-to-end whisper round-trip (RLX_WHISPER_DIR override, else
        // .cache/whisper-tiny). Confirms the fully-native pipeline is intelligible.
        let wd = std::env::var("RLX_WHISPER_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new("/Users/Shared/rlx-models/.cache/whisper-tiny").to_path_buf()
            });
        if wd.join("model.safetensors").exists() {
            use rlx_whisper::{SAMPLE_RATE as WR, WhisperRunner};
            let n = (audio.len() as u64 * WR as u64 / 24_000u64).max(1) as usize;
            let pcm: Vec<f32> = (0..n)
                .map(|i| {
                    let s = i as f64 * 24_000f64 / WR as f64;
                    let idx = s.floor() as usize;
                    let f = (s - idx as f64) as f32;
                    let a = audio[idx.min(audio.len() - 1)];
                    let b = audio[(idx + 1).min(audio.len() - 1)];
                    a + (b - a) * f
                })
                .collect();
            let mut w = WhisperRunner::builder()
                .weights(wd.join("model.safetensors"))
                .config_path(wd.join("config.json"))
                .tokenizer_path(wd.join("tokenizer.json"))
                .device(rlx_chatterbox::Device::Cpu)
                .language("en")
                .build()?;
            let transcript = w.transcribe_greedy(&pcm)?;
            println!("[native_smoke] target : {text}");
            println!("[native_smoke] whisper: {transcript}");
        } else {
            println!(
                "[native_smoke] (no whisper weights at {} — skipping transcript)",
                wd.display()
            );
        }
    }
    Ok(())
}

/// Load a reference wav (`RLX_REF` override, else `default_voice.wav`), else a tone.
fn load_reference(dir: &std::path::Path) -> (Vec<f32>, u32) {
    let p = std::env::var("RLX_REF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dir.join("default_voice.wav"));
    if let Ok(mut r) = hound::WavReader::open(&p) {
        let sr = r.spec().sample_rate;
        let ch = r.spec().channels as usize;
        let raw: Vec<f32> = match r.spec().sample_format {
            hound::SampleFormat::Int => r
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => r.samples::<f32>().filter_map(|s| s.ok()).collect(),
        };
        let mono: Vec<f32> = if ch > 1 {
            raw.chunks(ch)
                .map(|c| c.iter().sum::<f32>() / ch as f32)
                .collect()
        } else {
            raw
        };
        eprintln!(
            "[native_smoke] reference {} ({} samp @ {sr})",
            p.display(),
            mono.len()
        );
        return (mono, sr);
    }
    let n = 24_000 * 3;
    (
        (0..n)
            .map(|i| 0.05 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 24_000.0).sin())
            .collect(),
        24_000,
    )
}
