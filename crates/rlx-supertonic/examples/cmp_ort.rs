// Compare native-rlx vs ONNX-Runtime: wall-time (this prints) + peak RSS (run
// under `/usr/bin/time -l`). One path per process so max-RSS is attributable.
//   cargo run --release -p rlx-supertonic --features onnx --example cmp_ort -- native|ort [iters]
use rlx_runtime::Device;
use rlx_supertonic::{InferOpts, Supertonic, Voice};

fn rss_mb() -> u64 {
    // macOS: `ps -o rss=` reports KB.
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

/// macOS phys_footprint (real memory-in-use; excludes reclaimable pages) via
/// `vmmap --summary`. This is the metric that reflects MADV_FREE_REUSABLE.
fn phys_footprint_mb() -> String {
    let out = std::process::Command::new("vmmap")
        .args(["--summary", &std::process::id().to_string()])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .lines()
            .find(|l| l.contains("Physical footprint:") && !l.contains("Peak"))
            .map(|l| l.split(':').nth(1).unwrap_or("?").trim().to_string())
            .unwrap_or_else(|| "?".into()),
        Err(_) => "?".into(),
    }
}

fn main() -> anyhow::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "native".into());
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/supertonic-3");
    let text = "The quick brown fox jumps over the lazy dog near the river bank.";
    let tts = Supertonic::load_on(&dir, Device::Cpu)?;
    let voice = Voice::load(&dir.join("voice_styles/F1.json"))?;
    let opts = InferOpts {
        total_step: 8,
        speed: 1.0,
        seed: 42,
    };

    // Warm once (AOT compile / ort session init) — excluded from timing.
    let _ = match mode.as_str() {
        "ort" => tts.synthesize_ort(text, "en", &voice, &opts)?,
        _ => tts.synthesize(text, "en", &voice, &opts)?,
    };
    eprintln!("[{mode}] post-warmup RSS = {} MB", rss_mb());

    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    for _ in 0..iters {
        let a = match mode.as_str() {
            "ort" => tts.synthesize_ort(text, "en", &voice, &opts)?,
            _ => tts.synthesize(text, "en", &voice, &opts)?,
        };
        n = a.len();
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let audio_s = n as f64 / tts.sample_rate() as f64;
    eprintln!(
        "[{mode}] iters={iters} mean={:.3}s/synth  audio={audio_s:.2}s  RTF={:.1}x  steady-RSS={} MB  phys-footprint={}",
        dt,
        audio_s / dt,
        rss_mb(),
        phys_footprint_mb(),
    );
    Ok(())
}
