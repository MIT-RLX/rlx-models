//! Perf of the correct TSAC (= Descript-DAC-44kHz) codec per RLX backend.
//! Reports encode/decode wall time + real-time factor (RTF = audio_s / wall_s;
//! >1 = faster than real time), cold (incl. one-time graph compile) vs warm.
//! ```bash
//! RLX_DAC_DIR=$PWD/.cache/dac44 cargo run -p rlx-tsac --example correct_perf \
//!   --release --features native-codec,oracle,metal,mlx,gpu -- /tmp/tcmp/in44.wav
//! ```
use anyhow::Result;
use rlx_tsac::{SAMPLE_RATE, audio, correct, parse_tsac_device};
use std::time::Instant;

fn main() -> Result<()> {
    let in_wav = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/tcmp/in44.wav".into());
    let (pcm, _) = audio::load_wav_f32(std::path::Path::new(&in_wav), SAMPLE_RATE)?;
    let audio_s = pcm.len() as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "input: {} samples = {audio_s:.2}s @ {SAMPLE_RATE} Hz\n",
        pcm.len()
    );

    let extra: Vec<String> = std::env::args().skip(2).collect();
    let devices: Vec<String> = if extra.is_empty() {
        ["cpu", "metal", "mlx", "gpu"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        extra
    };

    eprintln!(
        "{:<7} {:>10} {:>8} {:>10} {:>8}",
        "device", "enc(warm)", "encRTF", "dec(warm)", "decRTF"
    );
    for dname in &devices {
        let dev = match parse_tsac_device(dname) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if !rlx_runtime::is_available(dev) {
            eprintln!("{dname:<7}  not available");
            continue;
        }
        let codec = match correct::open(dev) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{dname:<7}  open failed: {e}");
                continue;
            }
        };
        // cold (includes graph compile), then 2 warm iters.
        let t = Instant::now();
        let codes = codec.encode_pcm(&pcm, None)?;
        let enc_cold = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = codec.decode_codes(&codes)?;
        let dec_cold = t.elapsed().as_secs_f64();

        let mut enc_w = f64::MAX;
        let mut dec_w = f64::MAX;
        for _ in 0..2 {
            let t = Instant::now();
            let c = codec.encode_pcm(&pcm, None)?;
            enc_w = enc_w.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            let _ = codec.decode_codes(&c)?;
            dec_w = dec_w.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "{dname:<7} {:>9.0}ms {:>7.2}x {:>9.0}ms {:>7.2}x   (cold enc {:.0}ms dec {:.0}ms)",
            enc_w * 1e3,
            audio_s / enc_w,
            dec_w * 1e3,
            audio_s / dec_w,
            enc_cold * 1e3,
            dec_cold * 1e3,
        );
    }
    Ok(())
}
