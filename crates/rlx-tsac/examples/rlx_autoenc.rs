//! Diagnostic: does the DAC autoencoder (encoder → decoder, NO quantization)
//! reconstruct? Tells us whether the Bellard q8 weights want standard DAC
//! conventions (high SNR) before investing in the RVQ encoder.
//!
//! ```bash
//! RLX_TSAC_DIR=$PWD/.cache/tsac cargo run -p rlx-tsac --example rlx_autoenc \
//!   --release --features native-codec -- [frames]
//! ```
use anyhow::Result;
use rlx_runtime::Device;
use rlx_tsac::rlx_decode::{RlxDecoder, RlxEncoder};
use rlx_tsac::{SAMPLE_RATE, audio, default_tsac_dir};

fn snr_db(reference: &[f32], test: &[f32]) -> f32 {
    let n = reference.len().min(test.len());
    let (mut sig, mut err) = (0f64, 0f64);
    for i in 0..n {
        sig += (reference[i] as f64).powi(2);
        let d = (reference[i] - test[i]) as f64;
        err += d * d;
    }
    if err <= 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (sig / err).log10()) as f32
}

fn main() -> Result<()> {
    let frames: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let dir = default_tsac_dir();
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rlx-qwen3-tts/examples/audio/ask_not.wav");
    let (mut pcm, _) = audio::load_wav_f32(&src, SAMPLE_RATE)?;
    pcm.truncate((frames * 512).min(pcm.len()));
    eprintln!("input: {} samples ({} frames)", pcm.len(), pcm.len() / 512);

    {
        let enc = RlxEncoder::open(&dir, Device::Cpu)?;
        let z = enc.encode_latent(&pcm, 1)?;
        let n = z.len() as f64;
        let mean = z.iter().map(|&v| v as f64).sum::<f64>() / n;
        let rms = (z.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n).sqrt();
        let mx = z.iter().fold(0f32, |m, &v| m.max(v.abs()));
        eprintln!(
            "encoder z: dim={:?} mean={mean:.4} rms={rms:.4} max_abs={mx:.4}",
            z.dim()
        );
    }

    for faithful in [false, true] {
        let enc = RlxEncoder::open(&dir, Device::Cpu)?;
        let z = enc.encode_latent(&pcm, 1)?;
        let dec = RlxDecoder::open_mode(&dir, Device::Cpu, faithful)?;
        let recon = dec.decode_latent_direct(&z)?; // [Co, T]
        let ch0: Vec<f32> = recon.row(0).to_vec();
        let snr = snr_db(&pcm, &ch0);
        // best lag + gain-normalized SNR (is recon right up to shift/scale?)
        let (mut blag, mut bcorr) = (0i64, -2.0f32);
        for lag in -600i64..=600 {
            let (a, b): (&[f32], &[f32]) = if lag >= 0 {
                (&ch0[lag as usize..], &pcm[..])
            } else {
                (&ch0[..], &pcm[(-lag) as usize..])
            };
            let m = a.len().min(b.len());
            if m < 500 {
                continue;
            }
            let cc = audio::correlation(&a[..m], &b[..m]);
            if cc > bcorr {
                bcorr = cc;
                blag = lag;
            }
        }
        eprintln!(
            "faithful={faithful}: z{:?} recon[{}x{}] SNR={snr:.2}dB  bestlag={blag} corr={bcorr:.4}",
            z.dim(),
            recon.dim().0,
            recon.dim().1,
        );
    }
    Ok(())
}
