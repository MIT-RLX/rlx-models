//! Fast RLX-decoder check against a cached C-decoded oracle.
//!
//! ```bash
//! RLX_TSAC_DIR=$PWD/.cache/tsac cargo run -p rlx-tsac --example rlx_decode_check \
//!   --release --features native-codec -- <in.tsac> <oracle_c.wav> [device]
//! ```
use anyhow::Result;
use rlx_runtime::Device;
use rlx_tsac::rlx_decode::{RlxDecoder, read_codes};
use rlx_tsac::{audio, default_tsac_dir, parse_tsac_device};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tsac = std::path::PathBuf::from(&args[0]);
    let oracle = std::path::PathBuf::from(&args[1]);
    let device = args
        .get(2)
        .map(|s| parse_tsac_device(s))
        .transpose()?
        .unwrap_or(Device::Cpu);

    let dir = default_tsac_dir();
    let (codes, n_frames, n_cb) = read_codes(&tsac)?;
    eprintln!("codes: n_frames={n_frames} n_cb={n_cb}, device={device:?}");

    let dec = RlxDecoder::open(&dir, device)?;
    let pcm = dec.decode_codes(&codes, n_frames, n_cb)?;
    eprintln!("rlx out: [{}ch x {}]", pcm.dim().0, pcm.dim().1);
    let ch0: Vec<f32> = pcm.row(0).to_vec();

    let oracle_pcm = audio::load_pcm_from_wav(&oracle)?;
    let n = ch0.len().min(oracle_pcm.len());
    let cc = audio::correlation(&ch0[..n], &oracle_pcm[..n]);
    let mx = audio::max_abs_error(&ch0[..n], &oracle_pcm[..n]);
    let ms = audio::mse(&ch0[..n], &oracle_pcm[..n]);
    eprintln!(
        "RLX({device:?}) vs C: corr={cc:.5} mse={ms:.6} max_abs={mx:.5} (rlx_len={}, c_len={})",
        ch0.len(),
        oracle_pcm.len()
    );
    // Print a few aligned samples for eyeballing.
    eprintln!("rlx[100..108]={:?}", &ch0[100..108.min(ch0.len())]);
    eprintln!(
        "c  [100..108]={:?}",
        &oracle_pcm[100..108.min(oracle_pcm.len())]
    );

    // Lag search: is the discrepancy purely a time shift?
    let (mut best_lag, mut best_cc) = (0i64, -2.0f32);
    for lag in -4096i64..=4096 {
        let (a, b): (&[f32], &[f32]) = if lag >= 0 {
            let l = lag as usize;
            (
                &ch0[l..],
                &oracle_pcm[..oracle_pcm.len().min(ch0.len() - l)],
            )
        } else {
            let l = (-lag) as usize;
            (
                &ch0[..ch0.len().min(oracle_pcm.len() - l)],
                &oracle_pcm[l..],
            )
        };
        let m = a.len().min(b.len());
        if m < 1000 {
            continue;
        }
        let cc = audio::correlation(&a[..m], &b[..m]);
        if cc > best_cc {
            best_cc = cc;
            best_lag = lag;
        }
    }
    eprintln!("best lag={best_lag} corr={best_cc:.5}");

    // Where is the error? Per-8192-sample (16-frame batch) window correlation.
    let win = 8192;
    let mut worst = Vec::new();
    for (wi, start) in (0..n).step_by(win).enumerate() {
        let end = (start + win).min(n);
        if end - start < 64 {
            continue;
        }
        let cc = audio::correlation(&ch0[start..end], &oracle_pcm[start..end]);
        let mx = audio::max_abs_error(&ch0[start..end], &oracle_pcm[start..end]);
        if cc < 0.97 {
            worst.push((wi, start, cc, mx));
        }
    }
    eprintln!(
        "windows with corr<0.97: {} / {}",
        worst.len(),
        n.div_ceil(win)
    );
    for (wi, start, cc, mx) in worst.iter().take(12) {
        eprintln!("  win {wi} @{start}: corr={cc:.4} max_abs={mx:.4}");
    }
    Ok(())
}
