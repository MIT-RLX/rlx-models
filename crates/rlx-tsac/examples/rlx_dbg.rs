//! One-shot RLX vs C debug on a short clip (fast C decode). Dumps RVQ + output.
//! ```bash
//! RLX_DUMP=1 RLX_TSAC_DIR=$PWD/.cache/tsac cargo run -p rlx-tsac --example rlx_dbg \
//!   --release --features native-codec
//! ```
use anyhow::Result;
use rlx_runtime::Device;
use rlx_tsac::rlx_decode::{RlxDecoder, read_codes};
use rlx_tsac::{TsacBackendKind, TsacCodec, TsacOptions, audio, default_tsac_dir};

fn main() -> Result<()> {
    let dir = default_tsac_dir();
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rlx-qwen3-tts/examples/audio/ask_not.wav");

    // Build a short mono 44.1k clip (~24 frames) for a fast C decode.
    let (mut pcm, _sr) = audio::load_wav_f32(&src, rlx_tsac::SAMPLE_RATE)?;
    let frames: usize = std::env::var("RLX_DBG_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let keep = frames * 512; // frames * hop
    pcm.truncate(keep.min(pcm.len()));
    let short_raw = std::env::temp_dir().join("rlx_dbg_raw.wav");
    audio::write_wav_f32(&short_raw, &pcm, rlx_tsac::SAMPLE_RATE, 1)?;
    let short = std::env::temp_dir().join("rlx_dbg_short.wav");
    audio::prepare_tsac_wav(&short_raw, &short)?;

    let tsac = std::env::temp_dir().join("rlx_dbg.tsac");
    let oracle = std::env::temp_dir().join("rlx_dbg_c.wav");

    let c = TsacCodec::open_with_options(
        &dir,
        TsacOptions {
            device: Device::Cpu,
            backend: TsacBackendKind::Native,
            quality: Some(6),
            ..Default::default()
        },
    )?;
    c.encode(&short, &tsac)?;
    c.decode(&tsac, &oracle)?; // prints C_RVQ / C_M6 when RLX_DUMP set

    let (codes, n_frames, n_cb) = read_codes(&tsac)?;
    eprintln!("codes: n_frames={n_frames} n_cb={n_cb}");
    let dec = RlxDecoder::open(&dir, Device::Cpu)?;
    let out = dec.decode_codes(&codes, n_frames, n_cb)?; // prints RLX_RVQ
    let ch0: Vec<f32> = out.row(0).to_vec();
    let oracle_pcm = audio::load_pcm_from_wav(&oracle)?;
    // Is tsac-ng's OWN roundtrip (C enc -> C dec) correlated with the input?
    {
        let m = pcm.len().min(oracle_pcm.len());
        let cc = audio::correlation(&pcm[..m], &oracle_pcm[..m]);
        let (mut bl, mut bc) = (0i64, -2.0f32);
        for lag in -1024i64..=1024 {
            let (a, b): (&[f32], &[f32]) = if lag >= 0 {
                (&oracle_pcm[lag as usize..], &pcm[..])
            } else {
                (&oracle_pcm[..], &pcm[(-lag) as usize..])
            };
            let mm = a.len().min(b.len());
            if mm < 500 {
                continue;
            }
            let c = audio::correlation(&a[..mm], &b[..mm]);
            if c > bc {
                bc = c;
                bl = lag;
            }
        }
        eprintln!(
            "tsac-ng roundtrip (Cenc->Cdec) vs INPUT: corr={cc:.4} bestlag={bl} bestcorr={bc:.4}"
        );
    }
    let n = ch0.len().min(oracle_pcm.len());
    eprintln!("RLX[0..8]   = {:?}", &ch0[0..8.min(ch0.len())]);
    eprintln!(
        "oracle[0..8]= {:?}",
        &oracle_pcm[0..8.min(oracle_pcm.len())]
    );
    eprintln!(
        "corr={:.5} max_abs={:.5} (rlx_len={} c_len={})",
        audio::correlation(&ch0[..n], &oracle_pcm[..n]),
        audio::max_abs_error(&ch0[..n], &oracle_pcm[..n]),
        ch0.len(),
        oracle_pcm.len()
    );
    Ok(())
}
