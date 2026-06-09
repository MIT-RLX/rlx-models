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

//! Print per-frame VAD scores on assets/jfk (debug Silero and/or Earshot).

use rlx_vad::audio::{SAMPLE_RATE_16K, load_wav_mono_f32, resample_linear};
use std::path::PathBuf;

#[cfg(feature = "earshot")]
use rlx_vad::earshot::{Detector, FRAME_SAMPLES};
#[cfg(feature = "silero")]
use rlx_vad::silero::{SileroConfig, SileroSession, SileroWeights};

fn main() -> anyhow::Result<()> {
    if rlx_vad::enabled_backends().is_empty() {
        anyhow::bail!("no VAD backends enabled (use `--features earshot` and/or `silero`)");
    }

    let wav =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/jfk/jfk_rust_speech.wav");
    let (sr, pcm) = load_wav_mono_f32(&wav)?;
    let pcm = if sr == SAMPLE_RATE_16K {
        pcm
    } else {
        resample_linear(&pcm, sr, SAMPLE_RATE_16K)
    };

    #[cfg(feature = "silero")]
    let wpath = std::env::var("RLX_VAD_SILERO_WEIGHTS").ok();
    #[cfg(feature = "silero")]
    let w = match wpath {
        Some(p) => SileroWeights::load(p.as_ref())?,
        None => SileroWeights::embedded(),
    };
    #[cfg(feature = "silero")]
    let mut sil = SileroSession::new(w, SileroConfig::default());
    #[cfg(feature = "earshot")]
    let mut ear = Detector::default();

    #[cfg(all(feature = "earshot", feature = "silero"))]
    let hop = sil.frame_samples().max(FRAME_SAMPLES);
    #[cfg(all(feature = "earshot", not(feature = "silero")))]
    let hop = FRAME_SAMPLES;
    #[cfg(all(not(feature = "earshot"), feature = "silero"))]
    let hop = sil.frame_samples();

    let start = pcm.len() / 3;
    eprintln!(
        "speech slice frames around sample {start} (backends: {}):",
        rlx_vad::enabled_backends().join(", ")
    );
    for fi in 0..8 {
        let off = start + fi * hop;
        if off >= pcm.len() {
            break;
        }
        let chunk = &pcm[off..off + hop.min(pcm.len() - off)];

        #[cfg(feature = "silero")]
        {
            let mut buf = vec![0.0; sil.frame_samples()];
            buf[..chunk.len()].copy_from_slice(chunk);
            let sp = sil.predict_frame(&buf)?;
            #[cfg(not(feature = "earshot"))]
            eprintln!("  frame {fi} off={off}: silero={sp:.4}");
            #[cfg(feature = "earshot")]
            {
                let mut ef = [0.0f32; FRAME_SAMPLES];
                let n = chunk.len().min(FRAME_SAMPLES);
                ef[..n].copy_from_slice(&chunk[..n]);
                let ep = ear.predict_f32(&ef);
                eprintln!("  frame {fi} off={off}: silero={sp:.4} earshot={ep:.4}");
            }
        }

        #[cfg(all(feature = "earshot", not(feature = "silero")))]
        {
            let mut ef = [0.0f32; FRAME_SAMPLES];
            let n = chunk.len().min(FRAME_SAMPLES);
            ef[..n].copy_from_slice(&chunk[..n]);
            let ep = ear.predict_f32(&ef);
            eprintln!("  frame {fi} off={off}: earshot={ep:.4}");
        }
    }
    Ok(())
}
