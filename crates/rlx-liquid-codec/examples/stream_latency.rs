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

//! Streaming latency: per-frame time of [`EncodeStream::push`] (80 ms of
//! samples) and [`DecodeStream::push`] (one frame) after warm-up, against
//! whole-clip encode and decode.
//!
//! ```bash
//! cargo run -p rlx-liquid-codec --release --example stream_latency -- \
//!     [--weights DIR] [--wav clip.wav] [--device cpu]
//! ```
//!
//! [`EncodeStream::push`]: rlx_liquid_codec::EncodeStream::push
//! [`DecodeStream::push`]: rlx_liquid_codec::DecodeStream::push

use std::time::Instant;

use anyhow::Result;
use rlx_liquid_codec::{LiquidCodec, read_24k};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| default.to_string())
    };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let weights = opt("--weights", &format!("{root}/weights/audio/liquid-codec"));
    let wav = opt(
        "--wav",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clip_24k.wav"),
    );
    let device = rlx_cli::parse_device(&opt("--device", "cpu"))?;
    let codec = LiquidCodec::open(weights.as_ref(), device)?;
    let pcm = read_24k(wav.as_ref())?;
    let codes = codec.encode(&pcm)?;
    let n = codes.len();
    let t = Instant::now();
    let _ = codec.encode(&pcm)?;
    let whole_enc = t.elapsed();

    let fs = codec.config().frame_size();
    let mut enc = codec.encode_stream();
    let mut enc_ms = Vec::new();
    for (i, piece) in pcm.chunks(fs).enumerate() {
        let t = Instant::now();
        let got = enc.push(piece)?;
        if i > 1 && !got.is_empty() {
            enc_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    enc.finish()?;
    enc_ms.sort_by(f64::total_cmp);

    let t = Instant::now();
    let _ = codec.decode(&codes)?;
    let cold = t.elapsed();
    let t = Instant::now();
    let _ = codec.decode(&codes)?;
    let whole = t.elapsed();

    let mut warm = codec.decode_stream();
    let t = Instant::now();
    warm.push(&codes[..1])?;
    let first = t.elapsed();
    let mut per = Vec::with_capacity(n);
    for f in &codes[1..] {
        let t = Instant::now();
        warm.push(std::slice::from_ref(f))?;
        per.push(t.elapsed().as_secs_f64() * 1e3);
    }
    per.sort_by(f64::total_cmp);
    let pct = |p: f64| per[((per.len() - 1) as f64 * p) as usize];
    let frame_ms = 1e3 / codec.config().frame_rate() as f64;
    println!(
        "{device:?}: whole encode {whole_enc:.2?} warm; stream encode per frame median {:.2} ms, p90 {:.2} ms",
        enc_ms[enc_ms.len() / 2],
        enc_ms[(enc_ms.len() - 1) * 9 / 10]
    );
    println!(
        "{device:?}: {n} frames; whole decode {whole:.2?} warm ({cold:.2?} with compile, RTF {:.4})",
        whole.as_secs_f64() / (n as f64 * frame_ms / 1e3)
    );
    println!(
        "stream: first push {first:.2?} (compile); per frame median {:.2} ms, p90 {:.2} ms, max {:.2} ms ({frame_ms:.0} ms of audio each)",
        pct(0.5),
        pct(0.9),
        pct(1.0)
    );
    Ok(())
}
