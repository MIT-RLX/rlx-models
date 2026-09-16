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

//! Dump the frontend's per-frame intermediates, for stage-by-stage comparison
//! against the upstream C (see `tests/fixtures/README.md`).
//!
//! ```text
//! cargo run -p rlx-ten-vad --release --example dump_stages -- in.pcm out_prefix
//! ```
//! Writes `<prefix>.binpow.f32` (513/frame), `<prefix>.pitch.f32` (1/frame)
//! and `<prefix>.feat.f32` (3×41/frame), all little-endian f32.
//!
//! With `--score FEATS`, skips the frontend and instead scores a `.feat.f32`
//! file through the network, writing `<prefix>.prob.f32`. Feeding the reference
//! C's features in that way separates frontend error from network error.

use std::io::Write;

use rlx_ten_vad::frontend::{Frontend, pre_emphasis};
use rlx_ten_vad::{HOP_SIZE, TenVadWeights};

fn writer(prefix: &str, suffix: &str) -> std::io::BufWriter<std::fs::File> {
    std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.{suffix}")).expect("create"))
}

fn put(w: &mut impl Write, values: &[f32]) {
    for v in values {
        w.write_all(&v.to_le_bytes()).expect("write");
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let ["--score", feats, prefix] = args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        return score_features(feats, prefix);
    }
    let [input, prefix] = args.as_slice() else {
        anyhow::bail!(
            "usage: dump_stages <in.pcm> <out_prefix> | --score <feats.f32> <out_prefix>"
        );
    };

    let pcm: Vec<f32> = std::fs::read(input)?
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
        .collect();

    let mut frontend = Frontend::new(TenVadWeights::embedded().core());
    let (mut bin, mut pitch, mut feat) = (
        writer(prefix, "binpow.f32"),
        writer(prefix, "pitch.f32"),
        writer(prefix, "feat.f32"),
    );
    let mut emph = vec![0.0f32; HOP_SIZE];
    let mut prev = 0.0f32;
    let mut frames = 0usize;
    for raw in pcm.chunks_exact(HOP_SIZE) {
        pre_emphasis(raw, &mut prev, &mut emph);
        let info = frontend.push(raw, &emph);
        put(&mut bin, frontend.spectrum());
        put(&mut pitch, &[info.pitch.freq_hz]);
        put(&mut feat, frontend.context());
        frames += 1;
    }
    eprintln!("dumped {frames} frames");
    Ok(())
}

/// Score a `.feat.f32` dump (3×41 per frame) through the streaming graph.
fn score_features(path: &str, prefix: &str) -> anyhow::Result<()> {
    use rlx_ten_vad::model::{LstmState, Shape, TenVadModel};
    use rlx_ten_vad::{CONTEXT_FRAMES, FEATURE_LEN};

    let stride = CONTEXT_FRAMES * FEATURE_LEN;
    let feats: Vec<f32> = std::fs::read(path)?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    anyhow::ensure!(
        feats.len().is_multiple_of(stride),
        "{path} is not a multiple of {stride} floats"
    );

    let mut model = TenVadModel::new(
        rlx_runtime::Device::Cpu,
        Shape::Streaming,
        TenVadWeights::embedded(),
    )?;
    let mut state = LstmState::default();
    let mut out = writer(prefix, "prob.f32");
    for frame in feats.chunks_exact(stride) {
        put(&mut out, &[model.step(frame, &mut state)?]);
    }
    eprintln!("scored {} frames", feats.len() / stride);
    Ok(())
}
