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

use rlx_vad::silero::{SileroConfig, SileroSession, SileroWeights};

#[test]
fn silero_embedded_scores_in_unit_interval() {
    let w = SileroWeights::embedded();
    let mut session = SileroSession::new(w, SileroConfig::default());
    let frame = vec![0.0f32; session.frame_samples()];
    let p = session.predict_frame(&frame).expect("predict");
    assert!(p >= 0.0 && p <= 1.0, "prob={p}");
}

#[test]
fn silero_stream_probs_on_jfk() {
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/jfk/jfk_rust_speech.wav");
    if !wav.is_file() {
        return;
    }
    let (sr, pcm) = rlx_vad::load_wav_mono_f32(&wav).expect("wav");
    let pcm = if sr == rlx_vad::SAMPLE_RATE_16K {
        pcm
    } else {
        rlx_vad::resample_linear(&pcm, sr, rlx_vad::SAMPLE_RATE_16K)
    };
    let mut session = SileroSession::new(SileroWeights::embedded(), SileroConfig::default());
    let hop = session.frame_samples();
    let mut probs = Vec::new();
    for chunk in pcm.chunks(hop).take(30) {
        let mut buf = vec![0.0f32; hop];
        buf[..chunk.len()].copy_from_slice(chunk);
        probs.push(session.predict_frame(&buf).expect("predict"));
    }
    let mean: f32 = probs[20..30].iter().sum::<f32>() / 10.0;
    assert!(
        mean > 0.5,
        "speech region mean={mean}, sample={:?}",
        &probs[20..30]
    );
}
