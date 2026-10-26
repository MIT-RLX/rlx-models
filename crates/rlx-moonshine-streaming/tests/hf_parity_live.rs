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

//! Parity vs transformers' `MoonshineStreamingForConditionalGeneration`.
//!
//! ```sh
//! python3 scripts/moonshine_streaming_reference.py weights/asr/moonshine-streaming-tiny \
//!     assets/harness/jfk_16k.wav /tmp/moonshine-streaming-tiny.ref.json
//! RLX_MOONSHINE_STREAMING_DIR=$PWD/weights/asr/moonshine-streaming-tiny \
//! RLX_MOONSHINE_STREAMING_REF=/tmp/moonshine-streaming-tiny.ref.json \
//! RLX_MOONSHINE_STREAMING_WAV=$PWD/assets/harness/jfk_16k.wav \
//!     cargo test -p rlx-moonshine-streaming --release --test hf_parity_live -- --nocapture
//! ```
//! `RLX_TEST_DEVICE=metal` (with the matching feature) runs it on a GPU.

use rlx_moonshine_streaming::MoonshineStreaming;

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-6);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / scale
}

#[test]
fn matches_transformers() {
    let (Ok(dir), Ok(path), Ok(wav)) = (
        rlx_ir::env::var_result("RLX_MOONSHINE_STREAMING_DIR"),
        rlx_ir::env::var_result("RLX_MOONSHINE_STREAMING_REF"),
        rlx_ir::env::var_result("RLX_MOONSHINE_STREAMING_WAV"),
    ) else {
        eprintln!("skip: set RLX_MOONSHINE_STREAMING_{{DIR,REF,WAV}}");
        return;
    };
    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(rlx_runtime::Device::Cpu);
    let r: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read reference")).unwrap();
    let (pcm, rate) = rlx_core::asr_bench::read_wav_mono(std::path::Path::new(&wav)).unwrap();
    assert_eq!(rate, 16_000);
    assert_eq!(pcm.len() as u64, r["num_samples"].as_u64().unwrap());

    let mut model = MoonshineStreaming::open(&dir, device).expect("open");
    let enc = model.encode(&pcm).expect("encode");
    assert_eq!(
        enc.enc_seq as u64,
        r["encoder_seq"].as_u64().unwrap(),
        "encoder length"
    );

    let hidden = max_rel(&enc.hidden, &floats(&r["encoder_hidden"]));
    let states = max_rel(&enc.decoder_states, &floats(&r["decoder_encoder_states"]));
    eprintln!("encoder hidden max rel {hidden:.2e}, decoder states max rel {states:.2e}");
    // Measured 1.6e-6 – 3.5e-6 across tiny / small / medium on CPU;
    // dropping the per-layer sliding windows gives 0.70.
    assert!(hidden < 1e-4, "encoder hidden: {hidden}");
    assert!(states < 1e-4, "decoder states: {states}");

    let want: Vec<u32> = r["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let got = model.generate(&enc, want.len() - 1).expect("generate");
    eprintln!("rlx: {got:?}\nhf:  {want:?}\ntext: {:?}", r["text"]);
    assert_eq!(got, want, "greedy tokens");
}
