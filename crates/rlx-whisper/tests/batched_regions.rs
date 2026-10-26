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

#![cfg(feature = "tokenizer")]

//! Regions decoded together (batch > 1) transcribe as each does alone: the
//! same speech twice, a second of silence between, as two regions in one
//! batch. On a GPU this runs the resident K/V path at batch 2, each
//! sequence's rows kept apart. Opt-in: `RLX_TEST_WHISPER_DIR` and
//! `RLX_TEST_WHISPER_WAV`; `RLX_TEST_DEVICE` (`cpu`).

use rlx_whisper::audio::SpeechSegment;

#[test]
fn regions_in_one_batch_transcribe_as_each_alone() {
    let (Some(dir), Some(wav)) = (
        std::env::var_os("RLX_TEST_WHISPER_DIR"),
        std::env::var_os("RLX_TEST_WHISPER_WAV"),
    ) else {
        eprintln!("skip: set RLX_TEST_WHISPER_DIR and RLX_TEST_WHISPER_WAV");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let speech = rlx_whisper::audio::load_wav_mono_f32(std::path::Path::new(&wav)).unwrap();
    let device = std::env::var("RLX_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    let mut runner = rlx_whisper::WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(rlx_runtime::parse_device(&device).unwrap())
        .language("en")
        .build()
        .unwrap();
    let alone = runner.transcribe_greedy(&speech).unwrap();

    let gap = vec![0.0f32; 16_000];
    let pcm: Vec<f32> = [speech.as_slice(), &gap, &speech].concat();
    let second = speech.len() + gap.len();
    let regions = [
        SpeechSegment {
            start: 0,
            end: speech.len(),
        },
        SpeechSegment {
            start: second,
            end: second + speech.len(),
        },
    ];
    let together = runner
        .transcribe_regions_batched(&pcm, &regions, 1)
        .unwrap();
    eprintln!("{device}: alone {alone:?}\n  together {together:?}");
    assert_eq!(together.len(), 2);
    for t in &together {
        assert_eq!(t.trim(), alone.trim());
    }
}
