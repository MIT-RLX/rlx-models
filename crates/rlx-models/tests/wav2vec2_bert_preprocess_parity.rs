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

// Parity test: RLX log-mel vs HuggingFace SeamlessM4TFeatureExtractor.
//
// Requires Python + transformers:
//   pip install transformers numpy
//
// Run:
//   cargo test -p rlx-models wav2vec2_bert_preprocess_parity --release

mod compile_support;

use rlx_models::wav2vec2_bert::{LogMelExtractor, Wav2Vec2BertPreprocessConfig};
use std::process::Command;

const TOL: f32 = 2e-2;

#[test]
fn wav2vec2_bert_preprocess_parity_vs_hf() {
    if std::env::var("W2V_BERT_PREPROCESS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("skip wav2vec2_bert_preprocess_parity_vs_hf: set W2V_BERT_PREPROCESS_PARITY=1");
        return;
    }
    if Command::new("python3")
        .args(["-c", "import transformers"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(true)
    {
        eprintln!("skip: python3 + transformers not available");
        return;
    }

    let sr = 16_000usize;
    let waveform: Vec<f32> = (0..sr)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.25
                + (880.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.1
        })
        .collect();

    let ext = LogMelExtractor::new(Wav2Vec2BertPreprocessConfig::w2v_bert_2_0());
    let rlx = ext.extract(&waveform);

    let py = format!(
        r#"
import numpy as np
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
wav = np.array({wav:?}, dtype=np.float32)
fe = SeamlessM4TFeatureExtractor(
    feature_size=80, sampling_rate=16000, num_mel_bins=80, padding_value=1.0, stride=2
)
out = fe(wav, return_tensors="np", padding=False, do_normalize_per_mel_bins=True)
feat = out["input_features"][0]
mask = out["attention_mask"][0]
print(feat.shape[0], feat.shape[1])
for v in feat.flatten()[:8]:
    print(v)
print("---")
for v in feat.flatten()[-8:]:
    print(v)
print("MASK", int(mask.sum()))
"#,
        wav = waveform,
    );

    let out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "python failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    let shape: Vec<usize> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(shape[1], 160);
    // RLX keeps a fixed-size feature buffer; `shape[0]` is the valid
    // (unpadded) frame count from the HF reference.
    assert!(rlx.features.len() >= shape[0] * shape[1]);

    let mut hf_vals = Vec::new();
    for line in lines.by_ref() {
        if line == "---" {
            break;
        }
        hf_vals.push(line.parse::<f32>().unwrap());
    }
    for (i, (&a, &b)) in rlx
        .features
        .iter()
        .take(hf_vals.len())
        .take(8)
        .zip(hf_vals.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() <= TOL,
            "frame0 mismatch at {i}: rlx={a} hf={b}"
        );
    }

    let mask_line = text.lines().last().unwrap();
    let hf_mask: usize = mask_line.strip_prefix("MASK ").unwrap().parse().unwrap();
    assert_eq!(rlx.num_frames, hf_mask);
}
