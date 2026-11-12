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

//! Parity vs HF `M2M100ForConditionalGeneration` on the real checkpoint.
//!
//! ```sh
//! python3 scripts/nllb_reference.py weights/mt/nllb-200-distilled-600M /tmp/nllb.ref.json
//! RLX_NLLB_DIR=$PWD/weights/mt/nllb-200-distilled-600M RLX_NLLB_REF=/tmp/nllb.ref.json \
//!     cargo test -p rlx-nllb --release --test hf_parity_live -- --nocapture
//! ```
//! `RLX_TEST_DEVICE=metal` (with the matching feature) runs it on a GPU;
//! `just test-nllb-parity [DEVICE]` does all of the above.

use rlx_nllb::{GenerateConfig, NllbConfig, NllbModel, NllbTokenizer};
use serde_json::Value;
use std::path::Path;

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// `max |a - b| / max |b|` — scale-free, so one bound serves the encoder
/// (|x| ~ 1) and the decoder output (|x| up to ~30 after the last norm).
fn rel_max(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let scale = b.iter().map(|y| y.abs()).fold(0f32, f32::max);
    diff / scale
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max as f64
        + logits
            .iter()
            .map(|&l| ((l - max) as f64).exp())
            .sum::<f64>()
            .ln();
    logits.iter().map(|&l| (l as f64 - lse) as f32).collect()
}

#[test]
fn matches_hf_m2m100() {
    let (Ok(dir), Ok(path)) = (
        rlx_ir::env::var_result("RLX_NLLB_DIR"),
        rlx_ir::env::var_result("RLX_NLLB_REF"),
    ) else {
        eprintln!("skip: set RLX_NLLB_DIR and RLX_NLLB_REF");
        return;
    };
    let device = rlx_ir::env::var_result("RLX_TEST_DEVICE")
        .ok()
        .map(|d| rlx_cli::parse_device(&d).expect("RLX_TEST_DEVICE"))
        .unwrap_or(rlx_runtime::Device::Cpu);
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read reference")).unwrap();
    let dir = Path::new(&dir);
    let cfg = NllbConfig::from_hf_config_json(&dir.join("config.json")).expect("config");
    let d = cfg.d_model;
    let mut model = NllbModel::load(dir, cfg.clone(), device).expect("load");
    let tk = NllbTokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let beams = r["beams"].as_u64().unwrap() as usize;
    let max_new = r["max_new_tokens"].as_u64().unwrap() as usize;

    for case in r["cases"].as_array().unwrap() {
        let (text, src, tgt) = (
            case["text"].as_str().unwrap(),
            case["src"].as_str().unwrap(),
            case["tgt"].as_str().unwrap(),
        );
        let src_ids = tk.encode(text, src).expect("encode");
        assert_eq!(src_ids, ids(&case["src_ids"]), "{src}->{tgt}: source ids");
        let tgt_id = tk.lang_token_id(tgt).unwrap();
        assert_eq!(tgt_id, case["tgt_id"].as_u64().unwrap() as u32);

        // Encoder: the bucket-padded generation path, and the unpadded one.
        let enc = model.encode_tokens(&src_ids).expect("encode");
        let want_enc = floats(&case["encoder_hidden"]);
        let enc_err = rel_max(&enc.hidden, &want_enc);
        let full_enc_err = rel_max(&model.encode_full(&src_ids).expect("encode"), &want_enc);

        // Teacher-forced decoder over HF's greedy output, both spellings:
        // one causal pass, and token by token through the KV cache.
        let want_greedy = ids(&case["greedy_ids"]);
        let prefix = &want_greedy[..want_greedy.len() - 1];
        let want_dec = floats(&case["dec_hidden"]);
        let hidden = model.decode_hidden(prefix, &enc).expect("decode");
        let dec_err = rel_max(&hidden, &want_dec);
        let mut state = model.start_decode(&enc, 1, max_new + 1).expect("state");
        let mut stepped = Vec::with_capacity(hidden.len());
        for &t in prefix {
            stepped.extend(model.step_hidden(&mut state, &[t]).expect("step"));
        }
        let step_err = rel_max(&stepped, &want_dec);
        let mut logp_err = 0f32;
        for (t, topk) in case["step_topk"].as_array().unwrap().iter().enumerate() {
            let lp = log_softmax(&model.lm_logits(&stepped[t * d..(t + 1) * d]));
            for pair in topk.as_array().unwrap() {
                let (id, want) = (
                    pair[0].as_u64().unwrap() as usize,
                    pair[1].as_f64().unwrap(),
                );
                logp_err = logp_err.max((lp[id] as f64 - want).abs() as f32);
            }
        }

        // Free-running generation.
        let greedy_opts = GenerateConfig::new(max_new).with_forced_bos(tgt_id);
        let greedy = model.generate_greedy(&enc, &greedy_opts).expect("greedy");
        let beam_opts = greedy_opts.clone().with_beams(beams);
        let beam = model.generate_beam(&enc, &beam_opts).expect("beam");
        let want_beam = ids(&case["beam_ids"]);

        println!(
            "{src}->{tgt}: enc rel {enc_err:.2e} (unpadded {full_enc_err:.2e})  dec rel {dec_err:.2e} \
             (kv step {step_err:.2e})  top-8 logp {logp_err:.2e}  greedy {}  beam{beams} {}",
            if greedy == want_greedy { "==" } else { "!=" },
            if beam == want_beam { "==" } else { "!=" },
        );
        // Measured on CPU: enc/dec ≤ 1.2e-6, log-probs ≤ 8.5e-4 (the 256k-row
        // tied head sums in a different order than torch). Shifting the
        // sinusoidal positions by one row gives enc 3.4e-2 and log-probs 0.68
        // while greedy ids still match, so the numeric bounds carry the test.
        assert!(
            enc_err < 1e-4,
            "{src}->{tgt}: encoder rel err {enc_err:.3e}"
        );
        assert!(
            full_enc_err < 1e-4,
            "{src}->{tgt}: unpadded encoder rel err {full_enc_err:.3e}"
        );
        assert!(
            dec_err < 1e-4,
            "{src}->{tgt}: decoder rel err {dec_err:.3e}"
        );
        assert!(
            step_err < 1e-4,
            "{src}->{tgt}: KV-step decoder rel err {step_err:.3e}"
        );
        assert!(logp_err < 5e-3, "{src}->{tgt}: log-prob err {logp_err:.3e}");
        assert_eq!(greedy, want_greedy, "{src}->{tgt}: greedy ids");
        assert_eq!(beam, want_beam, "{src}->{tgt}: beam-{beams} ids");
    }
}
