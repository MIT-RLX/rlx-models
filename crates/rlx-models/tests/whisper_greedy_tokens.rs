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

//! Greedy token-id regression vs HF reference on JFK (whisper-tiny).
//!
//! ```sh
//! cargo test -p rlx-models --test whisper_greedy_tokens --release -- --nocapture
//! ```

use anyhow::Result;
use rlx_models::whisper::{WhisperConfig, WhisperRunner};
use rlx_runtime::Device;
use rlx_whisper::decode::{SuppressionMask, last_logits_row};
use std::path::PathBuf;

fn tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-tiny")
}

fn jfk_wav() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/whisper-bench/jfk_16k.wav")
}

#[test]
fn whisper_greedy_token_ids_match_hf_prefix() -> Result<()> {
    let dir = tiny_dir();
    let weights = dir.join("model.safetensors");
    let wav = jfk_wav();
    if !weights.is_file() || !wav.is_file() {
        eprintln!("skip: need weights + wav");
        return Ok(());
    }

    let cfg = WhisperConfig::from_file(&dir.join("config.json"))?;
    let suppression = SuppressionMask::from_config(&cfg);
    let eot = 50257u32;

    let pcm = rlx_models::whisper::load_wav_mono_f32(&wav)?;
    let mut runner = WhisperRunner::builder()
        .weights(&weights)
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;

    let mel = rlx_models::whisper::pcm_to_mel(&cfg, &pcm);
    let enc = runner.encode_mel(&mel)?;
    let cross = runner.cross_cache_batch(&enc, 1)?;
    let prompt = runner.build_prompt()?;
    let (prefill_logits, mut cache) = runner.prefill_prompt(&cross, &prompt, 1)?;

    let vocab = cfg.vocab_size;
    let mut row = last_logits_row(&prefill_logits, prompt.len(), vocab);
    let mut generated = Vec::new();
    let hf_prefix: [u32; 8] = [1968, 309, 15065, 505, 731, 420, 3171, 300];

    for step in 0..hf_prefix.len() {
        let next = suppression.argmax_next(&mut row, step == 0);
        eprintln!("step {step}: token={next}");
        generated.push(next);
        if next == eot {
            break;
        }
        let step_logits = runner.decode_one_step(&cross, next, &mut cache)?;
        row = if step_logits.len() == vocab {
            step_logits
        } else {
            last_logits_row(&step_logits, 1, vocab)
        };
    }

    eprintln!("generated={generated:?} hf={hf_prefix:?}");
    for (i, (&got, &want)) in generated.iter().zip(hf_prefix.iter()).enumerate() {
        assert_eq!(got, want, "greedy token mismatch at step {i}");
    }

    // Full decoder (no KV) should match HF for the third token.
    use rlx_core::flow_util::compile_built;
    use rlx_models::weight_map::WeightMap;
    use rlx_models::whisper::{
        WhisperWeightPrefix, build_whisper_cross_kv_built, build_whisper_decoder_built,
    };
    let mut wm = WeightMap::from_tensors(WeightMap::snapshot_from_path(weights.to_str().unwrap())?);
    let pfx = WhisperWeightPrefix::detect(&WeightMap::from_tensors(WeightMap::snapshot_from_path(
        weights.to_str().unwrap(),
    )?));
    let enc_seq = cfg.encoder_seq_len(mel.n_frames);
    let cross_built = build_whisper_cross_kv_built(&cfg, &mut wm, &pfx, 1, enc_seq)?;
    let cross_params = cross_built.params().clone();
    let mut cross_g = compile_built(cross_built, Device::Cpu)?;
    for (n, d) in &cross_params {
        cross_g.set_param(n, d);
    }
    let cross_out = cross_g.run(&[("encoder_hidden", &enc)]);
    let mut wm2 =
        WeightMap::from_tensors(WeightMap::snapshot_from_path(weights.to_str().unwrap())?);
    let ids: Vec<f32> = prompt
        .iter()
        .chain(&[1968, 309])
        .map(|&t| t as f32)
        .collect();
    let dec_seq = ids.len();
    let dec_built = build_whisper_decoder_built(&cfg, &mut wm2, &pfx, 1, dec_seq, enc_seq)?;
    let dec_params = dec_built.params().clone();
    let mut dec = compile_built(dec_built, Device::Cpu)?;
    for (n, d) in &dec_params {
        dec.set_param(n, d);
    }
    let mut dec_in: Vec<(&str, &[f32])> = vec![("token_ids", &ids), ("encoder_hidden", &enc)];
    let ck: Vec<String> = (0..cfg.decoder_layers)
        .flat_map(|i| [format!("cross_k_{i}"), format!("cross_v_{i}")])
        .collect();
    for i in 0..cfg.decoder_layers {
        dec_in.push((ck[2 * i].as_str(), cross_out[2 * i].as_slice()));
        dec_in.push((ck[2 * i + 1].as_str(), cross_out[2 * i + 1].as_slice()));
    }
    let full_logits = dec.run(&dec_in).remove(0);
    let full_row = last_logits_row(&full_logits, dec_seq, vocab);
    let full_next = suppression.argmax_next(&mut full_row.clone(), false);
    eprintln!("full decoder next token after prefix: {full_next} (incr step2 was 311)");
    assert_eq!(full_next, 15065, "full decoder should match HF at step 2");

    // Re-run incremental prefix and compare raw logits to full decoder row.
    let mel2 = rlx_models::whisper::pcm_to_mel(&cfg, &pcm);
    let enc2 = runner.encode_mel(&mel2)?;
    let cross2 = runner.cross_cache_batch(&enc2, 1)?;
    let (pre2, mut cache2) = runner.prefill_prompt(&cross2, &prompt, 1)?;
    let mut row2 = last_logits_row(&pre2, prompt.len(), vocab);
    for &tok in &[1968u32, 309] {
        row2 = {
            let lg = runner.decode_one_step(&cross2, tok, &mut cache2)?;
            if lg.len() == vocab {
                lg
            } else {
                last_logits_row(&lg, 1, vocab)
            }
        };
    }
    let mx: f32 = row2
        .iter()
        .zip(full_row.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "incr vs full logits len {} {} max_abs={mx:.6}",
        row2.len(),
        full_row.len()
    );
    assert!(
        mx < 0.1,
        "incremental logits diverge from full decoder: max_abs={mx}"
    );
    Ok(())
}
