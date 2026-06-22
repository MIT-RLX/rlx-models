// RLX — Qwen3-ASR parity harness against a transformers-library reference.
//
// Reads /tmp/ref/*.bin + meta.json (produced by ref_lib.py) and the real
// checkpoint, runs each RLX stage, and reports max-abs error + token-id match.
//
//   cargo run -p rlx-qwen3-asr --example parity -- \
//       /Users/Shared/qwen3-asr-0.6b /tmp/ref

use anyhow::{Context, Result};
use rlx_core::flow_util::compile_built;
use rlx_qwen3_asr::weights::{KEY_LM_HEAD, LanguageModelPrefixLoader, PREFIX_LANGUAGE_MODEL};
use rlx_qwen3_asr::{
    AsrTokenizer, AsrWeightStore, AudioGeometry, Qwen3AsrConfig, argmax_token,
    build_asr_prefill_built, build_encoder_built, pcm_to_log_mel,
};
use rlx_runtime::Device;
use std::path::PathBuf;

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn max_abs(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    let mut max = 0f32;
    let mut sum = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / a.len() as f32)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir: PathBuf = args
        .next()
        .unwrap_or("/Users/Shared/qwen3-asr-0.6b".into())
        .into();
    let ref_dir = args.next().unwrap_or("/tmp/ref".into());
    let dev_str = args.next().unwrap_or_else(|| "cpu".into());
    let device = rlx_cli::parse_standard_device("qwen3-asr", &dev_str)?;
    let stages_only = std::env::var("PARITY_STAGES_ONLY").is_ok();
    println!("[device] {device:?}");

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{ref_dir}/meta.json"))?)?;
    let cfg = Qwen3AsrConfig::from_file(&model_dir.join("config.json"))?;
    cfg.validate()?;
    let store = AsrWeightStore::open(&model_dir)?;
    let n_mels = cfg.audio.num_mel_bins;

    // ── Stage 1: mel ────────────────────────────────────────────────────
    let pcm = read_f32(&format!("{ref_dir}/pcm.bin"))?;
    let ref_mel = read_f32(&format!("{ref_dir}/mel.bin"))?;
    let my_mel = pcm_to_log_mel(&pcm, n_mels)?;
    let t = my_mel.n_frames;
    println!("[mel] frames: mine={} ref={}", t, ref_mel.len() / n_mels);
    let (mmax, mmean) = max_abs(&my_mel.data, &ref_mel);
    println!("[mel] max_abs={mmax:.3e} mean_abs={mmean:.3e}");

    // ── Stage 2: audio encoder (on the REFERENCE mel, to isolate it) ────
    let geom = AudioGeometry::new(&cfg.audio, t)?;
    let padded = geom.num_chunks * geom.max_chunk_len;
    let mut mel_in = vec![0f32; n_mels * padded];
    for m in 0..n_mels {
        mel_in[m * padded..m * padded + t].copy_from_slice(&ref_mel[m * t..(m + 1) * t]);
    }
    let mut wm = store.load_audio_weights()?;
    let enc = build_encoder_built(&cfg.audio, &mut wm, &geom)?;
    let params = enc.params().clone();
    let mut enc_c = compile_built(enc, device)?;
    for (n, d) in &params {
        enc_c.set_param(n, d);
    }
    let my_embeds = enc_c
        .run(&[("mel", mel_in.as_slice())])
        .into_iter()
        .next()
        .unwrap();
    let ref_embeds = read_f32(&format!("{ref_dir}/audio_embeds.bin"))?;
    let (emax, emean) = max_abs(&my_embeds, &ref_embeds);
    println!(
        "[encoder] tokens={} max_abs={emax:.3e} mean_abs={emean:.3e}",
        geom.num_audio_tokens
    );

    // ── Stage 3: decoder prefill on the REFERENCE fused embeds ──────────
    let fused = read_f32(&format!("{ref_dir}/fused_embeds.bin"))?;
    let seq = meta["input_ids_len"].as_u64().unwrap() as usize;
    let mut wm = store.load_prefixes(&[PREFIX_LANGUAGE_MODEL, KEY_LM_HEAD])?;
    let pf = {
        let mut loader = LanguageModelPrefixLoader::new(&mut wm);
        build_asr_prefill_built(
            &cfg.text,
            &mut loader,
            1,
            seq,
            matches!(device, Device::Metal),
        )?
    };
    let params = pf.params().clone();
    let mut pf_c = compile_built(pf, device)?;
    for (n, d) in &params {
        pf_c.set_param(n, d);
    }
    let my_logits = pf_c
        .run(&[("inputs_embeds", fused.as_slice())])
        .into_iter()
        .next()
        .unwrap();
    let ref_logits = read_f32(&format!("{ref_dir}/prefill_logits.bin"))?;
    let (lmax, lmean) = max_abs(&my_logits, &ref_logits);
    let my_arg = argmax_token(&my_logits);
    let ref_arg = meta["prefill_argmax"].as_u64().unwrap() as u32;
    println!(
        "[prefill] argmax mine={my_arg} ref={ref_arg} match={}",
        my_arg == ref_arg
    );
    println!("[prefill] logits max_abs={lmax:.3e} mean_abs={lmean:.3e}");

    if stages_only {
        return Ok(());
    }

    // ── Stage 4: end-to-end token ids ───────────────────────────────────
    let ref_ids: Vec<u32> = meta["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let tok = AsrTokenizer::from_model_dir(&model_dir)?;
    let n_audio = geom.num_audio_tokens;
    let my_prompt = tok.build_prompt(&cfg, "", n_audio)?;
    println!("[e2e] prompt ids match ref: {}", my_prompt == ref_ids);
    if my_prompt != ref_ids {
        println!("      mine={:?}\n      ref ={:?}", my_prompt, ref_ids);
    }

    let runner = rlx_qwen3_asr::AsrRunner::builder()
        .weights(&model_dir)
        .device(device)
        .max_new_tokens(64)
        .build()?;
    let toks = runner.generate(&my_prompt, &my_mel)?;
    let my_gen: Vec<u32> = toks[my_prompt.len()..].to_vec();
    let ref_gen: Vec<u32> = meta["gen_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .filter(|&t| t != 151643 && t != 151645) // drop eos for comparison
        .collect();
    println!("[e2e] gen ids mine={my_gen:?}");
    println!("[e2e] gen ids ref ={ref_gen:?}");
    println!("[e2e] MATCH = {}", my_gen == ref_gen);
    Ok(())
}
