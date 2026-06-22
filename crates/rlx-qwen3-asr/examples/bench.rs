// End-to-end Qwen3-ASR benchmark with per-stage timings.
//   cargo run --release -p rlx-qwen3-asr --features metal --example bench -- <model_dir> <device> [audio_s]
//
// Prints a machine-readable `BENCH device=… compute_ms=…` line plus a
// stage breakdown. Graphs are compiled once and reused so timings reflect
// backend compute, not the per-step weight reload in AsrRunner::generate.

use anyhow::Result;
use rlx_core::flow_util::compile_built;
use rlx_qwen3_asr::weights::{
    KEY_EMBED_TOKENS, KEY_LM_HEAD, LanguageModelPrefixLoader, PREFIX_LANGUAGE_MODEL,
};
use rlx_qwen3_asr::{
    AsrTokenizer, AsrWeightStore, AudioGeometry, Qwen3AsrConfig, argmax_token,
    build_asr_decode_built, build_asr_prefill_built, build_encoder_built, fuse_inputs_embeds,
    pcm_to_log_mel, rope_slice,
};
use rlx_runtime::Device;
use std::path::PathBuf;
use std::time::Instant;

fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f(); // warmup
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64 // ms/rep
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let model_dir: PathBuf = a
        .next()
        .unwrap_or("/Users/Shared/qwen3-asr-0.6b".into())
        .into();
    let dev_str = a.next().unwrap_or("cpu".into());
    let audio_s: f64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let device = rlx_cli::parse_standard_device("bench", &dev_str)?;
    let skip_fusion = matches!(device, Device::Metal);

    let cfg = Qwen3AsrConfig::from_file(&model_dir.join("config.json"))?;
    cfg.validate()?;
    let store = AsrWeightStore::open(&model_dir)?;
    let _h = cfg.text.hidden_size;
    let layers = cfg.text.num_hidden_layers;

    // Deterministic test audio.
    let sr = 16000usize;
    let n = (audio_s * sr as f64) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            use std::f32::consts::TAU;
            0.3 * (220.0 * TAU * t).sin() + 0.2 * (440.0 * TAU * t).sin()
        })
        .collect();

    // ── mel (host) ──
    let mel_ms = bench(5, || {
        let _ = pcm_to_log_mel(&pcm, cfg.audio.num_mel_bins).unwrap();
    });
    let mel = pcm_to_log_mel(&pcm, cfg.audio.num_mel_bins)?;
    let geom = AudioGeometry::new(&cfg.audio, mel.n_frames)?;
    let n_audio = geom.num_audio_tokens;
    eprintln!(
        "[stage] mel done: frames={} chunks={} tokens={} windows={:?}",
        mel.n_frames, geom.num_chunks, n_audio, geom.windows
    );

    // ── encoder ──
    let padded = geom.num_chunks * geom.max_chunk_len;
    let mut mel_in = vec![0f32; cfg.audio.num_mel_bins * padded];
    for m in 0..cfg.audio.num_mel_bins {
        mel_in[m * padded..m * padded + mel.n_frames]
            .copy_from_slice(&mel.data[m * mel.n_frames..(m + 1) * mel.n_frames]);
    }
    let mut wm = store.load_audio_weights()?;
    let enc_b = build_encoder_built(&cfg.audio, &mut wm, &geom)?;
    let enc_p = enc_b.params().clone();
    let mut enc = compile_built(enc_b, device)?;
    for (k, v) in &enc_p {
        enc.set_param(k, v);
    }
    let enc_ms = bench(5, || {
        let _ = enc.run(&[("mel", mel_in.as_slice())]);
    });
    let audio_embeds = enc
        .run(&[("mel", mel_in.as_slice())])
        .into_iter()
        .next()
        .unwrap();
    drop(enc);
    eprintln!(
        "[stage] encoder done: {} embeds",
        audio_embeds.len() / cfg.audio.output_dim
    );

    // ── fuse ──
    let tok = AsrTokenizer::from_model_dir(&model_dir)?;
    let prompt = tok.build_prompt(&cfg, "", n_audio)?;
    let seq = prompt.len();
    let mut ewm = store.load_keys(&[KEY_EMBED_TOKENS])?;
    let (embed, _) = ewm.take(KEY_EMBED_TOKENS)?;
    let fused = fuse_inputs_embeds(&cfg, &embed, &prompt, &audio_embeds)?;
    drop(ewm);

    // ── prefill (TTFT compute) ──
    let mut pwm = store.load_prefixes(&[PREFIX_LANGUAGE_MODEL, KEY_LM_HEAD])?;
    let pf_b = {
        let mut l = LanguageModelPrefixLoader::new(&mut pwm);
        build_asr_prefill_built(&cfg.text, &mut l, 1, seq, skip_fusion)?
    };
    let pf_p = pf_b.params().clone();
    let mut pf = compile_built(pf_b, device)?;
    for (k, v) in &pf_p {
        pf.set_param(k, v);
    }
    eprintln!("[stage] prefill compiled (seq={seq}), running…");
    let prefill_ms = bench(3, || {
        let _ = pf.run(&[("inputs_embeds", fused.as_slice())]);
    });
    eprintln!("[stage] prefill done");
    let outs = pf.run(&[("inputs_embeds", fused.as_slice())]);
    let next = argmax_token(&outs[0]);
    let kv: Vec<Vec<f32>> = outs[1..].to_vec();
    drop(pf);

    // ── decode (single reused graph at past=seq) ──
    let mut dwm = store.load_language_model_weights()?;
    let dec_b = {
        let mut l = LanguageModelPrefixLoader::new(&mut dwm);
        build_asr_decode_built(&cfg.text, &mut l, 1, seq, skip_fusion)?
    };
    let dec_p = dec_b.params().clone();
    let mut dec = compile_built(dec_b, device)?;
    for (k, v) in &dec_p {
        dec.set_param(k, v);
    }
    let key_past: Vec<String> = (0..layers)
        .flat_map(|i| [format!("past_k_{i}"), format!("past_v_{i}")])
        .collect();
    let (cos, sin) = rope_slice(&cfg.text, seq);
    let token_f = [next as f32];
    let step_ms = bench(10, || {
        let mut inp: Vec<(&str, &[f32])> = vec![
            ("input_ids", &token_f),
            ("rope_cos", cos.as_slice()),
            ("rope_sin", sin.as_slice()),
        ];
        for i in 0..layers {
            inp.push((key_past[2 * i].as_str(), kv[2 * i].as_slice()));
            inp.push((key_past[2 * i + 1].as_str(), kv[2 * i + 1].as_slice()));
        }
        let _ = dec.run(&inp);
    });
    drop(dec);

    let decode_tok_s = 1000.0 / step_ms;
    // Representative e2e compute for this clip: mel + enc + prefill + decode to
    // a typical transcript length (~ audio_s words * 1.4 tokens, min 8).
    let gen_len = ((audio_s * 2.5) as usize).max(8);
    let compute_ms = mel_ms + enc_ms + prefill_ms + step_ms * gen_len as f64;
    let rtf = (audio_s * 1000.0) / compute_ms;

    println!(
        "── Qwen3-ASR e2e bench: device={device:?} audio={audio_s}s frames={} audio_tokens={n_audio} prompt={seq} ──",
        mel.n_frames
    );
    println!("  mel         {mel_ms:8.2} ms");
    println!("  encoder     {enc_ms:8.2} ms  ({n_audio} tokens)");
    println!("  prefill TTFT{prefill_ms:8.2} ms  ({seq} tokens)");
    println!("  decode/step {step_ms:8.2} ms  ({decode_tok_s:.1} tok/s)");
    println!(
        "  e2e compute {compute_ms:8.2} ms  (mel+enc+prefill+{gen_len}×decode)  RTF={rtf:.1}×"
    );
    println!(
        "BENCH device={dev_str} compute_ms={compute_ms:.3} enc_ms={enc_ms:.3} prefill_ms={prefill_ms:.3} step_ms={step_ms:.3} tok_s={decode_tok_s:.2} rtf={rtf:.3}"
    );
    Ok(())
}
