// Localize qwen3-asr backend divergence: compare encoder, prefill, and one
// decode step between CPU and a target backend (RLX_ASR_DEVICE) on a real clip.
//   RLX_ASR_MODEL=/Users/Shared/qwen3-asr-0.6b RLX_ASR_WAV=clip.wav \
//   RLX_ASR_DEVICE=gpu cargo run -p rlx-qwen3-asr --example stage_probe \
//   --release --features metal,mlx,gpu,coreml
use anyhow::Result;
use rlx_core::asr_bench::load_clip_16k;
use rlx_core::flow_util::compile_built;
use rlx_qwen3_asr::weights::{
    KEY_EMBED_TOKENS, KEY_LM_HEAD, LanguageModelPrefixLoader, PREFIX_LANGUAGE_MODEL,
};
use rlx_qwen3_asr::{
    AsrTokenizer, AsrWeightStore, AudioGeometry, Qwen3AsrConfig, argmax_token,
    build_asr_decode_built_opts, build_asr_prefill_built, build_encoder_built, fuse_inputs_embeds,
    pcm_to_log_mel, rope_slice,
};
use rlx_runtime::Device;
use rlx_runtime::attn_mask::bucket_decode_mask;
use std::path::PathBuf;

fn stats(label: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let (mut maxd, mut sumd) = (0f32, 0f64);
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        maxd = maxd.max(d);
        sumd += d as f64;
    }
    eprintln!(
        "[probe] {label}: len {} vs {}  max|Δ|={maxd:.5}  mean|Δ|={:.6}",
        a.len(),
        b.len(),
        sumd / n.max(1) as f64
    );
}

fn run_encoder(
    cfg: &Qwen3AsrConfig,
    store: &AsrWeightStore,
    mel_in: &[f32],
    geom: &AudioGeometry,
    device: Device,
) -> Result<Vec<f32>> {
    let mut wm = store.load_audio_weights()?;
    let enc_b = build_encoder_built(&cfg.audio, &mut wm, geom)?;
    let enc_p = enc_b.params().clone();
    let mut enc = compile_built(enc_b, device)?;
    for (k, v) in &enc_p {
        enc.set_param(k, v);
    }
    Ok(enc.run(&[("mel", mel_in)]).into_iter().next().unwrap())
}

/// Prefill → all outputs (logits + KV cache).
fn run_prefill(
    cfg: &Qwen3AsrConfig,
    store: &AsrWeightStore,
    fused: &[f32],
    seq: usize,
    device: Device,
) -> Result<Vec<Vec<f32>>> {
    let skip_fusion = matches!(device, Device::Metal);
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
    Ok(pf.run(&[("inputs_embeds", fused)]))
}

/// One decode step with a fixed token + KV cache → next-token logits.
/// `custom`: use the Custom-mask decode graph (the real runtime path) vs the
/// built-in Causal graph.
fn run_decode(
    cfg: &Qwen3AsrConfig,
    store: &AsrWeightStore,
    token: u32,
    kv: &[Vec<f32>], // each layer K/V padded to `bucket` slots
    seq: usize,      // real past length (rope/mask position)
    bucket: usize,   // graph slot count (>= seq); padding region [seq,bucket) is masked
    device: Device,
    custom: bool,
) -> Result<Vec<Vec<f32>>> {
    let layers = cfg.text.num_hidden_layers;
    let skip_fusion = matches!(device, Device::Metal);
    let mut dwm = store.load_language_model_weights()?;
    let dec_b = {
        let mut l = LanguageModelPrefixLoader::new(&mut dwm);
        build_asr_decode_built_opts(&cfg.text, &mut l, 1, bucket, skip_fusion, custom)?
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
    let token_f = [token as f32];
    let mask = bucket_decode_mask(seq, bucket);
    let mut inp: Vec<(&str, &[f32])> = vec![
        ("input_ids", &token_f),
        ("rope_cos", cos.as_slice()),
        ("rope_sin", sin.as_slice()),
    ];
    if custom {
        inp.push(("mask", mask.as_slice()));
    }
    for i in 0..layers {
        inp.push((key_past[2 * i].as_str(), kv[2 * i].as_slice()));
        inp.push((key_past[2 * i + 1].as_str(), kv[2 * i + 1].as_slice()));
    }
    // Run the SAME compiled graph twice with identical inputs — the real
    // runtime reuses one cached compiled graph across decode steps, so any
    // cross-run state leak (sticky arena/KV binding) shows up here.
    let out1 = dec.run(&inp);
    let out2 = dec.run(&inp);
    let drift: f32 = out1[0]
        .iter()
        .zip(&out2[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("[probe]   reuse drift {device:?} (run2 vs run1, logits): {drift:.6}");
    Ok(out1) // [logits, new_k_0, new_v_0, ...]
}

fn main() -> Result<()> {
    let model_dir: PathBuf = std::env::var("RLX_ASR_MODEL")
        .unwrap_or_else(|_| "/Users/Shared/qwen3-asr-0.6b".into())
        .into();
    let wav = std::env::var("RLX_ASR_WAV").expect("set RLX_ASR_WAV");
    let dev_s = std::env::var("RLX_ASR_DEVICE").unwrap_or_else(|_| "gpu".into());
    let device = rlx_core::asr_bench::parse_device(&dev_s)?;

    let cfg = Qwen3AsrConfig::from_file(&model_dir.join("config.json"))?;
    let store = AsrWeightStore::open(&model_dir)?;
    let (pcm, _) = load_clip_16k(std::path::Path::new(&wav))?;
    let mel = pcm_to_log_mel(&pcm, cfg.audio.num_mel_bins)?;
    let geom = AudioGeometry::new(&cfg.audio, mel.n_frames)?;
    let n_audio = geom.num_audio_tokens;
    eprintln!(
        "[probe] frames={} chunks={} tokens={} windows={:?} device={device:?}",
        mel.n_frames, geom.num_chunks, n_audio, geom.windows
    );
    let padded = geom.num_chunks * geom.max_chunk_len;
    let mut mel_in = vec![0f32; cfg.audio.num_mel_bins * padded];
    for m in 0..cfg.audio.num_mel_bins {
        mel_in[m * padded..m * padded + mel.n_frames]
            .copy_from_slice(&mel.data[m * mel.n_frames..(m + 1) * mel.n_frames]);
    }

    // ── encoder: cpu vs target ──
    let enc_cpu = run_encoder(&cfg, &store, &mel_in, &geom, Device::Cpu)?;
    let enc_dev = run_encoder(&cfg, &store, &mel_in, &geom, device)?;
    stats("ENCODER", &enc_cpu, &enc_dev);

    // ── prefill on identical (cpu) input, cpu vs target ──
    let tok = AsrTokenizer::from_model_dir(&model_dir)?;
    let prompt = tok.build_prompt(&cfg, "", n_audio)?;
    let seq = prompt.len();
    let mut ewm = store.load_keys(&[KEY_EMBED_TOKENS])?;
    let (embed, _) = ewm.take(KEY_EMBED_TOKENS)?;
    let fused_cpu = fuse_inputs_embeds(&cfg, &embed, &prompt, &enc_cpu)?;

    let pf_cpu = run_prefill(&cfg, &store, &fused_cpu, seq, Device::Cpu)?;
    let pf_dev = run_prefill(&cfg, &store, &fused_cpu, seq, device)?;
    stats("PREFILL(logits)", &pf_cpu[0], &pf_dev[0]);
    // Prefill KV outputs (these become the decode's past KV). Same cpu input.
    let nkv = pf_cpu.len().min(pf_dev.len());
    let mut kv_max = 0f32;
    for o in 1..nkv {
        let m = pf_cpu[o]
            .iter()
            .zip(&pf_dev[o])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        kv_max = kv_max.max(m);
    }
    eprintln!(
        "[probe] PREFILL KV outputs max|Δ| across {} tensors = {kv_max:.5}",
        nkv - 1
    );
    // Also: decode using the TARGET's own prefill KV (the real pipeline path).
    let kv_dev: Vec<Vec<f32>> = pf_dev[1..].to_vec();
    let dd_devkv = run_decode(
        &cfg,
        &store,
        argmax_token(&pf_dev[0]),
        &kv_dev,
        seq,
        seq,
        device,
        true,
    )?;
    eprintln!(
        "[probe] decode w/ {device:?}-own prefill KV → token={} (cpu ref=6364-ish)",
        argmax_token(&dd_devkv[0])
    );
    let next = argmax_token(&pf_cpu[0]);
    eprintln!(
        "[probe] first-token argmax: cpu={} {device:?}={}",
        next,
        argmax_token(&pf_dev[0])
    );

    // ── one decode step on identical (cpu) KV + token, cpu vs target ──
    let kv_cpu: Vec<Vec<f32>> = pf_cpu[1..].to_vec();
    let kv_dim = kv_cpu[0].len() / seq;

    // (a) exact length (bucket == seq, mask all-valid)
    let dc = run_decode(&cfg, &store, next, &kv_cpu, seq, seq, Device::Cpu, true)?;
    let dd = run_decode(&cfg, &store, next, &kv_cpu, seq, seq, device, true)?;
    stats("DECODE custom exact (bucket=seq)", &dc[0], &dd[0]);
    eprintln!(
        "[probe] argmax: cpu={} {device:?}={}",
        argmax_token(&dc[0]),
        argmax_token(&dd[0])
    );

    // (b) padded bucket == next_pow2(seq) > seq — the REAL runtime path. KV is
    // zero-padded from `seq` to `bucket` slots; mask invalidates [seq,bucket).
    let bucket = seq.next_power_of_two().max(seq + 1);
    let kv_pad: Vec<Vec<f32>> = kv_cpu
        .iter()
        .map(|layer| {
            let mut p = vec![0f32; bucket * kv_dim];
            p[..seq * kv_dim].copy_from_slice(&layer[..seq * kv_dim]);
            p
        })
        .collect();
    eprintln!("[probe] bucket test: seq={seq} bucket={bucket} kv_dim={kv_dim}");
    let dc = run_decode(&cfg, &store, next, &kv_pad, seq, bucket, Device::Cpu, true)?;
    let dd = run_decode(&cfg, &store, next, &kv_pad, seq, bucket, device, true)?;
    stats("BUCKETED logits", &dc[0], &dd[0]);
    eprintln!(
        "[probe] logits argmax: cpu={} {device:?}={}",
        argmax_token(&dc[0]),
        argmax_token(&dd[0])
    );
    // Compare the decode graph's KV outputs (these feed the NEXT step).
    let n_out = dc.len().min(dd.len());
    for o in 1..n_out {
        let kind = if o % 2 == 1 { "K" } else { "V" };
        stats(
            &format!("BUCKETED out[{o}] ({kind} layer {})", (o - 1) / 2),
            &dc[o],
            &dd[o],
        );
    }
    Ok(())
}
