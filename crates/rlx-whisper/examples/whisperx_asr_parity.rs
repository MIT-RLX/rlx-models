//! `whisperx::ct2_beam_search` per VAD chunk against whisperX 3.8.6 / CTranslate2
//! (`tribev2-ref/asr/dump_asr.py`): mel, first encoder output, token ids, score, text.
//! usage: whisperx_asr_parity HF_DIR CT2_CONFIG STAGES_DIR NAME [--device metal]
use rlx_whisper::whisperx::{
    ByteLevelDecoder, Ct2Suppression, ct2_beam_search, ct2_beam_search_fast,
};

fn f32s(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (hf, ct2cfg, dir, name) = (&a[1], &a[2], &a[3], &a[4]);
    let device = match a
        .iter()
        .position(|x| x == "--device")
        .map(|i| a[i + 1].as_str())
    {
        Some("metal") => rlx_runtime::Device::Metal,
        _ => rlx_runtime::Device::Cpu,
    };
    let asr: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{name}.asr.json"))?)?;
    let audio = f32s(&format!("{dir}/{name}.audio.f32"));
    let mel_ref = f32s(&format!("{dir}/{name}.mel.f32"));
    let enc0 = f32s(&format!("{dir}/{name}.enc0.f32"));
    let sup = Ct2Suppression::from_config(&std::fs::read_to_string(ct2cfg)?)?;
    let dec = ByteLevelDecoder::from_tokenizer_json(&std::fs::read_to_string(format!(
        "{hf}/tokenizer.json"
    ))?)?;
    let mut runner = rlx_whisper::WhisperRunner::builder()
        .weights(format!("{hf}/model.safetensors"))
        .config_path(format!("{hf}/config.json"))
        .device(device)
        .build()?;
    let fast = if a.iter().any(|x| x == "--fast") {
        let t = std::time::Instant::now();
        let d = rlx_whisper::fast_decoder::FastDecoder::load(
            std::path::Path::new(&format!("{hf}/model.safetensors")),
            runner.config().decoder_attention_heads,
        )?;
        eprintln!("fast decoder loaded in {:.1}s", t.elapsed().as_secs_f64());
        Some(d)
    } else {
        None
    };
    let prompt: Vec<u32> = asr["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let cfg = runner.config().clone();
    let (mut same_ids, mut same_text) = (0, 0);
    let chunks = asr["chunks"].as_array().unwrap();
    for (k, c) in chunks.iter().enumerate() {
        let (s, e) = (c["start"].as_f64().unwrap(), c["end"].as_f64().unwrap());
        let seg = &audio[(s * 16000.0) as usize..((e * 16000.0) as usize).min(audio.len())];
        let t = std::time::Instant::now();
        let mel = rlx_whisper::audio::pcm_to_mel(&cfg, seg);
        let mref = &mel_ref[k * mel.data.len()..(k + 1) * mel.data.len()];
        let mel_d = mel
            .data
            .iter()
            .zip(mref)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let te = std::time::Instant::now();
        let enc = runner.encode_mel(&mel)?;
        let enc_s = te.elapsed().as_secs_f64();
        let tb = std::time::Instant::now();
        let enc_d = if k == 0 {
            enc.iter()
                .zip(&enc0)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        } else {
            f32::NAN
        };
        let ml = asr["max_length"].as_u64().unwrap() as usize;
        let r = match &fast {
            Some(d) => {
                let cross = runner.cross_cache_batch(&enc, 1)?;
                ct2_beam_search_fast(
                    d,
                    &cross.layers_k,
                    &cross.layers_v,
                    &prompt,
                    5,
                    1.0,
                    1.0,
                    ml,
                    &sup,
                    50257,
                )?
            }
            None => ct2_beam_search(&mut runner, &enc, &prompt, 5, 1.0, 1.0, ml, &sup, 50257)?,
        };
        let beam_s = tb.elapsed().as_secs_f64();
        eprintln!(
            "   encode {enc_s:.1}s beam {beam_s:.1}s ({} tokens)",
            r.ids.len()
        );
        let want_ids: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let text = dec.decode(
            &r.ids
                .iter()
                .copied()
                .filter(|&t| t < 50257)
                .collect::<Vec<_>>(),
        );
        let want_text = c["text"].as_str().unwrap();
        same_ids += (r.ids == want_ids) as usize;
        same_text += (text == want_text) as usize;
        println!(
            "{name}[{k}] ids {} text {} score {:.6} vs {:.6} | mel max|d| {mel_d:.2e} enc max|d| {enc_d:.2e} | {:.1}s",
            if r.ids == want_ids { "SAME" } else { "DIFF" },
            if text == want_text { "SAME" } else { "DIFF" },
            r.score,
            c["score"].as_f64().unwrap(),
            t.elapsed().as_secs_f64()
        );
        if text != want_text {
            println!("   rust: {text:?}\n   ct2 : {want_text:?}");
        }
    }
    println!(
        "{name}: {same_ids}/{} chunks identical ids, {same_text}/{} identical text",
        chunks.len(),
        chunks.len()
    );
    Ok(())
}
