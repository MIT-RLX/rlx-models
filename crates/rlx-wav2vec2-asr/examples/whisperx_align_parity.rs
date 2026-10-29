//! `whisperx::align` against whisperX 3.8.6's own output, per stage dump
//! (`tribev2-ref/asr/dump_stages.py`). With `--emissions ref` the stored
//! torchaudio emissions are used (algorithm parity); with `--model PTH` the
//! Rust wav2vec2 computes them (end-to-end alignment agreement).
//! usage: whisperx_align_parity STAGES_DIR NAME [--model PTH]
use rlx_wav2vec2_asr::whisperx::{AsrSegment, align};
use serde_json::Value;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (dir, name) = (&a[1], &a[2]);
    let model_path = a
        .iter()
        .position(|x| x == "--model")
        .map(|i| a[i + 1].clone());
    let st: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{name}.json"))?)?;
    let audio: Vec<f32> = std::fs::read(format!("{dir}/{name}.audio.f32"))?
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let shapes: Vec<Vec<usize>> =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{name}.em.json"))?)?;
    let raw: Vec<f32> = std::fs::read(format!("{dir}/{name}.em.f32"))?
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut refs = Vec::new();
    let mut at = 0;
    for s in &shapes {
        let n = s[0] * s[1];
        refs.push((raw[at..at + n].to_vec(), s[0]));
        at += n;
    }
    let segs: Vec<AsrSegment> = st["asr"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| AsrSegment {
            start: s["start"].as_f64().unwrap(),
            end: s["end"].as_f64().unwrap(),
            text: s["text"].as_str().unwrap().to_string(),
            avg_logprob: s["avg_logprob"].as_f64(),
        })
        .collect();
    let labels = rlx_wav2vec2_asr::model::LABELS;
    let mut model = match &model_path {
        Some(p) => Some(rlx_wav2vec2_asr::model::Wav2Vec2Ctc::open(
            std::path::Path::new(p),
            rlx_runtime::Device::Cpu,
        )?),
        None => None,
    };
    let mut k = 0;
    let mut worst = 0f32;
    let got = align(&segs, &audio, &labels, |pcm| {
        let r = refs[k].clone();
        k += 1;
        match model.as_mut() {
            None => Ok(r),
            Some(m) => {
                let (e, f) = m.emissions(pcm)?;
                if f == r.1 {
                    for (x, y) in e.iter().zip(&r.0) {
                        worst = worst.max((x - y).abs());
                    }
                } else {
                    eprintln!("frames {f} vs ref {}", r.1);
                }
                Ok((e, f))
            }
        }
    })?;
    let want = &st["aligned"]["segments"];
    let got_v = serde_json::to_value(&got.segments)?;
    let (mut words, mut exact, mut dmax, mut dsum) = (0, 0, 0f64, 0f64);
    let wget = |v: &Value| -> Vec<Value> {
        v.as_array()
            .unwrap()
            .iter()
            .flat_map(|s| s["words"].as_array().unwrap().clone())
            .collect()
    };
    let (gw, ww) = (wget(&got_v), wget(want));
    for (g, w) in gw.iter().zip(&ww) {
        words += 1;
        if g == w {
            exact += 1;
        }
        for key in ["start", "end"] {
            if let (Some(x), Some(y)) = (g[key].as_f64(), w[key].as_f64()) {
                dmax = dmax.max((x - y).abs());
                dsum += (x - y).abs();
            }
        }
    }
    let texts_equal = gw.iter().map(|w| w["word"].clone()).collect::<Vec<_>>()
        == ww.iter().map(|w| w["word"].clone()).collect::<Vec<_>>();
    let segs_equal = got_v.as_array().unwrap().len() == want.as_array().unwrap().len()
        && got_v
            .as_array()
            .unwrap()
            .iter()
            .zip(want.as_array().unwrap())
            .all(|(g, w)| {
                g["start"] == w["start"] && g["end"] == w["end"] && g["text"] == w["text"]
            });
    println!(
        "{name}: words {}/{} identical, words-text-equal {texts_equal}, segments-equal {segs_equal}, max |dt| {dmax:.3}s, mean |dt| {:.4}s{}",
        exact,
        ww.len().max(words),
        dsum / (2.0 * words.max(1) as f64),
        if model_path.is_some() {
            format!(", emission max |d| {worst:.2e}")
        } else {
            String::new()
        }
    );
    Ok(())
}
