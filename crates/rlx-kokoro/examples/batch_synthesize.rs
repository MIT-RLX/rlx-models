// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
//! Many utterances in one process (the model loads once): a job list on stdin,
//! one tab-separated line per WAV:
//!
//! ```text
//! out.wav <TAB> voice <TAB> speed <TAB> text
//! ```
//!
//! `voice` is a name (`af_heart`) or a blend of style vectors
//! (`af_heart*0.7+bm_george*0.3`; weights default to 1 and are normalised), which
//! gives new synthetic speakers, as for keyword-spotting datasets. A `text`
//! between slashes (`/hˈeɪ pˈiːʧi/`) is IPA and skips G2P. The espeak language
//! comes from the first voice. G2P output gets Kokoro's affricate ligatures
//! (`tʃ` → `ʧ`, `dʒ` → `ʤ`, as misaki's espeak fallback does) and then each
//! `--sub FROM=TO` (repeatable), for words espeak gets wrong (`--sub pˈiːkɪ=pˈiːʧi`
//! for "Peachy"). `--ipa` prints each line's phonemes as well.
//!
//! ```bash
//! cargo run --release -p rlx-kokoro --features metal --example batch_synthesize -- \
//!   --model weights/tts/kokoro-82m --device metal < jobs.tsv
//! ```

use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_kokoro::{Device, NativeKokoro, STYLE_DIM, VoiceBank, parse_device, write_wav};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let opt = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned());
    let model = PathBuf::from(opt("--model").unwrap_or_else(|| "weights/tts/kokoro-82m".into()));
    let device = opt("--device").map(|d| parse_device(&d)).transpose()?.unwrap_or(Device::Cpu);
    let show_ipa = args.iter().any(|a| a == "--ipa");
    let mut subs = vec![("tʃ".to_string(), "ʧ".to_string()), ("dʒ".to_string(), "ʤ".to_string())];
    for (i, a) in args.iter().enumerate() {
        if a == "--sub" {
            let s = args.get(i + 1).context("--sub needs FROM=TO")?;
            let (from, to) = s.split_once('=').context("--sub needs FROM=TO")?;
            subs.push((from.to_string(), to.to_string()));
        }
    }

    let tts = NativeKokoro::load(&model, device).context("load native Kokoro")?;
    let voices = VoiceBank::load_dir(&model.join("voices"))?;
    let (mut done, mut failed, mut audio_s) = (0usize, 0usize, 0f64);
    let t = std::time::Instant::now();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        match synth(&tts, &voices, &line, &subs, show_ipa) {
            Ok(n) => {
                done += 1;
                audio_s += n as f64 / f64::from(tts.sample_rate());
            }
            Err(e) => {
                failed += 1;
                eprintln!("[batch] {line}: {e:#}");
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    eprintln!("[batch] {done} written, {failed} failed: {audio_s:.1} s of audio in {secs:.1} s ({:.1}x RT)", audio_s / secs.max(1e-6));
    Ok(())
}

fn synth(tts: &NativeKokoro, voices: &VoiceBank, line: &str, subs: &[(String, String)], show_ipa: bool) -> Result<usize> {
    let mut f = line.splitn(4, '\t');
    let (Some(out), Some(voice), Some(speed), Some(text)) = (f.next(), f.next(), f.next(), f.next()) else {
        bail!("expected out<TAB>voice<TAB>speed<TAB>text");
    };
    let speed: f32 = speed.parse().context("speed")?;
    let blend = parse_blend(voice)?;
    let ipa = match text.strip_prefix('/').and_then(|t| t.strip_suffix('/')) {
        Some(ipa) => ipa.to_string(),
        None => {
            let lang = rlx_kokoro::config::voice_lang(&blend[0].0);
            let processed = rlx_kittentts::preprocess::TextPreprocessor::new().process(text);
            let mut ipa = rlx_kittentts::phonemize::phonemize_lang(lang, &processed).context("espeak phonemize")?;
            for (from, to) in subs {
                ipa = ipa.replace(from.as_str(), to);
            }
            ipa
        }
    };
    if show_ipa {
        eprintln!("[ipa] {text} -> {ipa}");
    }
    let content_len = tts.vocab().content_len(&ipa);
    anyhow::ensure!(content_len > 0, "no in-vocabulary phonemes in {ipa:?}");
    let mut style = vec![0f32; STYLE_DIM];
    let total: f32 = blend.iter().map(|(_, w)| w).sum();
    for (name, w) in &blend {
        let v = voices.get(name).with_context(|| format!("unknown voice {name}"))?;
        for (s, x) in style.iter_mut().zip(v.style_row(content_len)) {
            *s += x * w / total;
        }
    }
    let audio = tts.infer(&tts.vocab().to_input_ids(&ipa), &style, speed)?;
    write_wav(&audio, Path::new(out))?;
    Ok(audio.len())
}

/// `a*0.7+b*0.3` → `[(a, 0.7), (b, 0.3)]`.
fn parse_blend(spec: &str) -> Result<Vec<(String, f32)>> {
    let parts = spec
        .split('+')
        .map(|p| match p.split_once('*') {
            Some((n, w)) => Ok((n.trim().to_string(), w.trim().parse::<f32>().context("voice weight")?)),
            None => Ok((p.trim().to_string(), 1.0)),
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(!parts.is_empty() && parts.iter().all(|(_, w)| *w > 0.0), "bad voice blend {spec:?}");
    Ok(parts)
}
