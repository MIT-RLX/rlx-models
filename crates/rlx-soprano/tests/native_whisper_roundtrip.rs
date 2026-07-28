// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0

//! Fox sentence intelligibility: ORT backbone AR latents → native RLX Vocos
//! decoder → Whisper-tiny with **100%** content-word coverage.
//!
//! Also runs full `NativeSoprano::synthesize` (non-silent). Backbone attention
//! vs ORT is still WIP — see README.
//!
//! Needs `weights/tts/soprano`, Whisper cache, and
//! `python3` + `onnxruntime` + `tokenizers` + `numpy`.

use std::path::PathBuf;
use std::process::Command;

use rlx_runtime::Device;
use rlx_soprano::{DEFAULT_LOCAL_DIR, HIDDEN, InferOpts, NativeSoprano, peak_amplitude};
use rlx_whisper::{SAMPLE_RATE as WHISPER_RATE, WhisperRunner};

const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn env_dir(var: &str, default: &str) -> Option<PathBuf> {
    let candidates = [
        std::env::var(var).ok().map(PathBuf::from),
        Some(PathBuf::from(default)),
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(default),
        ),
    ];
    candidates.into_iter().flatten().find(|d| {
        d.join("soprano.rlxp").is_file()
            || d.join("soprano.gguf").is_file()
            || d.join("graphs/soprano_backbone_kv_fp32.rlxp").is_file()
            || d.join("onnx/soprano_backbone_kv_fp32.onnx").is_file()
    })
}

fn whisper_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RLX_WHISPER_DIR") {
        let p = PathBuf::from(d);
        if p.join("config.json").exists() {
            return Some(p);
        }
    }
    let roots = [
        PathBuf::from(".cache"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache"),
    ];
    for root in roots {
        for name in [
            "whisper-base.en",
            "whisper-small.en",
            "whisper-tiny.en",
            "whisper-tiny",
        ] {
            let p = root.join(name);
            if p.join("config.json").exists() {
                return Some(p);
            }
        }
    }
    None
}

fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64 / from as u64).max(1) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * from as f64 / to as f64;
            let idx = s.floor() as usize;
            let f = (s - idx as f64) as f32;
            let a = x[idx.min(x.len() - 1)];
            let b = x[(idx + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(str::to_string)
        .collect()
}

/// ORT backbone AR → latents `[T][512]` written as npy `[1,512,T]`.
fn ort_latents_npy(dir: &std::path::Path, text: &str, npy: &std::path::Path) -> bool {
    let py = r#"
import numpy as np, onnxruntime as ort, sys
from tokenizers import Tokenizer
dir, text, npy = sys.argv[1], sys.argv[2], sys.argv[3]
tok=Tokenizer.from_file(f'{dir}/tokenizer.json')
ids=list(tok.encode(f'[STOP][TEXT]{text}[START]').ids)
bb=ort.InferenceSession(f'{dir}/onnx/soprano_backbone_kv_fp32.onnx', providers=['CPUExecutionProvider'])
past={i: (np.zeros((1,1,0,128),np.float32), np.zeros((1,1,0,128),np.float32)) for i in range(17)}
feeds_past=lambda: {**{f'past_key_values.{i}.key':past[i][0] for i in range(17)}, **{f'past_key_values.{i}.value':past[i][1] for i in range(17)}}
cur=np.array([ids],np.int64); attn=np.ones((1,len(ids)),np.int64); pos=np.arange(len(ids))[None]
latents=[]; EOS=3
for step in range(96):
  outs=bb.run(None, {'input_ids':cur,'attention_mask':attn,'position_ids':pos, **feeds_past()})
  for i in range(17):
    past[i]=(outs[1+2*i], outs[2+2*i])
  next_id=int(np.argmax(outs[0][0,-1])); finished=next_id==EOS
  h=outs[-1][0,-1].astype(np.float32)
  if step>0 and not finished: latents.append(h.copy())
  total=past[0][0].shape[2]+1
  cur=np.array([[next_id]],np.int64); attn=np.ones((1,total),np.int64); pos=np.array([[total-1]],np.int64)
  if finished: break
T=len(latents); hs=np.zeros((1,512,T),np.float32)
for w,h in enumerate(latents): hs[0,:,w]=h
np.save(npy, hs)
print('ok', T)
"#;
    Command::new("python3")
        .args([
            "-c",
            py,
            dir.to_str().unwrap_or(""),
            text,
            npy.to_str().unwrap_or(""),
        ])
        .status()
        .map(|s| s.success() && npy.is_file())
        .unwrap_or(false)
}

fn load_npy_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    if &b[0..6] != b"\x93NUMPY" {
        return None;
    }
    let major = b[6];
    let (hlen, start) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10usize)
    } else {
        (
            u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize,
            12usize,
        )
    };
    Some(
        b[start + hlen..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[test]
fn soprano_sentence_whisper_roundtrip() {
    let Some(dir) = env_dir("RLX_SOPRANO_DIR", DEFAULT_LOCAL_DIR) else {
        eprintln!("skip: set RLX_SOPRANO_DIR / place weights at {DEFAULT_LOCAL_DIR}");
        return;
    };
    let Some(whisper) = whisper_dir() else {
        eprintln!("skip: set RLX_WHISPER_DIR");
        return;
    };
    let npy = std::env::temp_dir().join("soprano_ort_fox_hs.npy");
    if !ort_latents_npy(&dir, TEXT, &npy) {
        eprintln!("skip: python3 onnxruntime tokenizers numpy required");
        return;
    }
    let hs = load_npy_f32(&npy).expect("read latents");
    let t = hs.len() / HIDDEN;
    assert!(t > 4, "too few latents ({t})");
    let mut latents = Vec::with_capacity(t);
    for ti in 0..t {
        let mut v = vec![0f32; HIDDEN];
        for c in 0..HIDDEN {
            v[c] = hs[c * t + ti];
        }
        latents.push(v);
    }

    let model = NativeSoprano::open(&dir, Device::Cpu).expect("open");
    let pcm = model.decode_latents(&latents, true).expect("native decode");
    let peak = peak_amplitude(&pcm);
    assert!(peak > 0.05, "native decoder not audible (peak {peak:.4})");

    let pcm16k = resample(&pcm, model.sample_rate(), WHISPER_RATE as u32);
    let mut w = WhisperRunner::builder()
        .weights(whisper.join("model.safetensors"))
        .config_path(whisper.join("config.json"))
        .tokenizer_path(whisper.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .expect("whisper");
    let transcript = w.transcribe_greedy(&pcm16k).expect("transcribe");
    eprintln!("transcript: {transcript:?}");
    let want = words(TEXT);
    let got = words(&transcript);
    let missing: Vec<_> = want
        .iter()
        .filter(|w| !got.iter().any(|g| g == *w))
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "content-word coverage < 100%: missing {missing:?}; transcript={transcript:?}"
    );

    let pcm_hi = model
        .synthesize(
            "Hi.",
            &InferOpts {
                max_new_tokens: 32,
                greedy: true,
                ..Default::default()
            },
        )
        .expect("native synth");
    assert!(peak_amplitude(&pcm_hi) > 0.01, "native full synth silent");
}
