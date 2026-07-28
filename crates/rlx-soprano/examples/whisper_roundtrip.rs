//! Text-in → Soprano → Whisper text-out validation.
//!
//! Modes:
//! - Full native `synthesize` per device (default)
//! - `RLX_ORT_LATENTS=1`: ORT backbone latents → native decoder (isolates Vocos)
//!
//! ```bash
//! cargo run -p rlx-soprano --release --example whisper_roundtrip --features apple-silicon
//! RLX_ORT_LATENTS=1 cargo run -p rlx-soprano --release --example whisper_roundtrip --features apple-silicon
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use rlx_runtime::{Device, is_available};
use rlx_soprano::{DEFAULT_LOCAL_DIR, HIDDEN, InferOpts, NativeSoprano, peak_amplitude};

/// Intelligibility bar (pangram).
const DEFAULT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
/// Brand / name check — short “Hello from Soprano.” is often heard as
/// “Suprano” by Whisper-tiny; grounding the name in a longer phrase helps.
const BRAND_TEXT: &str = "Hello from the Soprano model.";

fn soprano_bundle_present(dir: &std::path::Path) -> bool {
    dir.join("soprano.rlxp").is_file()
        || dir.join("soprano.gguf").is_file()
        || dir.join("graphs/soprano_backbone_kv_fp32.rlxp").is_file()
        || dir.join("onnx/soprano_backbone_kv_fp32.onnx").is_file()
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("RLX_SOPRANO_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_LOCAL_DIR);
            soprano_bundle_present(&p).then_some(p)
        })
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/tts/soprano")
        });
    let text = std::env::var("RLX_TEXT").unwrap_or_else(|_| {
        if std::env::var_os("RLX_BRAND").is_some() {
            BRAND_TEXT.to_string()
        } else {
            DEFAULT_TEXT.to_string()
        }
    });
    let max_tokens: usize = std::env::var("RLX_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);
    let ort_latents = std::env::var_os("RLX_ORT_LATENTS").is_some();

    anyhow::ensure!(
        soprano_bundle_present(&dir),
        "missing soprano.rlxp / nested graphs / legacy onnx under {}",
        dir.display()
    );

    let mut whisper = load_whisper().ok_or_else(|| {
        anyhow::anyhow!("Whisper weights not found (RLX_WHISPER_DIR / .cache/whisper-*)")
    })?;

    let latents = if ort_latents {
        Some(ort_latents_vec(&dir, &text)?)
    } else {
        None
    };

    let devices = parse_devices();
    let mode = if ort_latents {
        "ORT latents → native decode"
    } else {
        "full native synthesize"
    };
    println!("== Soprano Whisper roundtrip ({mode}) ==");
    println!("text in: {text:?}");
    println!(
        "{:<8} {:>7} {:>8} {:>7}  whisper-out",
        "backend", "cov", "peak", "ms"
    );

    let opts = InferOpts {
        max_new_tokens: max_tokens,
        greedy: std::env::var_os("RLX_GREEDY").is_some()
            || std::env::var_os("RLX_ORT_LATENTS").is_none(),
        temperature: 0.3,
        top_p: 0.95,
        seed: 1337,
        ..Default::default()
    };

    for (dev, label) in devices {
        if dev != Device::Cpu && !is_available(dev) {
            println!("{label:<8}   n/a");
            continue;
        }
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let model = NativeSoprano::open(&dir, dev)?;
            let pcm = if let Some(ref lat) = latents {
                model.decode_latents(lat, true)?
            } else {
                model.synthesize(&text, &opts)?
            };
            let sr = model.sample_rate();
            Ok::<_, anyhow::Error>((pcm, sr))
        }));
        let (pcm, sr) = match result {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                println!("{label:<8}  err: {}", short(&e.to_string()));
                continue;
            }
            Err(_) => {
                println!("{label:<8}  panic");
                continue;
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let peak = peak_amplitude(&pcm);
        let (cov, out) = coverage(&mut whisper, &pcm, sr, &text);
        println!("{label:<8} {cov:>6.0}% {peak:>8.3} {ms:>7.0}  {out:?}");
        let _ = NativeSoprano::write_wav(&pcm, format!("/tmp/soprano_whisper_{label}.wav"), sr);
    }
    Ok(())
}

fn parse_devices() -> Vec<(Device, &'static str)> {
    let all = vec![
        (Device::Cpu, "CPU"),
        (Device::Metal, "Metal"),
        (Device::Mlx, "MLX"),
        (Device::Gpu, "wgpu"),
        (Device::Ane, "CoreML"),
    ];
    if let Ok(s) = std::env::var("RLX_DEVICES") {
        let want: Vec<&str> = s
            .split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .collect();
        return all
            .into_iter()
            .filter(|(_, l)| want.iter().any(|w| w.eq_ignore_ascii_case(l)))
            .collect();
    }
    all.into_iter()
        .filter(|(d, _)| *d == Device::Cpu || is_available(*d))
        .collect()
}

fn load_whisper() -> Option<rlx_whisper::WhisperRunner> {
    let d = std::env::var("RLX_WHISPER_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache");
            ["whisper-base.en", "whisper-tiny.en", "whisper-tiny"]
                .into_iter()
                .map(|n| c.join(n))
                .find(|p| p.join("model.safetensors").is_file())
        })?;
    rlx_whisper::WhisperRunner::builder()
        .weights(d.join("model.safetensors"))
        .config_path(d.join("config.json"))
        .tokenizer_path(d.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()
        .ok()
}

fn coverage(
    w: &mut rlx_whisper::WhisperRunner,
    pcm: &[f32],
    sr: u32,
    expect: &str,
) -> (f64, String) {
    let pcm16 = resample(pcm, sr, rlx_whisper::SAMPLE_RATE as u32);
    let Ok(t) = w.transcribe_greedy(&pcm16) else {
        return (0.0, String::new());
    };
    let want: Vec<_> = expect
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|x| x.len() >= 2)
        .map(str::to_string)
        .collect();
    let got = t.to_lowercase();
    let got_toks: Vec<&str> = got
        .split(|c: char| !c.is_alphanumeric())
        .filter(|x| x.len() >= 2)
        .collect();
    let hit = want
        .iter()
        .filter(|w| word_near_hit(w, &got, &got_toks))
        .count();
    (
        100.0 * hit as f64 / want.len().max(1) as f64,
        t.trim().to_string(),
    )
}

/// Exact substring match, or a transcript token within a small edit distance
/// (covers Whisper-tiny proper-noun slips like `Suprano` ≈ `Soprano`).
fn word_near_hit(want: &str, got: &str, got_toks: &[&str]) -> bool {
    if got.contains(want) {
        return true;
    }
    let max_ed = if want.len() >= 7 {
        2
    } else if want.len() >= 5 {
        1
    } else {
        0
    };
    if max_ed == 0 {
        return false;
    }
    got_toks.iter().any(|tok| {
        (tok.len() as isize - want.len() as isize).unsigned_abs() <= max_ed
            && edit_distance(want, tok) <= max_ed
    })
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (n, m) = (a.len(), b.len());
    let mut prev = (0..=m).collect::<Vec<_>>();
    let mut cur = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
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

fn short(s: &str) -> String {
    s.chars().take(96).collect()
}

fn ort_latents_vec(dir: &std::path::Path, text: &str) -> anyhow::Result<Vec<Vec<f32>>> {
    let npy = std::env::temp_dir().join("soprano_whisper_ort_hs.npy");
    let py = r#"
import numpy as np, onnxruntime as ort, sys
from tokenizers import Tokenizer
dir, text, npy = sys.argv[1], sys.argv[2], sys.argv[3]
tok=Tokenizer.from_file(f"{dir}/tokenizer.json")
ids=list(tok.encode(f"[STOP][TEXT]{text}[START]").ids)
bb=ort.InferenceSession(f"{dir}/onnx/soprano_backbone_kv_fp32.onnx", providers=["CPUExecutionProvider"])
past={i: (np.zeros((1,1,0,128),np.float32), np.zeros((1,1,0,128),np.float32)) for i in range(17)}
feeds_past=lambda: {**{f"past_key_values.{i}.key":past[i][0] for i in range(17)}, **{f"past_key_values.{i}.value":past[i][1] for i in range(17)}}
cur=np.array([ids],np.int64); attn=np.ones((1,len(ids)),np.int64); pos=np.arange(len(ids))[None]
latents=[]; EOS=3
for step in range(96):
  outs=bb.run(None, {"input_ids":cur,"attention_mask":attn,"position_ids":pos, **feeds_past()})
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
np.save(npy, hs); print("ok", T)
"#;
    let st = Command::new("python3")
        .args([
            "-c",
            py,
            dir.to_str().unwrap_or(""),
            text,
            npy.to_str().unwrap_or(""),
        ])
        .status()?;
    anyhow::ensure!(st.success() && npy.is_file(), "ORT latent export failed");
    let b = std::fs::read(&npy)?;
    let major = b[6];
    let (hlen, start) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let hs: Vec<f32> = b[start + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let t = hs.len() / HIDDEN;
    let mut latents = Vec::with_capacity(t);
    for ti in 0..t {
        let mut v = vec![0f32; HIDDEN];
        for c in 0..HIDDEN {
            v[c] = hs[c * t + ti];
        }
        latents.push(v);
    }
    anyhow::ensure!(t > 4, "too few ORT latents ({t})");
    Ok(latents)
}
