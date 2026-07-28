//! Partition vocoder vs prosody: inject ORT F0/N into the native generator.
//!
//! If ORT-F0+N → native vocoder is intelligible (Whisper hears "hello") while
//! free-run native is not, the vocoder is fine and the bug is in F0/N predict.
//!
//! ```bash
//! cargo run -p kitten_tts_mini_rlx --example ort_prosody_inject --release --features native
//! ```
//!
//! Writes `/tmp/kitten_native.wav`, `/tmp/kitten_ort_f0n.wav`, `/tmp/kitten_ort_ref.wav`.

use std::process::Command;
use std::time::Instant;

use kitten_tts_mini_rlx::GraphOptions;
use kitten_tts_mini_rlx::bundle_compile::{
    compile_from_bundle, compile_from_bundle_with_ort_concat, compile_from_bundle_with_ort_f0n,
    compile_from_bundle_with_ort_matmul1, compile_from_bundle_with_ort_matmul1_opts,
    run_parity_inputs_with_duration,
};
use rlx_runtime::Device;

const HELLO_IDS: [i64; 9] = [0, 50, 83, 156, 54, 57, 135, 10, 0];
const HELLO_DUR: [i64; 9] = [3, 2, 2, 3, 4, 4, 13, 2, 1];
const SAMPLE_RATE: u32 = 24_000;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn load_style_row() -> Vec<f32> {
    let script = r#"
import numpy as np, sys
z = np.load(sys.argv[1])
v = z['expr-voice-2-m'][int(sys.argv[2])]
sys.stdout.buffer.write(v.astype(np.float32).tobytes())
"#;
    let voices = repo_root().join(".cache/kittentts-mini-0.8/voices.npz");
    let out = Command::new("python3")
        .args(["-c", script, voices.to_str().unwrap(), "6"])
        .output()
        .expect("python style");
    out.stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct OrtProsody {
    f0: Vec<f32>,
    n: Vec<f32>,
    concat: Vec<f32>,
    concat_c: usize,
    concat_t: usize,
    matmul1: Vec<f32>,
    matmul1_c: usize,
    matmul1_t: usize,
    wave: Vec<f32>,
}

fn fetch_ort_prosody() -> anyhow::Result<OrtProsody> {
    let root = repo_root();
    let model = root.join(".cache/kittentts-mini-0.8/kitten_tts_mini_v0_8.onnx");
    let voices = root.join(".cache/kittentts-mini-0.8/voices.npz");
    let script = format!(
        r#"
import onnx, numpy as np, onnxruntime as ort
model = onnx.load({model:?})
existing = {{o.name for o in model.graph.output}}
outs = [o.name for o in model.graph.output]
wave_name = 'waveform' if 'waveform' in outs else outs[0]
want = [
    '/F0_proj/Conv_output_0',
    '/N_proj/Conv_output_0',
    '/decoder/Concat_output_0',
    '/MatMul_1_output_0',
    wave_name,
]
for name in want:
    if name not in existing:
        vi = model.graph.output.add()
        vi.name = name
onnx.save(model, '/tmp/kitten_ort_f0n.onnx')
z = np.load({voices:?})
style = z['expr-voice-2-m'][6:7].astype(np.float32)
ids = np.array([[{ids}]], dtype=np.int64)
sess = ort.InferenceSession('/tmp/kitten_ort_f0n.onnx', providers=['CPUExecutionProvider'])
inames = {{i.name for i in sess.get_inputs()}}
feed = {{'input_ids': ids, 'style': style, 'speed': np.array([1.0], np.float32)}}
feed = {{k: v for k, v in feed.items() if k in inames}}
res = sess.run(want, feed)
def dump(arr, path):
    a = np.asarray(arr, dtype=np.float32)
    open(path, 'wb').write(a.reshape(-1).tobytes())
    print(path, a.shape, float(np.max(np.abs(a))), flush=True)
    return a.shape
dump(res[0], '/tmp/kitten_ort_f0.bin')
dump(res[1], '/tmp/kitten_ort_n.bin')
cshape = dump(res[2], '/tmp/kitten_ort_concat.bin')
mshape = dump(res[3], '/tmp/kitten_ort_matmul1.bin')
dump(res[4], '/tmp/kitten_ort_wave.bin')
print('CONCAT_SHAPE', cshape[1], cshape[2], flush=True)
print('MATMUL1_SHAPE', mshape[1], mshape[2], flush=True)
"#,
        model = model,
        voices = voices,
        ids = HELLO_IDS
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    );
    let out = Command::new("python3").args(["-c", &script]).output()?;
    if !out.status.success() {
        anyhow::bail!(
            "ORT prosody fetch failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    eprintln!("{}", stdout.trim());
    let mut concat_c = 514usize;
    let mut concat_t = 34usize;
    let mut matmul1_c = 512usize;
    let mut matmul1_t = 34usize;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("CONCAT_SHAPE ") {
            let mut parts = rest.split_whitespace();
            if let (Some(c), Some(t)) = (parts.next(), parts.next()) {
                concat_c = c.parse().unwrap_or(concat_c);
                concat_t = t.parse().unwrap_or(concat_t);
            }
        }
        if let Some(rest) = line.strip_prefix("MATMUL1_SHAPE ") {
            let mut parts = rest.split_whitespace();
            if let (Some(c), Some(t)) = (parts.next(), parts.next()) {
                matmul1_c = c.parse().unwrap_or(matmul1_c);
                matmul1_t = t.parse().unwrap_or(matmul1_t);
            }
        }
    }
    let read_f32 = |path: &str| -> anyhow::Result<Vec<f32>> {
        let b = std::fs::read(path)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    };
    Ok(OrtProsody {
        f0: read_f32("/tmp/kitten_ort_f0.bin")?,
        n: read_f32("/tmp/kitten_ort_n.bin")?,
        concat: read_f32("/tmp/kitten_ort_concat.bin")?,
        concat_c,
        concat_t,
        matmul1: read_f32("/tmp/kitten_ort_matmul1.bin")?,
        matmul1_c,
        matmul1_t,
        wave: read_f32("/tmp/kitten_ort_wave.bin")?,
    })
}

fn wave_stats(label: &str, w: &[f32], trim: usize) {
    let w = &w[..w.len().min(trim)];
    let peak = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut zc = 0usize;
    for p in w.windows(2) {
        if (p[0] >= 0.0) != (p[1] >= 0.0) {
            zc += 1;
        }
    }
    let zc_s = zc as f32 / (w.len().max(1) as f32 / SAMPLE_RATE as f32);
    eprintln!(
        "  {label}: samples={} peak={peak:.4} zc={zc} zc/s={zc_s:.0}",
        w.len()
    );
}

fn write_wav(path: &str, pcm: &[f32], trim: usize) -> anyhow::Result<()> {
    let pcm = &pcm[..pcm.len().min(trim)];
    let mut bytes = Vec::with_capacity(44 + pcm.len() * 2);
    let data_len = (pcm.len() * 2) as u32;
    let sr = SAMPLE_RATE;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sr.to_le_bytes());
    bytes.extend_from_slice(&(sr * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn whisper_transcribe(wav: &str) -> Option<String> {
    let script = r#"
import sys
try:
    import whisper
except ImportError:
    print('NO_WHISPER')
    sys.exit(0)
m = whisper.load_model('tiny')
r = m.transcribe(sys.argv[1], language='en', fp16=False)
print(r.get('text','').strip())
"#;
    let out = Command::new("python3")
        .args(["-c", script, wav])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() || text == "NO_WHISPER" {
        None
    } else {
        Some(text)
    }
}

fn decode_wave(outs: &[(Vec<u8>, rlx_ir::DType)]) -> Vec<f32> {
    let (bytes, _) = &outs[0];
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn parse_device() -> Device {
    match std::env::var("KITTEN_PROBE_DEVICE")
        .unwrap_or_else(|_| "cpu".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "metal" => Device::Metal,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn main() -> anyhow::Result<()> {
    let device = parse_device();
    let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/rlx_bundle");
    let style = load_style_row();
    let token_len = HELLO_IDS.len();
    let seq = kitten_tts_mini_rlx::compile_profile::compile_slot_length(token_len);
    let mel = HELLO_DUR.iter().sum::<i64>() as usize;
    let trim = mel * 600;
    let max_wave = 55_200usize;
    let opts = GraphOptions {
        sequence_length: seq.max(token_len),
        max_waveform_samples: max_wave,
    };
    eprintln!("seq={seq} token_len={token_len} mel={mel} trim={trim} max_wave={max_wave}");

    eprintln!("fetching ORT F0/N + reference wave…");
    let ort = fetch_ort_prosody()?;
    wave_stats("ORT ref", &ort.wave, trim);
    write_wav("/tmp/kitten_ort_ref.wav", &ort.wave, trim)?;

    // --- A) free-run native (duration seeded to ORT) ---
    eprintln!("compiling native free-run…");
    let t0 = Instant::now();
    let mut native = compile_from_bundle(device, &bundle, &opts)?;
    eprintln!("  compile {:.1}s", t0.elapsed().as_secs_f64());
    let outs = run_parity_inputs_with_duration(
        &mut native,
        opts.sequence_length,
        token_len,
        &HELLO_IDS,
        &style,
        Some(&HELLO_DUR),
    );
    let native_wave = decode_wave(&outs);
    wave_stats("native", &native_wave, trim);
    write_wav("/tmp/kitten_native.wav", &native_wave, trim)?;

    // --- B) ORT F0+N → native vocoder ---
    eprintln!("compiling native + ORT F0/N inject…");
    let t1 = Instant::now();
    let mut injected = compile_from_bundle_with_ort_f0n(device, &bundle, &opts, &ort.f0, &ort.n)?;
    eprintln!("  compile {:.1}s", t1.elapsed().as_secs_f64());
    let outs = run_parity_inputs_with_duration(
        &mut injected,
        opts.sequence_length,
        token_len,
        &HELLO_IDS,
        &style,
        Some(&HELLO_DUR),
    );
    let inj_wave = decode_wave(&outs);
    wave_stats("ort_f0n→native", &inj_wave, trim);
    write_wav("/tmp/kitten_ort_f0n.wav", &inj_wave, trim)?;

    // --- C) ORT MatMul_1 only (ASR) → native F0/N + vocoder ---
    eprintln!("compiling native + ORT /MatMul_1 inject…");
    let t_m = Instant::now();
    let mut mm1_g = compile_from_bundle_with_ort_matmul1(
        device,
        &bundle,
        &opts,
        &ort.matmul1,
        ort.matmul1_c,
        ort.matmul1_t,
    )?;
    eprintln!("  compile {:.1}s", t_m.elapsed().as_secs_f64());
    let outs = run_parity_inputs_with_duration(
        &mut mm1_g,
        opts.sequence_length,
        token_len,
        &HELLO_IDS,
        &style,
        Some(&HELLO_DUR),
    );
    let mm1_wave = decode_wave(&outs);
    wave_stats("ort_matmul1→native", &mm1_wave, trim);
    write_wav("/tmp/kitten_ort_matmul1.wav", &mm1_wave, trim)?;

    // --- C2) ORT MatMul_1 + F0/N (native F0_conv/N_conv from ORT proj) ---
    eprintln!("compiling native + ORT MatMul_1 + F0/N…");
    let t_mf = Instant::now();
    let mut mm1f_g = compile_from_bundle_with_ort_matmul1_opts(
        device,
        &bundle,
        &opts,
        &ort.matmul1,
        ort.matmul1_c,
        ort.matmul1_t,
        Some(&ort.f0),
        Some(&ort.n),
    )?;
    eprintln!("  compile {:.1}s", t_mf.elapsed().as_secs_f64());
    let outs = run_parity_inputs_with_duration(
        &mut mm1f_g,
        opts.sequence_length,
        token_len,
        &HELLO_IDS,
        &style,
        Some(&HELLO_DUR),
    );
    let mm1f_wave = decode_wave(&outs);
    wave_stats("ort_mm1+f0n→native", &mm1f_wave, trim);
    write_wav("/tmp/kitten_ort_mm1_f0n.wav", &mm1f_wave, trim)?;

    // --- D) ORT decoder Concat → native encode/vocoder ---
    eprintln!("compiling native + ORT /decoder/Concat inject…");
    let t2 = Instant::now();
    let mut concat_g = compile_from_bundle_with_ort_concat(
        device,
        &bundle,
        &opts,
        &ort.concat,
        ort.concat_c,
        ort.concat_t,
        &ort.f0,
        &ort.n,
    )?;
    eprintln!("  compile {:.1}s", t2.elapsed().as_secs_f64());
    let outs = run_parity_inputs_with_duration(
        &mut concat_g,
        opts.sequence_length,
        token_len,
        &HELLO_IDS,
        &style,
        Some(&HELLO_DUR),
    );
    let concat_wave = decode_wave(&outs);
    wave_stats("ort_concat→native", &concat_wave, trim);
    write_wav("/tmp/kitten_ort_concat.wav", &concat_wave, trim)?;

    // Max-abs vs ORT reference (aligned at 0)
    let n = trim
        .min(ort.wave.len())
        .min(inj_wave.len())
        .min(native_wave.len())
        .min(concat_wave.len())
        .min(mm1_wave.len())
        .min(mm1f_wave.len());
    let mut max_nat = 0.0f32;
    let mut max_inj = 0.0f32;
    let mut max_mm1 = 0.0f32;
    let mut max_mm1f = 0.0f32;
    let mut max_cat = 0.0f32;
    for i in 0..n {
        max_nat = max_nat.max((native_wave[i] - ort.wave[i]).abs());
        max_inj = max_inj.max((inj_wave[i] - ort.wave[i]).abs());
        max_mm1 = max_mm1.max((mm1_wave[i] - ort.wave[i]).abs());
        max_mm1f = max_mm1f.max((mm1f_wave[i] - ort.wave[i]).abs());
        max_cat = max_cat.max((concat_wave[i] - ort.wave[i]).abs());
    }
    eprintln!(
        "vs ORT wave (first {n}): native={max_nat:.4}  f0n={max_inj:.4}  matmul1={max_mm1:.4}  mm1+f0n={max_mm1f:.4}  concat={max_cat:.4}"
    );

    eprintln!("\n=== Whisper (tiny) ===");
    for (label, path) in [
        ("ORT ref", "/tmp/kitten_ort_ref.wav"),
        ("native", "/tmp/kitten_native.wav"),
        ("ort_f0n→native", "/tmp/kitten_ort_f0n.wav"),
        ("ort_matmul1→native", "/tmp/kitten_ort_matmul1.wav"),
        ("ort_mm1+f0n→native", "/tmp/kitten_ort_mm1_f0n.wav"),
        ("ort_concat→native", "/tmp/kitten_ort_concat.wav"),
    ] {
        match whisper_transcribe(path) {
            Some(t) => eprintln!("  {label}: {t:?}"),
            None => eprintln!("  {label}: (whisper unavailable — listen to {path})"),
        }
    }

    eprintln!("\npartition guide:");
    eprintln!("  ort_f0n improves           → bug in F0/N predictor");
    eprintln!("  ort_matmul1 improves       → bug in ASR/alignment (MatMul_1)");
    eprintln!("  ort_mm1+f0n improves       → need both ASR + F0/N; F0_conv OK");
    eprintln!("  ort_concat improves (only) → bug in F0_conv/N_conv into Concat");
    eprintln!("  neither improves           → bug in native encode/decode/vocoder");
    Ok(())
}
