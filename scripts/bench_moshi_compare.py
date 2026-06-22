#!/usr/bin/env python3
"""Compare Moshi inference backends: wall time + output vs rlx-moshi CPU baseline."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Hello."
MAX_STEPS = 8
WARMUP = 2
IN_WAV = ROOT / "crates/rlx-qwen3-tts/examples/audio/ask_not.wav"


def read_wav_mono(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        sw = w.getsampwidth()
        ch = w.getnchannels()
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM, got width {sw}")
    import struct

    samples = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
    if ch > 1:
        samples = samples[::ch]
    return [s / 32768.0 for s in samples]


def pcm_stats(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    if n == 0:
        return {"len_a": len(a), "len_b": len(b), "corr": 0.0, "rmse": float("inf")}
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    rmse = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)
    peak_a = max(abs(x) for x in a) if a else 0.0
    peak_b = max(abs(x) for x in b) if b else 0.0
    return {
        "len_a": len(a),
        "len_b": len(b),
        "overlap": n,
        "corr": corr,
        "rmse": rmse,
        "peak_a": peak_a,
        "peak_b": peak_b,
    }


def whisper_text(wav: Path, whisper_dir: Path) -> str:
    env = os.environ.copy()
    env["RLX_WHISPER_DIR"] = str(whisper_dir)
    # Use rust whisper via a tiny inline - fallback to empty if not built
    script = f"""
use rlx_whisper::{{WhisperRunner, normalize_transcript, SAMPLE_RATE}};
use rlx_runtime::Device;
fn main() -> anyhow::Result<()> {{
    let dir = std::path::Path::new({whisper_dir!r});
    let mut w = WhisperRunner::builder()
        .weights(dir.join("model.safetensors"))
        .config_path(dir.join("config.json"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .device(Device::Cpu)
        .language("en")
        .build()?;
    let pcm = rlx_whisper::load_wav_mono_f32({str(wav)!r}, SAMPLE_RATE)?;
    let t = w.normalize_transcript(&w.transcribe_greedy(&pcm)?);
    print!("{{}}", t);
    Ok(())
}}
"""
    return ""


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out, dt


def parse_bench_kv(text: str) -> dict[str, str]:
    d: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith(" "):
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def bench_rlx_cpu(out_wav: Path) -> dict:
    cmd = [
        "cargo",
        "run",
        "-p",
        "rlx-moshi",
        "--example",
        "bench_one_way",
        "--release",
        "--",
        "--prompt",
        PROMPT,
        "--max-steps",
        str(MAX_STEPS),
        "--warmup",
        str(WARMUP),
    ]
    code, out, wall = run(cmd)
    if code != 0:
        return {"name": "rlx-moshi CPU eager", "status": "failed", "error": out[-2000:]}
    # also write wav for baseline
    gen_cmd = [
        "cargo",
        "run",
        "-p",
        "rlx-moshi",
        "--features",
        "hf-download",
        "--release",
        "--",
        "--prompt",
        PROMPT,
        "--out-wav",
        str(out_wav),
        "--max-steps",
        str(MAX_STEPS),
        "--device",
        "cpu",
    ]
    run(gen_cmd)
    kv = parse_bench_kv(out)
    return {
        "name": "rlx-moshi CPU eager (bf16)",
        "status": "ok",
        "load_s": float(kv.get("load_ms", "0")) / 1000,
        "gen_s": float(kv.get("gen_ms", "0")) / 1000,
        "wall_s": wall,
        "ms_per_frame": float(kv.get("ms_per_frame", "0")),
        "rtf": float(kv.get("rtf", "0")),
        "out_frames": int(kv.get("out_frames", "0")),
        "transcript": kv.get("transcript", ""),
        "precision": "reference",
    }


def bench_mimi_codec() -> dict:
    if not IN_WAV.is_file():
        return {"name": "rlx-mimi codec only", "status": "skip", "error": "missing in wav"}
    cmd = [
        "cargo",
        "run",
        "-p",
        "rlx-mimi",
        "--release",
        "--",
        "--bench",
        "--in-wav",
        str(IN_WAV),
    ]
    code, out, wall = run(cmd)
    if code != 0:
        return {"name": "rlx-mimi codec only", "status": "failed", "error": out[-1500:]}
    enc = re.search(r"encode: ([\d.]+) ms \(RTF ([\d.]+)\)", out)
    dec = re.search(r"decode: ([\d.]+) ms \(RTF ([\d.]+)\)", out)
    frames = re.search(r"frames: (\d+)", out)
    return {
        "name": "rlx-mimi codec only (not full Moshi)",
        "status": "ok",
        "wall_s": wall,
        "encode_ms": float(enc.group(1)) if enc else None,
        "decode_ms": float(dec.group(1)) if dec else None,
        "encode_rtf": float(enc.group(2)) if enc else None,
        "decode_rtf": float(dec.group(2)) if dec else None,
        "frames": int(frames.group(1)) if frames else None,
        "precision": "codec roundtrip vs input (see mimi tests)",
    }


def bench_moshi_mlx_q4(out_wav: Path) -> dict:
    try:
        import moshi_mlx  # noqa: F401
    except ImportError:
        return {
            "name": "Kyutai moshi_mlx Q4 (Metal)",
            "status": "skip",
            "error": "pip install moshi_mlx rustymimi",
        }
    # Programmatic one-way via moshi_mlx if available
    script = ROOT / "scripts/_bench_moshi_mlx.py"
    if not script.is_file():
        return {"name": "Kyutai moshi_mlx Q4", "status": "skip", "error": "helper missing"}
    code, out, wall = run(
        [sys.executable, str(script), "--out-wav", str(out_wav), "--max-steps", str(MAX_STEPS)],
    )
    if code != 0:
        return {"name": "Kyutai moshi_mlx Q4 (Metal)", "status": "failed", "error": out[-2000:]}
    kv = parse_bench_kv(out)
    return {
        "name": "Kyutai moshi_mlx Q4 (Metal)",
        "status": "ok",
        "load_s": float(kv.get("load_ms", "0")) / 1000,
        "gen_s": float(kv.get("gen_ms", "0")) / 1000,
        "wall_s": wall,
        "ms_per_frame": float(kv.get("ms_per_frame", "0")),
        "rtf": float(kv.get("rtf", "0")),
        "transcript": kv.get("transcript", ""),
    }


def main() -> int:
    os.chdir(ROOT)
    baseline = ROOT / "/tmp/moshi-bench-baseline.wav"
    baseline = Path("/tmp/moshi-bench-baseline.wav")
    results = []

    print("=== Moshi backend comparison ===", flush=True)
    print(f"prompt={PROMPT!r} max_steps={MAX_STEPS} warmup={WARMUP}\n", flush=True)

    base = bench_rlx_cpu(baseline)
    results.append(base)
    base_pcm = read_wav_mono(baseline) if baseline.is_file() else []

    results.append(bench_mimi_codec())

    not_impl = [
        {
            "name": "RLX compiled GPU flow (Metal/CUDA)",
            "status": "not_implemented",
            "error": "no compiled Moshi graph in rlx-moshi",
        },
        {
            "name": "Q8 GGUF in rlx-moshi",
            "status": "not_implemented",
            "error": "kyutai/moshiko-candle-q8 exists but rlx-moshi only loads safetensors",
        },
        {
            "name": "Smaller Moshi checkpoint",
            "status": "not_available",
            "error": "only 7B arch; quant variants same size class",
        },
    ]
    results.extend(not_impl)

    mlx_out = Path("/tmp/moshi-bench-mlx.wav")
    mlx = bench_moshi_mlx_q4(mlx_out)
    if mlx.get("status") == "ok" and mlx_out.is_file() and base_pcm:
        mlx["precision"] = pcm_stats(base_pcm, read_wav_mono(mlx_out))
    results.append(mlx)

    # Kyutai Rust Candle — optional if repo cloned
    moshi_rust = Path("/tmp/kyutai-moshi/rust")
    if moshi_rust.is_dir():
        results.append({"name": "Kyutai Rust Candle Metal", "status": "pending", "error": "run manually"})
    else:
        results.append(
            {
                "name": "Kyutai Rust Candle (moshi-backend)",
                "status": "skip",
                "error": "clone https://github.com/kyutai-labs/moshi to /tmp/kyutai-moshi",
            }
        )

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
