#!/usr/bin/env python3
# RLX cross-backend model harness — driver.
#
# Runs ON whatever host invokes it (Mac or the remote Linux+CUDA box). For each model
# in registry.toml it: resolves the backends this host supports, builds the crate
# once, ensures weights, runs real inference on every backend, and validates that
# the output is correct (semantic check + cross-backend parity vs the CPU baseline).
#
# Platform logic lives in ONE place (`host_backends` / `resolve`). The CUDA host is just a
# host that happens to report {cpu, wgpu, cuda, vulkan}. Stdlib only (tomllib +
# subprocess + wave) — no numpy/torch needed on the host.
#
# Env knobs:  TIER=1|2|all  ONLY=<name>[,<name>]  BACKENDS=cpu,wgpu,cuda,vulkan
#             ALL=1 (include tier-2 / NC / gated)  REGISTRY=<path>  OUT=<dir>
#             KEEP_BUILD_LOG=1  BUILD_TIMEOUT=1800

import os, sys, json, re, shlex, shutil, signal, subprocess, time, wave, math, platform
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    sys.exit("run_matrix.py needs Python 3.11+ (tomllib). Got " + sys.version.split()[0])

# ---------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent               # scripts/matrix
REPO = HERE.parents[1]                                # repo root
REGISTRY = Path(os.environ.get("REGISTRY", HERE / "registry.toml"))
OUT = Path(os.environ.get("OUT", HERE / "out"))
ART = OUT / "artifacts"
TARGET_DIR = Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))

# ------------------------------------------------------------------- backend tables
DEV_FEATURE = {  # rlx device name -> cargo feature that compiles it in ("" = base cpu)
    "cpu": "", "metal": "metal", "mlx": "mlx", "wgpu": "gpu",
    "coreml": "coreml", "cuda": "cuda", "vulkan": "vulkan", "rocm": "rocm",
}
DEV_CLI = {  # rlx device name -> the string passed to `--device`
    "cpu": "cpu", "metal": "metal", "mlx": "mlx", "wgpu": "gpu",
    "coreml": "coreml", "cuda": "cuda", "vulkan": "vulkan", "rocm": "rocm",
}
KIND_TIMEOUT = {"tts": 600, "asr": 180, "lm": 240, "vision": 240, "codec": 180}
BUILD_TIMEOUT = int(os.environ.get("BUILD_TIMEOUT", "1800"))


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------- host / device
def _have(cmd):
    return shutil.which(cmd) is not None


def _cmd_ok(argv):
    try:
        return subprocess.run(argv, capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


def host_backends():
    """cpu + whatever GPU backends this machine actually has. cpu is always first."""
    sysname = platform.system()
    if sysname == "Darwin":
        return ["cpu", "metal", "mlx", "wgpu", "coreml"]
    # Linux (and anything else that looks like it)
    order = ["cpu"]
    nvidia = _have("nvidia-smi") and _cmd_ok(["nvidia-smi", "-L"])
    # ROCm only counts with a real AMD compute device — /dev/kfd. `rocminfo`/`/opt/rocm`
    # can be present on an NVIDIA box and would wrongly pull in the rlx-rocm backend.
    amd = Path("/dev/kfd").exists()
    vulkan = _have("vulkaninfo")
    order.append("wgpu")               # wgpu rides on Vulkan/GL — assume present on any GPU host
    if nvidia:
        order.append("cuda")
    if vulkan:
        order.append("vulkan")
    if amd:
        order.append("rocm")
    return order


def adapter_names():
    """Best-effort human-readable GPU names for the report header."""
    names = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            names["cuda"] = r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            disc = [l.split("=", 1)[1].strip() for l in r.stdout.splitlines()
                    if "deviceName" in l]
            # prefer a discrete NVIDIA/AMD name over an integrated one
            pick = next((n for n in disc if "NVIDIA" in n or "Radeon" in n or "AMD" in n), None)
            names["vulkan"] = pick or (disc[0] if disc else "?")
            names.setdefault("wgpu", names["vulkan"])
    except Exception:
        pass
    return names


# --------------------------------------------------------------------- crate features
_PKG_TOML = None


def _pkg_toml_map():
    global _PKG_TOML
    if _PKG_TOML is None:
        _PKG_TOML = {}
        for toml in (REPO / "crates").glob("*/Cargo.toml"):
            try:
                with open(toml, "rb") as f:
                    data = tomllib.load(f)
                name = data.get("package", {}).get("name")
                if name:
                    _PKG_TOML[name] = (toml, data)
            except Exception:
                continue
    return _PKG_TOML


def crate_features(pkg):
    """Set of [features] keys a crate defines (so we know which backends it supports)."""
    entry = _pkg_toml_map().get(pkg)
    if not entry:
        return set()
    return set(entry[1].get("features", {}).keys())


# --------------------------------------------------------------------------- resolve
def resolve(model, host):
    """(feature_str, [devices]) for this model on this host. All platform logic here."""
    cf = crate_features(model["package"])
    skip = set(model.get("skip_backends", []))
    devices = []
    for d in host:
        if d in skip:
            continue
        if d == "cpu" or DEV_FEATURE[d] in cf:
            devices.append(d)
    feats = list(model.get("features_base", []))
    for d in devices:
        if d != "cpu" and DEV_FEATURE[d]:
            feats.append(DEV_FEATURE[d])
    feats = sorted(set(feats))
    return ",".join(feats), devices


# ----------------------------------------------------------------------------- cargo
def cargo_bin():
    c = shutil.which("cargo")
    if c:
        return c
    home = Path(os.path.expanduser("~/.cargo/bin/cargo"))
    if home.exists():
        return str(home)
    sys.exit("cargo not found (looked on PATH and ~/.cargo/bin)")


CARGO = cargo_bin()


def backend_env(dev, model_env=None):
    """Env overrides that pin GPU backends to the discrete NVIDIA GPU on a dual-GPU host."""
    env = dict(os.environ)
    env.setdefault("CARGO_TERM_COLOR", "never")
    env["CARGO_TARGET_DIR"] = str(TARGET_DIR)
    # make cargo/cuda reachable under non-interactive ssh
    extra_path = [str(Path(os.path.expanduser("~/.cargo/bin"))), "/usr/local/cuda/bin"]
    env["PATH"] = os.pathsep.join(extra_path + [env.get("PATH", "")])
    if dev == "cuda":
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            ["/usr/local/cuda/lib64", env.get("LD_LIBRARY_PATH", "")])
    elif dev == "wgpu":
        env.setdefault("WGPU_ADAPTER_NAME", "NVIDIA")
        if platform.system() == "Linux":
            env.setdefault("WGPU_BACKEND", "vulkan")
    # Per-model env: str table applied always, or {backend: {k:v}} / flat {k:v}.
    if model_env:
        if isinstance(model_env, dict) and any(k in DEV_CLI for k in model_env):
            # mixed: top-level keys that aren't backends apply always; backend tables merge in
            for k, v in model_env.items():
                if k in DEV_CLI:
                    if k == dev and isinstance(v, dict):
                        env.update({str(kk): str(vv) for kk, vv in v.items()})
                else:
                    env[str(k)] = str(v)
        elif isinstance(model_env, dict):
            env.update({str(k): str(v) for k, v in model_env.items()})
    return env


_BUILT = {}  # (pkg, bin, feats) -> (ok, logpath)


def build_once(model, feats):
    pkg, binname = model["package"], model["bin"]
    key = (pkg, binname, feats)
    if key in _BUILT:
        return _BUILT[key]
    argv = [CARGO, "build", "-q", "-p", pkg, "--bin", binname, "--release"]
    if feats:
        argv += ["--features", feats]
    log(f"  build: {pkg}:{binname}  features=[{feats}]")
    logpath = OUT / "build-logs" / f"{pkg}.{binname}.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=REPO, env=backend_env("cpu"),
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        ok = r.returncode == 0
        logpath.write_text(r.stdout + "\n" + r.stderr)
    except subprocess.TimeoutExpired:
        ok = False
        logpath.write_text("BUILD TIMEOUT after %ds\n" % BUILD_TIMEOUT)
    dt = time.time() - t0
    log(f"    -> {'ok' if ok else 'FAIL'} in {dt:.0f}s")
    _BUILT[key] = (ok, logpath)
    return _BUILT[key]


def bin_path(model):
    return TARGET_DIR / "release" / model["bin"]


def run_proc(argv, dev, timeout, cwd=REPO, model_env=None):
    """Run a compiled binary, kill its whole group on timeout. Returns (rc, out, err, ms)."""
    t0 = time.time()
    try:
        p = subprocess.Popen(argv, cwd=cwd, env=backend_env(dev, model_env), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            out, err = p.communicate()
            return (None, out or "", (err or "") + "\n[TIMEOUT]", (time.time() - t0) * 1000)
    except Exception as e:
        return (-1, "", f"[spawn error] {e}", (time.time() - t0) * 1000)
    return (rc, out, err, (time.time() - t0) * 1000)


# ------------------------------------------------------------------------- templates
def render(template, ctx):
    out = template
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def err_summary(err):
    """Pick the most informative line out of a failed run's stderr."""
    lines = [l.strip() for l in err.splitlines() if l.strip()]
    if not lines:
        return ""
    # a Rust panic's real message is the line AFTER "panicked at <file>:<line>:"
    for i, l in enumerate(lines):
        if "panicked at" in l:
            msg = lines[i + 1] if i + 1 < len(lines) and "backtrace" not in lines[i + 1] else l
            return ("panic: " + msg)[:200]
    for pat in ("error:", "Error:", "bail", "unsupported", "not supported",
                "no such", "No such", "failed", "Failed", "cannot"):
        for l in lines:
            if pat in l:
                return l[:200]
    return lines[-1][:200]


# ----------------------------------------------------------------------- validation
def read_wav_mono(path):
    """Minimal float reader for PCM16 / float32 WAV (stdlib wave can't do float32)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 44 or data[:4] != b"RIFF":
        return []
    # walk chunks
    fmt_tag, channels, bits = 1, 1, 16
    i = 12
    samples = b""
    while i + 8 <= len(data):
        cid = data[i:i + 4]
        clen = int.from_bytes(data[i + 4:i + 8], "little")
        body = data[i + 8:i + 8 + clen]
        if cid == b"fmt ":
            fmt_tag = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little") or 1
            bits = int.from_bytes(body[14:16], "little")
        elif cid == b"data":
            samples = body
        i += 8 + clen + (clen & 1)
    import struct
    vals = []
    if fmt_tag == 3 or bits == 32:      # float32
        n = len(samples) // 4
        vals = list(struct.unpack("<%df" % n, samples[:n * 4]))
    elif bits == 16:
        n = len(samples) // 2
        ints = struct.unpack("<%dh" % n, samples[:n * 2])
        vals = [x / 32768.0 for x in ints]
    if channels > 1 and vals:
        vals = [sum(vals[j:j + channels]) / channels for j in range(0, len(vals), channels)]
    return vals


def cosine(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = na = nb = 0.0
    for i in range(n):
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12)


def wordset(s):
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 2]


def coverage(reference, hypothesis):
    ref = wordset(reference)
    if not ref:
        return 0.0
    hyp = set(wordset(hypothesis))
    return sum(1 for w in ref if w in hyp) / len(ref)


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference, hypothesis):
    r, h = wordset(reference), wordset(hypothesis)
    if not r:
        return 1.0
    return _lev(r, h) / len(r)


_WHISPER = None  # (bin_path, weights_dir, feats) or False once we know it's unavailable


def whisper_transcribe(wav):
    """Transcribe a wav with the rlx-whisper bin (cpu) for TTS coverage. Best-effort."""
    global _WHISPER
    if _WHISPER is False:
        return None
    if _WHISPER is None:
        wdir = None
        for cand in [REPO / ".cache" / "whisper-tiny",
                     REPO / ".cache" / "whisper-base.en",
                     Path(os.environ.get("RLX_WHISPER_DIR", "/nonexistent"))]:
            if cand.exists():
                wdir = cand
                break
        model = {"package": "rlx-whisper", "bin": "rlx-whisper", "features_base": ["tokenizer"]}
        feats, _ = resolve(model, ["cpu"])
        ok, _lp = build_once(model, feats)
        if not (ok and wdir):
            log("  (whisper validator unavailable — coverage skipped, cosine decides)")
            _WHISPER = False
            return None
        _WHISPER = (bin_path(model), wdir)
    wbin, wdir = _WHISPER
    txt = ART / "whisper_tmp.txt"
    argv = [str(wbin), "--weights", str(wdir), "--wav", str(wav),
            "--device", "cpu", "--language", "en", "--output", str(txt)]
    rc, out, err, _ms = run_proc(argv, "cpu", 180)
    if rc == 0 and txt.exists():
        return txt.read_text(errors="ignore")
    return out or None


def validate(model, dev, rc, stdout, artifact, cpu_ref, ctx):
    """Return (band, detail, metrics). band in pass|warn|fail. cpu_ref is the cpu result dict."""
    kind = model["kind"]
    v = model.get("validate", {})
    m = {}
    if rc is None:
        return "fail", "timeout", m
    if rc != 0:
        return "fail", "run exited %s" % rc, m

    if kind == "lm":
        toks = stdout.split()
        m["n_tokens"] = len(toks)
        if len(toks) < int(v.get("min_new_tokens", 4)) or len(set(toks)) <= 1:
            return "fail", "degenerate output (%d toks)" % len(toks), m
        if cpu_ref and dev != "cpu":
            ref = cpu_ref.get("stdout", "").split()
            k = int(v.get("parity_prefix", 8))
            match = sum(1 for i in range(min(k, len(ref), len(toks))) if ref[i] == toks[i])
            m["prefix_match"] = f"{match}/{min(k, len(ref))}"
            if match == 0:
                return "warn", "no prefix match vs cpu", m
            if match < min(k, len(ref)):
                return "warn", "partial parity %s" % m["prefix_match"], m
        return "pass", "ok (%d toks)" % len(toks), m

    if kind == "asr":
        ref_txt = ctx.get("ref_text", "")
        hyp = artifact.read_text(errors="ignore") if artifact and artifact.exists() else stdout
        if not hyp.strip():
            return "fail", "empty transcript", m
        if ref_txt:
            w = wer(ref_txt, hyp)
            m["wer"] = round(w, 3)
            if w <= float(v.get("wer_pass", 0.15)):
                return "pass", "wer %.2f" % w, m
            if w <= float(v.get("wer_warn", 0.35)):
                return "warn", "wer %.2f" % w, m
            return "fail", "wer %.2f" % w, m
        return "pass", "%d chars" % len(hyp), m

    if kind == "tts":
        if not (artifact and artifact.exists()):
            return "fail", "no wav produced", m
        wav = read_wav_mono(artifact)
        m["samples"] = len(wav)
        if len(wav) < 1600 or max((abs(x) for x in wav), default=0) < 1e-4:
            return "fail", "silent/empty wav", m
        band, detail = "pass", "ok"
        # cross-backend cosine vs cpu (authoritative)
        if cpu_ref and dev != "cpu":
            ref_wav = cpu_ref.get("wav_samples")
            if ref_wav:
                c = cosine(wav, ref_wav)
                m["cosine"] = round(c, 5)
                cp, cw = float(v.get("cosine_pass", 0.90)), float(v.get("cosine_warn", 0.70))
                if c < cw:
                    band, detail = "fail", "cosine %.3f vs cpu" % c
                elif c < cp:
                    band, detail = "warn", "cosine %.3f vs cpu" % c
                else:
                    detail = "cosine %.3f" % c
        # whisper coverage on the cpu clip (semantic sanity; only softens, never the sole fail)
        if dev == "cpu":
            hyp = whisper_transcribe(artifact)
            if hyp is not None:
                cov = coverage(ctx.get("text", ""), hyp)
                m["coverage"] = round(cov, 3)
                cvp, cvw = float(v.get("coverage_pass", 0.6)), float(v.get("coverage_warn", 0.4))
                if cov < cvw and band == "pass":
                    band, detail = "warn", "coverage %.2f (whisper-tiny weak?)" % cov
        return band, detail, m

    if kind in ("vision", "codec"):
        produced = (artifact and artifact.exists() and any(artifact.iterdir())) if artifact and artifact.is_dir() \
            else (artifact and artifact.exists())
        if not produced and not stdout.strip():
            return "fail", "no output", m
        return "pass", "produced output", m

    return "pass", "ran", m


# --------------------------------------------------------------------------- one model
def process(model, host, defaults):
    name = model["name"]
    feats, devices = resolve(model, host)
    rows = []
    log(f"\n=== {name} ({model['kind']}, {model['package']}) — devices: {devices}")

    # tier / license gating
    if model.get("tier", 1) == 2 and os.environ.get("ALL") != "1":
        return [dict(model_row(model, d, "skip", "tier-2 (ALL=1 to run)", {}, 0)) for d in devices]
    lic = model.get("license", {})
    if lic.get("gated") and os.environ.get("ALL") != "1" and not os.environ.get("HF_TOKEN"):
        return [dict(model_row(model, d, "skip", "gated (needs HF_TOKEN)", {}, 0)) for d in devices]

    ok, logpath = build_once(model, feats)
    if not ok:
        return [model_row(model, d, "build_fail", f"see {logpath.name}", {}, 0) for d in devices]

    # weights
    wpath = model.get("weights", {}).get("path")
    if wpath and not (REPO / wpath).exists():
        dl = model.get("download", {}).get("mode", "none")
        if dl == "flag":
            log(f"  downloading weights -> {wpath}")
            run_proc([str(bin_path(model)), "--download"], "cpu", 1800)
        if not (REPO / wpath).exists():
            return [model_row(model, d, "no_weights", f"missing {wpath}", {}, 0) for d in devices]

    # build invocation context
    ctx = dict(defaults)
    ctx.update(model.get("inputs", {}))
    ctx["weights"] = wpath or ""
    base_extra = model.get("extra", "")
    gpu_extra = model.get("gpu_extra", "")   # appended only for non-cpu backends
    template = model.get("run") or defaults["templates"][model["template"]]

    cpu_ref = None
    for dev in devices:
        out_ext = {"tts": ".wav", "asr": ".txt", "lm": ".txt"}.get(model["kind"], "")
        if model["kind"] in ("vision", "codec"):
            artifact = ART / f"{name}.{dev}"
            artifact.mkdir(parents=True, exist_ok=True)
        else:
            artifact = ART / f"{name}.{dev}{out_ext}" if out_ext else None
        ctx["device"] = DEV_CLI[dev]
        ctx["out"] = str(artifact) if artifact else ""
        # gpu_extra: str -> applied to every non-cpu backend; table -> per-backend
        # (e.g. only cuda needs --packed to fit VRAM; vulkan is fine on the f32 path).
        if isinstance(gpu_extra, dict):
            dev_extra = gpu_extra.get(dev, "")
        else:
            dev_extra = gpu_extra if dev != "cpu" else ""
        ctx["extra"] = base_extra + (" " + dev_extra if dev_extra else "")
        rendered = render(template, ctx)
        argv = [str(bin_path(model))] + shlex.split(rendered)
        log(f"  [{dev}] {' '.join(argv[1:])}")
        rc, out, err, ms = run_proc(
            argv, dev, KIND_TIMEOUT.get(model["kind"], 240),
            model_env=model.get("env"))
        # persist full stdout+stderr for every run so failures are diagnosable
        (ART / f"{name}.{dev}.log").write_text(
            "$ %s\n\n--- stdout ---\n%s\n--- stderr ---\n%s" % (" ".join(argv), out, err))
        # lm/asr: stdout is the artifact if the bin didn't write a file
        if model["kind"] == "lm" and artifact:
            artifact.write_text(out)
        band, detail, metrics = validate(model, dev, rc, out, artifact, cpu_ref, ctx)
        if dev == "cpu":
            cpu_ref = {"stdout": out, "band": band}
            if model["kind"] == "tts" and artifact and artifact.exists():
                cpu_ref["wav_samples"] = read_wav_mono(artifact)
        if rc not in (0, None) and band == "fail":
            detail = err_summary(err) or detail
        rows.append(model_row(model, dev, band, detail, metrics, ms))
        log(f"       -> {band.upper()}  {detail}  ({ms:.0f} ms)")
    return rows


def model_row(model, dev, status, detail, metrics, ms):
    return {"model": model["name"], "kind": model["kind"], "package": model["package"],
            "backend": dev, "status": status, "detail": detail,
            "metrics": metrics, "ms": round(ms)}


# -------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "rb") as f:
        reg = tomllib.load(f)
    defaults = reg.get("defaults", {})
    defaults["templates"] = reg.get("templates", {})

    host = host_backends()
    if os.environ.get("BACKENDS"):
        want = [b.strip() for b in os.environ["BACKENDS"].split(",") if b.strip()]
        host = [b for b in host if b in want] or host
    tier = os.environ.get("TIER", "1")
    only = set(x.strip() for x in os.environ.get("ONLY", "").split(",") if x.strip())

    models = reg.get("models", [])
    if only:
        models = [m for m in models if m["name"] in only]
    elif tier != "all":
        models = [m for m in models if str(m.get("tier", 1)) == tier or os.environ.get("ALL") == "1"]

    adapters = adapter_names() if platform.system() == "Linux" else {}
    log(f"host={platform.system()} backends={host}")
    for d, n in adapters.items():
        log(f"  {d}: {n}")
    log(f"models: {[m['name'] for m in models]}")

    all_rows = []
    for m in models:
        try:
            all_rows.extend(process(m, host, defaults))
        except Exception as e:
            log(f"  !! {m['name']} crashed the driver: {e}")
            for d in host:
                all_rows.append(model_row(m, d, "fail", f"driver error: {e}", {}, 0))
        # persist after every model so a killed run still has partial results
        (OUT / "results.json").write_text(json.dumps(
            {"host": platform.system(), "backends": host, "adapters": adapters,
             "rows": all_rows}, indent=2))

    write_report(all_rows, host, adapters)
    log(f"\nreport -> {OUT/'report.md'}")


ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "build_fail": "🧱",
        "no_weights": "📦", "skip": "·", "timeout": "⏱"}


def write_report(rows, host, adapters):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["backend"]] = r
    lines = ["# Cross-backend harness report", ""]
    lines.append(f"- host: `{platform.system()}`  backends: {', '.join('`%s`' % b for b in host)}")
    for d, n in adapters.items():
        lines.append(f"- {d} adapter: `{n}`")
    lines.append("")
    header = ["model", "kind"] + host
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for model, cells in by_model.items():
        kind = next(iter(cells.values()))["kind"]
        row = [f"`{model}`", kind]
        for b in host:
            c = cells.get(b)
            if not c:
                row.append("n/a")
            else:
                mtxt = ""
                for key in ("cosine", "wer", "coverage", "prefix_match", "n_tokens"):
                    if key in c.get("metrics", {}):
                        mtxt = f" {c['metrics'][key]}"
                        break
                row.append(f"{ICON.get(c['status'], c['status'])}{mtxt}")
        lines.append("| " + " | ".join(row) + " |")
    # detail + what-to-fix
    reds = [r for r in rows if r["status"] in ("fail", "build_fail", "timeout")]
    if reds:
        lines += ["", "## Needs attention", ""]
        for r in reds:
            lines.append(f"- ❌ `{r['model']}` / {r['backend']}: {r['detail']}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
