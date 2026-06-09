#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Benchmark FFT implementations in Python / PyTorch vs optional native RLX sweep JSON.

Covers CPU (NumPy/SciPy/pure Python), PyTorch CPU/CUDA/MPS (CUDA uses cuFFT),
optional PyFFTW, and reports the linked linear-algebra backend (MKL/OpenBLAS/…).

Usage:
  python3 crates/rlx-fft/scripts/bench_fft_pytorch.py --n-fft 64,128 --batch 1,8,64
  python3 crates/rlx-fft/scripts/bench_fft_pytorch.py --json /tmp/fft-pytorch.json --html /tmp/fft-pytorch.html
  python3 crates/rlx-fft/scripts/bench_fft_pytorch.py --compare-rlx --rlx-json /tmp/fft-rlx.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import scipy.fft as scipy_fft
except ImportError:
    scipy_fft = None

try:
    import torch
except ImportError:
    torch = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import pyfftw
except ImportError:
    pyfftw = None


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class BenchRow:
    direction: str
    n_fft: int
    batch: int
    implementation: str
    device: str
    backend_note: str
    ms: float
    max_err: float | None = None
    iters: int = 0


@dataclass
class BenchReport:
    iters: int
    warmup: int
    seed: int
    numpy_config: str
    torch_version: str | None
    torch_cuda: bool
    torch_mps: bool
    cupy: bool
    pyfftw: bool
    elapsed_ms: float
    rows: list[BenchRow] = field(default_factory=list)
    rlx_rows: list[dict[str, Any]] = field(default_factory=list)


def parse_csv_ints(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    da = np.asarray(a)
    db = np.asarray(b)
    if np.iscomplexobj(da) or np.iscomplexobj(db):
        return float(np.max(np.abs(da.astype(np.complex128) - db.astype(np.complex128))))
    return float(np.max(np.abs(da.astype(np.float64) - db.astype(np.float64))))


def pure_python_dft_real(signal: np.ndarray, n_fft: int) -> np.ndarray:
    """O(n^2) reference — small n_fft only."""
    batch = signal.shape[0]
    out = np.zeros((batch, n_fft), dtype=np.complex64)
    k = np.arange(n_fft, dtype=np.float64)
    n = np.arange(n_fft, dtype=np.float64)
    for b in range(batch):
        x = signal[b].astype(np.float64)
        for i in range(n_fft):
            angle = -2.0 * math.pi * k * n[i] / n_fft
            out[b, i] = np.sum(x * np.exp(1j * angle))
    return out


def make_signal(batch: int, n_fft: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, n_fft), dtype=np.float32)


def make_spectrum(batch: int, n_fft: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    re = rng.standard_normal((batch, n_fft), dtype=np.float32)
    im = rng.standard_normal((batch, n_fft), dtype=np.float32)
    return re + 1j * im


def time_call(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def numpy_config_summary() -> str:
    parts = [f"numpy {np.__version__}"]
    try:
        for key in ("blas", "lapack", "fft"):
            info = np.__config__.get_info(key)  # type: ignore[attr-defined]
            if info:
                parts.append(f"{key}={info}")
    except Exception:
        pass
    return " | ".join(parts) if len(parts) > 1 else parts[0]


def bench_forward(
    signal: np.ndarray,
    n_fft: int,
    batch: int,
    warmup: int,
    iters: int,
) -> tuple[list[BenchRow], np.ndarray]:
    rows: list[BenchRow] = []
    ref = np.fft.fft(signal.astype(np.float64), axis=-1).astype(np.complex64)

    def add(name: str, device: str, note: str, fn: Callable[[], np.ndarray | None]) -> None:
        out_holder: dict[str, np.ndarray] = {}

        def run() -> None:
            out_holder["y"] = fn()

        ms = time_call(run, warmup, iters)
        err = None
        if "y" in out_holder and out_holder["y"] is not None:
            y = np.asarray(out_holder["y"])
            if y.shape == ref.shape:
                err = max_abs_err(y, ref)
            elif y.shape[-1] == n_fft // 2 + 1:
                # rfft: compare to ref[..., :n//2+1]
                err = max_abs_err(y, ref[..., : y.shape[-1]])
        rows.append(
            BenchRow(
                direction="forward",
                n_fft=n_fft,
                batch=batch,
                implementation=name,
                device=device,
                backend_note=note,
                ms=ms,
                max_err=err,
                iters=iters,
            )
        )

    add("numpy_fft", "cpu", "NumPy pocketfft/MKL", lambda: np.fft.fft(signal, axis=-1))
    add("numpy_rfft", "cpu", "NumPy rfft (half spectrum)", lambda: np.fft.rfft(signal, axis=-1))

    if scipy_fft is not None:
        add("scipy_fft", "cpu", "SciPy FFT", lambda: scipy_fft.fft(signal, axis=-1))
        add("scipy_rfft", "cpu", "SciPy rfft", lambda: scipy_fft.rfft(signal, axis=-1))

    if n_fft <= 32 and batch <= 4:
        add(
            "pure_python_dft",
            "cpu",
            "O(n^2) Python loops",
            lambda: pure_python_dft_real(signal, n_fft),
        )

    if pyfftw is not None:
        a = pyfftw.empty_aligned((batch, n_fft), dtype="float32")
        a[:] = signal
        fft_obj = pyfftw.builders.fft(a, axis=-1, threads=os.cpu_count() or 1)

        def pyfftw_run() -> np.ndarray:
            return np.asarray(fft_obj())

        add("pyfftw_fft", "cpu", f"FFTW threads={os.cpu_count()}", pyfftw_run)

    if torch is not None:
        t_cpu = torch.from_numpy(signal)

        def torch_cpu_fft() -> np.ndarray:
            return torch.fft.fft(t_cpu, dim=-1).numpy()

        def torch_cpu_rfft() -> np.ndarray:
            return torch.fft.rfft(t_cpu, dim=-1).numpy()

        add("torch_cpu_fft", "cpu", "PyTorch CPU", torch_cpu_fft)
        add("torch_cpu_rfft", "cpu", "PyTorch CPU rfft", torch_cpu_rfft)

        if torch.cuda.is_available():
            t_cuda = t_cpu.cuda()

            def torch_cuda_fft() -> np.ndarray:
                return torch.fft.fft(t_cuda, dim=-1).cpu().numpy()

            def torch_cuda_rfft() -> np.ndarray:
                return torch.fft.rfft(t_cuda, dim=-1).cpu().numpy()

            add(
                "torch_cuda_fft",
                "cuda",
                "PyTorch CUDA (cuFFT)",
                torch_cuda_fft,
            )
            add(
                "torch_cuda_rfft",
                "cuda",
                "PyTorch CUDA rfft (cuFFT)",
                torch_cuda_rfft,
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            def torch_mps_fft() -> np.ndarray:
                t = torch.from_numpy(signal).to("mps")
                return torch.fft.fft(t, dim=-1).detach().cpu().numpy()

            def torch_mps_rfft() -> np.ndarray:
                t = torch.from_numpy(signal).to("mps")
                return torch.fft.rfft(t, dim=-1).detach().cpu().numpy()

            add("torch_mps_fft", "mps", "PyTorch Metal (Apple GPU)", torch_mps_fft)
            add("torch_mps_rfft", "mps", "PyTorch MPS rfft", torch_mps_rfft)

    if cp is not None:
        g_signal = cp.asarray(signal)

        def cupy_fft() -> np.ndarray:
            return cp.asnumpy(cp.fft.fft(g_signal, axis=-1))

        def cupy_rfft() -> np.ndarray:
            return cp.asnumpy(cp.fft.rfft(g_signal, axis=-1))

        add("cupy_fft", "cuda", "CuPy cuFFT (direct CUDA)", cupy_fft)
        add("cupy_rfft", "cuda", "CuPy rfft", cupy_rfft)

    return rows, ref


def bench_inverse(
    spectrum: np.ndarray,
    n_fft: int,
    batch: int,
    warmup: int,
    iters: int,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    ref = np.fft.ifft(spectrum.astype(np.complex64), axis=-1).real.astype(np.float32)

    def add(name: str, device: str, note: str, fn: Callable[[], np.ndarray | None]) -> None:
        out_holder: dict[str, np.ndarray] = {}

        def run() -> None:
            out_holder["y"] = fn()

        ms = time_call(run, warmup, iters)
        err = None
        if "y" in out_holder and out_holder["y"] is not None:
            y = np.asarray(out_holder["y"]).real.astype(np.float32)
            err = max_abs_err(y, ref)
        rows.append(
            BenchRow(
                direction="inverse",
                n_fft=n_fft,
                batch=batch,
                implementation=name,
                device=device,
                backend_note=note,
                ms=ms,
                max_err=err,
                iters=iters,
            )
        )

    add("numpy_ifft", "cpu", "NumPy ifft", lambda: np.fft.ifft(spectrum, axis=-1).real)
    if scipy_fft is not None:
        add(
            "scipy_ifft",
            "cpu",
            "SciPy ifft",
            lambda: scipy_fft.ifft(spectrum, axis=-1).real,
        )

    if torch is not None:
        t_spec = torch.from_numpy(spectrum)

        def torch_cpu_ifft() -> np.ndarray:
            return torch.fft.ifft(t_spec, dim=-1).real.numpy()

        add("torch_cpu_ifft", "cpu", "PyTorch CPU", torch_cpu_ifft)

        if torch.cuda.is_available():
            t_cuda = t_spec.cuda()
            add(
                "torch_cuda_ifft",
                "cuda",
                "PyTorch CUDA (cuFFT)",
                lambda: torch.fft.ifft(t_cuda, dim=-1).real.cpu().numpy(),
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            add(
                "torch_mps_ifft",
                "mps",
                "PyTorch Metal",
                lambda: torch.fft.ifft(torch.from_numpy(spectrum).to("mps"), dim=-1)
                .real.detach()
                .cpu()
                .numpy(),
            )

    if cp is not None:
        g_spec = cp.asarray(spectrum)
        add(
            "cupy_ifft",
            "cuda",
            "CuPy cuFFT",
            lambda: cp.asnumpy(cp.fft.ifft(g_spec, axis=-1).real),
        )

    return rows


def rlx_cargo_features() -> str:
    if os.environ.get("RLX_FFT_FEATURES"):
        return os.environ["RLX_FFT_FEATURES"]
    if sys.platform == "darwin":
        return "apple-silicon"
    return "cpu,cuda,gpu"


def run_rlx_sweep(
    n_ffts: list[int],
    batches: list[int],
    iters: int,
    json_out: Path,
    devices: str,
) -> list[dict[str, Any]]:
    feats = rlx_cargo_features()
    cmd = [
        "cargo",
        "run",
        "-p",
        "rlx-fft",
        "--release",
        "--features",
        feats,
        "--",
        "bench-sweep",
        "--n-fft",
        ",".join(str(n) for n in n_ffts),
        "--batch",
        ",".join(str(b) for b in batches),
        "--devices",
        devices,
        "--iters",
        str(iters),
        "--both-dirs",
        "--json",
        str(json_out),
    ]
    print(f"[rlx] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    data = json.loads(json_out.read_text())
    return data.get("rows", [])


def print_table(report: BenchReport) -> None:
    print(
        f"\n=== FFT Python/PyTorch bench (iters={report.iters}, warmup={report.warmup}) ===\n"
    )
    print(f"NumPy: {report.numpy_config}")
    print(
        f"PyTorch: {report.torch_version}  cuda={report.torch_cuda}  mps={report.torch_mps}  "
        f"cupy={report.cupy}  pyfftw={report.pyfftw}\n"
        "Note: PyTorch CUDA and CuPy use cuFFT. cuDNN has no standalone FFT API.\n"
    )
    hdr = f"{'dir':<8} {'n':>4} {'batch':>5} {'impl':<22} {'device':<6} {'ms':>10} {'max_err':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(
        report.rows,
        key=lambda x: (x.direction, x.n_fft, x.batch, x.device, x.implementation),
    ):
        err = f"{r.max_err:.3e}" if r.max_err is not None else "-"
        print(
            f"{r.direction:<8} {r.n_fft:>4} {r.batch:>5} {r.implementation:<22} {r.device:<6} {r.ms:>10.4f} {err:>12}"
        )
    if report.rlx_rows:
        print("\n--- RLX native (from bench-sweep) ---")
        for r in report.rlx_rows:
            err = r.get("max_err")
            err_s = f"{err:.3e}" if err is not None else "-"
            print(
                f"{r.get('direction','?'):<8} {r.get('n_fft',0):>4} {r.get('batch',0):>5} "
                f"{r.get('implementation','?'):<22} {r.get('device','?'):<6} "
                f"{float(r.get('ms',0)):>10.4f} {err_s:>12}"
            )
    print(f"\nTotal: {report.elapsed_ms:.1f} ms")


def write_html(path: Path, report: BenchReport) -> None:
    rows_json = json.dumps([asdict(r) for r in report.rows])
    rlx_json = json.dumps(report.rlx_rows)
    meta = {
        "iters": report.iters,
        "warmup": report.warmup,
        "numpy_config": str(report.numpy_config),
        "torch_version": report.torch_version,
        "torch_cuda": report.torch_cuda,
        "torch_mps": report.torch_mps,
        "pyfftw": report.pyfftw,
    }
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>FFT Python/PyTorch vs RLX</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1117; color: #e8eaed; }}
.card {{ background: #1a1d27; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; }}
h1 {{ font-size: 1.35rem; }}
.meta {{ color: #9aa0a6; font-size: 0.9rem; }}
canvas {{ max-height: 420px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ border: 1px solid #333; padding: 0.35rem 0.5rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style></head><body>
<h1>FFT benchmark: Python / PyTorch / optional RLX</h1>
<div class="card meta" id="meta"></div>
<div class="card"><canvas id="chart"></canvas></div>
<div class="card"><table id="tbl"><thead><tr>
<th>dir</th><th>n_fft</th><th>batch</th><th>impl</th><th>device</th><th>ms</th><th>max_err</th>
</tr></thead><tbody></tbody></table></div>
<script>
const META = {json.dumps(meta)};
const ROWS = {rows_json};
const RLX = {rlx_json};
document.getElementById('meta').textContent =
  `iters=${{META.iters}} warmup=${{META.warmup}} | PyTorch ${{META.torch_version||'n/a'}} cuda=${{META.torch_cuda}} mps=${{META.torch_mps}} pyfftw=${{META.pyfftw}}`;
const tbody = document.querySelector('#tbl tbody');
for (const r of [...ROWS, ...RLX.map(x => ({{...x, implementation: 'rlx:'+x.implementation}}))]) {{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{r.direction||'?'}}</td><td>${{r.n_fft}}</td><td>${{r.batch}}</td>
    <td>${{r.implementation}}</td><td>${{r.device||'-'}}</td><td>${{Number(r.ms).toFixed(4)}}</td>
    <td>${{r.max_err != null ? Number(r.max_err).toExponential(3) : '-'}}</td>`;
  tbody.appendChild(tr);
}}
const fwd = ROWS.filter(r => r.direction === 'forward');
const labels = [...new Set(fwd.map(r => `${{r.implementation}}@${{r.device}} (n=${{r.n_fft}} b=${{r.batch}})`))];
const data = labels.map(l => {{
  const r = fwd.find(x => `${{x.implementation}}@${{x.device}} (n=${{x.n_fft}} b=${{x.batch}})` === l);
  return r ? r.ms : null;
}});
new Chart(document.getElementById('chart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'forward ms/iter', data, backgroundColor: '#59a14f' }}] }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e8eaed' }} }} }},
    scales: {{ x: {{ ticks: {{ color: '#9aa0a6', maxRotation: 45 }} }}, y: {{ title: {{ display: true, text: 'ms', color: '#9aa0a6' }}, ticks: {{ color: '#9aa0a6' }} }} }}
  }}
}});
</script></body></html>"""
    path.write_text(html)
    print(f"wrote {path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark FFT: NumPy/SciPy/PyTorch vs optional RLX")
    ap.add_argument("--n-fft", type=parse_csv_ints, default=[64, 128])
    ap.add_argument("--batch", type=parse_csv_ints, default=[1, 8, 64])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--forward-only", action="store_true", help="skip inverse FFT")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--html", type=Path, default=None)
    ap.add_argument("--compare-rlx", action="store_true", help="also run rlx-fft bench-sweep")
    ap.add_argument("--rlx-json", type=Path, default=Path("/tmp/fft-rlx-sweep.json"))
    ap.add_argument(
        "--rlx-devices",
        default="cpu,metal",
        help="devices for rlx bench-sweep (cuda if on NVIDIA rig)",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    report = BenchReport(
        iters=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        numpy_config=str(numpy_config_summary()),
        torch_version=torch.__version__ if torch else None,
        torch_cuda=bool(torch and torch.cuda.is_available()),
        torch_mps=bool(
            torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        pyfftw=pyfftw is not None,
        cupy=cp is not None,
        elapsed_ms=0.0,
    )

    for n_fft in args.n_fft:
        for batch in args.batch:
            signal = make_signal(batch, n_fft, args.seed + n_fft + batch)
            fwd_rows, _ = bench_forward(
                signal, n_fft, batch, args.warmup, args.iters
            )
            report.rows.extend(fwd_rows)
            if not args.forward_only:
                spectrum = make_spectrum(batch, n_fft, args.seed + n_fft + batch)
                report.rows.extend(
                    bench_inverse(spectrum, n_fft, batch, args.warmup, args.iters)
                )

    if args.compare_rlx:
        if args.rlx_json.is_file():
            report.rlx_rows = json.loads(args.rlx_json.read_text()).get("rows", [])
            print(f"[rlx] loaded {len(report.rlx_rows)} rows from {args.rlx_json}", file=sys.stderr)
        else:
            try:
                report.rlx_rows = run_rlx_sweep(
                    args.n_fft,
                    args.batch,
                    max(10, args.iters // 5),
                    args.rlx_json,
                    args.rlx_devices,
                )
            except subprocess.CalledProcessError as e:
                print(f"[rlx] bench-sweep failed: {e}", file=sys.stderr)

    report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print_table(report)

    if args.json:
        payload = asdict(report)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}", file=sys.stderr)

    if args.html:
        write_html(args.html, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
