#!/usr/bin/env python3
"""Render the rlx-fft precision/speed benchmark chart to docs/precision-benchmark.svg.

Data is the output of two benchmarks (committed, reproducible):
  * `precision_fft::tests::comprehensive_precision_benchmark` (eager-CPU precision ladder)
  * `rlx-fft bench-sweep --with-butterfly-compiled` (hardware x batch, native f32)

This is a pure-stdlib generator (no matplotlib) so it runs anywhere; edit the
DATA tables below and re-run to refresh the SVG:

    python3 scripts/plot_precision_bench.py
"""
import math, os

# ── Panel 1: precision ladder (eager CPU, n=256) ──────────────────────────────
#   (label, base, bits, roundtrip_err, us_per_fft)
LADDER = [
    ("f8 x1",  "f8",  4,   2.0e-1, 320),
    ("f8 x2",  "f8",  8,   1.9e-2, 2773),
    ("f16 x1", "f16", 11,  1.8e-3, 319),
    ("f16 x2", "f16", 22,  1.2e-4, 3150),
    ("f32 x1", "f32", 24,  1.8e-7, 939),
    ("f32 x2", "f32", 48,  4.7e-15, 36),
    ("f32 x3", "f32", 72,  5.7e-23, 3027),
    ("f32 x4", "f32", 96,  1.3e-30, 4730),
    ("f64 x1", "f64", 53,  2.2e-16, 11),
    ("f64 x2", "f64", 106, 1.8e-31, 35),
]
BASE_COLOR = {"f8": "#e15759", "f16": "#f28e2b", "f32": "#4e79a7", "f64": "#59a14f"}

# ── Panel 2: hardware x batch (native f32 butterfly, ms/iter, n=256) ───────────
#   device -> [(batch, ms), ...]   (Metal warm; cold first-compile reading dropped)
HW = {
    "CPU rlx":  ([32, 1024, 8192, 32768, 65536, 131072], [0.080, 2.44, 21.2, 86.2, 177.8, 348.6], "#4e79a7"),
    "Metal rlx":([1024, 8192, 32768, 65536, 131072],     [0.666, 2.85, 8.58, 11.8, 20.6], "#59a14f"),
    "wgpu rlx": ([32, 1024, 8192, 32768],                [1.87, 2.59, 11.06, 29.59], "#b07aa1"),
    "rustfft":  ([32, 1024, 8192, 32768, 65536, 131072], [0.030, 0.654, 5.35, 22.33, 45.6, 89.1], "#79706e"),
}
WGPU_CAP = 65536  # wgpu: max_compute_workgroups_per_dimension = 65535 → batch ≥ 65536 is rejected

W, H = 912, 770
def digits(e): return -math.log10(e)

def esc(s): return s.replace("x", "×")

out = []
def emit(s): out.append(s)

emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
emit(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
emit(f'<text x="{W/2}" y="32" text-anchor="middle" font-size="20" font-weight="700" '
     f'fill="#222">rlx-fft &#8212; precision ladder &amp; hardware scaling (n=256)</text>')

# ===== Panel 1 =====
L, R, T, B = 80, 790, 78, 360
def x1(b): return L + b/110*(R-L)
def y1(d): return B - d/34*(B-T)
emit(f'<text x="{(L+R)/2}" y="62" text-anchor="middle" font-size="15" font-weight="600" '
     f'fill="#333">compensated multi-limb FFT reaches the precision of its total bit-width</text>')
# grid + axes
for d in range(0, 33, 4):
    y = y1(d)
    emit(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="#eee"/>')
    emit(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#666">{d}</text>')
for b in [0, 16, 32, 48, 64, 80, 96, 110]:
    x = x1(b)
    emit(f'<text x="{x:.1f}" y="{B+18}" text-anchor="middle" font-size="11" fill="#666">{b}</text>')
emit(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="#999"/>')
emit(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="#999"/>')
emit(f'<text x="{(L+R)/2}" y="{B+36}" text-anchor="middle" font-size="12" fill="#444">total mantissa bits (base bits &#215; limbs)</text>')
emit(f'<text x="22" y="{(T+B)/2}" text-anchor="middle" font-size="12" fill="#444" '
     f'transform="rotate(-90 22 {(T+B)/2})">accurate decimal digits (&#8722;log&#8321;&#8320; err)</text>')
# theoretical line: digits = log10(2) * bits
emit(f'<line x1="{x1(0):.1f}" y1="{y1(0):.1f}" x2="{x1(110):.1f}" y2="{y1(0.30103*110):.1f}" '
     f'stroke="#bbb" stroke-width="1.5" stroke-dasharray="5 4"/>')
emit(f'<text x="{x1(56):.1f}" y="{y1(0.30103*56)-8:.1f}" font-size="10.5" fill="#999" '
     f'transform="rotate(-15 {x1(56):.1f} {y1(0.30103*56):.1f})">information limit (log&#8321;&#8320;2 &#215; bits)</text>')
# points + labels
LBL = {  # per-point label offset (dx, dy, anchor)
    "f8 x1": (8, 4, "start"), "f8 x2": (8, 4, "start"),
    "f16 x1": (8, 4, "start"), "f16 x2": (8, 4, "start"),
    "f32 x1": (8, 14, "start"), "f32 x2": (10, 14, "start"),
    "f32 x3": (-8, 16, "end"), "f32 x4": (-10, -8, "end"),
    "f64 x1": (10, -8, "start"), "f64 x2": (-10, 16, "end"),
}
for label, base, bits, err, us in LADDER:
    x, y = x1(bits), y1(digits(err))
    emit(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{BASE_COLOR[base]}" stroke="#fff" stroke-width="1"/>')
    dx, dy, anc = LBL[label]
    emit(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anc}" font-size="10.5" '
         f'fill="#333">{esc(label)} &#183; {us}&#181;s</text>')
# legend (upper-left, in the empty high-digit / low-bit region)
lx, ly = L+18, T+14
for i, base in enumerate(["f8", "f16", "f32", "f64"]):
    yy = ly + i*18
    emit(f'<circle cx="{lx}" cy="{yy}" r="5" fill="{BASE_COLOR[base]}"/>')
    emit(f'<text x="{lx+12}" y="{yy+4}" font-size="11" fill="#444">{base} base</text>')

# ===== Panel 2 =====
L2, R2, T2, B2 = 80, 790, 478, 712
def x2(b): return L2 + (math.log2(b)-5)/12*(R2-L2)
def y2(ms): return B2 - (math.log10(ms)+1.7)/4.4*(B2-T2)
emit(f'<text x="{(L2+R2)/2}" y="462" text-anchor="middle" font-size="15" font-weight="600" '
     f'fill="#333">native f32 throughput: GPU overtakes CPU once the batch amortizes launch</text>')
for ms in [0.1, 1, 10, 100]:
    y = y2(ms)
    emit(f'<line x1="{L2}" y1="{y:.1f}" x2="{R2}" y2="{y:.1f}" stroke="#eee"/>')
    lab = f'{ms:g}'
    emit(f'<text x="{L2-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#666">{lab} ms</text>')
for b in [32, 1024, 8192, 32768, 131072]:
    x = x2(b)
    emit(f'<line x1="{x:.1f}" y1="{T2}" x2="{x:.1f}" y2="{B2}" stroke="#f4f4f4"/>')
    bl = f'{b//1024}k' if b >= 1024 else str(b)
    emit(f'<text x="{x:.1f}" y="{B2+18}" text-anchor="middle" font-size="11" fill="#666">{bl}</text>')
emit(f'<text x="{(L2+R2)/2}" y="{B2+34}" text-anchor="middle" font-size="11" fill="#666">batch</text>')
# wgpu hard dispatch cap
xc = x2(WGPU_CAP)
emit(f'<line x1="{xc:.1f}" y1="{T2}" x2="{xc:.1f}" y2="{B2}" stroke="#b07aa1" stroke-width="1.2" stroke-dasharray="3 3" opacity="0.7"/>')
emit(f'<text x="{xc-5:.1f}" y="{T2+14:.1f}" text-anchor="end" font-size="10" fill="#b07aa1">wgpu cap 65535</text>')
emit(f'<line x1="{L2}" y1="{T2}" x2="{L2}" y2="{B2}" stroke="#999"/>')
emit(f'<line x1="{L2}" y1="{B2}" x2="{R2}" y2="{B2}" stroke="#999"/>')
emit(f'<text x="22" y="{(T2+B2)/2}" text-anchor="middle" font-size="12" fill="#444" '
     f'transform="rotate(-90 22 {(T2+B2)/2})">ms / iter (log, lower is faster)</text>')
for i, (name, (bs, ms, color)) in enumerate(HW.items()):
    pts = " ".join(f'{x2(b):.1f},{y2(m):.1f}' for b, m in zip(bs, ms))
    dash = ' stroke-dasharray="6 3"' if name == "rustfft" else ''
    emit(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
    for b, m in zip(bs, ms):
        emit(f'<circle cx="{x2(b):.1f}" cy="{y2(m):.1f}" r="4" fill="{color}"/>')
    # label at right end
    emit(f'<text x="{x2(bs[-1])+6:.1f}" y="{y2(ms[-1])+4:.1f}" font-size="11" '
         f'fill="{color}" font-weight="600">{name}</text>')
# annotate Metal crossover
emit(f'<text x="{x2(8192):.1f}" y="{y2(2.85)-12:.1f}" text-anchor="end" font-size="10" '
     f'fill="#59a14f">beats rustfft &#64; 8k</text>')

emit('</svg>')

dst = os.path.join(os.path.dirname(__file__), "..", "docs", "precision-benchmark.svg")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    f.write("\n".join(out))
print(f"wrote {os.path.normpath(dst)}  ({len(out)} elements)")
