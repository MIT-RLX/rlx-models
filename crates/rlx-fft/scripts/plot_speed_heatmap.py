#!/usr/bin/env python3
"""Render docs/heatmap-speed.svg — rlx_op_fft latency over (n_fft × batch) per device.

Input: /tmp/grid5.csv with columns `device,n_fft,batch,ms,max_err,status` — one
row per cell, produced by the per-cell probe in README "Reproduce" (each cell in
its own process so a backend cap/panic isolates). status ∈ {ok, fail, skip}:
  * ok   → timed; ms + max_err present
  * fail → backend rejected the cell (wgpu dispatch envelope) → drawn `cap`
  * skip → over the memory/work budget for this run → drawn `—`

Also prints GitHub-flavored markdown tables (one per device) to stdout for the
README. Pure stdlib; re-run to refresh:

    python3 scripts/plot_speed_heatmap.py > /tmp/heatmap_tables.md
"""
import csv, math, os, sys

NS = [64, 256, 1024, 4096]                      # rows  (top → bottom)
BS = [32, 256, 2048, 16384, 131072]             # cols  (left → right)
# row0 = CPU-executed trio (rustfft reference, rlx-cpu, ANE host-fallback);
# row1 = GPU trio (Metal, MLX, wgpu).
DEVICES = [("rustfft", "rustfft (reference)"), ("cpu", "CPU (rlx-cpu)"), ("ane", "ANE / CoreML"),
           ("metal", "Metal"), ("mlx", "MLX"), ("wgpu", "wgpu")]

# ── load ─────────────────────────────────────────────────────────────────────
cell = {}  # (device, n, b) -> (ms|None, status)
for r in csv.DictReader(open("/tmp/grid5.csv")):
    ms = float(r["ms"]) if r["ms"].strip() else None
    cell[(r["device"], int(r["n_fft"]), int(r["batch"]))] = (ms, r["status"])

vals = [v for (v, s) in cell.values() if v]
LO, HI = math.log10(min(vals)), math.log10(max(vals))

def color(v):
    t = (math.log10(v) - LO) / (HI - LO)
    stops = [(0.0, (26, 152, 80)), (0.5, (254, 224, 139)), (1.0, (215, 48, 39))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1 or t1 == 1.0:
            f = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            rgb = tuple(round(a + (b - a) * max(0, min(1, f))) for a, b in zip(c0, c1))
            break
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "#%02x%02x%02x" % rgb, ("#222" if lum > 140 else "#fff")

def fmt(v):
    if v is None: return "—"
    if v < 1:   return f"{v:.2f}"
    if v < 10:  return f"{v:.1f}"
    return f"{v:.0f}"

def blabel(b): return f"{b//1024}k" if b >= 1024 else str(b)

# ── SVG (panels in a 3-wide grid: row0 cpu/metal/mlx, row1 wgpu/ane + colorbar) ─
CW, CH, LABW = 42, 31, 38
PITCH_X, PITCH_Y = LABW + 5 * CW + 30, 20 + 4 * CH + 40
X0, Y0 = 55, 70
W = X0 + 3 * PITCH_X + 80
H = Y0 + 2 * PITCH_Y
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="#fff"/>',
       f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" '
       f'fill="#222">FFT latency over n_fft × batch (ms/iter, lower=greener)</text>',
       f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12" fill="#666">'
       f'top row runs on CPU (rustfft = optimized reference; ANE host-fallback ≈ CPU) · bottom row on GPU · wgpu `cap` = exceeds dispatch envelope</text>']

for idx, (dev, title) in enumerate(DEVICES):
    r, c = divmod(idx, 3)
    px, py = X0 + c * PITCH_X, Y0 + r * PITCH_Y
    gx, gy = px + LABW, py + 20
    out.append(f'<text x="{gx + 5*CW/2:.0f}" y="{py+12}" text-anchor="middle" font-size="14" font-weight="600" fill="#333">{title}</text>')
    for ri, n in enumerate(NS):
        out.append(f'<text x="{gx-6:.0f}" y="{gy+ri*CH+CH/2+4:.0f}" text-anchor="end" font-size="11" fill="#555">{n}</text>')
    for ci, b in enumerate(BS):
        out.append(f'<text x="{gx+ci*CW+CW/2:.0f}" y="{gy+4*CH+15:.0f}" text-anchor="middle" font-size="10" fill="#555">{blabel(b)}</text>')
    for ri, n in enumerate(NS):
        for ci, b in enumerate(BS):
            x, y = gx + ci * CW, gy + ri * CH
            v, status = cell.get((dev, n, b), (None, "skip"))
            if v is None:
                lbl = "cap" if status == "fail" else "—"
                out.append(f'<rect x="{x}" y="{y}" width="{CW-2}" height="{CH-2}" fill="#eceff1" stroke="#fff"/>')
                out.append(f'<text x="{x+CW/2-1:.0f}" y="{y+CH/2+3:.0f}" text-anchor="middle" font-size="9" fill="#90a4ae">{lbl}</text>')
            else:
                fill, tc = color(v)
                out.append(f'<rect x="{x}" y="{y}" width="{CW-2}" height="{CH-2}" fill="{fill}" stroke="#fff"/>')
                out.append(f'<text x="{x+CW/2-1:.0f}" y="{y+CH/2+4:.0f}" text-anchor="middle" font-size="9.5" fill="{tc}">{fmt(v)}</text>')
    out.append(f'<text x="{px+8:.0f}" y="{gy+2*CH:.0f}" text-anchor="middle" font-size="10" fill="#888" transform="rotate(-90 {px+8:.0f} {gy+2*CH:.0f})">n_fft</text>')
    out.append(f'<text x="{gx+5*CW/2:.0f}" y="{gy+4*CH+30:.0f}" text-anchor="middle" font-size="10" fill="#888">batch</text>')

# colorbar to the right of the 3×2 panel grid, vertically centered
cbx, cbw, cbh = X0 + 3 * PITCH_X + 16, 18, 4 * CH
cby = Y0 + PITCH_Y - 2 * CH + 20
out.append('<defs><linearGradient id="cb" x1="0" y1="1" x2="0" y2="0">'
           '<stop offset="0" stop-color="#1a9850"/><stop offset="0.5" stop-color="#fee08b"/>'
           '<stop offset="1" stop-color="#d73027"/></linearGradient></defs>')
out.append(f'<text x="{cbx}" y="{cby-6}" font-size="11" font-weight="600" fill="#333">ms/iter</text>')
out.append(f'<rect x="{cbx}" y="{cby}" width="{cbw}" height="{cbh}" fill="url(#cb)" stroke="#ccc"/>')
out.append(f'<text x="{cbx+cbw+4}" y="{cby+8}" font-size="10" fill="#555">{10**HI:.0f}</text>')
out.append(f'<text x="{cbx+cbw+4}" y="{cby+cbh/2+4:.0f}" font-size="10" fill="#555">{10**((HI+LO)/2):.1f}</text>')
out.append(f'<text x="{cbx+cbw+4}" y="{cby+cbh}" font-size="10" fill="#555">{10**LO:.2f}</text>')
out.append('</svg>')

dst = os.path.join(os.path.dirname(__file__), "..", "docs", "heatmap-speed.svg")
open(dst, "w").write("\n".join(out))
print(f"wrote {os.path.normpath(dst)} ({len(out)} elements)", file=sys.stderr)

# ── markdown tables ──────────────────────────────────────────────────────────
for dev, title in DEVICES:
    metric = "rustfft" if dev == "rustfft" else "`rlx_op_fft`"
    print(f"\n**{title}** — {metric} ms/iter")
    print("| n_fft ╲ batch | " + " | ".join(blabel(b) for b in BS) + " |")
    print("|---|" + "---|" * len(BS))
    for n in NS:
        cells = []
        for b in BS:
            v, status = cell.get((dev, n, b), (None, "skip"))
            cells.append(("⛔ cap" if status == "fail" else "—") if v is None else fmt(v))
        print(f"| **{n}** | " + " | ".join(cells) + " |")
