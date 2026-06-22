#!/usr/bin/env python3
"""Generate ASR backend-comparison charts from `ASRBENCH ...` lines.

Reads a log of `ASRBENCH crate=.. device=.. config=.. wer=.. cer=.. rtfx=..
bsf=.. wall_s=.. rss_mb=..` lines (the uniform output of
`rlx_core::asr_bench::run_asr_bench`) and writes PNG charts to docs/asr/.

Usage: python3 scripts/asr_charts.py <agg.txt> [out_dir]
"""
import sys
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICES = ["cpu", "metal", "mlx", "gpu"]
DEV_COLORS = {"cpu": "#6b7280", "metal": "#2563eb", "mlx": "#16a34a", "gpu": "#d97706"}


def parse(path):
    rows = []
    pat = re.compile(r"(\w+)=(\S+)")
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ASRBENCH"):
            continue
        d = dict(pat.findall(line))
        def num(k):
            v = d.get(k, "na")
            try:
                return float(v)
            except ValueError:
                return None
        rows.append({
            "crate": d.get("crate", "?").replace("rlx-", ""),
            "device": d.get("device", "?"),
            "config": d.get("config", "?"),
            "wer": num("wer"), "cer": num("cer"), "rtfx": num("rtfx"),
            "bsf": num("bsf"), "wall_s": num("wall_s"), "rss_mb": num("rss_mb"),
        })
    return rows


def grouped_bar(rows, metric, title, ylabel, out, lower_better, logy=False):
    batch = [r for r in rows if r["config"] == "batch"]
    crates = sorted({r["crate"] for r in batch})
    if not crates:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(crates) * 1.7), 4.6))
    n = len(DEVICES)
    w = 0.8 / n
    for i, dev in enumerate(DEVICES):
        xs, ys = [], []
        for j, c in enumerate(crates):
            v = next((r[metric] for r in batch if r["crate"] == c and r["device"] == dev), None)
            if v is not None:
                xs.append(j + (i - (n - 1) / 2) * w)
                ys.append(v)
        if xs:
            bars = ax.bar(xs, ys, w, label=dev, color=DEV_COLORS[dev])
            for b, y in zip(bars, ys):
                ax.annotate(f"{y:.1f}", (b.get_x() + b.get_width() / 2, y),
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(crates)))
    ax.set_xticklabels(crates, fontsize=9)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(f"{title}  ({'lower=better' if lower_better else 'higher=better'})", fontsize=11)
    ax.legend(title="backend", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def leaderboard(rows):
    """Print fastest / least-memory / most-accurate rankings (batch rows)."""
    batch = [r for r in rows if r["config"] == "batch" and r["rtfx"] is not None]
    if not batch:
        return
    crates = sorted({r["crate"] for r in batch})
    print("\n=== ASR leaderboard (batch) ===")
    print(f"{'model':<14} {'WER%':>6} {'fastest RTF':>16} {'least mem':>18}")
    for c in crates:
        rs = [r for r in batch if r["crate"] == c]
        wer = next((r["wer"] for r in rs if r["wer"] is not None), None)
        fast = max(rs, key=lambda r: r["rtfx"])
        lean = min((r for r in rs if r["rss_mb"]), key=lambda r: r["rss_mb"], default=fast)
        wers = "n/a" if wer is None else f"{wer:.2f}"  # already a percentage
        print(f"{c:<14} {wers:>6} {fast['rtfx']:>9.2f}x @{fast['device']:<5}"
              f" {lean['rss_mb'] / 1024:>10.2f} GB @{lean['device']}")
    fast = max(batch, key=lambda r: r["rtfx"])
    lean = min((r for r in batch if r["rss_mb"]), key=lambda r: r["rss_mb"])
    acc = min((r for r in batch if r["wer"] is not None), key=lambda r: r["wer"])
    print(f"-> fastest:     {fast['crate']} @{fast['device']}  RTF {fast['rtfx']:.2f}x")
    print(f"-> least mem:   {lean['crate']} @{lean['device']}  {lean['rss_mb'] / 1024:.2f} GB")
    print(f"-> most accurate: WER {acc['wer']:.2f}% ({acc['crate']})")


def main():
    agg = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rlxbench/asr/AGG.txt"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/asr")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse(agg)
    if not rows:
        print("no ASRBENCH rows found in", agg)
        return
    grouped_bar(rows, "wer", "ASR accuracy — WER % (batch, JFK clip)", "WER %",
                out_dir / "asr_wer.png", lower_better=True)
    grouped_bar(rows, "rtfx", "ASR speed — RTFx (batch)", "RTFx (audio÷wall)",
                out_dir / "asr_rtfx.png", lower_better=False, logy=True)
    grouped_bar(rows, "rss_mb", "ASR memory — peak RSS (batch)", "peak RSS (MB)",
                out_dir / "asr_rss.png", lower_better=True)
    print(f"parsed {len(rows)} rows across "
          f"{len({r['crate'] for r in rows})} crates")
    leaderboard(rows)


if __name__ == "__main__":
    main()
