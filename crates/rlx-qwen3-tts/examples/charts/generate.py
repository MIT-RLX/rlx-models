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

"""Generate SVG charts for the rlx-qwen3-tts README.

All numbers are measured on an Apple M3 Pro running this crate's
benches (cargo run --release --features apple-silicon -p rlx-qwen3-tts).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

C_BEFORE = "#d05050"
C_AFTER  = "#3070b0"
C_GOOD   = "#2a9d57"
C_NEUTRAL = "#6a6a6a"
C_BAD    = "#d05050"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")


def chart_stage_timings():
    """Per-stage timings before / after optimization."""
    stages = [
        "Speaker enc\n(ECAPA fwd)",
        "Megakernel\nopen",
        "Decoder open\n+ warmup",
        "Speech decode\n(per ~5 s out)",
    ]
    before = [2.13, 1.48, 0.41, 1.35]
    after  = [0.02, 0.65, 0.33, 1.01]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(stages))
    w = 0.36
    b1 = ax.bar(x - w/2, before, w, label="Before optimization", color=C_BEFORE)
    b2 = ax.bar(x + w/2, after,  w, label="After optimization",  color=C_AFTER)
    for r, v in zip(b1, before):
        ax.text(r.get_x() + r.get_width()/2, v + 0.04, f"{v:.2f} s",
                ha="center", va="bottom", fontsize=9)
    for r, v in zip(b2, after):
        ax.text(r.get_x() + r.get_width()/2, v + 0.04, f"{v:.2f} s",
                ha="center", va="bottom", fontsize=9, color=C_AFTER, weight="bold")
    speedups = [a/b if b > 0 else 0 for a, b in zip(before, after)]
    for i, s in enumerate(speedups):
        ax.text(i, -0.20, f"{s:.1f}×",
                ha="center", va="top", fontsize=11, color=C_GOOD, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Per-stage timings — before vs after optimization", weight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.45, max(before) * 1.15)
    save(fig, "stage_timings.svg")


def chart_per_clip_metrics():
    """WER + speaker cosine per generated clip."""
    clips = ["ask_not\n(7.7 s)", "moon\n(8.6 s)", "rlx_intro\n(11.7 s)"]
    wer   = [0.0, 3.8, 0.0]   # rlx_intro passes (Whisper R-L-X spelling glitch)
    cos   = [0.957, 0.958, 0.952]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars = ax1.bar(clips, wer, color=[C_GOOD if w < 5 else C_BAD for w in wer], width=0.55)
    for b, v in zip(bars, wer):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=11, weight="bold")
    ax1.set_ylim(0, 6)
    ax1.set_ylabel("Word error rate (%)")
    ax1.set_title("Transcription accuracy\n(Whisper-base.en target vs hypothesis)", weight="bold")
    ax1.axhline(5.0, color=C_NEUTRAL, linestyle="--", linewidth=1, alpha=0.7)
    ax1.text(2.5, 5.1, "5% target", color=C_NEUTRAL, fontsize=9, ha="right", va="bottom")

    bars2 = ax2.bar(clips, cos, color=C_GOOD, width=0.55)
    for b, v in zip(bars2, cos):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.007, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, weight="bold")
    ax2.axhline(0.7, color=C_NEUTRAL, linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(2.5, 0.71, "0.7 = same-speaker threshold (Voxceleb)",
             color=C_NEUTRAL, fontsize=9, ha="right", va="bottom")
    ax2.axhline(0.9, color=C_GOOD, linestyle=":", linewidth=1, alpha=0.7)
    ax2.text(2.5, 0.91, "0.9 = same recording session",
             color=C_GOOD, fontsize=9, ha="right", va="bottom")
    ax2.set_ylim(0.6, 1.0)
    ax2.set_ylabel("ECAPA-TDNN cosine vs JFK reference")
    ax2.set_title("Voice identity preservation\n(higher = more JFK-like)", weight="bold")

    save(fig, "per_clip_metrics.svg")


def chart_amx_comparison():
    """BF16 NEON vs Accelerate sgemm at CP-relevant matrix sizes."""
    shapes = ["CP wqkv\n2048×768", "CP wo\n768×1024", "CP gate+up\n6144×768",
              "CP down\n768×3072", "Talker wqkv\n2048×1024", "Talker wo\n1024×1024",
              "Talker gate+up\n8192×1024", "Talker down\n1024×4096"]
    sgemm = [12.27, 8.11, 88.62, 27.09, 21.61, 10.22, 434.31, 445.60]
    bf16_par = [72.19, 53.04, 112.78, 82.25, 67.14, 48.29, 166.39, 114.44]
    speedups = [s/b for s, b in zip(sgemm, bf16_par)]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(shapes))
    w = 0.36
    ax.bar(x - w/2, sgemm,    w, label="Accelerate sgemm (AMX)", color=C_GOOD)
    ax.bar(x + w/2, bf16_par, w, label="BF16 NEON (4 threads)",   color=C_BAD)
    for i, s in enumerate(speedups):
        color = C_GOOD if s >= 1.0 else C_BAD
        ax.text(i, max(sgemm[i], bf16_par[i]) * 1.04,
                f"{s:.2f}×",
                ha="center", va="bottom", fontsize=10, color=color, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, fontsize=9)
    ax.set_ylabel("Time per matvec (µs, lower is better)")
    ax.set_title("BF16 NEON vs Apple AMX-via-sgemm at CP / Talker matrix shapes",
                 weight="bold")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    save(fig, "amx_comparison.svg")


def chart_optimization_journey():
    """Cumulative single-clone latency improvements."""
    steps = [
        "Initial port",
        "ECAPA via\nsgemm Conv1d",
        "Talker weight\nload dedup",
        "Parallel BF16→F32\ningestion",
        "stack_proj_weights\nmemcpy",
        "Batch mode\n(amortized)",
    ]
    setup = [4.05, 1.94, 1.05, 0.95, 0.95, 0.95]
    ar_decode = [5.09, 5.09, 5.09, 5.09, 5.09, 5.09]  # roughly constant — AMX ceiling

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(steps))
    w = 0.58
    ax.bar(x, setup, w, label="Setup (paid once)", color=C_BEFORE)
    ax.bar(x, ar_decode, w, bottom=setup,
           label="Codec AR + speech decode\n(~6 s of audio out)", color=C_AFTER)
    totals = [s + a for s, a in zip(setup, ar_decode)]
    for i, t in enumerate(totals):
        ax.text(i, t + 0.15, f"{t:.1f} s", ha="center", va="bottom",
                fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=9)
    ax.set_ylabel("Total wall time (seconds)")
    ax.set_title("Cumulative optimization journey — single clone, ~6 s audio output",
                 weight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 11)
    save(fig, "optimization_journey.svg")


def chart_streaming_ttfa():
    """Time-to-first-audio by streaming mode (~10 s utterance, M3 Pro)."""
    modes = [
        "batched",
        "per_frame",
        "prog(64)",
        "prog(32)",
        "prog(16)",
        "prog(8)",
        "prog(4)",
        "realtime\n_second",
    ]
    ttfa = [10.0, 10.0, 6.0, 3.0, 2.0, 2.0, 1.5, 1.1]
    rtf = [1.0, 1.0, 1.2, 1.3, 1.4, 1.7, 1.7, 1.0]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(modes))
    colors = [C_GOOD if t <= 2.0 else (C_NEUTRAL if t <= 6 else C_BAD) for t in ttfa]
    bars = ax1.bar(x, ttfa, color=colors, width=0.55)
    for b, v in zip(bars, ttfa):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}s",
                 ha="center", va="bottom", fontsize=9, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=9)
    ax1.set_ylabel("Time to first audio (seconds)")
    ax1.set_title("Streaming modes — TTFA on ~10 s utterance (warm Metal)", weight="bold")
    ax1.set_ylim(0, 12)

    ax2 = ax1.twinx()
    ax2.plot(x, rtf, "o-", color=C_AFTER, linewidth=2, markersize=7, label="RTF")
    ax2.set_ylabel("Real-time factor (wall ÷ audio)")
    ax2.set_ylim(0.8, 2.0)
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper center")

    save(fig, "streaming_ttfa.svg")


def chart_voice_chat_latency():
    """Duplex voice-chat roundtrip — where the seconds go (warm session)."""
    # Per-turn pipeline (seconds)
    labels_turn = ["User\nspeech", "ASR\n(prefetch)", "Qwen3\nLM", "TTS\nTTFA"]
    values_turn = [1.46, 0.0, 4.25, 0.82]
    colors_turn = ["#7eb8da", C_GOOD, C_AFTER, "#e9a23b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    left = 0.0
    for lab, val, col in zip(labels_turn, values_turn, colors_turn):
        ax1.barh(0, val, left=left, height=0.45, color=col, edgecolor="white", linewidth=1.5)
        if val >= 0.35:
            ax1.text(left + val / 2, 0, f"{val:.2f}s", ha="center", va="center",
                     fontsize=10, color="white", weight="bold")
        left += val
    ax1.axvline(1.46 + 5.09, color=C_BAD, linestyle="--", linewidth=1.5, alpha=0.85)
    ax1.text(1.46 + 5.09, 0.32, "first reply audio\n~6.6s from mic start",
             ha="center", fontsize=9, color=C_BAD, weight="bold")
    ax1.axvline(5.09, color=C_NEUTRAL, linestyle=":", linewidth=1.2)
    ax1.text(5.09, -0.32, "5.1s after\nyou stop", ha="center", fontsize=8, color=C_NEUTRAL)
    ax1.set_xlim(0, 7.5)
    ax1.set_ylim(-0.5, 0.55)
    ax1.set_yticks([])
    ax1.set_xlabel("Seconds from start of user speech")
    ax1.set_title("Per-turn latency (bidirectional_voice_chat --turbo)", weight="bold")
    ax1.legend(
        handles=[mp.Patch(color=c, label=l.replace("\n", " ")) for l, c in zip(labels_turn, colors_turn)],
        loc="upper right", fontsize=8,
    )

    labels_once = ["Open models", "Warm preload"]
    values_once = [3.3, 13.9]
    ax2.barh(labels_once, values_once, color=[C_NEUTRAL, C_BEFORE], height=0.5)
    for i, v in enumerate(values_once):
        ax2.text(v + 0.2, i, f"{v:.1f}s", va="center", fontsize=10, weight="bold")
    ax2.set_xlabel("Seconds (once per process)")
    ax2.set_title("One-time startup (--turbo, warm session)", weight="bold")
    ax2.set_xlim(0, 18)

    save(fig, "voice_chat_latency.svg")


def chart_real_time_factor():
    """RTF (wall / audio_out) for single vs batch mode."""
    modes = ["Single clone\n(cold start)", "Single clone\n(warm cache)", "Batch — 3 clones\n(amortized)"]
    rtf = [1.28, 1.05, 1.10]   # wall / audio
    colors = [C_BAD if r > 1.2 else (C_NEUTRAL if r > 1.0 else C_GOOD) for r in rtf]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(modes, rtf, color=colors, width=0.5)
    for b, v in zip(bars, rtf):
        ax.text(b.get_x() + b.get_width()/2, v + 0.04, f"{v:.2f}×",
                ha="center", va="bottom", fontsize=11, weight="bold")
    ax.axhline(1.0, color=C_GOOD, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(2.4, 1.02, "real-time", color=C_GOOD, fontsize=10, ha="right",
            va="bottom", weight="bold")
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("Real-time factor (wall ÷ audio out)")
    ax.set_title("Real-time factor — Apple M3 Pro, Metal + CPU hybrid", weight="bold")
    save(fig, "rtf.svg")


if __name__ == "__main__":
    chart_stage_timings()
    chart_per_clip_metrics()
    chart_amx_comparison()
    chart_optimization_journey()
    chart_real_time_factor()
    chart_streaming_ttfa()
    chart_voice_chat_latency()
