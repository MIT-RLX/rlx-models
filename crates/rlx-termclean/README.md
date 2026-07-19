# rlx-termclean

Extract clean text from terminal/TUI screens — drop the **chrome** (borders,
ANSI, box-drawing, padding, scrollbars, status/help bars), keep the **content** —
and reconstruct whole documents from scrolled captures. Built to run **thousands
of live sessions at once**: pure-`std`, branch-light, batched, multicore.

The production path is **rule-based code** (a char-class classifier the ML
rediscovered), with an optional **Metal-trained tagger** for the last few points
of accuracy on ambiguous panels/dashboards.

## Pipeline

```
raw frame ──▶ fastclean ──▶ [typeclass] ──▶ stitch ──▶ whole document
             drop chrome    route by type   overlap+vote  (scroll-reconstructed,
                                                            error-corrected)
```

## Components

### `fastclean` — per-frame cleaner (pure `std`)
Strips ANSI + chrome-glyph classes (box-drawing `U+2500–259F`, blocks, braille) +
column padding + ≥4-char border runs + bare pager prompts (`:`, `(END)`, `~`);
keeps the text span between the first and last real character.
**content-F1 0.945** on 32 real apps at **467k frames/sec, 155 MB/s single core**
(2.14 µs/frame); `clean_batch` reuses scratch buffers and `clean_batch_par` fans
across cores.

### `typeclass` — 5-way line classifier (pure `std`)
`JSON | code | text | file | UI`, ~96–98% accuracy, 1–2M lines/sec — routes each
line (drop UI, keep/tag the rest).

### `stitch` — whole-document reconstruction (pure `std`)
A scrolling TUI shows a *moving window* over a document; consecutive captures
overlap. `stitch` aligns each frame to the accumulated document by its scroll
overlap and adds only the newly-revealed lines:

- **Both scroll directions** — down → append, up → prepend — plus revisit dedup
  and unrelated-jump handling.
- **Error correction** — overlapping frames give redundant copies of each line,
  so majority voting (line-level, then per-column) votes out a transient glitch
  present in a minority of frames, *even when no single frame is fully clean*.
- **Streaming** `Stitcher` — `push`/`push_raw` one frame at a time; O(H²) per
  frame in the frame height H, independent of document length.
- **Batched + parallel** — `stitch_sessions_par` reconstructs many independent
  sessions across cores (pure-`std` oversubscribed threads).

Throughput: **310k frames/sec across 14 cores**; 1000 live sessions reconstructed
per frame-tick in **~3 ms**.

### tagger — ML content/chrome model (optional, Metal)
A char-embedding + `NLAYERS`-deep **ReZero-gated, tanh-bounded-attention** encoder
with a per-char content/chrome head and a **vertical-consistency** input feature,
trained on **real captures from 32 TUI apps**. **content-F1 0.967** — beats the
rule (0.945) on the hard side-by-side-panel and dashboard apps. Trains on Metal
and ships as a loadable weights bundle (see [`weights/`](weights/README.md)).

## Benchmarks

| stage | accuracy | throughput |
|-------|----------|------------|
| `fastclean` | content-F1 0.945 | 467k frames/s (1 core) |
| `typeclass` | 96–98% | 1–2M lines/s |
| `stitch` (clean + reconstruct) | exact reconstruction | 40k fps (1 core) / 310k (14 cores) |
| tagger (Metal) | content-F1 0.967 | 4.3M chars/s inference |

## Binaries

| binary | what it does |
|--------|--------------|
| `rlx-termclean-gen` | generate the synthetic dataset → `./data` |
| `rlx-termclean-bench-clean` | cleaner throughput + accuracy |
| `rlx-termclean-bench-type` | classifier throughput + accuracy |
| `rlx-termclean-stitch <frames-dir>` | reconstruct a document from scrolled frame captures |
| `rlx-termclean-bench-stitch` | streaming + batched stitch throughput |
| `rlx-termclean-ablation-par` | parallel-strategy ablation (workers × granularity × skew) |
| `rlx-termclean-train-multi` *(feature `train`)* | train the tagger on real captures; `--save <dir>` writes a weights bundle |

## Features

- **default** — pure-`std`, zero dependencies (fast compile, no workspace
  resolution risk).
- **`train`** — the Metal tagger training stack (`rlx-tensor`).
- **`rayon`** — work-stealing parallel variants. The *default* parallelism is
  pure-`std` oversubscribed threads, which the ablation shows **matches or beats
  rayon** on both uniform and skewed workloads here — so rayon is opt-in only.

---

## The synthetic data engine

The task has a **known corruption function** — rendering clean text *into* a TUI —
so we invert it: generate clean text, render it into a realistic terminal screen,
and record provenance as we draw. Every example is perfectly labeled with **no
human annotation**.

`cargo run -p rlx-termclean --bin rlx-termclean-gen --release` writes into
`./data`: `train.jsonl` / `val.jsonl` / `test.jsonl` (default 20000/2500/2500),
`preview.txt` (rendered screens, raw ANSI kept), and `manifest.json`. Override
with `-- --train N --val N --test N --seed U64 --out DIR`. Fully reproducible from
`--seed`, so the JSONL is not committed.

### Record schema

```json
{
  "id": 22501,
  "kind": "table",          // layout family
  "content_type": "table",  // prose | code | log | kv | table | list | label
  "width": 46, "ansi": false, "style": "box=heavy",
  "input":  "…rendered screen with chrome + content + ANSI…",
  "target": "…clean reflowed text…",
  "tags":   "XXCCCCXX…"     // one marker per input char: C=content, X=chrome
}
```

**Invariant (asserted in tests):** `input` and `tags` have identical
Unicode-scalar length; `tags` is only `C`/`X`. `target` is *not* char-aligned to
`input` (reflow, panel reorder, padding-collapse break 1:1). Two supervision
signals: `tags` → extractive per-char head; `target` → seq2seq reflow decoder.

**Faithfulness rule:** `target` never contains text not visible on screen —
truncated (`…`) lines keep only the visible prefix, so the model is never trained
to hallucinate cut-off characters.

### Layout families

`panel` (bordered), `wrap` (reflow), `table`, `keyvalue`, `list`, `code`
(line-number gutter), `statusbar`, `truncate`, `split` (de-interleave two
panels), `progress` (bars/spinners), plus interaction states `tab` and `scroll`
(scrollbar gutter + "N more"). Chrome is drawn from a deliberately broad glyph set
(`src/symbols.rs`: 6 box styles, blocks/shading, braille+ASCII spinners, 12
bullets, arrows, 30 SGR strings); content is peppered with unicode (`src/corpus.rs`)
so the model learns those glyphs are content to keep.

## Design notes

- **Pure `std`, zero dependencies** (default). RNG is seeded SplitMix64
  (`src/rng.rs`); JSONL is hand-serialized (`src/record.rs`); parallelism uses
  `std::thread::scope` (no rayon).
- Tests (`cargo test -p rlx-termclean`) cover the dataset alignment invariant,
  the cleaner, and the stitcher (exact reconstruction both directions, dedup,
  error-correction voting, streaming≡batch, parallel≡sequential).

## Known coverage gaps

Horizontal scroll (column-axis stitching), append-only streams (`tail -f`),
independent per-panel scroll (mc/lazygit), wide/CJK double-width alignment for
char-voting, terminal-resize reflow, and a dynamic-vs-document gate (so live
monitors like `htop` aren't mis-stitched).
