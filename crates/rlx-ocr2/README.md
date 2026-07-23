# rlx-ocr2

A native [RLX](https://github.com/) OCR pipeline — no ONNX / `ort` runtime. Every
model stage is built as a native `rlx-ir` graph, so the whole pipeline runs on any
RLX backend: **CPU, Metal, MLX, CUDA, wgpu, Vulkan, CoreML/ANE**.

```text
image ─▶ detector ─▶ line grouping ─▶ per-line recognizer ─▶ [rescoring] ─▶ text
        (heatmaps)      (boxes)          (CRNN + CTC)        (n-gram + lexicon)
```

## Stages

| stage | what it is |
|-------|------------|
| **detector** (`detection`) | CRAFT-style text detector — grouped-conv encoder + squeeze-excite + FPN, 7 heatmap heads. The pipeline builds a head-pruned graph (only `region_score` + `link_score_horizontal`, ~28 % fewer ops). |
| **grouping** (`grouping`) | region + link heatmaps → text-line bounding boxes. |
| **recognizer** (`recognition` / `runner`) | CRNN + CTC — VGG conv front → 2× bidirectional LSTM(128) → FC → 439 classes, greedy- or beam-decoded. Compiled per line width and cached. |
| **rescoring** (`beam` / `ngram` / `rescore`) | CTC prefix-beam N-best, each candidate rescored by a memory-mapped n-gram model + a lexicon trie; best wins. Fixes single-character slips (e.g. `Worlc → World`). |

## Usage

### Library

```rust
use rlx_ocr2::{Ocr2, Rescorer};
use rlx_runtime::Device;

let ocr = Ocr2::load(recipe, det_weights, rec_weights, codemap, Device::Metal)?
    .with_rescorer(Rescorer::load_en(Some(ngram_bin), Some(lexicon_tsv))?);

for line in ocr.recognize_image(image_path)? {
    let (x0, y0, x1, y1) = line.bbox;
    println!("[{x0},{y0} {x1},{y1}] {}", line.text);
}
```

### CLI

```sh
# full page, with correction, on Metal
OCR2_DEVICE=metal cargo run --release -p rlx-ocr2 -- \
    image recipe.json det.safetensors rec.safetensors codemap.txt page.png ngram.bin lexicon.tsv

# a single pre-cropped text line
cargo run --release -p rlx-ocr2 -- line rec.safetensors codemap.txt line.png
```

The optional `ngram.bin` / `lexicon.tsv` arguments enable beam-search correction;
omit them for raw CTC output.

## Backends

Default is CPU. Select a GPU backend with a feature flag:

```sh
cargo build -p rlx-ocr2 --features metal      # or: mlx, cuda, gpu (wgpu), vulkan, coreml
```

Convenience groups: `all-backends`, `metal-mlx`, plus `blas-accelerate` for a faster
CPU BLAS on Apple.

## Environment knobs

| var | effect |
|-----|--------|
| `OCR2_DEVICE` | backend for the CLI: `cpu` (default) `metal` `mlx` `cuda` `gpu` `vulkan` `coreml` |
| `OCR2_TIMING` | print per-stage timings |
| `OCR2_REPEAT` | run the CLI pipeline N times in-process (warm-timing aid) |
| `OCR2_NO_FUSION` | disable conv+bias+act fusion in the detector |
| `OCR2_LEX_W` | override the lexicon rescoring weight (default `3.0`) |
| `OCR2_RESCORE_DEBUG` | print each beam candidate's component scores |

## Assets

| file | role |
|------|------|
| `recipe.json` | detector op recipe (interpreted into an rlx-ir graph) |
| `det.safetensors`, `rec.safetensors` | detector / recognizer weights |
| `codemap.txt` | recognizer class index → Unicode codepoint (blank ≥ `0xFFFE`) |
| `ngram.bin` | packed, memory-mapped n-gram model (optional) |
| `lexicon.tsv` | `word \t log-prob` lexicon (optional) |

The n-gram model uses a compact, zero-copy **memory-mapped** layout (magic
`RLXNGRM1`): a fixed header + a sorted, fixed-width record array read straight from
the `mmap` via `bytemuck` and searched by binary search — no heap parse. See the
format spec in [`src/ngram.rs`](src/ngram.rs).

## Tests

Numeric parity tests (`tests/`) check each stage against stored fixtures. The
detector/recognizer tests are env-gated on fixture paths; the n-gram test ships a
fixture and always runs:

```sh
cargo test -p rlx-ocr2
```

## License

See the workspace license.
