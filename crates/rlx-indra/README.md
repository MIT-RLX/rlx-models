# rlx-indra

[Indra OCR](https://github.com/datalab-to/indra) (Datalab) on RLX: page images in,
markdown / text / JSON lines with boxes out.

Every network is a **native RLX graph** over safetensors weights, and the page
pipeline around them is **Rust**. There is no ONNX Runtime, no OpenCV and no Python at
inference time; ONNX Runtime appears only in the parity tests, as the reference.

| Network | Upstream file | Graph |
|---|---|---|
| line detector (MobileNet + FPN, pixel shuffle) | `det.onnx` | [`graph/det.rs`](src/graph/det.rs) |
| line recognizers: base (tall-window context branch), CJK, RTL, world | `rec*.onnx` | [`graph/rec.rs`](src/graph/rec.rs) |
| script router | `script.onnx` | [`graph/classifier.rs`](src/graph/classifier.rs) |
| page orientation | `orient.onnx` | [`graph/classifier.rs`](src/graph/classifier.rs) |
| reading order + layout heads (transformer) | `order.onnx` | [`graph/order.rs`](src/graph/order.rs) |

Host side ([`pipeline.rs`](src/pipeline.rs)): deskew search and orientation, adaptive
re-detection, line crops ([`lines.rs`](src/lines.rs)), script routing with the
unsure-line fallback, CTC greedy decoding plus the n-gram beam search for unsure lines
([`lm.rs`](src/lm.rs)), gutter and gap splitting, list-marker joining, right-to-left
reordering ([`text.rs`](src/text.rs)), reading order ([`order.rs`](src/order.rs)) and
markdown with HTML tables ([`layout.rs`](src/layout.rs)). The OpenCV operations the
upstream pipeline uses (`resize`, `warpAffine`, connected components) are reimplemented
to OpenCV's fixed-point arithmetic in [`imgproc.rs`](src/imgproc.rs).

## Usage

```bash
just fetch-indra                     # download v1.1.0, export to .cache/indra
just indra -- page.png               # markdown to stdout
just indra -- scan.jpg --format json --device metal
```

`rlx-indra-convert <upstream-dir> <out-dir>` writes `<net>.safetensors` for each
network (initializers renamed after their module, e.g. `rec.blocks.0.se.fc1.weight`),
`lm*.safetensors` for the n-gram tables, and copies `meta.json`.

```rust
use rlx_indra::{Indra, Options, load_gray};

let ocr = Indra::load(".cache/indra", Options::default())?;
let page = ocr.read(&load_gray("page.png".as_ref())?, true)?;
println!("{}", page.markdown.unwrap_or_else(|| page.text()));
```

CLI flags: `--model-dir` (default `$INDRA_MODEL_DIR`), `--device`, `--format md|text|json`,
`--out DIR`, `--keep-furniture`, `--table-heads`, `--no-turn`, `--timings`.

## Checks

| Test | Compares | Needs |
|---|---|---|
| `indra_parity_ort` (`--features parity-ort`) | each graph vs its `.onnx` on ONNX Runtime | `INDRA_UPSTREAM_DIR`, `INDRA_MODEL_DIR` |
| `imgproc_cv_reference` | `imgproc` vs OpenCV fixtures (`scripts/gen_cv_fixtures.py`) | `INDRA_CV_FIXTURES` |
| `indra_reference` | whole pages vs upstream `indra-ocr --format json` (`scripts/gen_reference.sh`) | `INDRA_MODEL_DIR`, `INDRA_REFERENCE_DIR` |

`just test-indra-parity` runs the first; `just bench-indra` (`tests/indra_bench.rs`)
times each graph against ONNX Runtime on page-sized shapes. Measured on the release: detector max |Δ|
2.1e-5, recognizers identical argmax and top-8 with log-prob |Δ| ≤ 4.2e-5, script /
orientation |Δp| ≤ 2.7e-7, order heads |Δ| ≤ 9.1e-6 (padded batch). `warpAffine` and
`INTER_AREA` agree with OpenCV to the level (bit-exact on the enlarging and rotation
cases); 8-bit `INTER_LINEAR` differs by up to 3 levels from arm64 OpenCV builds, which
route it to a HAL (Carotene / KleidiCV) rather than the generic fixed-point path
implemented here.

## Not ported

- PDF input: the upstream converter renders pages with pdfium and can read the PDF
  text layer. Here input is images (PNG, JPEG, TIFF first frame); render PDFs first.
- Math (`--math`: `mathcls`, `math_enc`, `math_dec_*`), off by default upstream.
- CenterMask (`mask.onnx`), not in the v1.1.0 release.

Color images are reduced to gray with OpenCV's `cvtColor` weights; upstream decodes
straight to gray (libpng / libjpeg), so color inputs can differ by a gray level.

## License

Code: GPL-3.0 (this workspace). The Indra model weights are under Datalab's modified
OpenRAIL-M license (see the model card); converting them does not change it.
