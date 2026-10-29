#!/usr/bin/env bash
# Upstream Indra outputs for tests/indra_reference.rs (one .jsonl per image).
#
#   pip install indra-ocr==1.1.2
#   crates/rlx-indra/scripts/gen_reference.sh <upstream-model-dir> <out-dir> IMAGE...
#
# The model dir is the datalab-to/indra-ocr v1.1.0 snapshot (the ONNX files);
# run the test with INDRA_REFERENCE_DIR=<out-dir>.
set -euo pipefail
models="$1"; out="$2"; shift 2
mkdir -p "$out"
INDRA_OCR_MODEL_DIR="$models" indra-ocr "$@" --format json --cores 1 --out "$out"
