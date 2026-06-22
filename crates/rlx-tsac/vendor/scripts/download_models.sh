#!/bin/sh
# download_models.sh — Download TSAC model files for tsac-ng
#
# Sources (tried in order):
#   1. Hugging Face Hub: https://huggingface.co/Hope2333/tsac-ng-models
#   2. GitHub Releases:  https://github.com/Hope2333/tsac-ng/releases
#
# Usage:
#   ./scripts/download_models.sh [output_dir]
#
# Default output: /usr/share/tsac (requires sudo) or ./models/tsac

set -e

MODEL_DIR="${1:-/usr/share/tsac}"
HF_BASE="https://huggingface.co/Hope2333/tsac-ng-models/resolve/main"
GH_BASE="https://github.com/Hope2333/tsac-ng/releases/download/v0.1.0-models"

MODELS="
dac_mono_q8.bin
dac_stereo_q8.bin
tsac_mono_q8.bin
tsac_stereo_q8.bin
"

# Try to create target directory
mkdir -p "$MODEL_DIR" 2>/dev/null || {
    echo "Cannot create $MODEL_DIR. Trying ./models/tsac/"
    MODEL_DIR="./models/tsac"
    mkdir -p "$MODEL_DIR"
}

download_file() {
    local name="$1"
    local url="$2"
    local dest="$MODEL_DIR/$name"

    if [ -f "$dest" ]; then
        echo "  [skip] $name (already exists)"
        return 0
    fi

    echo "  Downloading $name from $url ..."
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$dest" "$url" || return 1
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$dest" "$url" || return 1
    else
        echo "  Error: wget or curl required"
        return 1
    fi
    echo "  [ok] $name"
}

echo "tsac-ng model downloader"
echo "Target: $MODEL_DIR"
echo ""

for model in $MODELS; do
    # Try Hugging Face first
    if download_file "$model" "$HF_BASE/$model?download=true"; then
        continue
    fi
    # Fall back to GitHub Release
    if download_file "$model" "$GH_BASE/$model"; then
        continue
    fi
    echo "  [FAIL] $model — could not download from any source"
    echo "  Please download manually and place in $MODEL_DIR"
done

echo ""
echo "Done. Model files in $MODEL_DIR"
ls -lh "$MODEL_DIR"/*.bin 2>/dev/null || echo "(no .bin files found)"
