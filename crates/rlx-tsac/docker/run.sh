#!/usr/bin/env bash
# Docker image with Bellard `tsac` + standalone `tsac-ng` only (linux/amd64).
# RLX native codec and perf orchestration run on the host.
#
#   bash crates/rlx-tsac/docker/run.sh build
#   cargo run -p rlx-tsac --example perf_bench --release --features native-codec -- \
#     --in-wav speech.wav
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${RLX_TSAC_REF_IMAGE:-rlx-tsac-ref}"
PLATFORM="${RLX_TSAC_DOCKER_PLATFORM:-linux/amd64}"

build_image() {
  echo "building ${IMAGE} (${PLATFORM}) from ${CRATE_ROOT} ..."
  local backup=""
  if [[ -f "$CRATE_ROOT/.dockerignore" ]]; then
    backup="$(mktemp)"
    cp "$CRATE_ROOT/.dockerignore" "$backup"
  fi
  cp "$SCRIPT_DIR/dockerignore" "$CRATE_ROOT/.dockerignore"
  cleanup_ignore() {
    if [[ -n "$backup" ]]; then
      mv "$backup" "$CRATE_ROOT/.dockerignore"
    else
      rm -f "$CRATE_ROOT/.dockerignore"
    fi
  }
  trap cleanup_ignore RETURN

  docker build --platform "$PLATFORM" \
    -t "$IMAGE" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$CRATE_ROOT"
}

if [[ "${1:-}" == "build" ]]; then
  build_image
  exit 0
fi

if [[ "${1:-}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage:
  bash crates/rlx-tsac/docker/run.sh build
  cargo run -p rlx-tsac --example perf_bench --release --features native-codec -- --in-wav WAV

Docker runs reference binaries only (bellard + tsac-ng). RLX runs on the host.

Env:
  RLX_TSAC_REF_IMAGE          image tag (default: rlx-tsac-ref)
  RLX_TSAC_DOCKER_PLATFORM    platform (default: linux/amd64)
EOF
  exit 0
fi

echo "run perf bench on the host — Docker is reference-only:" >&2
echo "  cargo run -p rlx-tsac --example perf_bench --release --features native-codec -- --in-wav WAV" >&2
exit 1
