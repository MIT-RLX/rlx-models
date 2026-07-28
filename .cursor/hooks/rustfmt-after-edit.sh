#!/usr/bin/env bash
# afterFileEdit / afterTabFileEdit: rustfmt edited .rs files and record them
# for the stop-hook clippy gate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MARKER_DIR="${TMPDIR:-/tmp}/rlx-models-cursor-lint"
MARKER="$MARKER_DIR/touched-rs.txt"
mkdir -p "$MARKER_DIR"

INPUT="$(cat || true)"
FILE_PATH="$(printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print(data.get("file_path") or "")
')"

if [ -z "$FILE_PATH" ]; then
  printf '%s\n' '{}'
  exit 0
fi

case "$FILE_PATH" in
  *.rs) ;;
  *)
    printf '%s\n' '{}'
    exit 0
    ;;
esac

if [ ! -f "$FILE_PATH" ]; then
  printf '%s\n' '{}'
  exit 0
fi

# Prefer cargo-aware rustfmt when available; fall back to rustfmt.
if command -v rustfmt >/dev/null 2>&1; then
  rustfmt --edition 2024 "$FILE_PATH" >/dev/null 2>&1 || rustfmt "$FILE_PATH" >/dev/null 2>&1 || true
fi

# Only track in-tree workspace files for the stop-hook clippy gate.
case "$FILE_PATH" in
  "$ROOT"/*)
    touch "$MARKER"
    if ! grep -Fqx "$FILE_PATH" "$MARKER" 2>/dev/null; then
      printf '%s\n' "$FILE_PATH" >>"$MARKER"
    fi
    ;;
esac

printf '%s\n' '{}'
exit 0
