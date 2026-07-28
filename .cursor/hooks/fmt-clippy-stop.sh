#!/usr/bin/env bash
# stop: after an agent turn that edited Rust, run fmt+clippy and auto-follow-up
# with failures so the agent can fix them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MARKER_DIR="${TMPDIR:-/tmp}/rlx-models-cursor-lint"
MARKER="$MARKER_DIR/touched-rs.txt"

INPUT="$(cat || true)"

eval "$(printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
status = data.get("status") or "completed"
loop_count = int(data.get("loop_count") or 0)
# Emit shell-safe assignments
import shlex
print("STATUS=" + shlex.quote(str(status)))
print("LOOP_COUNT=" + shlex.quote(str(loop_count)))
')"

emit_empty() {
  printf '%s\n' '{}'
  exit 0
}

# Only enforce on clean completions.
if [ "$STATUS" != "completed" ]; then
  emit_empty
fi

# Cap auto-follow-ups (hooks.json also sets loop_limit).
if [ "${LOOP_COUNT:-0}" -ge 3 ]; then
  emit_empty
fi

if [ ! -s "$MARKER" ]; then
  emit_empty
fi

# Snapshot + clear marker so success paths do not re-trigger forever.
FILES="$(tr '\n' ' ' <"$MARKER" | sed 's/[[:space:]]*$//')"
cp "$MARKER" "$MARKER.prev" 2>/dev/null || true
: >"$MARKER"

if [ -z "$FILES" ]; then
  emit_empty
fi

# Ensure cargo/rustfmt from the usual install locations are visible.
export PATH="${HOME}/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"

OUT_FILE="$(mktemp "${TMPDIR:-/tmp}/rlx-models-lint.XXXXXX")"
set +e
# shellcheck disable=SC2086
./scripts/rust-lint-gate.sh --files $FILES >"$OUT_FILE" 2>&1
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  rm -f "$OUT_FILE" "$MARKER.prev"
  emit_empty
fi

# Restore marker so the next corrective turn still scopes correctly.
if [ -f "$MARKER.prev" ]; then
  cat "$MARKER.prev" >>"$MARKER"
  rm -f "$MARKER.prev"
fi

python3 - "$OUT_FILE" "$RC" <<'PY'
import json, sys
path, code = sys.argv[1], int(sys.argv[2])
with open(path, "r", errors="replace") as f:
    out = f.read()
# Keep follow-up payload bounded for context.
out = out[-12000:]
msg = (
    "Automated rust lint gate failed after your last turn "
    "(`scripts/rust-lint-gate.sh`, same bar as `just lint` / publish).\n\n"
    f"Exit code: {code}\n\n"
    "Fix fmt/clippy issues below, then continue. Do not git commit.\n\n"
    "```text\n"
    f"{out}\n"
    "```\n"
)
print(json.dumps({"followup_message": msg}, ensure_ascii=False))
PY
rm -f "$OUT_FILE"
exit 0
