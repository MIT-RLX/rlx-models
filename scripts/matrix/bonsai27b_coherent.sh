#!/usr/bin/env bash
# Run Bonsai-27B on every available RLX backend and require coherent
# multi-word detokenized output.
#
# Usage:
#   scripts/matrix/bonsai27b_coherent.sh              # local backends
#   RLX_CUDA_HOST=user@host scripts/matrix/bonsai27b_coherent.sh     # remote (expects synced trees)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GGUF="${BONSAI_GGUF:-$REPO/weights/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf}"
PROMPT="${BONSAI_PROMPT:-What is the capital of France? Reply with one short sentence.}"
SYSTEM="${BONSAI_SYSTEM:-You are a helpful assistant. Answer clearly in English.}"
MAX_TOKENS="${BONSAI_MAX_TOKENS:-32}"
MAX_SEQ="${BONSAI_MAX_SEQ:-64}"
OUT_DIR="${BONSAI_OUT:-$HERE/out/bonsai27b}"
FEATURES="${BONSAI_FEATURES:-runner,minicpm5,bonsai,qwen35,apple-silicon}"

mkdir -p "$OUT_DIR"
cd "$REPO"

if [ ! -s "$GGUF" ]; then
  echo "missing GGUF: $GGUF" >&2
  exit 2
fi

export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export RLX_LOW_MEM_COMPILE=1
export RLX_DEQUANT_CACHE=0
export RLX_CUDA_NO_CUDNN="${RLX_CUDA_NO_CUDNN:-1}"

echo ">> building rlx-run ($FEATURES)"
cargo build --release -p rlx-models --bin rlx-run --features "$FEATURES" \
  2>&1 | tee "$OUT_DIR/build.log" | tail -5

BIN="$REPO/target/release/rlx-run"
# Discover backends: prefer explicit BONSAI_BACKENDS, else probe.
if [ -n "${BONSAI_BACKENDS:-}" ]; then
  IFS=',' read -r -a BACKENDS <<< "$BONSAI_BACKENDS"
else
  BACKENDS=(cpu)
  for d in metal mlx coreml wgpu cuda vulkan; do
    if "$BIN" bonsai --help >/dev/null 2>&1; then
      # probe by dry-run inspect of device parse via a 0-token attempt is heavy;
      # use runtime availability printed by a tiny one-shot when possible.
      :
    fi
    case "$d" in
      metal)  [[ "$(uname -s)" == Darwin ]] && BACKENDS+=(metal) ;;
      mlx)    [[ "$(uname -s)" == Darwin ]] && BACKENDS+=(mlx) ;;
      coreml) [[ "$(uname -s)" == Darwin ]] && BACKENDS+=(coreml) ;;
      wgpu)   BACKENDS+=(wgpu) ;;
      cuda)   command -v nvidia-smi >/dev/null && BACKENDS+=(cuda) ;;
      vulkan) command -v vulkaninfo >/dev/null 2>&1 && BACKENDS+=(vulkan) ;;
    esac
  done
fi

echo ">> backends: ${BACKENDS[*]}"
PASS=0
FAIL=0
SUMMARY="$OUT_DIR/summary.md"
{
  echo "# Bonsai-27B coherent backend check"
  echo
  echo "- gguf: \`$GGUF\`"
  echo "- prompt: $PROMPT"
  echo "- max_tokens: $MAX_TOKENS  max_seq: $MAX_SEQ"
  echo
  echo "| backend | status | words | text |"
  echo "|---------|--------|-------|------|"
} > "$SUMMARY"

word_count() {
  # Count whitespace-separated tokens with at least one letter.
  python3 -c 'import re,sys; t=sys.stdin.read(); print(len(re.findall(r"[A-Za-z][A-Za-z'\''-]*", t)))'
}

for dev in "${BACKENDS[@]}"; do
  log="$OUT_DIR/${dev}.log"
  echo ">> === $dev ==="
  set +e
  "$BIN" bonsai --weights "$GGUF" --packed --device "$dev" \
    --max-seq "$MAX_SEQ" --max-tokens "$MAX_TOKENS" \
    --temperature 0.0 --seed 0 \
    --system "$SYSTEM" \
    --prompt "$PROMPT" \
    >"$log" 2>&1
  rc=$?
  set -e

  text="$(python3 - <<PY
import re, pathlib
p = pathlib.Path("$log")
s = p.read_text(errors="replace")
m = re.search(r"\[rlx-qwen35\] qwen35: text>\n(.*?)(?:\n\[|\Z)", s, re.S)
if m:
    print(m.group(1).strip())
else:
    m2 = re.search(r'\[rlx-qwen35\] qwen35: text: "(.*)"', s)
    if m2:
        print(m2.group(1).encode("utf-8").decode("unicode_escape"))
PY
)"
  words="$(printf '%s' "$text" | word_count)"
  # Coherence: at least 2 alphabetic words, and not empty/oom/panic.
  ok=0
  if [ "$rc" -eq 0 ] && [ "${words:-0}" -ge 2 ] && [ -n "$text" ]; then
    ok=1
  fi
  if [ "$ok" -eq 1 ]; then
    echo "PASS $dev ($words words): $text"
    echo "| $dev | PASS | $words | $(printf '%s' "$text" | tr '|' '/' | tr '\n' ' ') |" >> "$SUMMARY"
    PASS=$((PASS + 1))
  else
    echo "FAIL $dev rc=$rc words=${words:-0}"
    echo "| $dev | FAIL (rc=$rc) | ${words:-0} | _(see ${dev}.log)_ |" >> "$SUMMARY"
    FAIL=$((FAIL + 1))
    tail -20 "$log" || true
  fi
done

echo
echo ">> result: $PASS passed, $FAIL failed — $SUMMARY"
cat "$SUMMARY"
[ "$FAIL" -eq 0 ] && [ "$PASS" -gt 0 ]
