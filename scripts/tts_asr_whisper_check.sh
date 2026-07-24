#!/usr/bin/env bash
# TTS → Whisper exact-match (+ optional ASR CLI) validation against Hub-published weights.
#
# Requires word-for-word transcript parity after alphanumeric normalization
# (case/punct ignored). Uses openai/whisper-base.en and the default 30 s pad path.
#
# Usage:
#   just tts-asr-whisper-check
#   TEXT='…' bash scripts/tts_asr_whisper_check.sh          # single sentence
#   SUITE=1 bash scripts/tts_asr_whisper_check.sh           # default longer suite
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WAV="${WAV:-/tmp/rlx_tts_whisper_check.wav}"
WAV16="${WAV16:-/tmp/rlx_tts_whisper_check_16k.wav}"
WHISPER_DIR="${RLX_WHISPER_DIR:-.cache/whisper-base.en}"
RUN_ID="${RUN_ID:-$$}"
WHISPER_ERR="${WHISPER_ERR:-/tmp/rlx_whisper_check_${RUN_ID}.err}"
ASR_LOG="${ASR_LOG:-/tmp/rlx_asr_check_${RUN_ID}.log}"

# Longer sentences that must round-trip 100% (override with TEXT=… for one-off).
DEFAULT_SUITE=(
  "Hello everyone."
  "Hello from our system."
  "The quick brown fox jumps over the lazy dog near the riverbank at sunrise."
  "Please open the window and let some fresh air into the room before we start the meeting this afternoon."
  "Tomorrow morning the weather will be cloudy with a chance of light rain along the coast."
  "Artificial intelligence models can turn written text into natural sounding speech for many applications."
)

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing $1" >&2; exit 1; }; }

mkdir -p weights/tts/rlx-tts weights/asr .cache/whisper-base.en

if [[ ! -f weights/tts/rlx-tts/rlx-tts.gguf ]]; then
  echo "==> fetch-rlx-tts"
  just fetch-rlx-tts
fi
if [[ ! -f weights/asr/model.gguf ]]; then
  echo "==> fetch-rlx-asr"
  just fetch-rlx-asr
fi
if [[ ! -s "$WHISPER_DIR/model.safetensors" ]]; then
  echo "==> fetch-whisper-base ($WHISPER_DIR)"
  if command -v hf >/dev/null 2>&1; then
    hf download openai/whisper-base.en --local-dir "$WHISPER_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download openai/whisper-base.en --local-dir "$WHISPER_DIR"
  else
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("openai/whisper-base.en", local_dir="$WHISPER_DIR")
PY
  fi
fi
test -s "$WHISPER_DIR/model.safetensors"
test -s "$WHISPER_DIR/config.json"
test -s "$WHISPER_DIR/tokenizer.json"

run_one() {
  local text="$1"
  local wav="$2"
  local wav16="$3"
  local run_id="$4"
  local whisper_err="/tmp/rlx_whisper_check_${run_id}.err"
  local asr_log="/tmp/rlx_asr_check_${run_id}.log"

  echo "==> rlx-tts synthesize: $text"
  cargo run -p rlx-tts --release -- --text "$text" --out "$wav"
  test -s "$wav"
  echo "    wrote $wav ($(du -h "$wav" | awk '{print $1}'))"

  echo "==> resample 24 kHz → 16 kHz (Whisper + ASR)"
  python3 - <<PY
import struct, wave, pathlib
src = pathlib.Path("$wav")
dst = pathlib.Path("$wav16")
with wave.open(str(src), "rb") as w:
    ch, sw, sr, n, _, _ = w.getparams()
    assert sw == 2
    raw = w.readframes(n)
pcm = struct.unpack("<" + "h" * (len(raw)//2), raw)
if ch == 2:
    pcm = [(pcm[i] + pcm[i+1]) // 2 for i in range(0, len(pcm), 2)]
peak = max(abs(s) for s in pcm) or 1
gain = min(0.9 * 32767 / peak, 8.0)
pcm = [max(-32767, min(32767, int(s * gain))) for s in pcm]
ratio = 16000 / sr
out_n = int(len(pcm) * ratio)
out = []
for i in range(out_n):
    x = i / ratio
    j = int(x)
    f = x - j
    a = pcm[j] if j < len(pcm) else 0
    b = pcm[j+1] if j+1 < len(pcm) else a
    out.append(int(a * (1-f) + b * f))
with wave.open(str(dst), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(struct.pack("<" + "h"*len(out), *out))
print(f"    {sr} Hz → 16000 Hz, {len(out)} samples ({len(out)/16000:.2f}s), peak_gain={gain:.2f} → {dst}")
PY

  echo "==> whisper-base.en transcribe ($WHISPER_DIR)"
  cargo run -p rlx-whisper --release --bin rlx-whisper -- \
    --weights "$WHISPER_DIR/model.safetensors" \
    --config "$WHISPER_DIR/config.json" \
    --tokenizer "$WHISPER_DIR/tokenizer.json" \
    --wav "$wav16" --lang en \
    >"/tmp/rlx_whisper_check_${run_id}.out" 2>"$whisper_err" || {
      echo "whisper failed:" >&2
      cat "$whisper_err" >&2
      exit 1
    }
  local transcript
  transcript="$(python3 - <<PY
from pathlib import Path
err = Path("$whisper_err").read_text(errors="replace")
marker = "transcribed in"
idx = err.rfind(marker)
body = err[idx:] if idx >= 0 else err
lines = body.splitlines()
if lines and "transcribed in" in lines[0]:
    lines = lines[1:]
text = " ".join(ln.strip() for ln in lines if ln.strip()).strip()
print(text)
PY
)"
  echo "    expected : $text"
  echo "    whisper  : $transcript"

  EXPECTED="$text" GOT="$transcript" python3 - <<'PY'
import os, re, sys
expected = os.environ["EXPECTED"]
got = os.environ["GOT"]

def words(s: str) -> list[str]:
    return [w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if w]

ew, gw = words(expected), words(got)
print(f"    words expected={ew}")
print(f"    words whisper ={gw}")
if ew != gw:
    for i, (a, b) in enumerate(zip(ew, gw)):
        if a != b:
            print(f"FAIL: first word mismatch at [{i}]: expected '{a}' got '{b}'", file=sys.stderr)
            break
    else:
        if len(ew) != len(gw):
            print(
                f"FAIL: length mismatch expected={len(ew)} got={len(gw)}",
                file=sys.stderr,
            )
    print("FAIL: whisper transcript is not a 100% word match", file=sys.stderr)
    sys.exit(1)
print("    whisper OK (100% word match)")
PY

  echo "==> rlx-asr transcribe"
  cargo run -p rlx-asr --release --bin rlx-asr -- \
    transcribe --wav "$wav16" 2>&1 | tee "$asr_log" | tail -5
  python3 - <<PY
from pathlib import Path
log = Path("$asr_log").read_text(errors="replace")
print("    asr log bytes:", len(log))
low = log.lower()
if "model.gguf" in low and ("not found" in low or "no such file" in low):
    raise SystemExit("FAIL: asr weight missing")
if "panic" in low:
    raise SystemExit("FAIL: asr panicked")
print("    asr CLI completed")
PY
  echo
}

if [[ -n "${TEXT:-}" ]]; then
  run_one "$TEXT" "$WAV" "$WAV16" "$RUN_ID"
  echo "PASS: TTS → Whisper 100% match + ASR CLI check"
else
  n=${#DEFAULT_SUITE[@]}
  i=0
  for text in "${DEFAULT_SUITE[@]}"; do
    i=$((i + 1))
    echo "======== suite $i/$n ========"
    run_one "$text" "/tmp/rlx_tts_suite_${i}.wav" "/tmp/rlx_tts_suite_${i}_16k.wav" "suite_${i}"
  done
  echo "PASS: TTS → Whisper 100% match on $n longer sentences + ASR CLI check"
fi
