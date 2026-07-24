#!/usr/bin/env bash
# Multi-model TTS short/long → WAV → Whisper coverage (+ optional rlx-asr CLI).
#
# Steps:
#   1. Ensure whisper-base.en + rlx-tts/rlx-asr Hub weights
#   2. Exact 100% suite for rlx-tts (scripts/tts_asr_whisper_check.sh)
#   3. rlx-tts-bench matrix: available adapters × short+long × Whisper, WAVs under OUT_DIR
#   4. Run rlx-asr on every generated WAV
#
# Env:
#   OUT_DIR      default /tmp/rlx-tts-asr-validate
#   MODELS       comma list or "available" (default) / "all"
#   FEATURES     cargo features for rlx-tts-bench (default rlx-tts,matrix-onnx,apple-silicon)
#   SKIP_ASR=1   skip rlx-asr pass
#   SKIP_EXACT=1 skip rlx-tts exact suite
#   MIN_COVERAGE default 0.5 (Whisper content-word coverage gate via --fail-under-fox for fox)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-/tmp/rlx-tts-asr-validate}"
FEATURES="${FEATURES:-rlx-tts,matrix-onnx,apple-silicon}"
MODELS="${MODELS:-available}"
MIN_FOX="${MIN_FOX:-4}"
export RLX_WHISPER_DIR="${RLX_WHISPER_DIR:-.cache/whisper-base.en}"

SHORT_TEXT="${SHORT_TEXT:-The quick brown fox jumps over the lazy dog near the riverbank at sunrise.}"
LONG_TEXT="${LONG_TEXT:-Please open the window and let some fresh air into the room before we start the meeting this afternoon. Tomorrow morning the weather will be cloudy with a chance of light rain along the coast.}"

mkdir -p "$OUT_DIR/wav" "$OUT_DIR/asr"

echo "==> weights / whisper"
just fetch-rlx-tts >/dev/null
just fetch-rlx-asr >/dev/null
if [[ ! -s "$RLX_WHISPER_DIR/model.safetensors" ]]; then
  just fetch-whisper-base
fi

if [[ "${SKIP_EXACT:-0}" != "1" ]]; then
  echo "==> rlx-tts exact Whisper suite (100% word match)"
  bash scripts/tts_asr_whisper_check.sh | tee "$OUT_DIR/rlx_tts_exact.log"
fi

echo "==> discover adapters"
LIST_OUT="$OUT_DIR/adapters.txt"
cargo run -p rlx-tts-bench --release --features "$FEATURES" -- list | tee "$LIST_OUT"

if [[ "$MODELS" == "available" ]]; then
  # Pick models whose list line contains "OK " (skip fake).
  MODELS_CSV=$(awk '/^[^ ]/ && $1!="model" && $1!="fake" && /OK / {printf "%s%s", (n++?",":""), $1}' "$LIST_OUT")
  if [[ -z "$MODELS_CSV" ]]; then
    echo "no available models with weights" >&2
    exit 1
  fi
elif [[ "$MODELS" == "all" ]]; then
  MODELS_CSV=all
else
  MODELS_CSV="$MODELS"
fi
echo "    models: $MODELS_CSV"

echo "==> synth short+long + Whisper → $OUT_DIR"
cargo run -p rlx-tts-bench --release --features "$FEATURES" -- \
  run -m "$MODELS_CSV" -d cpu \
  --phrases short,long \
  --text-short "$SHORT_TEXT" \
  --text-long "$LONG_TEXT" \
  --whisper --noise --no-isolate \
  --out-dir "$OUT_DIR" \
  --timeout-secs "${TIMEOUT_SECS:-900}" \
  2>&1 | tee "$OUT_DIR/bench.log" || {
    echo "WARN: tts-bench exited non-zero (some models may have failed); continuing" >&2
  }

echo "==> WAV inventory"
find "$OUT_DIR/wav" -type f -name '*.wav' | sort | tee "$OUT_DIR/wavs.txt"
WAV_N=$(wc -l < "$OUT_DIR/wavs.txt" | tr -d ' ')
echo "    $WAV_N wav files"

if [[ "${SKIP_ASR:-0}" != "1" ]]; then
  echo "==> rlx-asr on generated WAVs"
  ASR_OK=0
  ASR_FAIL=0
  : > "$OUT_DIR/asr/summary.tsv"
  echo -e "wav\tok\tnote" >> "$OUT_DIR/asr/summary.tsv"
  while IFS= read -r wav; do
    base=$(basename "$wav" .wav)
    # Resample to 16 kHz mono for ASR
    wav16="$OUT_DIR/asr/${base}_16k.wav"
    python3 - <<PY
import struct, wave, pathlib
src, dst = pathlib.Path("$wav"), pathlib.Path("$wav16")
with wave.open(str(src), "rb") as w:
    ch, sw, sr, n, *_ = w.getparams()
    assert sw == 2, sw
    raw = w.readframes(n)
pcm = struct.unpack("<" + "h"*(len(raw)//2), raw)
if ch == 2:
    pcm = [(pcm[i]+pcm[i+1])//2 for i in range(0,len(pcm),2)]
ratio = 16000 / sr
out_n = int(len(pcm) * ratio)
out = []
for i in range(out_n):
    x = i / ratio
    j = int(x)
    f = x - j
    a = pcm[j] if j < len(pcm) else 0
    b = pcm[j+1] if j+1 < len(pcm) else a
    out.append(int(a*(1-f)+b*f))
with wave.open(str(dst), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack("<"+"h"*len(out), *out))
PY
    log="$OUT_DIR/asr/${base}.log"
    if cargo run -p rlx-asr --release --bin rlx-asr -- \
         transcribe --wav "$wav16" >"$log" 2>&1; then
      if rg -qi 'panic|not found|no such file' "$log"; then
        echo -e "${base}\t0\terror" >> "$OUT_DIR/asr/summary.tsv"
        ASR_FAIL=$((ASR_FAIL+1))
      else
        echo -e "${base}\t1\tok" >> "$OUT_DIR/asr/summary.tsv"
        ASR_OK=$((ASR_OK+1))
      fi
    else
      echo -e "${base}\t0\texit" >> "$OUT_DIR/asr/summary.tsv"
      ASR_FAIL=$((ASR_FAIL+1))
    fi
  done < "$OUT_DIR/wavs.txt"
  echo "    asr ok=$ASR_OK fail=$ASR_FAIL"
fi

echo
echo "PASS: validation artifacts in $OUT_DIR"
echo "  - rlx_tts_exact.log (100% suite)"
echo "  - results.jsonl / BACKENDS.md / report.html"
echo "  - wav/ ($WAV_N files)"
[[ "${SKIP_ASR:-0}" != "1" ]] && echo "  - asr/summary.tsv"
