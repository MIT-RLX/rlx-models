#!/usr/bin/env bash
# Run encode + decode for a reference TSAC binary inside Docker.
#
# Usage:
#   bench_ref.sh ENGINE WORKDIR IN_WAV [QUALITY] [FAST]
#
# ENGINE: bellard | tsac-ng
# FAST:   1 = -f (no transformer), 0 = full quality path
#
# Prints key=value lines for the host orchestrator.

set -euo pipefail

ENGINE="${1:?engine (bellard|tsac-ng)}"
WORKDIR="${2:?work dir (mounted at /data)}"
IN_WAV="${3:?input wav}"
QUALITY="${4:-9}"
FAST="${5:-1}"

TSAC_DIR="/opt/tsac"
case "$ENGINE" in
  bellard)
    BIN="$TSAC_DIR/tsac"
    ;;
  tsac-ng)
    BIN="/usr/local/bin/tsac-ng"
    ;;
  *)
    echo "unknown engine: $ENGINE (expected bellard or tsac-ng)" >&2
    exit 1
    ;;
esac

OUT_TSAC="$WORKDIR/${ENGINE}.tsac"
OUT_WAV="$WORKDIR/${ENGINE}_roundtrip.wav"

ARGS=(-q "$QUALITY")
if [[ "$ENGINE" == "bellard" && "$FAST" == "1" ]]; then
  ARGS+=(-f)
fi

# tsac-ng needs explicit model paths; fast encode is not implemented yet.
if [[ "$ENGINE" == "tsac-ng" ]]; then
  channels=$(python3 - <<'PY' "$IN_WAV"
import struct, sys
with open(sys.argv[1], "rb") as f:
    f.seek(22)
    ch = struct.unpack("<H", f.read(2))[0]
print(ch)
PY
)
  if [[ "$channels" -gt 1 ]]; then
    ARGS+=(-m "$TSAC_DIR/dac_stereo_q8.bin")
  else
    ARGS+=(-m "$TSAC_DIR/dac_mono_q8.bin")
  fi
fi

cd "$TSAC_DIR"
export LD_LIBRARY_PATH="/usr/local/lib:$TSAC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ENC_START=$(date +%s%N)
"$BIN" "${ARGS[@]}" c "$IN_WAV" "$OUT_TSAC"
ENC_END=$(date +%s%N)

DEC_START=$(date +%s%N)
"$BIN" "${ARGS[@]}" d "$OUT_TSAC" "$OUT_WAV"
DEC_END=$(date +%s%N)

BYTES=$(stat -c%s "$OUT_TSAC")
ENCODE_MS=$(awk "BEGIN {printf \"%.3f\", ($ENC_END-$ENC_START)/1e6}")
DECODE_MS=$(awk "BEGIN {printf \"%.3f\", ($DEC_END-$DEC_START)/1e6}")

echo "ENGINE=$ENGINE"
echo "ENCODE_MS=$ENCODE_MS"
echo "DECODE_MS=$DECODE_MS"
echo "BYTES=$BYTES"
echo "TSAC_PATH=$OUT_TSAC"
echo "WAV_PATH=$OUT_WAV"
