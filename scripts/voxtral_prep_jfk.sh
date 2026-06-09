#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

set -euo pipefail

# Download a public-domain JFK inaugural address recording (Wikimedia Commons),
# convert to 24kHz mono WAV, and chop into fixed-length training clips.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_DIR="${OUT_DIR:-"$ROOT/.cache/voxtral/jfk"}"
CLIPS_DIR="${CLIPS_DIR:-"$OUT_DIR/wavs"}"
SEGMENT_SEC="${SEGMENT_SEC:-6}"

SRC_URL="${SRC_URL:-https://commons.wikimedia.org/wiki/Special:FilePath/JFK_inaugural_address.ogg}"

mkdir -p "$OUT_DIR" "$CLIPS_DIR"

SRC_OGG="$OUT_DIR/jfk_inaugural_address.ogg"
FULL_WAV="$OUT_DIR/jfk_inaugural_24k_mono.wav"

echo "[voxtral-jfk] download: $SRC_URL"
curl -L --fail --retry 3 --retry-delay 2 -o "$SRC_OGG" "$SRC_URL"

echo "[voxtral-jfk] convert: 24kHz mono wav"
ffmpeg -y -hide_banner -loglevel error \
  -i "$SRC_OGG" \
  -ac 1 -ar 24000 -c:a pcm_s16le \
  "$FULL_WAV"

echo "[voxtral-jfk] segment: ${SEGMENT_SEC}s clips -> $CLIPS_DIR"
rm -f "$CLIPS_DIR"/jfk_*.wav
ffmpeg -y -hide_banner -loglevel error \
  -i "$FULL_WAV" \
  -f segment -segment_time "$SEGMENT_SEC" -reset_timestamps 1 \
  -ac 1 -ar 24000 -c:a pcm_s16le \
  "$CLIPS_DIR/jfk_%04d.wav"

echo "[voxtral-jfk] done: $(ls "$CLIPS_DIR" | wc -l | tr -d ' ') wav clips"

