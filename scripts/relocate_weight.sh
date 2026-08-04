#!/usr/bin/env bash
# Relocate one weight tree to the external FOUR drive and symlink it back in
# place, freeing internal disk while keeping every `weights/<rel>` path valid.
#
# Safe + idempotent: rsync -> byte-exact + file-count verify -> only then swap
# the source for a symlink. If verification fails, the original is left intact.
#
# Usage: scripts/relocate_weight.sh <relpath-under-weights>   e.g. tts/metavoice
set -euo pipefail

REL="${1:?usage: relocate_weight.sh <relpath under weights/>}"
ROOT="/Users/Shared/rlx-models"
STORE="/Volumes/FOUR/weights"
SRC="$ROOT/weights/$REL"
DST="$STORE/$REL"

[ -d "/Volumes/FOUR" ] || { echo "FOUR not mounted"; exit 1; }

# Already a symlink (relocated, or an internal alias) -> nothing to do.
if [ -L "$SRC" ]; then echo "skip (already symlink): $REL"; exit 0; fi
if [ ! -e "$SRC" ]; then echo "skip (missing): $REL"; exit 0; fi

mkdir -p "$(dirname "$DST")"
echo "rsync $REL -> $DST"
rsync -a --delete "$SRC/" "$DST/"

bytes() { find "$1" -type f -print0 | xargs -0 stat -f%z 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
count() { find "$1" -type f | wc -l | tr -d ' '; }
sb=$(bytes "$SRC"); db=$(bytes "$DST")
sc=$(count "$SRC"); dc=$(count "$DST")
if [ "$sb" != "$db" ] || [ "$sc" != "$dc" ]; then
  echo "VERIFY FAILED $REL: src=${sb}b/${sc}f dst=${db}b/${dc}f (original kept)"; exit 1
fi

rm -rf "$SRC"
ln -s "$DST" "$SRC"
echo "OK relocated $REL: ${sc} files, ${sb} bytes -> $DST (symlinked)"
