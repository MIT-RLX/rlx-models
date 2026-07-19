#!/usr/bin/env bash
# Mirror the two working trees (rlx-models + its sibling ../rlx) to msi.
# We rsync working trees (not git) because both have uncommitted changes newer
# than msi and we must not commit. weights/, target/, .cache/ are EXCLUDED so
# --delete can never touch msi's 12G of downloaded weights or its build cache.
set -euo pipefail

HOST="${MSI_HOST:-msi}"
LOCAL_MODELS="${LOCAL_MODELS:-/Users/Shared/rlx-models}"
LOCAL_RLX="${LOCAL_RLX:-/Users/Shared/rlx}"
REMOTE_MODELS="${REMOTE_MODELS:-rlx-models}"     # relative to remote $HOME
REMOTE_RLX="${REMOTE_RLX:-rlx}"

EXCLUDES=(--exclude 'target/' --exclude 'weights/' --exclude '.cache/'
          --exclude '.git/' --exclude '.venv-*/' --exclude 'scripts/matrix/out/'
          --exclude '__pycache__/')

# safety: refuse to run with an empty exclude list (would let --delete wipe weights)
[ ${#EXCLUDES[@]} -ge 3 ] || { echo "refusing: exclude list too short"; exit 1; }
[ -d "$LOCAL_RLX" ]    || { echo "missing $LOCAL_RLX"; exit 1; }
[ -d "$LOCAL_MODELS" ] || { echo "missing $LOCAL_MODELS"; exit 1; }

echo ">> syncing $LOCAL_RLX -> $HOST:~/$REMOTE_RLX"
rsync -az --delete "${EXCLUDES[@]}" "$LOCAL_RLX/"    "$HOST:$REMOTE_RLX/"
echo ">> syncing $LOCAL_MODELS -> $HOST:~/$REMOTE_MODELS"
rsync -az --delete "${EXCLUDES[@]}" "$LOCAL_MODELS/" "$HOST:$REMOTE_MODELS/"
echo ">> sync done"
