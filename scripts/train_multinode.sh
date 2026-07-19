#!/usr/bin/env bash
# Launch a data-parallel rlx-tune training run across physical machines.
#
# Rank 0 runs locally; ranks 1..N-1 run on SSH hosts. Every node needs the
# example binary already built (same relative path, or set REMOTE_BIN) and must
# be mutually reachable on the LAN at the addresses in PEERS. This is the same
# protocol the `--nnodes` launcher uses on one host — only the peer IPs change.
#
# Usage:
#   PEERS="192.168.0.10:29500,192.168.0.11:29500" \
#   HOSTS="_,user@192.168.0.11" \
#     scripts/train_multinode.sh target/release/examples/mnist --steps 400 --overlap --shard
#
#   PEERS  comma list host:port, one per rank; rank i binds PEERS[i].
#   HOSTS  comma list, one per rank; "_" = local (rank 0), else an ssh target.
#   REMOTE_BIN  binary path on remote nodes (default: same path as $1).
#
# Tip: instead of PEERS you can set DISCOVER=1 on every rank for UDP
# auto-discovery (no hand-wired IPs) — see rlx-driver's Node::from_env.
set -euo pipefail

BIN="${1:?usage: train_multinode.sh <example-binary> [args...]}"; shift
ARGS=("$@")
: "${PEERS:?set PEERS=host0:port,host1:port,...}"
: "${HOSTS:?set HOSTS=_,user@host1,...}"
REMOTE_BIN="${REMOTE_BIN:-$BIN}"

IFS=',' read -r -a HOST_ARR <<< "$HOSTS"
WORLD="${#HOST_ARR[@]}"
echo "launching $WORLD ranks | PEERS=$PEERS"

pids=()
for rank in "${!HOST_ARR[@]}"; do
  host="${HOST_ARR[$rank]}"
  if [[ "$host" == "_" ]]; then
    echo "  rank $rank: local"
    RANK="$rank" WORLD="$WORLD" PEERS="$PEERS" "$BIN" "${ARGS[@]}" &
    pids+=("$!")
  else
    echo "  rank $rank: $host ($REMOTE_BIN)"
    ssh -o BatchMode=yes "$host" \
      "RANK=$rank WORLD=$WORLD PEERS='$PEERS' '$REMOTE_BIN' ${ARGS[*]}" &
    pids+=("$!")
  fi
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit "$fail"
