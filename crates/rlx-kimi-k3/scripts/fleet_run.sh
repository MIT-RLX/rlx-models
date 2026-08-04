#!/bin/bash
# Run the orchestrator against the live fleet, then read each worker's shutdown
# timing line and derive per-engine cold-paging throughput (17.5 MB / packed expert).
set -u
PEERS="192.168.99.148:29500,192.168.99.76:29501,192.168.99.161:29502,192.168.99.76:29503,192.168.99.161:29504,192.168.99.161:29505"
BIN="./target/debug/examples/kimi_k3_cluster"
SH="${SH:-1:0-370,3:370-430,2:466-706,4:706-806,5:806-896}"
BODY="${BODY:-cpu}"; LAYERS="${LAYERS:-8}"; REPEAT="${REPEAT:-1}"
OUT="${1:-/tmp/orch.log}"
echo "### expert-run body=$BODY layers=$LAYERS repeat=$REPEAT"
echo "### shards=$SH"
$BIN expert-run --peers "$PEERS" --model-dir /Volumes/FOUR/kimi --tokens 1,100,5000 \
  --layers "$LAYERS" --gen 1 --shards "$SH" --device "$BODY" --repeat "$REPEAT" 2>"$OUT"
echo "=== [bench] ==="; grep -aE "^\[bench\]" "$OUT"
echo "=== per-engine worker paging (MB/s @ 17.5MB/expert) ==="
hosts=(x msi amd msi amd amd); logs=(x w1_cuda w2_mi100 w3_cpu w4_780m w5_cpu)
engs=(x "msi CUDA" "amd MI100" "msi CPU " "amd 780M" "amd CPU ")
for r in 1 2 3 4 5; do
  ln=$(ssh -n "${hosts[$r]}" "grep -a 'rank $r] shutdown' ~/${logs[$r]}.log | tail -1")
  paged=$(echo "$ln" | sed -nE 's/.*: ([0-9]+) paged.*/\1/p')
  pg=$(echo "$ln"    | sed -nE 's/.*PAGING ([0-9.]+)s.*/\1/p')
  cp=$(echo "$ln"    | sed -nE 's/.*COMPUTE ([0-9.]+)s.*/\1/p')
  if [ -n "${paged:-}" ] && [ -n "${pg:-}" ]; then
    awk -v e="${engs[$r]}" -v p="$paged" -v s="$pg" -v c="${cp:-0}" 'BEGIN{
      mbps = (s>0)? p*17.5/s : 0; mspe = (p>0)? s*1000/p : 0;
      printf "  r%d %s: %4d paged  PAGING %6.2fs (%5.1f ms/exp, %6.0f MB/s)  COMPUTE %6.2fs\n", '"$r"', e, p, s, mspe, mbps, c }'
  else
    echo "  r$r ${engs[$r]}: (no shutdown line) -> $ln"
  fi
done
