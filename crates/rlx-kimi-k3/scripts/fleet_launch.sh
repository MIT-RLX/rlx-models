#!/bin/bash
# Kimi-K3 multi-engine fleet — one worker rank per compute ENGINE per node.
#   msi: CUDA(r1) + CPU(r3)      amd: MI100(r2) + 780M(r4) + CPU(r5)
#   XDNA is dark (no XRT overlay configured → is_available()=false).
# Per-worker stderr → ~/wN.log so the shutdown "PAGING Xs + COMPUTE Ys | N paged"
# line is readable for per-engine load balancing.
# Env (optional, exported before calling): WENV="RLX_KIMI_PACKED_EXPERTS=1 ..." to
# switch the worker expert path; default (empty) = f32 GroupedMatMul path.
set -u
PEERS="192.168.99.148:29500,192.168.99.76:29501,192.168.99.161:29502,192.168.99.76:29503,192.168.99.161:29504,192.168.99.161:29505"
BIN="${BIN:-./target/debug/examples/kimi_k3_cluster}"   # set BIN=./target/release/... for release workers
WENV="${WENV:-}"
# rank:lo-hi per engine — REBALANCE HERE. [430,466) → Mac-local overflow.
R1_LO=0;   R1_HI=370   # msi CUDA
R2_LO=466; R2_HI=706   # amd MI100
R3_LO=370; R3_HI=430   # msi CPU
R4_LO=706; R4_HI=806   # amd 780M
R5_LO=806; R5_HI=896   # amd CPU

listen(){ ssh -n "$1" "ss -tln|grep -q ':$2 '"; }
allup(){ listen amd 29505 && listen amd 29504 && listen msi 29503 && listen amd 29502 && listen msi 29501; }

# Workers persist after ssh returns via `setsid bash -lc` — the login shell supplies
# the CUDA/ROCm loader paths (proven by the bash -lc selfcheck); bare setsid missed them.
W(){ # host "extra_env" rank lo hi device logfile settle
  local host="$1" env="$2" rank="$3" lo="$4" hi="$5" dev="$6" log="$7" settle="$8"
  ssh -n "$host" "setsid bash -lc 'cd ~/rlx-models && $env $WENV $BIN expert-worker --rank $rank --peers $PEERS --dest ~/kimi-experts --lo $lo --hi $hi --device $dev' </dev/null >~/$log 2>&1 &"
  sleep "$settle"
}
for attempt in 1 2 3; do
  ssh -n msi 'pkill -9 -f "[k]imi_k3_cluster"' 2>/dev/null; ssh -n amd 'pkill -9 -f "[k]imi_k3_cluster"' 2>/dev/null; sleep 3
  W amd ""                       5 $R5_LO $R5_HI cpu  w5_cpu.log   3
  W amd "HIP_VISIBLE_DEVICES=1"  4 $R4_LO $R4_HI rocm w4_780m.log  6
  W msi ""                       3 $R3_LO $R3_HI cpu  w3_cpu.log   3
  W amd "HIP_VISIBLE_DEVICES=0"  2 $R2_LO $R2_HI rocm w2_mi100.log 8
  W msi ""                       1 $R1_LO $R1_HI cuda w1_cuda.log  6
  ok=0; for w in $(seq 1 10); do if allup; then ok=1; break; fi; sleep 2; done
  [ "$ok" = 1 ] && { echo "fleet up (attempt $attempt)"; break; }
  echo "fleet incomplete (attempt $attempt), retrying"
done
allup || { echo "FLEET FAILED"; exit 1; }
echo "### fleet ready (WENV=${WENV:-none})"
