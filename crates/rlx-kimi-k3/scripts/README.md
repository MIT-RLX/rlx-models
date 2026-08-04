# Kimi-K3 cluster fleet scripts

Operational tooling for the disaggregated expert-parallel cluster (see
`../docs/MOE_EXPERT_OPTIMIZATION.md`). The node IPs, ports, and expert-range topology are
**cluster-specific** — edit `PEERS` and the `R{1..5}` ranges for your fleet.

## `fleet_launch.sh` — start one worker rank per compute engine

Launches the 5-engine fleet in descending-rank order (so each worker's higher-rank connect
targets are already listening), with a retry-until-all-listening loop. Workers persist via
`setsid bash -lc` (the login shell supplies the CUDA/ROCm loader paths; a bare `setsid`
misses them, and non-interactive ssh has no cargo/CUDA in PATH).

```sh
# f32 default path:
BIN=./target/release/examples/kimi_k3_cluster scripts/fleet_launch.sh
# packed MXFP4 path (native GPU kernel / fused CPU kernel):
BIN=./target/release/examples/kimi_k3_cluster \
  WENV="RLX_KIMI_PACKED_EXPERTS=1 RLX_KIMI_EXPERT_CACHE=8192" scripts/fleet_launch.sh
```

Env: `BIN` (worker binary path on each node, default debug), `WENV` (extra env prepended to
every worker command).

## `fleet_run.sh` — run the orchestrator + report per-engine timing

Runs `expert-run` (rank 0, the Mac backbone) against the live fleet, prints the `[bench]`
line, then reads each worker's shutdown `PAGING/COMPUTE` line and derives per-engine
cold-paging throughput (17.5 MB / packed expert).

```sh
BODY=cpu LAYERS=8 REPEAT=1 scripts/fleet_run.sh /tmp/orch.log
```

Env: `BODY` (orchestrator backbone device), `LAYERS`, `REPEAT`, `SH` (shard map override).

## Notes

- Never probe worker ports with `nc -z` — it connects as a fake peer and kills the worker
  mesh. Use passive `ss`/logs (the scripts already do).
- `expert-run` shuts the workers down at the end (which prints their timing lines), so the
  fleet is single-use per launch; re-launch for another measurement.
