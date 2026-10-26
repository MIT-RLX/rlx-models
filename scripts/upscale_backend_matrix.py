#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Which backends can run each rlx-upscale architecture, decided statically.

"Does DRCT run on ROCm?" is answerable from the op set alone — no ROCm box
required — because the compiler fails a graph that needs an op the target
cannot reach. But *reach* is not the same as *claim*: an op is runnable on a
backend by any of three routes, and a matrix that only reads `SUPPORTED_OPS`
reports false gaps for the other two.

  1. the backend claims it in `SUPPORTED_OPS`
  2. the legalize loop in `rlx-compile/src/rewrite.rs` has an explicit lowering
     that fires when the target does not claim it (`Pad`, `Roll`, `Slice`, …)
  3. the op carries `OpCaps::FUSED` in `rlx-ir/src/capability.rs`, so the
     unfuse pass decomposes it into primitives (`SelectiveScan`, `Lstm`, …)

Routes 2 and 3 are deliberate: CoreML, for instance, leaves `SelectiveScan`
*unclaimed on purpose* so the rewriter produces the compact `Op::Scan` form,
because the Metal shader compiler is superlinear in program size. Reading only the claim
set would call that a gap when it is a considered optimization.

This reads the op set from `cargo run --example op_coverage --json` and all
three support sources from the RLX tree, then prints the matrix and the precise
residue for every genuinely red cell.

`--check` is a *regression* guard, not a demand that every cell be green. Some
gaps are known and deliberate — an int8/fp16 NPU has no FP32 datapath, so an
f32 upscaler was never going to run there. Those are listed in `EXPECTED_GAPS`
and `NOT_AN_F32_TARGET`; anything outside them fails.

Usage:
  python3 scripts/upscale_backend_matrix.py                 # matrix
  python3 scripts/upscale_backend_matrix.py --check         # exit 1 on a NEW gap
  python3 scripts/upscale_backend_matrix.py --rlx ../rlx    # tree location
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

# Backend → (crate-relative file holding SUPPORTED_OPS, cargo feature on
# rlx-upscale that turns it on). `None` means the feature does not exist yet.
BACKENDS = [
    ("CPU", "crates/backends/rlx-cpu/src/supported_ops.rs", None),
    ("Metal", "crates/backends/rlx-metal/src/supported_ops.rs", "metal"),
    ("MLX", "crates/backends/rlx-mlx/src/supported_ops.rs", "mlx"),
    ("wgpu", "crates/backends/rlx-wgpu/src/supported_ops.rs", "gpu"),
    ("CUDA", "crates/backends/rlx-cuda/src/supported_ops.rs", "cuda"),
    ("ROCm", "crates/backends/rlx-rocm/src/supported_ops.rs", "rocm"),
    ("Vulkan", "crates/backends/rlx-vulkan/src/backend.rs", "vulkan"),
    ("ANE", "crates/backends/rlx-coreml/src/supported_ops.rs", "coreml"),
    ("OneAPI", "crates/backends/rlx-oneapi/src/backend.rs", None),
    ("QNN", "crates/backends/rlx-qnn/src/supported_ops.rs", None),
    ("TPU", "crates/backends/rlx-tpu/src/supported_ops.rs", None),
    # XDNA and WebGL declare an *empty* `SUPPORTED_OPS` in their own crate and
    # keep the live list elsewhere; read it where it actually is.
    ("XDNA", "crates/core/rlx-runtime/src/backend/xdna_backend.rs", None, "fn supported_ops"),
    ("WebGL", "crates/backends/rlx-webgl/src/plan.rs", None, "fn supported_ops"),
]

# Backends with no `SUPPORTED_OPS` at all, and why an f32 image upscaler is not
# a target for them. Listed so "all backends" has a definite meaning rather
# than quietly meaning "the ones the matrix happens to know about".
NO_OP_SET = {
    "rlx-cortexm": "INT8 kernels for ARMv7E-M microcontrollers — not an f32 target",
    "rlx-cerebras": "per-graph CSL synthesis for the Wafer-Scale Engine",
    "rlx-fpga": "per-graph datapath synthesis",
    # Not a compute backend: `SUPPORTED_OPS` is genuinely empty and `compile()`
    # panics with a diagnostic. It tunnels to a real GPU rather than running
    # anything itself, so "supports nothing" is the correct answer, not a gap.
    "rlx-egpu": "transport stub for a tunnelled GPU — compile() panics by design",
}


def extract_const_idents(path: Path, const_name: str = "SUPPORTED_OPS") -> set[str]:
    """Collect the `OpKind::` identifiers listed in a `SUPPORTED_OPS` slice."""
    text = path.read_text()
    m = re.search(rf"\bconst {re.escape(const_name)}\b[^=]*=", text)
    if not m:
        raise SystemExit(f"{const_name} not found in {path}")
    rest = re.sub(r"//.*?$", "", text[m.end():], flags=re.M)
    depth, started, buf = 0, False, []
    for ch in rest:
        if ch == "[":
            depth += 1
            started = True
            continue
        if ch == "]":
            depth -= 1
            if started and depth == 0:
                break
            continue
        if started and depth > 0:
            buf.append(ch)
    idents = set()
    for tok in re.findall(r"(?:OpKind::)?([A-Za-z_][A-Za-z0-9_]*)", "".join(buf)):
        if tok not in {"OpKind", "use", "rlx_ir"}:
            idents.add(tok)
    return idents


def extract_fn_idents(path: Path, anchor: str) -> set[str]:
    """Collect `OpKind::` identifiers from the slice literal after `anchor`.

    Used where the op list is the body of a `supported_ops()` function rather
    than a `const` — XDNA and WebGL both keep an empty `SUPPORTED_OPS` in their
    own crate and the real list somewhere else, which reads as "supports
    nothing" if you only look at the const.
    """
    text = path.read_text()
    m = re.search(re.escape(anchor), text)
    if not m:
        raise SystemExit(f"{anchor} not found in {path}")
    # Skip past the signature: `-> &'static [OpKind]` contains a bracket pair
    # that closes immediately, so scanning from the anchor finds the *return
    # type* and reports an empty op set — which reads exactly like a backend
    # that supports nothing.
    body = text.find("{", m.end())
    if body == -1:
        raise SystemExit(f"no body after {anchor} in {path}")
    rest = re.sub(r"//.*?$", "", text[body:], flags=re.M)
    depth, started, buf = 0, False, []
    for ch in rest:
        if ch == "[":
            depth += 1
            started = True
            continue
        if ch == "]":
            depth -= 1
            if started and depth == 0:
                break
            continue
        if started and depth > 0:
            buf.append(ch)
    return {
        t for t in re.findall(r"(?:OpKind::)?([A-Za-z_][A-Za-z0-9_]*)", "".join(buf))
        if t not in {"OpKind", "use", "rlx_ir"}
    }


def lowerable_in_legalize(rlx: Path) -> set[str]:
    """Ops the legalize loop rewrites when the target does not claim them."""
    p = rlx / "crates/core/rlx-compile/src/rewrite.rs"
    return set(re.findall(r"bad\.contains\(&OpKind::([A-Za-z_][A-Za-z0-9_]*)\)", p.read_text()))


def fused_kinds(rlx: Path) -> set[str]:
    """Ops carrying `OpCaps::FUSED` — the unfuse pass can decompose these.

    The capability table writes these as match arms, and the FUSED arm is a
    fifteen-line `A | B | ... => OpCaps::FUSED`. Walking *backwards* from each
    `=>` is what makes this reliable: a forward non-greedy match returns only
    the last identifier of the arm, which silently drops `SelectiveScan` and
    reports MambaIRv2 as unsupported everywhere it is merely decomposed.
    """
    text = (rlx / "crates/core/rlx-ir/src/capability.rs").read_text()
    kinds: set[str] = set()
    for m in re.finditer(r"=>\s*\{?\s*(OpCaps::[^\n]*(?:\n[^\n=]*?)??)(?:,|\n)", text):
        if "FUSED" not in m.group(1):
            continue
        # Pattern lists contain only identifiers, `|` and whitespace, so walk
        # back until something else (a `,`, `}` or `:` ending the prior arm).
        i = m.start()
        j = i
        while j > 0 and (text[j - 1].isalnum() or text[j - 1] in "_| \t\r\n"):
            j -= 1
        kinds.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text[j:i]))
    # `FUSION_BOUNDARY` / `BLAS` are flag names that can appear in a union
    # expression, never op kinds.
    return {k for k in kinds if not k.isupper()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlx", default="../rlx", help="path to the RLX tree")
    ap.add_argument("--check", action="store_true", help="exit 1 on any gap")
    args = ap.parse_args()
    rlx = (HERE / args.rlx).resolve()
    if not rlx.is_dir():
        raise SystemExit(f"RLX tree not found at {rlx}")

    out = subprocess.run(
        ["cargo", "run", "-q", "-p", "rlx-upscale", "--example", "op_coverage", "--", "--json"],
        cwd=HERE, capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit("op_coverage failed")
    data = json.loads(out.stdout)
    per_arch: dict[str, list[str]] = data["per_arch"]

    lowerable = lowerable_in_legalize(rlx)
    fused = fused_kinds(rlx)
    reachable = lowerable | fused

    support: dict[str, set[str]] = {}
    claimed: dict[str, set[str]] = {}
    missing_backend: list[str] = []
    for entry in BACKENDS:
        name, rel = entry[0], entry[1]
        anchor = entry[3] if len(entry) > 3 else None
        p = rlx / rel
        if not p.exists():
            missing_backend.append(f"{name} ({rel})")
            continue
        claimed[name] = (
            extract_fn_idents(p, anchor) if anchor else extract_const_idents(p)
        )
        # A backend can run anything it claims, plus anything the compiler
        # rewrites away before legalization.
        support[name] = claimed[name] | reachable

    names = [e[0] for e in BACKENDS if e[0] in support]
    w = max(len(a) for a in per_arch)
    print(f"{'arch':<{w}} " + " ".join(f"{n:>7}" for n in names))
    gaps: dict[tuple[str, str], list[str]] = {}
    for arch, ops in sorted(per_arch.items()):
        cells = []
        for n in names:
            miss = sorted(set(ops) - support[n])
            if miss:
                gaps[(arch, n)] = miss
            cells.append(f"{'·' if miss else 'ok':>7}")
        print(f"{arch:<{w}} " + " ".join(cells))

    if missing_backend:
        print("\nbackends whose SUPPORTED_OPS could not be read:")
        for m in missing_backend:
            print(f"  {m}")

    # Account for every backend crate in the tree, not just the ones with an
    # op set — otherwise "all backends" silently means "all the ones listed".
    present = {d.name for d in (rlx / "crates/backends").iterdir() if d.is_dir()}
    support_crates = {
        "rlx-gpu-dispatch", "rlx-gpu-host", "rlx-gpu-kernels", "rlx-mlx-sys",
    }
    covered = {
        "rlx-cpu", "rlx-metal", "rlx-mlx", "rlx-wgpu", "rlx-cuda", "rlx-rocm",
        "rlx-vulkan", "rlx-coreml", "rlx-oneapi", "rlx-qnn", "rlx-tpu",
        "rlx-xdna", "rlx-webgl",
    }
    unaccounted = present - support_crates - covered - set(NO_OP_SET)
    print("\nnot in the matrix (no declared op set):")
    for name in sorted(NO_OP_SET):
        if name in present:
            print(f"  {name:<16} {NO_OP_SET[name]}")
    if unaccounted:
        print("\nUNACCOUNTED backend crates — the matrix has fallen behind the tree:")
        for name in sorted(unaccounted):
            print(f"  {name}")

    print(
        f"\n'ok' = claimed by the backend, or rewritten before it "
        f"({len(lowerable)} legalize lowerings + {len(fused)} fused decompositions)"
    )

    # Where a cell is green only because of a rewrite, say so — it is the
    # difference between a native kernel and a decomposition, which matters for
    # speed even when it does not matter for correctness.
    via_rewrite: dict[str, set[str]] = {}
    for arch, ops in per_arch.items():
        for n in names:
            for op in set(ops) - claimed[n]:
                if op in reachable:
                    via_rewrite.setdefault(op, set()).add(n)
    if via_rewrite:
        print("\nrun via rewrite rather than a native kernel:")
        for op, backends in sorted(via_rewrite.items()):
            route = "legalize lowering" if op in lowerable else "fused decomposition"
            print(f"  {op:<18} {route:<22} on {', '.join(sorted(backends))}")

    # Not every backend is a target for an f32 image upscaler. Saying so keeps
    # a red cell meaningful: an NPU with no FP32 datapath is not a gap to close,
    # it is the wrong machine for this workload.
    NOT_AN_F32_TARGET = {
        "QNN": "Hexagon NPU — an int8/fp16 engine",
        "XDNA": "AIE-ML tile array — int8/bf16 MACs, no FP32 datapath (its own note)",
    }
    # Gaps that are known, understood and accepted. Keyed by backend → the ops
    # it is allowed to be missing. A gap outside this table is a regression.
    EXPECTED_GAPS = {
        # WebGL2 has no practical sort primitive; only MambaIRv2 needs one.
        "WebGL": {"ArgSort"},
    }
    unexpected = {
        (arch, backend): miss
        for (arch, backend), miss in gaps.items()
        if backend not in NOT_AN_F32_TARGET
        and not set(miss) <= EXPECTED_GAPS.get(backend, set())
    }

    if gaps:
        blocked = {b for _, b in gaps}
        aside = sorted(blocked & set(NOT_AN_F32_TARGET))
        if aside:
            print("\nred cells on backends that are not f32 targets anyway:")
            for b in aside:
                print(f"  {b:<7} {NOT_AN_F32_TARGET[b]}")
        print("\ngaps (architecture → backend: missing ops)")
        by_op: dict[str, set[str]] = {}
        for (arch, backend), miss in sorted(gaps.items()):
            print(f"  {arch:<12} {backend:<7} {' '.join(miss)}")
            for op in miss:
                by_op.setdefault(op, set()).add(backend)
        print("\nby op — this is the work list:")
        for op, backends in sorted(by_op.items()):
            print(f"  {op:<18} missing on {', '.join(sorted(backends))}")
    else:
        print("\nevery architecture is supported on every backend listed.")

        if unexpected:
            print("\nUNEXPECTED — not in EXPECTED_GAPS / NOT_AN_F32_TARGET:")
            for (arch, backend), miss in sorted(unexpected.items()):
                print(f"  {arch:<12} {backend:<7} {' '.join(miss)}")
        else:
            print("\nevery gap above is expected and classified.")

    return 1 if (args.check and (unexpected or missing_backend or unaccounted)) else 0


if __name__ == "__main__":
    sys.exit(main())
