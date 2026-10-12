#!/usr/bin/env python3
"""Lint a crate in its non-default feature states.

Progress lines are flushed as they are produced: a sweep is minutes long and
its output is usually piped, where block buffering would hide which package is
being built when something hangs.

`just lint` builds default features only, so code behind an optional feature is
never compiled by any gate. That hides a whole class of breakage — a field
declared `#[cfg(feature = "x")]` and used without the gate compiles for whoever
turned `x` on and fails for everyone else. A 2026-08 sweep found ~90 such errors
across the workspace; `rlx-qwen3`'s `kv_store` was another.

    scripts/check-feature-states.py rlx-qwen3 rlx-serve
    scripts/check-feature-states.py --changed      # packages you are editing
    scripts/check-feature-states.py --all          # every package with features

Each package is checked twice: its **default** features (what `just lint`
builds) and its default features **plus every optional extra** except those
needing a toolchain this host does not have.

It deliberately does not test `--no-default-features`. Most crates here declare
a default that their core path requires — `rlx-whisper` is
`default = ["tokenizer", "timestamps"]` with the tokenizer behind a `dep:` —
so a featureless build fails by design, and reporting that is noise rather than
a finding. Pass `--bare` for a crate that genuinely claims to support it.

Exits non-zero if any state fails.
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Features whose transitive closure needs a toolchain that is not present on
# every dev machine. Skipped rather than failed: a red build here would mean
# "you do not have CUDA", which is not a finding.
NEEDS_TOOLCHAIN = ("cuda", "rocm", "vulkan", "blas-mkl", "mkl", "xdna", "hip", "tensorrt")


def metadata() -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def changed_packages(pkgs: dict) -> list[str]:
    """Packages with uncommitted changes — what you are actually working on."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    dirs = set()
    for line in out.stdout.splitlines():
        path = line[3:].strip().strip('"')
        parts = pathlib.Path(path).parts
        if len(parts) >= 2 and parts[0] == "crates":
            dirs.add(parts[1])
    return sorted(n for n, d in pkgs.items() if d["dir"] in dirs)


def run(pkg: str, features: list[str], bare: bool) -> tuple[bool, str]:
    cmd = ["cargo", "clippy", "-p", pkg, "--all-targets", "--no-deps"]
    if bare:
        cmd.append("--no-default-features")
    if features:
        cmd += ["--features", ",".join(features)]
    cmd += ["--", "-D", "warnings"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        return True, ""
    first = next(
        (l for l in r.stderr.splitlines() if l.startswith("error")),
        f"exit {r.returncode}",
    )
    return False, first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="*")
    ap.add_argument("--all", action="store_true", help="every package with features")
    ap.add_argument("--changed", action="store_true", help="packages with uncommitted changes")
    ap.add_argument("--bare", action="store_true",
                    help="also check --no-default-features (only meaningful for a crate "
                         "that claims to support it)")
    args = ap.parse_args()

    md = metadata()
    pkgs = {
        p["name"]: {
            "features": sorted(p["features"]),
            "dir": pathlib.Path(p["manifest_path"]).parent.name,
        }
        for p in md["packages"]
    }

    if args.changed:
        names = changed_packages(pkgs)
    elif args.all:
        names = sorted(n for n, d in pkgs.items() if d["features"])
    else:
        names = args.packages
    if not names:
        print("no packages selected (pass names, --changed or --all)")
        return 0

    states_n = 3 if args.bare else 2
    print(f"checking {len(names)} package(s) in {states_n} feature states each", flush=True)
    failures = []
    for name in names:
        if name not in pkgs:
            print(f"{name}: not a workspace package", file=sys.stderr)
            return 2
        own = [f for f in pkgs[name]["features"]
               if not any(t in f for t in NEEDS_TOOLCHAIN) and f != "default"]
        skipped = len(pkgs[name]["features"]) - len(own) - 1
        states = [("default", [], False), ("default + extras", own, False)]
        if args.bare:
            states.insert(0, ("no features", [], True))
        marks = []
        for label, feats, bare in states:
            ok, why = run(name, feats, bare)
            marks.append("ok" if ok else "FAIL")
            if not ok:
                failures.append(f"{name} [{label}]: {why}")
        note = f" ({skipped} toolchain features skipped)" if skipped > 0 else ""
        print(f"{name:28} {' / '.join(marks)}{note}", flush=True)

    if failures:
        print("\nfailures:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nfeature states OK — {len(names)} package(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
