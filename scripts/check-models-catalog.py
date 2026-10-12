#!/usr/bin/env python3
"""Verify MODELS.md against `cargo metadata` — the catalog claims to be generated
from each crate's Cargo.toml, but it is hand-maintained and drifts every release.

Checks, in the order they have actually bitten:

  1. every publishable model crate has a row (missing crates are invisible);
  2. no row names a crate that does not exist;
  3. a row never claims a backend the crate has no feature for;
  4. a row never silently omits a backend the crate does have — unless the cell
     carries a prose caveat (`verified` / `wired` / `untested` / …), which is a
     deliberate status note rather than a feature list;
  5. the Categories table counts and total match the real row counts;
  6. the headline family count in MODELS.md and every repeat in README agree;
  7. `STUB` markers agree with the crate description.

Exit 0 clean, 1 on any finding. Run from the repo root:

    python3 scripts/check-models-catalog.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Crates that are deliberately absent from the catalog: shared infrastructure,
# servers, tooling and base/shared-arch crates. MODELS.md documents this rule in
# its header ("Shared infrastructure, servers, and benchmark crates are not
# listed here"); this list is what that sentence means concretely.
EXEMPT_EXACT = {
    "rlx-models", "rlx-models-core", "rlx-cli", "rlx-serve", "rlx-eval",
    "rlx-assets", "rlx-vision", "rlx-text", "rlx-audio-blocks", "rlx-tune",
    "rlx-pkg", "rlx-embed", "rlx-guardrails", "rlx-llama-base", "rlx-vlm-base",
    "rlx-model-hub", "rlx-openai", "rlx-protocol", "rlx-onnx-decompose",
    "rlx-quant-calib", "rlx-sam-ir", "rlx-fft", "rlx-ssm",
}
EXEMPT_PREFIX = ("bench_", "kitten_")

FEATURE_LABEL = {
    "metal": "Metal", "mlx": "MLX", "cuda": "CUDA", "rocm": "ROCm",
    "gpu": "wgpu", "vulkan": "Vulkan", "coreml": "CoreML",
}
ALL7 = {"Metal", "MLX", "CUDA", "ROCm", "wgpu", "Vulkan"}
# Cells that describe verification status rather than the feature set.
CAVEAT = re.compile(r"verified|wired|untested|parity|stub|Fit:|bare metal|prefill|torch", re.I)
SKIP_SECTIONS = {"Backends", "Categories"}


def metadata():
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode != 0:
        sys.exit(f"cargo metadata failed:\n{out.stderr.strip()}")
    return {p["name"]: p for p in json.loads(out.stdout)["packages"]}


def parse_rows(text):
    """-> [(section, crate, description, backend_cell, n_cells)] for every row.

    `n_cells` is reported so callers can catch an unescaped `|` inside a cell.
    Markdown splits on it, so the row silently gains columns and the backend
    cell this function returns is a *truncated* prefix — the backend checks
    below then validate text that is not what the row says. GitHub renders the
    row wrong too.
    """
    rows, section = [], None
    for line in text.split("\n"):
        head = re.match(r"^## (.+)$", line)
        if head:
            section = head.group(1).strip()
            continue
        if not (section and line.startswith("|")):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0]:
            continue
        if set(cells[0]) <= set("-: ") or cells[0].lower() in ("crate", "category"):
            continue
        rows.append((section, cells[0].strip("`"), cells[1], cells[2], len(cells)))
    return rows


def claimed_backends(cell):
    found = set(ALL7) if re.search(r"\*\*all\s*7\*\*", cell, re.I) else set()
    for label in FEATURE_LABEL.values():
        # Case-sensitive: cells capitalise backend names, so this must not fire
        # on ordinary prose like "bare metal".
        if re.search(rf"\b{re.escape(label)}\b", cell):
            found.add(label)
    return found


def main():
    pkgs = metadata()
    text = (ROOT / "MODELS.md").read_text()
    rows = parse_rows(text)
    problems = []

    listed = {crate for _, crate, _, _, _ in rows}
    publishable = {
        n for n, p in pkgs.items()
        if p.get("publish") is None
        and n not in EXEMPT_EXACT
        and not n.startswith(EXEMPT_PREFIX)
    }
    for crate in sorted(publishable - listed):
        problems.append(f"missing from MODELS.md: {crate}")
    for section, crate, _, _, _ in rows:
        if crate not in pkgs and not (ROOT / "crates" / crate).is_dir():
            problems.append(f"row names a crate that does not exist: {crate} [{section}]")

    for section, crate, desc, cell, ncells in rows:
        if ncells != 3:
            problems.append(
                f"{crate}: table row has {ncells} cells, expected 3 — an "
                f"unescaped '|' inside a cell (escape it as '\\|' or reword). "
                f"The backend cell is truncated at the stray pipe, so the "
                f"checks below are reading only part of it [{section}]"
            )
        if crate not in pkgs:
            continue  # workspace-excluded (e.g. embedded targets): no metadata to compare
        feats = set(pkgs[crate].get("features", {}))
        actual = {lbl for k, lbl in FEATURE_LABEL.items() if k in feats}
        claimed = claimed_backends(cell)
        if claimed - actual:
            problems.append(
                f"{crate}: claims {sorted(claimed - actual)} but has no such feature — cell {cell!r}"
            )
        if actual - claimed and not CAVEAT.search(cell):
            problems.append(
                f"{crate}: omits {sorted(actual - claimed)} that the crate has — cell {cell!r}"
            )
        crate_desc = pkgs[crate].get("description") or ""
        if ("STUB" in desc.upper()) != ("STUB" in crate_desc.upper()):
            problems.append(f"{crate}: STUB marker disagrees with its Cargo.toml description")

    counts = {}
    for section, *_ in rows:
        counts[section] = counts.get(section, 0) + 1
    fam_sections = {
        s: n for s, n in counts.items()
        if s not in SKIP_SECTIONS | {"Training crates", "Interpretability crates"}
    }
    total = sum(fam_sections.values())

    for line in text.split("\n"):
        m = re.match(r"^\| \[(.+?)\]\(#.*?\) \| (\d+) \|$", line)
        if m and m.group(1) in counts and int(m.group(2)) != counts[m.group(1)]:
            problems.append(
                f"Categories table: {m.group(1)} says {m.group(2)}, actual rows {counts[m.group(1)]}"
            )
        m = re.match(r"^\| \*\*Total\*\* \| \*\*(\d+)\*\* \|$", line)
        if m and int(m.group(1)) != total:
            problems.append(f"Categories table total says {m.group(1)}, actual {total}")

    for name in ("MODELS.md", "README.md"):
        body = (ROOT / name).read_text()
        # Only the headline claims — not prose like "across 4 families" that
        # happens to be counting something else entirely (README's test suites).
        stated_counts = re.findall(r"(\d+) model famil(?:y|ies)", body)
        stated_counts += re.findall(r"all (\d+) families", body)
        for stated in stated_counts:
            if int(stated) != total:
                problems.append(f"{name}: claims {stated} families, actual {total}")

    if problems:
        print(f"MODELS.md catalog: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"MODELS.md catalog OK — {total} model families across {len(fam_sections)} categories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
