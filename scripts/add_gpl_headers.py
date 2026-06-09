#!/usr/bin/env python3
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

"""Add GPLv3 file headers to rlx-models sources (same text as sibling rlx repo)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUST_LINES = [
    "// RLX — versatile ML compiler + runtime.",
    "// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.",
    "//",
    "// This program is free software: you can redistribute it and/or modify",
    "// it under the terms of the GNU General Public License as published by",
    "// the Free Software Foundation, version 3.",
    "//",
    "// This program is distributed in the hope that it will be useful,",
    "// but WITHOUT ANY WARRANTY; without even the implied warranty of",
    "// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the",
    "// GNU General Public License for more details.",
    "//",
    "// You should have received a copy of the GNU General Public License",
    "// along with this program. If not, see <https://www.gnu.org/licenses/>.",
    "",
]

HASH_LINES = [
    line.replace("//", "#", 1) if line.startswith("//") else line for line in RUST_LINES
]

SKIP_DIRS = {
    ".git",
    ".cache",
    ".claude",
    "target",
    "weights",
    "docker",
}

SKIP_DIR_PREFIXES = (".venv",)

EXTENSIONS = {".rs", ".sh", ".py"}

RLX_TAG = "RLX — versatile ML compiler + runtime."


def format_header(path: Path) -> str:
    lines = RUST_LINES if path.suffix == ".rs" else HASH_LINES
    return "\n".join(lines) + "\n"


def is_header_comment_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") or s.startswith("#")


def strip_all_rlx_headers(text: str) -> tuple[str | None, str]:
    """Remove every RLX/GPL preamble block; return (shebang, body)."""
    shebang = None
    rest = text
    if rest.startswith("#!"):
        first_nl = rest.find("\n")
        if first_nl != -1:
            shebang = rest[: first_nl + 1]
            rest = rest[first_nl + 1 :]

    while True:
        rest = rest.lstrip("\n")
        lines = rest.splitlines(keepends=True)
        if not lines:
            break
        first = lines[0]
        if RLX_TAG not in first:
            break

        idx = 1
        saw_gpl = False
        while idx < len(lines):
            line = lines[idx]
            stripped = line.strip()
            if "license header truncated" in line:
                idx += 1
                break
            if "along with this program" in line:
                idx += 1
                break
            if "This program is free software" in line:
                saw_gpl = True
            if (
                saw_gpl
                and stripped == ""
                and idx + 1 < len(lines)
                and RLX_TAG not in lines[idx + 1]
                and not is_header_comment_line(lines[idx + 1])
            ):
                idx += 1
                break
            if (
                not saw_gpl
                and "Copyright (C) 2026" in lines[idx - 1]
                and stripped == ""
            ):
                idx += 1
                break
            idx += 1

        rest = "".join(lines[idx:])

    return shebang, rest


def process_file(path: Path, dry_run: bool) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "skip-binary"
    shebang, body = strip_all_rlx_headers(text)
    if not body.strip():
        return "skip-empty-body"
    header = format_header(path)
    new_text = (shebang or "") + header + body
    if new_text == text:
        return None

    if not dry_run:
        path.write_text(new_text, encoding="utf-8", newline="\n")
    return "update"


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
        if any(part.startswith(prefix) for prefix in SKIP_DIR_PREFIXES):
            return True
    return False


def iter_source_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in EXTENSIONS:
            continue
        if should_skip(path):
            continue
        if path.name == "add_gpl_headers.py":
            continue
        out.append(path)
    return sorted(out)


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    changed = 0
    skipped = 0
    for path in iter_source_files():
        result = process_file(path, dry_run)
        rel = path.relative_to(ROOT)
        if result == "update":
            changed += 1
            print(f"{'would update' if dry_run else 'updated'}: {rel}")
        elif result == "skip-binary":
            skipped += 1
            if dry_run:
                print(f"skip (not utf-8): {rel}")
        elif result == "skip-empty-body":
            skipped += 1
            print(f"skip (empty body — restore from git before re-running): {rel}")
    print(f"{'would update' if dry_run else 'updated'} {changed} file(s)")
    if skipped:
        print(f"skipped {skipped} non-utf-8 file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
