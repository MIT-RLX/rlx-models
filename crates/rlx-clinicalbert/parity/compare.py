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

"""Compare HF reference vs RLX outputs for encoder + (optional) pooler + MLM."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def diff(name: str, hf: np.ndarray, rlx: np.ndarray) -> bool:
    flat_hf = hf.reshape(-1)
    if rlx.size != flat_hf.size:
        print(f"[{name}] FAIL: size mismatch HF={flat_hf.size} RLX={rlx.size}")
        return False
    rlx2 = rlx.reshape(hf.shape)
    d = flat_hf - rlx
    max_abs = float(np.max(np.abs(d)))
    mean_abs = float(np.mean(np.abs(d)))
    rel = float(np.max(np.abs(d) / (np.abs(flat_hf) + 1e-6)))
    cos = float(np.dot(flat_hf, rlx) / (np.linalg.norm(flat_hf) * np.linalg.norm(rlx) + 1e-12))
    print(f"[{name}]  shape={hf.shape}  max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}  rel={rel:.4e}  cos={cos:.10f}")
    if name == "mlm":
        # MLM logits are large in absolute value; check top-1 token agreement.
        hf_top1 = np.argmax(hf, axis=-1)
        rlx_top1 = np.argmax(rlx2, axis=-1)
        agree = float(np.mean(hf_top1 == rlx_top1))
        print(f"[{name}]  top1 token agreement: {agree*100:.2f}%")
        return cos > 0.9999 and agree >= 0.99
    return cos > 0.9999 and max_abs < 1e-3


def main() -> int:
    ref_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/rlx-clinicalbert-parity")

    results = []
    def check(name: str, hf_path: str, rlx_path: str):
        hf_p = ref_dir / hf_path
        rlx_p = ref_dir / rlx_path
        if not (hf_p.is_file() and rlx_p.is_file()):
            print(f"[{name}]  skipped (missing: hf={hf_p.is_file()} rlx={rlx_p.is_file()})")
            return
        hf = np.load(hf_p).astype(np.float32)
        rlx = np.fromfile(rlx_p, dtype=np.float32)
        results.append((name, diff(name, hf, rlx)))

    check("encoder", "hidden_states.npy", "hidden_states_rlx.bin")
    check("pooler",  "pooler_output.npy", "pooler_output_rlx.bin")
    check("mlm",     "mlm_logits.npy",    "mlm_logits_rlx.bin")

    print()
    bad = [n for n, ok in results if not ok]
    if not results:
        print("VERDICT: no comparisons run")
        return 1
    if bad:
        print(f"VERDICT: FAIL on {bad}")
        return 2
    print("VERDICT: PARITY across all checked outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
