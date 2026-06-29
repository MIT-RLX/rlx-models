#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Single-sample AIF probe (used by RLX when RLX_QWEN25_VL_AIF_PYTHON=1).
#
# Env:
#   RLX_QWEN25_VL_HF_DIR / RLX_QWEN25_VL_DOWNLOAD
#   RLX_QWEN25_VL_IMAGE
#   RLX_QWEN25_VL_PROMPT   user question text
#   RLX_QWEN25_VL_OUT_DIR  output directory
#   RLX_QWEN25_VL_SAMPLE_ID optional prefix (default "sample")
#   RLX_QWEN25_VL_DEVICE   cpu | cuda | mps

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "crates/rlx-models/tests/qwen25_vl_parity_helpers"))

from aif_probe import resolve_model_dir, run_hf_prefill_probe, save_probe_sample  # noqa: E402


def main() -> None:
    image = os.environ["RLX_QWEN25_VL_IMAGE"]
    out_dir = os.environ["RLX_QWEN25_VL_OUT_DIR"]
    question = os.environ.get("RLX_QWEN25_VL_PROMPT", "Describe this image.")
    sample_id = os.environ.get("RLX_QWEN25_VL_SAMPLE_ID", "sample")
    device = os.environ.get("RLX_QWEN25_VL_DEVICE", "cpu")
    vlmevalkit = os.environ.get("RLX_QWEN25_VL_VLMEVALKIT", "0") == "1"

    result = run_hf_prefill_probe(
        model_dir=resolve_model_dir(),
        image_path=image,
        question=question,
        device=device,
        vlmevalkit=vlmevalkit,
    )
    save_probe_sample(out_dir, sample_id, result)
    ratio = result.get("probe", {}).get("mask_ratio")
    print(f"probe ok sample={sample_id} ratio={ratio}")


if __name__ == "__main__":
    main()
